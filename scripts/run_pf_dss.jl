# --- Activate project ---
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using OpenDSSDirect
using PowerModelsDistribution
using Ipopt
import MathOptInterface as MOI
using JSON3
using Statistics
using Logging
using Dates
using LinearAlgebra: norm   # ← 某些 PMD 变换会用到
# EN: Some PMD transformations may use `norm`.

# =========================================================
# PF Runner for OpenDSS / PMD (Julia side)
# ---------------------------------------------------------
# 设计目标：
# EN: Design goals:
# - OpenDSSDirect：确定性快照潮流，支持注入/电压汇总/线路潮流
# EN: OpenDSSDirect: deterministic snapshot power flow with support for injections / voltage summary / line flows.
# - PMD：优化型潮流（鲁棒），保留模型遍历、状态摘要与电压摘要
# EN: PMD: optimization-based (robust) power flow; keep model sweep, status summary, and voltage summary.
# - 始终输出 JSON：对 Symbol / 枚举 / 非有限浮点做统一清洗，外层兜底
# EN: Always output JSON: sanitize Symbols / enums / non-finite floats; robust fallback at top level.
# =========================================================

# 仅输出 Warning 以上，避免污染 stdout
# EN: Only print warnings or higher to avoid polluting stdout.
global_logger(SimpleLogger(stderr, Logging.Warn))

# ---------- JSON sanitize ----------
# 递归将对象转成“可被 JSON3 写出”的值：
# EN: Recursively convert objects into values that JSON3 can serialize:
# - 非有限浮点 → nothing
# EN: - Non-finite floats → nothing
# - Missing/Nothing → nothing
# EN: - Missing / Nothing → nothing
# - Symbol / MOI.TerminationStatusCode → string(x)
# EN: - Symbol / MOI.TerminationStatusCode → string(x)
# - Dict/Vector 递归处理，字典键统一转成 String
# EN: - Dict/Vector handled recursively; dict keys coerced to String.
json_sanitize(x::AbstractFloat) = isfinite(x) ? x : nothing
json_sanitize(x::Missing) = nothing
json_sanitize(x::Nothing) = nothing
json_sanitize(x::Integer) = x
json_sanitize(x::Bool) = x
json_sanitize(x::AbstractString) = x
json_sanitize(x::Symbol) = string(x)
json_sanitize(x::MOI.TerminationStatusCode) = string(x)
json_sanitize(v::AbstractVector) = [json_sanitize(e) for e in v]
function json_sanitize(d::AbstractDict)
    out = Dict{String,Any}()
    for (k,v) in d
        out[string(k)] = json_sanitize(v)
    end
    out
end
json_sanitize(x) = try string(x) catch; "<?>"; end  # 终极兜底
# EN: Last-resort fallback: stringify anything else.

# 统一输出（可写文件或 stdout），并对编码做兜底，保证永远输出 JSON
# EN: Unified output (to file or stdout). Encoding-safe fallback to guarantee JSON is always produced.
function output_json(obj, out_path::AbstractString)
    local s::String
    try
        s = JSON3.write(json_sanitize(obj))
    catch e
        s = JSON3.write(json_sanitize(Dict(
            "status" => "error",
            "error_type" => "JSON_ENCODE_ERROR",
            "detail" => sprint(showerror, e)
        )))
    end
    if !isempty(out_path)
        open(out_path, "w") do io; write(io, s); end
    else
        println(s)
    end
end

# ==================== OpenDSSDirect branch ====================

# 若目标母线无既有 Load，则临时新建三相 wye 负荷
# EN: If the target bus has no existing Load, create a temporary three-phase wye load.
function apply_injections!(injections::AbstractDict{<:AbstractString,<:Any})
    function _ensure_load_on_bus!(bus::AbstractString, pkW::Float64, qkVAr::Float64)
        try
            OpenDSSDirect.Circuit.SetActiveBus(bus)
        catch
            return false
        end
        kvbase_ll = try OpenDSSDirect.Bus.kVBase() catch; 0.0 end
        if !(isfinite(kvbase_ll)) || kvbase_ll <= 0
            kvbase_ll = 0.4
        end
        kv_ln = kvbase_ll / sqrt(3)
        safe_bus = replace(String(bus), "." => "_", " " => "_")
        lname = "inj_" * safe_bus
        OpenDSSDirect.Text.Command("New Load.$lname bus1=$bus phases=3 conn=wye kV=$(kv_ln) kW=$(pkW) kvar=$(qkVAr)")
        return true
    end

    # 采集各母线已有负荷
    # EN: Collect existing loads per bus.
    loads_on_bus = Dict{String, Vector{String}}()
    i = OpenDSSDirect.Loads.First()
    while i > 0
        lname = OpenDSSDirect.Loads.Name()
        OpenDSSDirect.Circuit.SetActiveElement("Load." * lname)
        bns = OpenDSSDirect.CktElement.BusNames()
        rawbus = isempty(bns) ? "" : split(bns[1], '.')[1]
        if !isempty(rawbus)
            get!(loads_on_bus, rawbus, String[]) |> push!(lname)
        end
        i = OpenDSSDirect.Loads.Next()
    end

    # 施加注入：若该母线有多个负荷，等比例平摊 ΔP/ΔQ
    # EN: Apply injections: if a bus has multiple loads, distribute ΔP/ΔQ proportionally (evenly).
    for (bus, inj) in injections
        if inj === nothing || length(inj) < 2; continue; end
        ΔP = Float64(inj[1]); ΔQ = Float64(inj[2])
        if haskey(loads_on_bus, bus) && !isempty(loads_on_bus[bus])
            names = loads_on_bus[bus]; n = max(length(names), 1)
            for lname in names
                OpenDSSDirect.Circuit.SetActiveElement("Load." * lname)
                kw   = OpenDSSDirect.Loads.kW()   + ΔP / n
                kvar = OpenDSSDirect.Loads.kvar() + ΔQ / n
                OpenDSSDirect.Loads.kW(kw)
                OpenDSSDirect.Loads.kvar(kvar)
            end
        else
            _ = _ensure_load_on_bus!(bus, ΔP, ΔQ)
        end
    end
end

# 用 puVmagAngle 直接拿到 p.u. 电压
# EN: Use `puVmagAngle` to directly obtain per-unit voltages.
function collect_bus_voltages()
    bus_res = Dict{String,Any}()
    for bus in OpenDSSDirect.Circuit.AllBusNames()
        OpenDSSDirect.Circuit.SetActiveBus(bus)
        va    = OpenDSSDirect.Bus.VMagAngle()
        pu_va = OpenDSSDirect.Bus.puVmagAngle()
        phases = Int(length(va) ÷ 2)
        arr = Vector{Any}()
        for i in 1:phases
            v      = va[2*i-1]
            ang    = va[2*i]
            vpu    = pu_va[2*i-1]
            push!(arr, Dict("phase"=>i, "V"=>v, "angle_deg"=>ang, "Vpu"=>vpu))
        end
        bus_res[bus] = arr
    end
    bus_res
end

# 线路潮流（按端聚合）
# EN: Line power flows (aggregated per terminal).
function collect_line_flows()
    res = Dict{String,Any}()
    i = OpenDSSDirect.Lines.First()
    while i > 0
        lname = OpenDSSDirect.Lines.Name()
        name  = "line." * lname
        OpenDSSDirect.Circuit.SetActiveElement(name)

        nterms = try OpenDSSDirect.CktElement.NumTerminals() catch; 0 end
        pqs    = try OpenDSSDirect.CktElement.Powers() catch; Float64[] end  # [P1,Q1,P2,Q2,...]
        npairs = Int(length(pqs) ÷ 2)

        p1 = 0.0; q1 = 0.0
        p2 = 0.0; q2 = 0.0
        if npairs > 0
            per_term = nterms > 0 ? max(1, Int(floor(npairs / nterms))) : npairs
            for idx in 1:min(per_term, npairs)
                p1 += float(pqs[2*idx-1]); q1 += float(pqs[2*idx])
            end
            if nterms >= 2 && npairs >= 2*per_term
                base = per_term
                for idx in 1:per_term
                    p2 += float(pqs[2*(base+idx)-1]); q2 += float(pqs[2*(base+idx)])
                end
            end
        end
        res[name] = Dict(
            "term1"=>Dict("P_kW"=>p1, "Q_kvar"=>q1),
            "term2"=>Dict("P_kW"=>p2, "Q_kvar"=>q2)
        )
        i = OpenDSSDirect.Lines.Next()
    end
    return res
end

# 汇总（与 Python 端格式对齐）
# EN: Summarize results (aligned with Python-side schema).
function make_summary(bus_voltages::Dict{String,Any}; vmin=0.95, vmax=1.05)
    per_bus = Dict{String,Any}()
    global_min = +Inf; worst_bus = ""
    for (bus, arr) in bus_voltages
        vpus = [x["Vpu"] for x in arr if !(x["Vpu"] isa Float64 && isnan(x["Vpu"]))]
        if isempty(vpus)
            per_bus[bus] = Dict("min_vpu"=>NaN, "max_vpu"=>NaN, "under_limit"=>false, "over_limit"=>false)
            continue
        end
        mn = minimum(vpus); mx = maximum(vpus)
        under = any(x -> x < vmin, vpus)
        over  = any(x -> x > vmax, vpus)
        per_bus[bus] = Dict("min_vpu"=>mn, "max_vpu"=>mx, "under_limit"=>under, "over_limit"=>over)
        if mn < global_min
            global_min = mn; worst_bus = bus
        end
    end

    P_loss_kW = 0.0; Q_loss_kvar = 0.0
    L = OpenDSSDirect.Circuit.Losses()
    if L isa Complex
        P_loss_kW = real(L) / 1000.0
        Q_loss_kvar = imag(L) / 1000.0
    elseif L isa AbstractVector && length(L) >= 2
        P_loss_kW = float(L[1]) / 1000.0
        Q_loss_kvar = float(L[2]) / 1000.0
    end

    Dict(
        "voltage_summary_per_bus" => per_bus,
        "worst_bus_min_vpu" => Dict("bus"=>worst_bus, "min_vpu"=>global_min),
        "system_losses" => Dict("P_loss_kW"=>P_loss_kW, "Q_loss_kvar"=>Q_loss_kvar),
        "limits" => Dict("vmin"=>vmin, "vmax"=>vmax)
    )
end

# ==================== PMD branch（鲁棒求解 + 电压摘要） ====================
# EN: PMD branch (robust solve + voltage summary).
const PMD = PowerModelsDistribution

# 枚举可用模型符号
# EN: Enumerate available model symbols.
function _available_model_syms()
    [s for s in names(PMD) if occursin("PowerModel", string(s))]
end

# 排序：常用 → 其它；EN（带显式中性线）放后
# EN: Ordering: common → others; EN (explicit neutral) placed later.
function _ordered_models()
    syms = _available_model_syms()
    pick(pat) = [x for x in syms if occursin(pat, string(x))]
    bf   = pick(r"^BF.*PowerModel$")
    acru = pick(r"^ACRU.*PowerModel$")
    acpu = vcat(pick(r"^ACPU.*PowerModel$"), pick(r"^ACPP.*PowerModel$"))
    ivru = pick(r"^IVRU.*PowerModel$")
    others = [x for x in syms if !(x in bf) && !(x in acru) && !(x in acpu) && !(x in ivru)]
    en = [x for x in vcat(bf,acru,acpu,ivru,others) if occursin("EN", string(x))]
    non_en = [x for x in vcat(bf,acru,acpu,ivru,others) if !occursin("EN", string(x))]

    seen = Set{Symbol}(); out = Symbol[]
    for x in vcat(non_en, en)
        if !(x in seen); push!(out, x); push!(seen, x); end
    end
    out
end

# 规范化：把任何“数组/字典风格”的数值拿成向量
# EN: Normalize: coerce any array/dict-style numeric value into a Vector.
_as_vec(x) = x isa AbstractVector ? [try Float64(v) catch; NaN end for v in x] :
             x isa AbstractDict && haskey(x,"values") ? _as_vec(x["values"]) :
             Float64[]

# 从 PMD 结果摘取解向量
# EN: Extract solution vectors from PMD results.
function _pmd_pick_solution(res)
    if haskey(res, "solution") && res["solution"] isa AbstractDict
        return res["solution"]
    elseif haskey(res, "nw") && res["nw"] isa AbstractDict && !isempty(res["nw"])
        first_key = first(keys(res["nw"]))
        return res["nw"][first_key]["solution"]
    end
    return Dict{String,Any}()
end

function pmd_collect_bus_vm(res)::Dict{String,Vector{Float64}}
    # --- 0) 直接处理顶层 bus_vm_pu（你当前 PMD 返回的就是这个） ---
    # EN: (0) Directly handle top-level `bus_vm_pu` (current PMD may already return this).
    if haskey(res, "bus_vm_pu")
        vmmap = get(res, "bus_vm_pu", Dict{String,Any}())
        out = Dict{String,Vector{Float64}}()
        for (b, v_any) in vmmap
            out[string(b)] = _as_vec(v_any)
        end
        return out
    end

    # --- 1) 常见的单场景解：res["solution"]["bus"][bus]["vm"/"vm_pu"] ---
    # EN: (1) Common single-scenario solution: res["solution"]["bus"][bus]["vm" | "vm_pu"].
    sol = _pmd_pick_solution(res)
    if !isempty(sol)
        buses = get(sol, "bus", Dict{String,Any}())
        out = Dict{String,Vector{Float64}}()
        for (b, props_any) in buses
            props = props_any isa AbstractDict ? props_any::AbstractDict : Dict{String,Any}()
            v = haskey(props, "vm")    ? _as_vec(props["vm"]) :
                haskey(props, "vm_pu") ? _as_vec(props["vm_pu"]) : Float64[]
            # 回退：vr/vi → |V|
            # EN: Fallback: derive |V| from vr/vi.
            if isempty(v)
                vr = haskey(props,"vr") ? _as_vec(props["vr"]) : Float64[]
                vi = haskey(props,"vi") ? _as_vec(props["vi"]) : Float64[]
                if !isempty(vr) && !isempty(vi)
                    n = min(length(vr), length(vi))
                    v = [ hypot(vr[i], vi[i]) for i in 1:n ]
                end
            end
            out[string(b)] = v
        end
        return out
    end

    # --- 2) 兜底 ---
    # EN: (2) Fallback.
    return Dict{String,Vector{Float64}}()
end


function make_summary_from_pmd_vm(vm_map::Dict{String,Vector{Float64}}; vmin=0.95, vmax=1.05)
    per_bus = Dict{String,Any}()
    global_min = +Inf
    worst_bus = ""
    for (bus, vlist) in vm_map
        vgood = [v for v in vlist if isfinite(v)]
        if isempty(vgood)
            per_bus[bus] = Dict("min_vpu"=>NaN, "max_vpu"=>NaN, "under_limit"=>false, "over_limit"=>false)
            continue
        end
        mn = minimum(vgood); mx = maximum(vgood)
        under = any(<(vmin), vgood)  # any v < vmin
        over  = any(>(vmax), vgood)  # any v > vmax
        per_bus[bus] = Dict("min_vpu"=>mn, "max_vpu"=>mx, "under_limit"=>under, "over_limit"=>over)
        if mn < global_min
            global_min = mn; worst_bus = bus
        end
    end
    return Dict(
        "voltage_summary_per_bus" => per_bus,
        "worst_bus_min_vpu" => Dict("bus"=>worst_bus, "min_vpu"=>global_min),
        "system_losses" => Dict("P_loss_kW"=>nothing, "Q_loss_kvar"=>nothing)  # 需要的话后续再补
        # EN: System losses could be added later if needed.
    )
end

# 单次尝试：求解并规范化易炸字段
# EN: Single attempt: solve and normalize fields that are brittle to serialize.
function _try_solve_once(data::Dict{String,Any}, model_sym::Symbol)
    mtype = try getfield(PMD, model_sym) catch; nothing end
    if mtype === nothing
        return Dict("termination_status"=>"MODEL_NOT_FOUND", "model"=>string(model_sym))
    end
    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level"    => 0,
        "sb"             => "yes",
        "tol"            => 1e-6,
        "acceptable_tol" => 1e-4,
        "max_iter"       => 500
    )
    try
        t0 = time()
        res = if isdefined(PMD, :solve_mc_pf)
            PMD.solve_mc_pf(data, mtype, solver)
        elseif isdefined(PMD, :run_mc_pf)
            PMD.run_mc_pf(data, mtype, PMD.with_optimizer(solver))
        else
            Dict("termination_status"=>"NO_SOLVE_API")
        end
        # 统一转字符串，避免 JSON3 推断类型炸栈
        # EN: Convert to strings to avoid JSON3 type inference issues.
        if haskey(res, "termination_status"); res["termination_status"] = string(res["termination_status"]); end
        if haskey(res, "model");              res["model"]              = string(res["model"]);              end
        res["model"] = string(model_sym)
        res["solver_time_sec"] = time() - t0
        return res
    catch e
        return Dict(
            "termination_status"=>"ERROR",
            "model"=>string(model_sym),
            "error"=>sprint(showerror, e)
        )
    end
end

# 从 DSS 构建 PMD 数据，附注入与电压边界
# EN: Build PMD data from DSS; apply injections and voltage bounds.
function _build_data(master_dss::String, injections::AbstractDict{<:AbstractString,<:Any};
                     vm_lb::Float64, vm_ub::Float64, sbase::Float64)
    data = PMD.parse_file(
        master_dss;
        transformations = [
            PMD.make_lossless!,
            (PMD.apply_voltage_bounds!, "vm_lb"=>vm_lb, "vm_ub"=>vm_ub)
        ]
    )
    get!(data, "settings", Dict{String,Any}())
    data["settings"]["sbase_default"] = sbase

    if haskey(data, "load") && !isempty(injections)
        for (_lid, ld) in data["load"]
            bus = string(get(ld, "bus", ""))
            if !isempty(bus) && haskey(injections, bus)
                inj = injections[bus]
                if length(inj) == 2
                    ld["pd"] = get(ld, "pd", 0.0) + Float64(inj[1]) / 1000.0  # kW -> MW
                    ld["qd"] = get(ld, "qd", 0.0) + Float64(inj[2]) / 1000.0  # kvar -> MVAr
                end
            end
        end
    end
    data
end

# 多重容错求解：边界×基值×模型；成功即停，失败返回最后尝试结果
# EN: Multi-layer fault-tolerant solve: bounds × sbase × models; stop on first success, otherwise return last attempt.
function run_pf_with_pmd(master_dss::String, injections::AbstractDict{<:AbstractString,<:Any})
    bounds = [(0.90,1.10), (0.80,1.20)]
    sbases = [1.0, 10.0]
    models = _ordered_models()
    tried = Vector{Dict{String,Any}}()
    last_res = Dict{String,Any}()
    last_used = Dict{String,Any}()

    for (lb,ub) in bounds
        for sbase in sbases
            data = _build_data(master_dss, injections; vm_lb=lb, vm_ub=ub, sbase=sbase)
            last_used = Dict("vm_lb"=>lb,"vm_ub"=>ub,"sbase"=>sbase)
            for ms in models
                res = _try_solve_once(data, ms)
                push!(tried, Dict("model"=>get(res, "model", "?"), "status"=>string(get(res, "termination_status", ""))))
                last_res = res
                ts = string(get(res, "termination_status", ""))
                if ts in ("LOCALLY_SOLVED","OPTIMAL","SUCCESS")
                    res["tried_models"] = tried
                    res["used_bounds"] = last_used
                    return res
                end
            end
        end
    end
    last_res["tried_models"] = tried
    last_res["used_bounds"] = last_used
    return last_res
end

# ==================== Main ====================
"""
Usage: julia run_pf_dss.jl <master_dss> [injections_json] [mode] [engine] [out_path]
- master_dss: OpenDSS master 文件路径
- injections_json: 如 {"bus":[ΔP_kW,ΔQ_kVAr], ...}
- mode: "raw" | "summary"
- engine: "opendss" | "pmd"
- out_path: 如提供则写文件，否则打印到 stdout
"""
# EN: Do not modify the docstring above; it documents CLI usage in Chinese.
function main()
    if length(ARGS) < 1
        println("Usage: julia run_pf_dss.jl <master_dss> [injections_json] [mode] [engine] [out_path]")
        return
    end

    master   = replace(ARGS[1], "\\" => "/")
    injections_json = length(ARGS) >= 2 ? ARGS[2] : "{}"
    mode     = length(ARGS) >= 3 ? String(ARGS[3]) : "raw"
    engine   = length(ARGS) >= 4 ? String(ARGS[4]) : "opendss"
    out_path = length(ARGS) >= 5 ? String(ARGS[5]) : ""

    inj_raw = try JSON3.read(injections_json) catch; nothing end
    injections = Dict{String,Any}()
    if inj_raw !== nothing
        for (k,v) in pairs(inj_raw); injections[string(k)] = v; end
    end

    if engine == "pmd"
        # —— PMD：先求解，再尽量给出电压摘要（若有解向量） —— #
        # EN: PMD: first solve, then provide a voltage summary if a solution vector exists.
       # —— PMD：先求解 —— 
       # EN: PMD: solve first.
        res = run_pf_with_pmd(master, injections)

        # 只做可行性与元信息，不做电压统计（避免未归一化的数值误导）
        # EN: Report feasibility and metadata only; skip voltage stats to avoid misleading non-normalized values.
        ts_str  = try string(get(res, "termination_status", nothing)) catch; nothing end
        mdl_str = try string(get(res, "model", nothing))              catch; nothing end

        usedb = try get(res, "used_bounds", nothing) catch; nothing end
        usedb_sanitized = usedb === nothing ? nothing : Dict(
            "vm_lb" => get(usedb, "vm_lb", nothing),
            "vm_ub" => get(usedb, "vm_ub", nothing),
            "sbase" => get(usedb, "sbase", nothing)
        )

        tried = try get(res, "tried_models", nothing) catch; nothing end
        tried_flat = tried === nothing ? nothing :
            [ string(get(t, "model", "?")) * ":" * string(get(t, "status", "?")) for t in tried ]

        obj_val = try
            v = get(res, "objective", nothing)
            v isa Number ? v : nothing
        catch; nothing end

        summ = Dict(
            "termination_status" => ts_str,
            "model"              => mdl_str,
            "objective"          => obj_val,
            "used_bounds"        => usedb_sanitized,
            "tried_models"       => tried_flat,
            "solver_time_sec"    => get(res, "solver_time_sec", nothing),
        )

        if mode == "summary"
            output_json(Dict("summary"=>summ), out_path)
        else
            # raw：仅返回可行性与(若有)未归一的电压向量，字段名保持 bus_vm_pu，供后续校准
            # EN: For "raw": return feasibility and (if present) unnormalized voltage vectors under `bus_vm_pu` for later calibration.
            raw_out = Dict(
                "termination_status" => ts_str,
                "model"              => mdl_str,
                "used_bounds"        => usedb_sanitized,
                "tried_models"       => tried_flat,
                "solver_time_sec"    => get(res, "solver_time_sec", nothing),
            )
            # 尝试附上 bus_vm_pu（不参与统计）
            # EN: Attempt to attach `bus_vm_pu` (excluded from statistics).
            vm_map = pmd_collect_bus_vm(res)
            if !isempty(vm_map)
                raw_out["bus_vm_pu"] = vm_map
            end
            output_json(raw_out, out_path)
        end

        return
    end
end

# Entrypoint: never leave stdout empty
try
    main()
catch e
    output_json(Dict(
        "status" => "error",
        "error_type" => string(typeof(e)),
        "detail" => sprint(showerror, e)
    ), "")
end
