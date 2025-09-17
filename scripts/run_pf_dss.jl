# --- Activate project ---
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using OpenDSSDirect
using PowerModelsDistribution
using Ipopt
using JSON3
using Statistics
using Logging
using Dates
using LinearAlgebra: norm   # ← 必须在顶层

# =========================================================
# PF Runner for OpenDSS / PMD (Julia side)
# ---------------------------------------------------------
# This script is invoked by a Python FastAPI service as a
# subprocess. It compiles an OpenDSS master file, optionally
# applies load injections, runs a power flow either via:
#   - OpenDSSDirect (deterministic, snapshot PF), or
#   - PowerModelsDistribution (optimization-based PF),
# and returns JSON to stdout or a given output file.
#
# Design goals:
# - No noisy stdout: only emit structured JSON unless errors.
# - Robust JSON: sanitize NaN/±Inf/Missing to JSON null.
# - Defensive parsing: tolerate missing fields/odd returns.
# - PMD branch tries multiple model classes and settings to
#   improve solve robustness; it never throws—returns status.
# =========================================================

# Only output warnings and above to stderr to avoid polluting stdout
global_logger(SimpleLogger(stderr, Logging.Warn))

# ---------- JSON sanitize ----------
# Map various Julia types into JSON-safe values.
# - Non-finite floats -> nothing (null)
# - Missing/Nothing -> nothing (null)
# - Vectors/Dicts -> recursively sanitize
# - Fallback -> string(x) (never throw)
json_sanitize(x::AbstractFloat) = isfinite(x) ? x : nothing
json_sanitize(x::Missing) = nothing
json_sanitize(x::Nothing) = nothing
json_sanitize(x::Integer) = x
json_sanitize(x::Bool) = x
json_sanitize(x::AbstractString) = x
json_sanitize(v::AbstractVector) = [json_sanitize(e) for e in v]
function json_sanitize(d::AbstractDict)
    out = Dict{String,Any}()
    for (k,v) in d
        out[string(k)] = json_sanitize(v)   # stringify keys to be safe for JSON
    end
    out
end
json_sanitize(x) = try string(x) catch; "<?>"; end  # ultimate fallback

# Unified output: if out_path is given, write there; else print to stdout
function output_json(obj, out_path::AbstractString)
    s = JSON3.write(json_sanitize(obj))
    if !isempty(out_path)
        open(out_path, "w") do io; write(io, s); end
    else
        println(s)
    end
end

# ==================== OpenDSSDirect branch ====================
# 改造点：若目标母线没有既有 Load，则自动创建一个临时 Load.inj_<bus>（三相wye，kV=LL/√3）
# Apply/realize bus-level injections. If a bus has existing Loads, add deltas.
# If not, create a temporary Load.inj_<bus> connected wye with a reasonable kV.
# injections: Dict{String, Any} like bus => [ΔP_kW, ΔQ_kVAr]
function apply_injections!(injections::AbstractDict{<:AbstractString,<:Any})
    # Helper: ensure there is at least one load on a bus; if none, create one.
    function _ensure_load_on_bus!(bus::AbstractString, pkW::Float64, qkVAr::Float64)
        # Activate bus, get base voltage
        try
            OpenDSSDirect.Circuit.SetActiveBus(bus)
        catch
            return false
        end
        kvbase_ll = try
            OpenDSSDirect.Bus.kVBase()
        catch
            0.0
        end
        # Default if unknown: assume LV 0.4 kV LL
        if !(isfinite(kvbase_ll)) || kvbase_ll <= 0
            kvbase_ll = 0.4
        end
        # For wye-connected load, OpenDSS expects kV (LN). Convert LL -> LN.
        kv_ln = kvbase_ll / sqrt(3)

        # Create a unique load name; sanitize bus name
        safe_bus = replace(String(bus), "." => "_", " " => "_")
        lname = "inj_" * safe_bus

        # Create/overwrite an injection load at this bus
        OpenDSSDirect.Text.Command("New Load.$lname bus1=$bus phases=3 conn=wye kV=$(kv_ln) kW=$(pkW) kvar=$(qkVAr)")
        return true
    end

    # Index existing loads by their raw bus (left of the dot)
    loads_on_bus = Dict{String, Vector{String}}()
    i = OpenDSSDirect.Loads.First()
    while i > 0
        lname = OpenDSSDirect.Loads.Name()
        OpenDSSDirect.Circuit.SetActiveElement("Load." * lname)

        # BusNames() returns e.g. ["632.1.2.3"]; take the part before the dot
        bns = OpenDSSDirect.CktElement.BusNames()
        rawbus = isempty(bns) ? "" : split(bns[1], '.')[1]

        if !isempty(rawbus)
            get!(loads_on_bus, rawbus, String[]) |> push!(lname)
        end
        i = OpenDSSDirect.Loads.Next()
    end

    # Apply each injection
    for (bus, inj) in injections
        if inj === nothing || length(inj) < 2; continue; end
        ΔP = Float64(inj[1]); ΔQ = Float64(inj[2])

        if haskey(loads_on_bus, bus) && !isempty(loads_on_bus[bus])
            # Add deltas to all loads on this bus proportionally (equal split)
            names = loads_on_bus[bus]
            n = max(length(names), 1)
            for lname in names
                OpenDSSDirect.Circuit.SetActiveElement("Load." * lname)
                kw   = OpenDSSDirect.Loads.kW()   + ΔP / n
                kvar = OpenDSSDirect.Loads.kvar() + ΔQ / n
                OpenDSSDirect.Loads.kW(kw)
                OpenDSSDirect.Loads.kvar(kvar)
            end
        else
            # No loads on this bus: create a temporary injection load
            _ = _ensure_load_on_bus!(bus, ΔP, ΔQ)
        end
    end
end

# Voltage collection:
# Use OpenDSS's puVmagAngle() to directly obtain per-unit magnitudes,
# avoiding ambiguity with LN/LL base voltages.
function collect_bus_voltages()
    bus_res = Dict{String,Any}()
    for bus in OpenDSSDirect.Circuit.AllBusNames()
        OpenDSSDirect.Circuit.SetActiveBus(bus)

        # VMagAngle: [V1,deg1,V2,deg2,...] in physical volts
        va    = OpenDSSDirect.Bus.VMagAngle()

        # puVmagAngle: [Vpu1,deg1,Vpu2,deg2,...] directly in p.u.
        pu_va = OpenDSSDirect.Bus.puVmagAngle()

        phases = Int(length(va) ÷ 2)
        # 用 length(va) ÷ 2 就能得到母线上有多少相（phases）。
        arr = Vector{Any}()
        for i in 1:phases
            v      = va[2*i-1]
            ang    = va[2*i]
            vpu    = pu_va[2*i-1]
            push!(arr, Dict("phase"=>i, "V"=>v, "angle_deg"=>ang, "Vpu"=>vpu))
        end
        # 初始化一个空数组 arr，用来存储每相的电压结果。
        bus_res[bus] = arr
    end
    bus_res
end

# Robust line flow aggregation:
# OpenDSSDirect.CktElement.Powers() returns a flattened [P1,Q1,P2,Q2,...]
# list over all terminals/phases. This function aggregates by terminal
# defensively, avoiding out-of-bounds when lengths mismatch metadata.
function collect_line_flows()
    res = Dict{String,Any}()

    i = OpenDSSDirect.Lines.First()
    while i > 0
        lname = OpenDSSDirect.Lines.Name()
        name  = "line." * lname

        # Set active element so Powers() reads the correct component
        OpenDSSDirect.Circuit.SetActiveElement(name)

        nterms = try OpenDSSDirect.CktElement.NumTerminals() catch; 0 end
        pqs    = try OpenDSSDirect.CktElement.Powers() catch; Float64[] end  # [P1,Q1,P2,Q2,...]
        npairs = Int(length(pqs) ÷ 2)

        p1 = 0.0; q1 = 0.0
        p2 = 0.0; q2 = 0.0

        if npairs > 0
            # Estimate pairs per terminal; ensure at least 1
            per_term = nterms > 0 ? max(1, Int(floor(npairs / nterms))) : npairs

            # Aggregate terminal 1
            for idx in 1:min(per_term, npairs)
                p1 += float(pqs[2*idx-1]); q1 += float(pqs[2*idx])
            end

            # Aggregate terminal 2 if present and sufficient pairs exist
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

# Summarization:
# - Compute per-bus min/max Vpu and flags for under/over limits.
# - Convert system Losses() from W/var to kW/kvar.
function make_summary(bus_voltages::Dict{String,Any}; vmin=0.95, vmax=1.05)
    per_bus = Dict{String,Any}()
    global_min = +Inf; worst_bus = ""
    for (bus, arr) in bus_voltages
        vpus = [x["Vpu"] for x in arr if !isnan(x["Vpu"])]
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

    # Losses(): some builds return Complex, others return a 2-length vector
    P_loss_kW = 0.0; Q_loss_kvar = 0.0
    L = OpenDSSDirect.Circuit.Losses()
    if isa(L, Complex)
        P_loss_kW = real(L) / 1000.0
        Q_loss_kvar = imag(L) / 1000.0
    elseif isa(L, AbstractVector) && length(L) >= 2
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

# ==================== PMD branch (auto model search + fallbacks) ====================
const PMD = PowerModelsDistribution

# Enumerate available PowerModel symbols exported by PMD (e.g., ACRUPowerModel, ACPU…, BF…)
function _available_model_syms()
    [s for s in names(PMD) if occursin("PowerModel", string(s))]
end

# Order candidate models:
# - Prefer common formulations (BF/ACRU/ACPU/IVRU)
# - Defer models with "EN" (ExplicitNeutral) to the end as they are more restrictive
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

    # Deduplicate while preserving order
    seen = Set{Symbol}(); out = Symbol[]
    for x in vcat(non_en, en)
        if !(x in seen); push!(out, x); push!(seen, x); end
    end
    out
end

# Single attempt with a given model symbol:
# - Builds an Ipopt optimizer with silent output.
# - Uses PMD.solve_mc_pf (or run_mc_pf for older PMD versions).
# - Returns a JSON-like Dict with status; does NOT throw.
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
        res["model"] = string(model_sym)
        res["solver_time_sec"] = time() - t0
        return res
    catch e
        # Trap any error and return it as a JSON field instead of throwing
        return Dict(
            "termination_status"=>"ERROR",
            "model"=>string(model_sym),
            "error"=>sprint(showerror, e)
        )
    end
end

# Build PMD data from an OpenDSS master, with transformations and optional injections:
# - make_lossless!: remove line shunts for numerical stability
# - apply_voltage_bounds!: enforce bounds in the model
# - sbase_default: set per-unit base for PMD
# - injections: adjust load pd/qd (kW/kvar -> MW/MVAr for PMD data)
function _build_data(master_dss::String, injections::AbstractDict{<:AbstractString,<:Any}; vm_lb::Float64, vm_ub::Float64, sbase::Float64)
    data = PMD.parse_file(
        master_dss;
        transformations = [
            PMD.make_lossless!,
            (PMD.apply_voltage_bounds!, "vm_lb"=>vm_lb, "vm_ub"=>vm_ub)
        ]
    )

    # Ensure settings sub-dict exists; set sbase default
    get!(data, "settings", Dict{String,Any}())
    data["settings"]["sbase_default"] = sbase

    # Apply injections by bus if provided (convert kW/kvar to MW/MVAr)
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

# Public PMD solve wrapper:
# - Iterates across voltage bounds (tight -> wide), then sbase choices (small -> large),
#   then candidate models. Stops at first "good" termination status.
# - On total failure, returns the last result plus diagnostics (no exception).
function run_pf_with_pmd(master_dss::String, injections::AbstractDict{<:AbstractString,<:Any})
    bounds = [(0.90,1.10), (0.80,1.20)]       # Add wider ranges if needed, e.g., (0.70,1.30)
    sbases = [1.0, 10.0]                      # Try smaller then larger base
    models = _ordered_models()                # Auto-ordered model list (EN last)
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
    # If all attempts fail, return last result enriched with diagnostics
    last_res["tried_models"] = tried
    last_res["used_bounds"] = last_used
    return last_res
end
# === PMD voltage extraction & summary (add this) ===
function _pmd_pick_solution(res)
    if haskey(res, "solution") && res["solution"] isa AbstractDict
        return res["solution"]
    elseif haskey(res, "nw") && res["nw"] isa AbstractDict && !isempty(res["nw"])
        # pick first network if multi-scenario
        first_key = first(keys(res["nw"]))
        return res["nw"][first_key]["solution"]
    end
    return Dict{String,Any}()
end

# normalize any array-like to Float64 vector
_as_vec(x) = x isa AbstractVector ? [try Float64(v) catch; NaN end for v in x] :
             x isa AbstractDict && haskey(x,"values") ? _as_vec(x["values"]) :
             Float64[]

function pmd_collect_bus_vm(res)::Dict{String,Vector{Float64}}
    sol = _pmd_pick_solution(res)
    buses = get(sol, "bus", Dict{String,Any}())
    out = Dict{String,Vector{Float64}}()
    for (b, props) in buses
        # common keys seen in PMD: "vm" (per-unit magnitudes), sometimes nested
        vm = haskey(props, "vm") ? _as_vec(props["vm"]) :
             haskey(props, "vm_pu") ? _as_vec(props["vm_pu"]) : Float64[]
        out[string(b)] = vm
    end
    return out
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
        "system_losses" => Dict("P_loss_kW"=>nothing, "Q_loss_kvar"=>nothing),  # 可选：后续再加
    )
end

# ==================== Main ====================
"""
Usage: julia run_pf_dss.jl <master_dss> [injections_json] [mode] [engine] [out_path]
- master_dss: path to OpenDSS master file
- injections_json: JSON string like {"632":[ΔP_kW,ΔQ_kVAr], ...}
- mode   : "raw" | "summary"
- engine : "opendss" | "pmd"
- out_path: if provided, write JSON to this file; else print to stdout

Exit behavior:
- Never throws for expected conditions; errors are encoded in JSON fields
  (except for fatal argument issues).
"""
function main()
    if length(ARGS) < 1
        println("Usage: julia run_pf_dss.jl <master_dss> [injections_json] [mode] [engine] [out_path]")
        return
    end

    # Normalize arguments and defaults
    master   = replace(ARGS[1], "\\" => "/")
    injections_json = length(ARGS) >= 2 ? ARGS[2] : "{}"
    mode     = length(ARGS) >= 3 ? String(ARGS[3]) : "raw"
    engine   = length(ARGS) >= 4 ? String(ARGS[4]) : "opendss"
    out_path = length(ARGS) >= 5 ? String(ARGS[5]) : ""

    # Parse injections JSON defensively; coerce all keys to String
    inj_raw = try JSON3.read(injections_json) catch; nothing end
    injections = Dict{String,Any}()
    if inj_raw !== nothing
        for (k,v) in pairs(inj_raw); injections[string(k)] = v; end
    end

    if engine == "pmd"
        # PMD path: robustness-first, do not throw; return detailed status
        res = run_pf_with_pmd(master, injections)
        if mode == "summary"
            summ = Dict(
                "termination_status" => get(res, "termination_status", nothing),
                "objective" => get(res, "objective", nothing),
                "model" => get(res, "model", nothing),
                "used_bounds" => get(res, "used_bounds", nothing),
                "tried_models" => get(res, "tried_models", nothing),
                "solver_time_sec" => get(res, "solver_time_sec", nothing)
            )
            output_json(Dict("summary"=>summ), out_path)
        else
            output_json(res, out_path)
        end
        return
    else
        # OpenDSS path: compile, optionally apply injections, then solve
        OpenDSSDirect.Basic.ClearAll()
        OpenDSSDirect.Text.Command("compile $master")

        if !isempty(injections); apply_injections!(injections); end
        OpenDSSDirect.Solution.Solve()

        buses = collect_bus_voltages()
        if mode == "summary"
            summ = make_summary(buses)
            output_json(Dict("summary"=>summ), out_path)
        else
            lines = collect_line_flows()
            output_json(Dict("bus_vmag_angle_pu"=>buses, "line_flows"=>lines), out_path)
        end
    end
end

# Entrypoint: always run main() on script execution.
main()
