# -*- coding: utf-8 -*-
"""
PF Service (pandapower + OpenDSS/PMD via Julia)

Endpoints
- GET  /health
- GET  /health/deps
- GET  /profiles/slp
- POST /grid/register      : load pp (.json/.p), auto-fix slack, persist pickle, return grid_id + hints
- POST /grid/fix_slack     : re-run slack fixer on an existing grid_id
- GET  /bus/list           : list bus indices & names
- POST /pf/run             : single snapshot PF (pandapower/opendss/pmd)
- POST /pf/run_timeseries  : timeseries PF with P/Q profiles (per-hour try/except; never 500 for PP branch)

Registry
- saved_json/grid_registry.json
- per grid: saved_json/grids/<grid_id>/net.p

Slack policy
- build connected components (lines/transformers)
- slack = highest vn_kv in largest component (tie -> smallest index), ensure single ext_grid(vm_pu=1.0)
- recommend_injection = lowest vn_kv in the same component (tie -> smallest index)

Compatibility patch
- tolerate pandapower version mismatches (e.g., add trafo['tap_phase_shifter'])
"""

from __future__ import annotations
import os, json, uuid, time, re, subprocess, shutil
from typing import Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import pandapower as pp
    import pandas as pd
except Exception as e:
    raise SystemExit(f"[FATAL] pandapower import failed: {e}")

# ---------- paths / registry ----------
APP_ROOT = os.path.abspath(os.getcwd())
REG_DIR = os.path.join(APP_ROOT, "saved_json")
os.makedirs(REG_DIR, exist_ok=True)
REG_FILE = os.path.join(REG_DIR, "grid_registry.json")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _is_windows_abs(p: str) -> bool:
    # Detect "C:\..." or "D:/..." when running on Linux
    return bool(re.match(r"^[A-Za-z]:[\\/]", p or ""))

def _abs_norm_path(p: str) -> str:
    """Normalize; if relative, make it absolute under APP_ROOT.
    Avoid mangling Windows-absolute paths when running inside Linux."""
    if not p:
        return p
    p = p.replace("\\", "/")
    if os.path.isabs(p) or _is_windows_abs(p):
        return p
    return os.path.abspath(os.path.join(APP_ROOT, p)).replace("\\", "/")

def _load_registry() -> Dict[str, dict]:
    if not os.path.exists(REG_FILE):
        return {}
    try:
        with open(REG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_registry(reg: Dict[str, dict]) -> None:
    _ensure_dir(REG_DIR)
    with open(REG_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

# ---------- net IO ----------
def _pp_load(path: str):
    path = _abs_norm_path(path)
    if not os.path.exists(path):
        raise HTTPException(400, f"pandapower file not found: {path}")
    if path.lower().endswith(".json"):
        return pp.from_json(path)
    return pp.from_pickle(path)

def _pp_save_pickle(net, out_path: str) -> str:
    if not out_path.lower().endswith(".p"):
        out_path = out_path + ".p"
    pp.to_pickle(net, out_path)
    return out_path

# ---------- compatibility patch ----------
def _pp_apply_compat_patches(net) -> None:
    """Bridge column differences across pandapower versions so runpp doesn't KeyError."""
    import pandas as _pd

    def _ensure(df: Optional[_pd.DataFrame], col: str, default):
        if df is None:
            return
        if col not in df.columns:
            df[col] = default
        else:
            try:
                df[col].fillna(default, inplace=True)
            except Exception:
                pass

    # Trafo / Trafo3w
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        _ensure(net.trafo, "tap_phase_shifter", False)
        _ensure(net.trafo, "in_service", True)
        _ensure(net.trafo, "tap_step", 0)
        _ensure(net.trafo, "tap_side", "hv")
        _ensure(net.trafo, "tap_neutral", 0)
        _ensure(net.trafo, "tap_min", -10)
        _ensure(net.trafo, "tap_max", 10)

    if hasattr(net, "trafo3w") and net.trafo3w is not None and len(net.trafo3w):
        _ensure(net.trafo3w, "tap_phase_shifter", False)
        _ensure(net.trafo3w, "in_service", True)

    # Lines
    if hasattr(net, "line") and net.line is not None and len(net.line):
        _ensure(net.line, "in_service", True)

# ---------- graph & fix slack ----------
def _build_components(net) -> List[set]:
    adj: Dict[int, set] = {}

    def add_edge(a: int, b: int):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    if hasattr(net, "line") and len(net.line):
        for i in net.line.index:
            a = int(net.line.at[i, "from_bus"]); b = int(net.line.at[i, "to_bus"])
            add_edge(a, b)
    if hasattr(net, "trafo") and len(net.trafo):
        for i in net.trafo.index:
            a = int(net.trafo.at[i, "hv_bus"]); b = int(net.trafo.at[i, "lv_bus"])
            add_edge(a, b)
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        for i in net.trafo3w.index:
            hv = int(net.trafo3w.at[i, "hv_bus"]); mv = int(net.trafo3w.at[i, "mv_bus"]); lv = int(net.trafo3w.at[i, "lv_bus"])
            add_edge(hv, mv); add_edge(hv, lv); add_edge(mv, lv)

    seen: set = set()
    comps: List[set] = []
    for b in net.bus.index:
        b = int(b)
        if b in seen: continue
        stack = [b]; comp = set()
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); comp.add(u)
            for v in adj.get(u, ()):
                if v not in seen: stack.append(v)
        comps.append(comp)
    comps.sort(key=lambda s: len(s), reverse=True)
    return comps

def _pick_slack_bus(net, comp: set) -> int:
    vn = net.bus.loc[list(comp), "vn_kv"].astype(float) if len(comp) else pd.Series([], dtype=float)
    if len(vn):
        max_vn = float(vn.max())
        cand = vn.index[vn == max_vn].tolist()
        return int(min(cand))
    return int(min(comp)) if len(comp) else 0

def _recommend_injection_bus(net, comp: set) -> int:
    vn = net.bus.loc[list(comp), "vn_kv"].astype(float) if len(comp) else pd.Series([], dtype=float)
    if len(vn):
        min_vn = float(vn.min())
        cand = vn.index[vn == min_vn].tolist()
        return int(min(cand))
    return int(min(comp)) if len(comp) else 0

def _ensure_single_slack_on_bus(net, bus_idx: int):
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
    pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="auto_slack")

def _fix_net_slack(net) -> Tuple[int, int]:
    """Return (slack_bus_idx, inject_bus_idx)."""
    comps = _build_components(net)
    if not comps:
        raise HTTPException(400, "Empty bus set in network.")
    main_comp = comps[0]
    slack_bus = _pick_slack_bus(net, main_comp)
    _ensure_single_slack_on_bus(net, slack_bus)
    inj_bus = _recommend_injection_bus(net, main_comp)
    try:
        pp.runpp(net, calculate_voltage_angles=True, init="flat", enforce_q_lims=True, numba=False)
    except Exception:
        pass  # do not block registration
    return (slack_bus, inj_bus)

# ---------- helpers for PF ----------
def _pp_bus_index_by_name(net, name: str) -> Optional[int]:
    if "name" in net.bus.columns:
        m = net.bus.index[net.bus["name"] == name]
        if len(m) > 0:
            return int(m[0])
    try:
        idx = int(name)
        if idx in net.bus.index:
            return idx
    except Exception:
        pass
    return None

def _pp_apply_injection_once(net, injections: Dict[str, List[float]]):
    if not injections: return
    for bus, arr in injections.items():
        if not arr or len(arr) < 2: continue
        dp_kw, dq_kvar = float(arr[0]), float(arr[1])
        bidx = _pp_bus_index_by_name(net, bus)
        if bidx is None:
            raise HTTPException(400, f"injection bus not found: {bus}")
        pp.create_load(net, bus=bidx, p_mw=max(dp_kw/1000.0, 0.0), q_mvar=dq_kvar/1000.0, name=f"inj_{bus}")

def _pp_summary(net, vmin: float, vmax: float) -> dict:
    vm = net.res_bus["vm_pu"].astype(float)
    per_bus = {}
    worst_min = float("+inf"); worst_bus = ""
    for idx in net.bus.index:
        name = str(net.bus.at[idx,"name"]) if "name" in net.bus.columns else str(idx)
        v = float(vm.at[idx])
        under = v < vmin; over = v > vmax
        per_bus[name] = {"min_vpu": v, "max_vpu": v, "under_limit": under, "over_limit": over}
        if v < worst_min: worst_min = v; worst_bus = name
    P_loss_mw = 0.0; Q_loss_mvar = 0.0
    if "res_line" in net and len(net.res_line):
        P_loss_mw += float(net.res_line["pl_mw"].sum()); Q_loss_mvar += float(net.res_line["ql_mvar"].sum())
    if "res_trafo" in net and len(net.res_trafo):
        P_loss_mw += float(net.res_trafo["pl_mw"].sum()); Q_loss_mvar += float(net.res_trafo["ql_mvar"].sum())
    if "res_trafo3w" in net and len(net.res_trafo3w):
        P_loss_mw += float(net.res_trafo3w["pl_mw"].sum()); Q_loss_mvar += float(net.res_trafo3w["ql_mvar"].sum())
    return {
        "voltage_summary_per_bus": per_bus,
        "worst_bus_min_vpu": {"bus": worst_bus, "min_vpu": (None if worst_min == float("+inf") else worst_min)},
        "system_losses": {"P_loss_kW": P_loss_mw*1000.0, "Q_loss_kvar": Q_loss_mvar*1000.0},
        "limits": {"vmin": vmin, "vmax": vmax},
    }

def _run_julia_pmd(master_path: str, injections: Dict[str, List[float]], mode: str = "summary") -> dict:
    """
    调用 Julia 脚本 /app/scripts/run_pf_dss.jl，engine 固定传 'pmd'
    - master_path: DSS 主文件容器内路径
    - injections: 形如 {"bus1":[P_kW,Q_kVAr], ...}
    - mode: "summary" | "raw"
    返回：结构化 JSON（不抛异常）
    """
    jl = os.environ.get("JULIA_EXE") or "julia"
    script = "/app/scripts/run_pf_dss.jl"  # ← 你的脚本会以 -v 映射到这里
    inj_json = json.dumps(injections or {}, ensure_ascii=False)
    cmd = [jl, script, master_path, inj_json, (mode or "summary"), "pmd"]

    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",   # ★ 修正：统一按 UTF-8 解码
        errors="ignore",    # ★ 修正：忽略无法解码的字节（避免 GBK 报错）
        check=False,
    )
    elapsed = time.perf_counter() - t0

    if p.returncode != 0:
        return {
            "status": "error",
            "engine": "pmd",
            "mode": mode,
            "elapsed_sec": elapsed,
            "error_type": "JuliaError",
            "stderr": p.stderr,
        }

    try:
        obj = json.loads(p.stdout.strip() or "{}")
        return {
            "status": "ok",
            "engine": "pmd",
            "mode": mode,
            "elapsed_sec": elapsed,
            "result": obj
        }
    except Exception:
        return {
            "status": "error",
            "engine": "pmd",
            "mode": mode,
            "elapsed_sec": elapsed,
            "error_type": "JSONParseError",
            "stdout": p.stdout,
            "stderr": p.stderr,
        }

def _aggregate(summary_obj: dict, vmin: float, vmax: float) -> dict:
    bus_sum = (summary_obj or {}).get("voltage_summary_per_bus", {}) or {}
    min_v = +1e9; max_v = -1e9; under = 0; over = 0
    for _, ent in bus_sum.items():
        mn = ent.get("min_vpu"); mx = ent.get("max_vpu")
        if isinstance(mn, (int, float)):
            min_v = min(min_v, mn)
            if mn < vmin: under += 1
        if isinstance(mx, (int, float)):
            max_v = max(max_v, mx)
            if mx > vmax: over += 1
    if min_v == +1e9: min_v = None
    if max_v == -1e9: max_v = None
    losses = (summary_obj or {}).get("system_losses", {}) or {}
    return {
        "min_vpu": min_v, "max_vpu": max_v,
        "under_voltage_bus_count": under, "over_voltage_bus_count": over,
        "P_loss_kW": losses.get("P_loss_kW"), "Q_loss_kvar": losses.get("Q_loss_kvar"),
    }

# ---------- Julia runner (OpenDSS / PMD) ----------
JULIA_EXE = os.environ.get("JULIA_EXE") or "julia"
JL_RUNNER = os.path.join(APP_ROOT, "scripts", "run_pf_dss.jl").replace("\\", "/")

def _run_julia_pf(master_dss: str, injections: Dict[str, List[float]], mode: str, engine: str) -> dict:
    """Call Julia script to run OpenDSS / PMD PF. Returns JSON object (raises HTTPException on fatal)."""
    master = _abs_norm_path(master_dss or "")
    if not master or not os.path.exists(master):
        raise HTTPException(400, f"OpenDSS master not found: {master}")
    payload = json.dumps(injections or {}, ensure_ascii=False)
    args = [JULIA_EXE, JL_RUNNER, master, payload, mode, engine]
    try:
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # ★ 修正：UTF-8
            errors="ignore",    # ★ 修正：忽略错误
            check=False
        )
    except Exception as e:
        raise HTTPException(500, f"failed to spawn julia: {e}")
    stdout = (p.stdout or "").strip()
    if not stdout:
        err = (p.stderr or "").strip()
        raise HTTPException(500, f"julia returned empty stdout; stderr={err[:400]}")
    try:
        data = json.loads(stdout)
    except Exception:
        raise HTTPException(500, f"invalid JSON from julia: {stdout[:400]}")
    return data

def _aggregate_opendss_summary(summ: dict, vmin: float, vmax: float) -> dict:
    # OpenDSS summary keys are aligned with _pp_summary → we can reuse _aggregate
    return _aggregate(summ or {}, vmin, vmax)

# ---------- API models ----------
class GridRegisterReq(BaseModel):
    name: Optional[str] = None
    pp_json_path: Optional[str] = None
    pp_pickle_path: Optional[str] = None  # .p

class GridRegisterDSSReq(BaseModel):
    name: Optional[str] = None
    dss_master_path: str  # 容器内路径，如 /work/my_dss/master.dss

class PFRequest(BaseModel):
    grid_id: str
    injections: Dict[str, List[float]] = Field(default_factory=dict)
    mode: Literal["raw", "summary"] = "summary"
    engine: Literal["pandapower", "opendss", "pmd"] = "pandapower"
    vmin: float = 0.95
    vmax: float = 1.05
    dss_master_path: Optional[str] = None  # required when engine is opendss/pmd

class TSProfile(BaseModel):
    P_kW: List[float]
    Q_kVAr: List[float]

class PFTimeSeriesRequest(BaseModel):
    grid_id: str
    engine: Literal["pandapower", "opendss", "pmd"] = "pandapower"
    mode: Literal["raw","summary"] = "summary"
    hours: int = 24
    profiles: Dict[str, TSProfile] = Field(default_factory=dict)
    vmin: float = 0.95
    vmax: float = 1.05
    dss_master_path: Optional[str] = None  # required when engine is opendss/pmd

# ---------- utilities ----------
def _normalize_profiles_to_hours(profiles: Dict[str, TSProfile], hours: int) -> Dict[str, Dict[str, List[float]]]:
    """Pydantic TSProfile -> plain lists; truncate/pad to 'hours'."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for bus, prof in (profiles or {}).items():
        P = [float(x) for x in (prof.P_kW or [])]
        Q = [float(x) for x in (prof.Q_kVAr or [])]
        if len(P) < hours: P = P + [0.0] * (hours - len(P))
        if len(Q) < hours: Q = Q + [0.0] * (hours - len(Q))
        if len(P) > hours: P = P[:hours]
        if len(Q) > hours: Q = Q[:hours]
        out[str(bus)] = {"P_kW": P, "Q_kVAr": Q}
    return out

# ---------- FastAPI app ----------
app = FastAPI(title="PF Service (PP + OpenDSS/PMD)", version="0.7")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/deps")
def health_deps():
    import shutil as _sh
    return {
        "ok": True,
        "julia_found": bool(_sh.which(JULIA_EXE)),
        "pandapower_found": True
    }

@app.get("/profiles/slp")
def slp(code: str, date: str, hours: int = 24):
    # simple residential-like profile; normalized; interp for non-24
    base = [0.015,0.014,0.013,0.013,0.013,0.015,0.020,0.030,0.045,0.050,0.045,0.040,
            0.038,0.037,0.038,0.040,0.045,0.055,0.060,0.055,0.045,0.035,0.028,0.020]
    if hours != 24:
        import numpy as np
        x = np.linspace(0, 23, 24); y = np.array(base, dtype=float)
        x2 = np.linspace(0, 23, hours); y2 = np.interp(x2, x, y)
        y = y2 / y2.sum()
        prof = y.tolist()
    else:
        s = sum(base); prof = [x/s for x in base]
    return {"code": code, "date": date, "hours": hours, "source": "fallback", "profile": prof}

@app.post("/grid/register")
def grid_register(req: GridRegisterReq):
    src = req.pp_pickle_path or req.pp_json_path
    if not src:
        raise HTTPException(400, "Provide pp_pickle_path (.p) or pp_json_path.")
    net = _pp_load(src)
    _pp_apply_compat_patches(net)
    sidx, iidx = _fix_net_slack(net)

    grid_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(REG_DIR, "grids", grid_id)
    _ensure_dir(out_dir)
    pp_path = os.path.join(out_dir, "net.p").replace("\\", "/")
    _pp_save_pickle(net, pp_path)

    reg = _load_registry()
    reg[grid_id] = {
        "name": req.name or f"pp-grid-{grid_id}",
        "pp_path": pp_path,
        "engine_hint": "pandapower",
        "slack_bus_index": sidx,
        "recommend_inj_index": iidx,
    }
    _save_registry(reg)
    return {"grid_id": grid_id, "pp_path": pp_path, "name": reg[grid_id]["name"],
            "slack_bus_index": sidx, "recommend_inj_index": iidx}

@app.post("/grid/fix_slack")
def grid_fix_slack(grid_id: str):
    reg = _load_registry()
    info = reg.get(grid_id)
    if not info:
        raise HTTPException(404, f"grid_id not found: {grid_id}")
    net = _pp_load(info["pp_path"])
    _pp_apply_compat_patches(net)
    sidx, iidx = _fix_net_slack(net)
    _pp_save_pickle(net, info["pp_path"])  # overwrite
    info["slack_bus_index"] = sidx
    info["recommend_inj_index"] = iidx
    _save_registry(reg)
    name = lambda idx: (str(net.bus.at[idx,"name"]) if "name" in net.bus.columns else str(idx))
    return {"status": "ok", "grid_id": grid_id, "slack_bus_index": sidx, "slack_bus_name": name(sidx),
            "recommend_inj_index": iidx, "recommend_inj_name": name(iidx)}

@app.get("/bus/list")
def bus_list(grid_id: str):
    info = _load_registry().get(grid_id)
    if not info: raise HTTPException(404, f"grid_id not found: {grid_id}")
    net = _pp_load(info["pp_path"])
    _pp_apply_compat_patches(net)
    rows = []
    for idx in net.bus.index:
        rows.append({"index": int(idx),
                     "name": (str(net.bus.at[idx,"name"]) if "name" in net.bus.columns else str(idx)),
                     "vn_kv": float(net.bus.at[idx,"vn_kv"])})
    return {"grid_id": grid_id, "buses": rows}

@app.post("/pf/run")
def run_pf(req: PFRequest):
    # --- OpenDSS / PMD branch ---
    if req.engine in ("opendss", "pmd"):
        if not req.dss_master_path:
            raise HTTPException(400, "opendss/pmd requires dss_master_path")
        t0 = time.perf_counter()
        data = _run_julia_pf(req.dss_master_path, req.injections or {}, req.mode, req.engine)
        elapsed = time.perf_counter() - t0
        if req.mode == "raw":
            return {"status": "ok", "engine": req.engine, "mode": "raw",
                    "elapsed_sec": elapsed, "result": data}
        else:
            if req.engine == "opendss":
                summ = (data or {}).get("summary") or {}
                agg = _aggregate_opendss_summary(summ, req.vmin, req.vmax)
                return {"status": "ok", "engine": "opendss", "mode": "summary",
                        "elapsed_sec": elapsed, "result": {"summary": summ, "aggregate": agg}}
            else:
                summ = (data or {}).get("summary") or data or {}
                return {"status": "ok", "engine": "pmd", "mode": "summary",
                        "elapsed_sec": elapsed, "result": {"summary": summ}}

    # --- pandapower branch ---
    info = _load_registry().get(req.grid_id)
    if not info: raise HTTPException(404, f"grid_id not found: {req.grid_id}")
    net = _pp_load(info["pp_path"]).deepcopy()
    _pp_apply_compat_patches(net)
    _pp_apply_injection_once(net, req.injections or {})
    t0 = time.perf_counter()
    try:
        pp.runpp(net, calculate_voltage_angles=True, init="flat", enforce_q_lims=True, numba=False)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return {"status": "error", "engine": "pandapower", "mode": req.mode,
                "elapsed_sec": time.perf_counter() - t0, "error_type": type(e).__name__, "error": str(e)}

    if req.mode == "raw":
        bus_res = {}
        for idx in net.bus.index:
            name = str(net.bus.at[idx,"name"]) if "name" in net.bus.columns else str(idx)
            vm = float(net.res_bus.at[idx,"vm_pu"]); va = float(net.res_bus.at[idx,"va_degree"])
            bus_res[name] = [{"phase": 1, "V": None, "angle_deg": va, "Vpu": vm}]
        return {"status": "ok", "engine": "pandapower", "mode": "raw",
                "elapsed_sec": elapsed, "result": {"bus_vmag_angle_pu": bus_res}}
    else:
        summ = _pp_summary(net, req.vmin, req.vmax)
        return {"status": "ok", "engine": "pandapower", "mode": "summary",
                "elapsed_sec": elapsed, "result": {"summary": summ}}

@app.post("/pf/run_timeseries")
def run_pf_timeseries(req: PFTimeSeriesRequest):
    if req.hours <= 0:
        raise HTTPException(400, "hours must be positive")

    # --- OpenDSS / PMD branch ---
    if req.engine in ("opendss", "pmd"):
        if not req.dss_master_path:
            raise HTTPException(400, "opendss/pmd requires dss_master_path")
        profs = _normalize_profiles_to_hours(req.profiles or {}, req.hours)

        series = {"min_vpu": [], "max_vpu": [], "P_loss_kW": [], "Q_loss_kvar": [],
                  "under_voltage_bus_count": [], "over_voltage_bus_count": [],
                  "termination_status": [], "solver_time_sec": []}
        feasible = 0
        worst_bus_min = +1e9
        t_total = time.perf_counter()

        for h in range(req.hours):
            inj = {bus: [float(v["P_kW"][h]), float(v["Q_kVAr"][h])] for bus, v in profs.items()}
            t0 = time.perf_counter()
            try:
                data = _run_julia_pf(req.dss_master_path, inj, "summary", req.engine)
                series["solver_time_sec"].append(time.perf_counter() - t0)

                if req.engine == "opendss":
                    summ = (data or {}).get("summary") or {}
                    m = _aggregate_opendss_summary(summ, req.vmin, req.vmax)
                    for k in ("min_vpu","max_vpu","P_loss_kW","Q_loss_kvar","under_voltage_bus_count","over_voltage_bus_count"):
                        series[k].append(m[k])
                    ok = (
                        (m["min_vpu"] is not None)
                        and (m["max_vpu"] is not None)
                        and (m["under_voltage_bus_count"] == 0)
                        and (m["over_voltage_bus_count"] == 0)
                    )
                    series["termination_status"].append("OK" if ok else "CHECK")
                    if m["min_vpu"] is not None:
                        worst_bus_min = min(worst_bus_min, m["min_vpu"])
                    if ok: feasible += 1
                else:
                    # PMD: only record termination_status, no voltages available here
                    summ = (data or {}).get("summary") or {}
                    ts = str(summ.get("termination_status") or summ.get("status") or "UNKNOWN")
                    series["termination_status"].append(ts)
                    for k in ("min_vpu","max_vpu","P_loss_kW","Q_loss_kvar","under_voltage_bus_count","over_voltage_bus_count"):
                        series[k].append(None)
            except Exception as e:
                series["solver_time_sec"].append(time.perf_counter() - t0)
                series["termination_status"].append(f"ERROR:{type(e).__name__}")
                for k in ("min_vpu","max_vpu","P_loss_kW","Q_loss_kvar","under_voltage_bus_count","over_voltage_bus_count"):
                    series[k].append(None)

        summary = {"hours": req.hours, "hours_feasible": feasible,
                   "worst_bus_min_vpu": (None if worst_bus_min == +1e9 else worst_bus_min),
                   "engine": req.engine, "mode": req.mode,
                   "v_limits": {"vmin": req.vmin, "vmax": req.vmax}}
        return {"status": "ok", "engine": req.engine, "hours": req.hours,
                "elapsed_sec_total": time.perf_counter() - t_total,
                "series": series, "summary": summary,
                "meta": {"dss_master_path": _abs_norm_path(req.dss_master_path)}}

    # --- pandapower branch ---
    profs = _normalize_profiles_to_hours(req.profiles or {}, req.hours)
    reg = _load_registry()
    info = reg.get(req.grid_id)
    if not info:
        raise HTTPException(404, f"grid_id not found: {req.grid_id}")
    path = info["pp_path"]

    # validate that all bus keys exist
    net0 = _pp_load(path)
    _pp_apply_compat_patches(net0)
    for b in profs.keys():
        if _pp_bus_index_by_name(net0, b) is None:
            raise HTTPException(400, f"profile bus not found: {b}")

    series = {"min_vpu": [], "max_vpu": [], "P_loss_kW": [], "Q_loss_kvar": [],
              "under_voltage_bus_count": [], "over_voltage_bus_count": [],
              "termination_status": [], "solver_time_sec": []}
    feasible = 0
    worst_bus_min = +1e9
    t_total = time.perf_counter()

    for h in range(req.hours):
        net = _pp_load(path).deepcopy()
        _pp_apply_compat_patches(net)
        inj = {bus: [float(v["P_kW"][h]), float(v["Q_kVAr"][h])] for bus, v in profs.items()}
        try:
            _pp_apply_injection_once(net, inj)
        except HTTPException as e:
            series["termination_status"].append(f"ERROR:{e.detail}")
            for k in ("min_vpu","max_vpu","P_loss_kW","Q_loss_kvar","under_voltage_bus_count","over_voltage_bus_count"):
                series[k].append(None)
            series["solver_time_sec"].append(0.0)
            continue

        t0 = time.perf_counter()
        try:
            pp.runpp(net, calculate_voltage_angles=True, init="flat", enforce_q_lims=True, numba=False)
            series["solver_time_sec"].append(time.perf_counter() - t0)
            summ = _pp_summary(net, req.vmin, req.vmax)
            m = _aggregate(summ, req.vmin, req.vmax)
            for k in ("min_vpu","max_vpu","P_loss_kW","Q_loss_kvar","under_voltage_bus_count","over_voltage_bus_count"):
                series[k].append(m[k])
            ok = (
                (m["min_vpu"] is not None)
                and (m["max_vpu"] is not None)
                and (m["under_voltage_bus_count"] == 0)
                and (m["over_voltage_bus_count"] == 0)
            )
            series["termination_status"].append("OK" if ok else "CHECK")
            if m["min_vpu"] is not None:
                worst_bus_min = min(worst_bus_min, m["min_vpu"])
            if ok: feasible += 1
        except Exception as e:
            series["solver_time_sec"].append(time.perf_counter() - t0)
            series["termination_status"].append(f"ERROR:{type(e).__name__}")
            for k in ("min_vpu","max_vpu","P_loss_kW","Q_loss_kvar","under_voltage_bus_count","over_voltage_bus_count"):
                series[k].append(None)

    summary = {"hours": req.hours, "hours_feasible": feasible,
               "worst_bus_min_vpu": (None if worst_bus_min == +1e9 else worst_bus_min),
               "engine": "pandapower", "mode": req.mode,
               "v_limits": {"vmin": req.vmin, "vmax": req.vmax}}
    return {"status": "ok", "engine": "pandapower", "hours": req.hours,
            "elapsed_sec_total": time.perf_counter() - t_total,
            "series": series, "summary": summary, "meta": {"pp_path": path}}

# ==== DSS/PMD bridge (append-only) ==========================================
from pydantic import BaseModel, Field
from fastapi import HTTPException
import subprocess, json, os, shutil, tempfile
from typing import Dict, List, Literal

class DSSPFRequest(BaseModel):
    master_dss: str                                  # 容器内可见路径，如 /work/ieee13/Master.dss
    injections: Dict[str, List[float]] = Field(default_factory=dict)  # {"632":[ΔP_kW, ΔQ_kVAr], ...}
    mode: Literal["raw","summary"] = "summary"       # 与 julia 脚本一致
    engine: Literal["opendss","pmd"] = "pmd"         # 选择 OpenDSS 或 PMD
    timeout_sec: int = 120                           # 运行超时保护

def _which_julia() -> str:
    cand = os.environ.get("JULIA_EXE") or "julia"
    if shutil.which(cand):
        return cand
    # 兜底：常见安装位置（可按需扩展）
    for p in ["/usr/local/julia/bin/julia", "/usr/bin/julia"]:
        if os.path.exists(p):
            return p
    raise HTTPException(500, "JULIA_EXE not found in PATH/ENV")

def _ensure_path_exists(path: str):
    # 仅检查容器内可见路径；客户端应传 /work/... 或 /app/scripts/...
    if not os.path.exists(path):
        raise HTTPException(400, f"master_dss not found in container: {path}")

@app.post("/pf/run_dss")
def pf_run_dss(req: DSSPFRequest):
    julia = _which_julia()
    script = os.environ.get("JULIA_DSS_SCRIPT", "/app/scripts/run_pf_dss.jl")
    if not os.path.exists(script):
        raise HTTPException(500, f"Julia script not found: {script}")

    _ensure_path_exists(req.master_dss)

    # 准备参数
    inj_json = json.dumps(req.injections or {}, ensure_ascii=False)
    cmd = [julia, script, req.master_dss, inj_json, req.mode, req.engine]

    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # ★ 修正：UTF-8
            errors="ignore",    # ★ 修正：忽略错误
            timeout=max(10, int(req.timeout_sec)),
            check=False
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Julia PF timed out")

    if r.returncode != 0:
        raise HTTPException(500, f"Julia exited {r.returncode}: {r.stderr.strip()}")

    out = (r.stdout or "").strip()
    if not out:
        raise HTTPException(500, f"Julia returned empty stdout. stderr={r.stderr[:500]}")

    try:
        payload = json.loads(out)
    except Exception as e:
        raise HTTPException(500, f"Invalid JSON from Julia: {e}; first 500 chars: {out[:500]}")

    return {
        "status": "ok",
        "engine": f"dss:{req.engine}",
        "mode": req.mode,
        "result": payload
    }
