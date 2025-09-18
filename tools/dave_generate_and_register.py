# tools/dave_generate_and_register.py
# -*- coding: utf-8 -*-
"""
DAVE → pandapower → PF 注册（极简稳妥版）

特点：
- 默认只拉取 roads，电压层级用 mv（避免 LV 拓扑递归/几何问题）
- 运行期补丁：
  * shapely.union_all 安全化
  * nearest_road_points 安全化（空道路/CRS 自动处理）
  * 禁用 DAVE 的 power_processing（用恒等函数返回原 net，不再返回 None）
  * 修补 dave_core 的 pp_to_json（字符串缩进），并在本脚本保存时完全绕过 pp.to_json
- 避免 DAVE 内部保存：save_data=False；由本脚本用 PPJSONEncoder 手工写 JSON
- 自动补 ext_grid（若 DAVE 结果无参考母线）
- 支持 --reuse_from 直接注册已有 net_power.json
- 失败自动写 error_log.txt（带 stage/traceback/env）

运行示例（PowerShell 一行）：
  $env:OVERPASS_API_URL="https://lz4.overpass-api.de/api/interpreter"; `
  $env:OSMNX_OVERPASS_ENDPOINT=$env:OVERPASS_API_URL; `
  python tools/dave_generate_and_register.py --selector own_area `
    --own_area_geojson data/aoi/micro_bremen.geojson `
    --geodata roads --power-levels mv --no_probe
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
import importlib
from pathlib import Path
from typing import Dict, List, Optional

# ===== TLS / 证书 =====
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
    print("[TLS] Using OS trust store via 'truststore'.")
except Exception:
    try:
        import certifi  # type: ignore
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass

# ===== HTTP Cache =====
try:
    import requests_cache  # type: ignore
    _CACHE_DIR = Path(__file__).resolve().parents[1] / "intermediate" / "dave_cache"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    requests_cache.install_cache(str(_CACHE_DIR / "http_cache"), expire_after=None)
    print(f"[CACHE] HTTP cache active → {_CACHE_DIR / 'http_cache.sqlite'}")
except Exception as e:
    print(f"[CACHE] disabled: {e}")

import requests  # noqa

API_DEFAULT = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "intermediate" / "dave_export"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ===== DAVE 顶层入口 =====
from dave_core.create import create_grid  # type: ignore

# ===== 可选：OSMnx 设置 =====
try:
    import osmnx as ox  # type: ignore
    ox.settings.timeout = int(os.environ.get("OSMNX_TIMEOUT", 180))
    ox.settings.retry_on_rate_limit = True
    ox.settings.log_console = False
    print(f"[OSMNX] timeout={ox.settings.timeout}, retries=on, rate_limit=True")
except Exception:
    pass


# ================== 小工具 ==================
def _port_open(p: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, p), timeout=timeout):
            return True
    except Exception:
        return False

def _maybe_enable_proxy(proxy_cli: Optional[str]) -> Optional[str]:
    if proxy_cli:
        os.environ["HTTP_PROXY"] = proxy_cli
        os.environ["HTTPS_PROXY"] = proxy_cli
        return proxy_cli
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "PROXY"):
        v = os.environ.get(k)
        if v:
            os.environ["HTTP_PROXY"] = v
            os.environ["HTTPS_PROXY"] = v
            return v
    for p in (7890, 1080, 8889, 8888, 8080):
        if _port_open(p):
            v = f"http://127.0.0.1:{p}"
            os.environ["HTTP_PROXY"] = v
            os.environ["HTTPS_PROXY"] = v
            return v
    return None

def _pick_overpass_endpoint() -> None:
    ep = os.environ.get("OVERPASS_API_URL") or os.environ.get("OSMNX_OVERPASS_ENDPOINT")
    if ep:
        print(f"[OVERPASS] using endpoint(from env): {ep}")
        return
    for cand in [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://overpass.osm.jp/api/interpreter",
    ]:
        try:
            r = requests.get(cand.rsplit("/", 1)[0] + "/status", timeout=5, verify=False)
            if r.status_code in (200, 301, 302):
                os.environ["OVERPASS_API_URL"] = cand
                os.environ["OSMNX_OVERPASS_ENDPOINT"] = cand
                print(f"[OVERPASS] using endpoint: {cand}")
                return
        except Exception:
            pass
    print("[OVERPASS] no endpoint verified; DAVE default will be used")

def _write_error_log(out_dir: Path, err: BaseException, context: Dict) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "error_log.txt"
        with p.open("w", encoding="utf-8") as f:
            f.write("=== ERROR CONTEXT ===\n")
            for k, v in (context or {}).items():
                f.write(f"{k}: {v}\n")
            f.write("\n=== TRACEBACK ===\n")
            f.write("".join(traceback.format_exception(type(err), err, err.__traceback__)))
            f.write("\n\n=== ENV ===\n")
            for key in ("OVERPASS_API_URL", "OSMNX_OVERPASS_ENDPOINT", "HTTP_PROXY", "HTTPS_PROXY"):
                f.write(f"{key}={os.environ.get(key)}\n")
        print(f"[ERROR] details written → {p}")
    except Exception:
        pass

def _read_area_from_geojson(path: str):
    import geopandas as gpd  # type: ignore
    g = gpd.read_file(path)
    if g.empty:
        raise RuntimeError(f"AOI empty: {path}")
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    geom = getattr(g.geometry, "union_all", None)
    geom = geom() if callable(geom) else g.unary_union
    if getattr(geom, "geom_type", "") == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda x: x.area)
    return geom


# ======== 运行期补丁 ========
def _install_runtime_patches() -> None:
    # -- shapely.union_all 安全化 --
    import shapely
    from shapely.geometry.base import BaseGeometry
    from shapely.geometry import GeometryCollection
    try:
        _orig_union_all = shapely.union_all
    except Exception:
        from shapely.set_operations import union_all as _orig_union_all  # type: ignore

    def _safe_union_all(seq, *args, **kwargs):
        flat: List[BaseGeometry] = []
        def _collect(x):
            if x is None:
                return
            if isinstance(x, BaseGeometry):
                if not getattr(x, "is_empty", False):
                    flat.append(x)
                return
            if hasattr(x, "__iter__"):
                for y in x:
                    _collect(y)
        _collect(seq)
        if not flat:
            return GeometryCollection()
        return _orig_union_all(flat, *args, **kwargs)

    shapely.union_all = _safe_union_all  # type: ignore
    print("[PATCH] shapely.union_all → safe")

    # -- nearest_road_points 安全化（CRS/空道路） --
    try:
        import geopandas as gpd  # type: ignore
        from shapely.ops import nearest_points  # type: ignore

        def _nearest_safe(points, roads):
            pts = gpd.GeoSeries(points, crs=getattr(points, "crs", None))
            rds = gpd.GeoSeries(roads,  crs=getattr(roads,  "crs", None))
            if rds.isna().all() or rds.empty:
                return pts
            rds = rds[~rds.isna()]
            pts_m = pts.to_crs(3857) if (pts.crs and pts.crs.is_geographic) else pts
            rds_m = rds.to_crs(3857) if (rds.crs and rds.crs.is_geographic) else rds
            multi = _safe_union_all(rds_m.values)
            if getattr(multi, "is_empty", False):
                return pts
            res = pts_m.apply(lambda x: nearest_points(x, multi)[1])
            try:
                if pts.crs and pts.crs.is_geographic:
                    res = res.set_crs(3857).to_crs(pts.crs)
            except Exception:
                pass
            return res

        try:
            import dave_core.geography.geo_utils as gu  # type: ignore
            gu.nearest_road_points = _nearest_safe  # type: ignore
        except Exception:
            pass
        try:
            import dave_core.topology.low_voltage as lv  # type: ignore
            lv.nearest_road_points = _nearest_safe  # type: ignore
        except Exception:
            pass
        print("[PATCH] nearest_road_points → safe")
    except Exception:
        pass

    # -- 禁用 power_processing：恒等补丁 --
    for name in [
        "dave_core.converter.process_pandapower",
        "dave_core.converter.process_panda",
        "dave_core.converter.create_pandapower",
        "dave_core.converter",
    ]:
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(m, "power_processing"):
            def _pp_identity(*args, **kwargs):
                for obj in args:
                    if hasattr(obj, "bus") and hasattr(obj, "res_bus"):
                        return obj
                return args[0] if args else None
            setattr(m, "power_processing", _pp_identity)
            print(f"[PATCH] power_processing → identity in {name}")

    # -- 修补 dave_core.io.file_io.pp_to_json（indent 用字符串，兜底） --
    try:
        import dave_core.io.file_io as fio  # type: ignore
        from pandapower.file_io import PPJSONEncoder  # type: ignore
        import json as _json

        def _pp_to_json_safe(net, file_path, *args, **kwargs):
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                _json.dump(net, f, cls=PPJSONEncoder, indent="  ", ensure_ascii=False)

        if hasattr(fio, "pp_to_json"):
            fio.pp_to_json = _pp_to_json_safe  # type: ignore
            print("[PATCH] dave_core.io.file_io.pp_to_json → safe (indent='  ')")
    except Exception as e:
        print(f"[PATCH] skip pp_to_json patch: {e}")


# ======== pandapower 保存 / 修复 ========
def _ensure_ext_grid(net) -> None:
    import pandas as pd  # type: ignore
    import pandapower as pp  # type: ignore
    need = (not hasattr(net, "ext_grid")) or net.ext_grid is None or len(net.ext_grid) == 0
    if not need:
        return
    if hasattr(net, "bus") and isinstance(net.bus, pd.DataFrame) and "vn_kv" in net.bus.columns and len(net.bus):
        bus_idx = int(net.bus["vn_kv"].fillna(-1).idxmax())
    else:
        bus_idx = 0
    pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="ext_grid_auto")
    print(f"[PP ] ext_grid auto-added at bus {bus_idx}")

def _save_pp_net_to_json(net, out_path: Path) -> str:
    # 不用 pandapower.to_json（它内部用 indent=int，触发某些环境的 json bug）
    from pandapower.file_io import PPJSONEncoder  # type: ignore
    import json as _json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(net, f, cls=PPJSONEncoder, indent="  ", ensure_ascii=False)
        return str(out_path.resolve())
    except Exception:
        # 兜底：写 pickle
        from pandapower import to_pickle  # type: ignore
        pkl = out_path.with_suffix(".p")
        to_pickle(net, str(pkl))
        print(f"[PP ] fallback saved pickle → {pkl}")
        return str(pkl.resolve())


# ======== PF API ========
def _register_pp_json(api: str, path: str, name: str) -> str:
    r = requests.post(f"{api.rstrip('/')}/grid/register", json={"name": name, "pp_json_path": path}, timeout=180)
    r.raise_for_status()
    js = r.json() or {}
    gid = js.get("grid_id")
    if not gid:
        raise RuntimeError(f"register failed: {js}")
    return gid

def _probe_buses(api: str, grid_id: str) -> List[str]:
    r = requests.post(
        f"{api.rstrip('/')}/pf/run",
        json={"grid_id": grid_id, "engine": "pandapower", "mode": "raw", "injections": {}},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json() or {}
    return list(((data.get("result") or {}).get("bus_vmag_angle_pu") or {}).keys())


# ======== 可选 24h 时序（若服务支持） ========
def _fetch_slp(api: str, code: str, date: str, hours: int) -> List[float]:
    # 优先尝试 BDEW（demandlib，含季节/日型），失败回退到 fallback SLP
    try:
        r = requests.get(f"{api.rstrip('/')}/profiles/bdew",
                         params={"code": code, "date": date, "hours": hours}, timeout=30)
        if r.status_code == 200:
            js = r.json() or {}
            prof = js.get("profile")
            if isinstance(prof, list) and len(prof) == hours and (abs(sum(prof) - 1.0) < 1e-6 or sum(prof) > 0):
                # 再归一化一次以防数值误差
                s = float(sum(prof)) or 1.0
                return [float(x)/s for x in prof]
    except Exception:
        pass
    # 回退：用现有 /profiles/slp（演示曲线）
    try:
        r = requests.get(f"{api.rstrip('/')}/profiles/slp",
                         params={"code": code, "date": date, "hours": hours}, timeout=30)
        if r.status_code == 200:
            js = r.json() or {}
            prof = js.get("profile")
            if isinstance(prof, list) and len(prof) == hours:
                return [float(x) for x in prof]
    except Exception:
        pass
    # 最后兜底：正弦日曲线
    import math
    raw = [max(0.0, 0.4 + 0.6 * math.sin((i + 7) / hours * math.pi)) for i in range(hours)]
    s = sum(raw) or 1.0
    return [v / s for v in raw]

    try:
        r = requests.get(f"{api.rstrip('/')}/profiles/slp", params={"code": code, "date": date, "hours": hours}, timeout=30)
        if r.status_code == 200:
            js = r.json() or {}
            prof = js.get("profile")
            if isinstance(prof, list) and len(prof) == hours:
                return [float(x) for x in prof]
    except Exception:
        pass
    import math
    raw = [max(0.0, 0.4 + 0.6 * math.sin((i + 7) / hours * math.pi)) for i in range(hours)]
    s = sum(raw) or 1.0
    return [v / s for v in raw]

def _scale_daily_by_annual(slp: List[float], annual_kwh: float) -> List[float]:
    daily = annual_kwh / 365.0
    return [round(daily * w, 6) for w in slp]

def _q_from_p_pf(p_kw: float, pf: float) -> float:
    import math
    pf = min(max(pf, 1e-3), 0.999999)
    phi = math.acos(pf)
    return round(p_kw * math.tan(phi), 6)

def _build_profiles_from_specs(api: str, specs: List[str], date: str, hours: int) -> Dict[str, Dict[str, List[float]]]:
    # "bus:SLP:annual_kWh:pf" 例："0:H0:3500:0.95"
    profiles: Dict[str, Dict[str, List[float]]] = {}
    for spec in specs:
        bus, code, annual_s, pf_s = spec.split(":")
        annual = float(annual_s); pf = float(pf_s)
        slp = _fetch_slp(api, code, date, hours)
        P = _scale_daily_by_annual(slp, annual)
        Q = [_q_from_p_pf(p, pf) for p in P]
        profiles[bus] = {"P_kW": P, "Q_kVAr": Q}
    return profiles

def _run_timeseries(api: str, grid_id: str, specs: List[str], date: str, hours: int,
                    vmin: float, vmax: float, out_dir: Path) -> None:
    profiles = _build_profiles_from_specs(api, specs, date, hours)
    payload = {
        "grid_id": grid_id, "engine": "pandapower", "mode": "summary",
        "hours": int(hours), "vmin": float(vmin), "vmax": float(vmax),
        "profiles": profiles
    }
    p = out_dir / "timeseries_payload.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PF ] Saved payload → {p}")

    r = requests.post(f"{api.rstrip('/')}/pf/run_timeseries", json=payload, timeout=3600)
    r.raise_for_status()
    res = r.json() or {}
    summ = res.get("summary", {}) or {}
    series = res.get("series", {}) or {}
    print("\n=== 24h RESULT (key metrics) ===")
    print(f"engine                  : {summ.get('engine')}")
    print(f"hours_feasible          : {summ.get('hours_feasible')}/{summ.get('hours')}")
    print(f"worst_bus_min_vpu       : {summ.get('worst_bus_min_vpu')}")
    print(f"max under_voltage_count : {max(series.get('under_voltage_bus_count') or [0])}")
    print(f"max over_voltage_count  : {max(series.get('over_voltage_bus_count') or [0])}")
    print(f"payload path            : {p}\n")


# ================= 主流程 =================
def run_register_only(
    *,
    api: str,
    selector: str,
    geodata: List[str],
    power_levels: List[str],
    own_area_geojson: Optional[str],
    town: Optional[List[str]],
    postalcode: Optional[List[str]],
    nuts: Optional[List[str]],
    nuts_year: str,
) -> Dict[str, str]:
    _install_runtime_patches()
    _pick_overpass_endpoint()

    out_dir = OUT_ROOT / time.strftime("dave_export_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    create_kw = dict(
        geodata=geodata,
        power_levels=power_levels,
        gas_levels=[],
        convert_power=["pandapower"],
        convert_gas=[],
        opt_model=False,
        combine_areas=[],
        transformers=True,
        renewable_powerplants=False,
        conventional_powerplants=False,
        loads=False,
        compressors=False, sinks=False, sources=False,
        output_folder=str(out_dir), output_format="json",
        save_data=False,  # 关键：不让 DAVE 内部保存，避免其 json.dumps(indent=2) 路径
    )

    print(f"[DAVE] selector={selector} geodata={geodata} power_levels={power_levels} loads=False")

    # 选区 → DAVE
    try:
        if selector == "own_area":
            if not own_area_geojson:
                # 默认 1km×1km micro AOI（Bremen）
                micro = ROOT / "data" / "aoi" / "micro_bremen.geojson"
                if not micro.exists():
                    micro.parent.mkdir(parents=True, exist_ok=True)
                    micro.write_text(
                        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[8.795,53.075],[8.805,53.075],[8.805,53.085],[8.795,53.085],[8.795,53.075]]]}}]}',
                        encoding="utf-8",
                    )
                own_area_geojson = str(micro)
            geom = _read_area_from_geojson(own_area_geojson)
            grid_data, net, *_ = create_grid(own_area=geom, **create_kw)  # type: ignore
        elif selector == "town":
            if not town:
                raise SystemExit("town 需要 --town Bremen ...")
            grid_data, net, *_ = create_grid(town_name=town, **create_kw)  # type: ignore
        elif selector == "postalcode":
            if not postalcode:
                raise SystemExit("postalcode 需要 --postalcode ...")
            grid_data, net, *_ = create_grid(postalcode=postalcode, **create_kw)  # type: ignore
        elif selector == "nuts":
            if not nuts:
                raise SystemExit("nuts 需要 --nuts DE501 ...")
            grid_data, net, *_ = create_grid(nuts_region=(nuts, nuts_year), **create_kw)  # type: ignore
        else:
            raise SystemExit(f"unsupported selector: {selector}")
    except Exception as e:
        _write_error_log(out_dir, e, {"stage": "create_grid", "selector": selector})
        raise SystemExit(f"DAVE 生成失败：{type(e).__name__}: {e}")

    # 保存 net + 补 ext_grid
    try:
        if net is None:
            raise RuntimeError("DAVE returned net=None (unexpected with convert_power=['pandapower'])")
        _ensure_ext_grid(net)
        pp_json_abs = _save_pp_net_to_json(net, out_dir / "net_power.json")
        print(f"[PP ] saved Pandapower net → {pp_json_abs}")
    except Exception as e:
        _write_error_log(out_dir, e, {"stage": "save_pandapower"})
        raise

    # 注册到 PF 服务
    try:
        gid = _register_pp_json(api, pp_json_abs, f"dave_grid_{out_dir.name}")
        print(f"[PF ] Registered pandapower grid_id = {gid}")
    except Exception as e:
        _write_error_log(out_dir, e, {"stage": "grid_register", "pp_json": pp_json_abs})
        raise

    print("\n=== REGISTER-ONLY DONE ===")
    print(f"grid_id      : {gid}")
    print(f"pp_json_path : {pp_json_abs}")
    print(f"export_dir   : {out_dir}\n")

    return {"grid_id": gid, "pp_json_path": pp_json_abs, "export_dir": str(out_dir)}


# ================= CLI =================
def main():
    ap = argparse.ArgumentParser(description="DAVE→pandapower→PF register (MV-only, safe patches)")
    ap.add_argument("--api", default=API_DEFAULT)

    # 选区
    ap.add_argument("--selector", choices=["own_area", "postalcode", "town", "nuts"], default="own_area")
    ap.add_argument("--nuts", nargs="+")
    ap.add_argument("--nuts-year", default="2021")
    ap.add_argument("--own_area_geojson")
    ap.add_argument("--town", nargs="+")
    ap.add_argument("--postalcode", nargs="+")

    # geodata/电压层级
    ap.add_argument("--geodata", nargs="+", default=["roads"])
    ap.add_argument("--power-levels", nargs="+", default=["mv"])

    # 其它
    ap.add_argument("--no_probe", action="store_true")
    ap.add_argument("--reuse_from")
    ap.add_argument("--proxy")

    # 可选时序（若 PF 服务支持）
    ap.add_argument("--run24", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--buses", nargs="+", default=["0:H0:3500:0.95"])
    ap.add_argument("--vmin", type=float, default=0.95)
    ap.add_argument("--vmax", type=float, default=1.05)

    args = ap.parse_args()
    api = args.api.rstrip("/")

    proxy = _maybe_enable_proxy(args.proxy)
    print(f"[PROXY] {'enabled → ' + proxy if proxy else 'not enabled'}")

    # 复用现成 net（跳过 DAVE）
    if args.reuse_from:
        pp_json_abs = str(Path(args.reuse_from).resolve())
        if not Path(pp_json_abs).exists():
            raise SystemExit(f"--reuse_from 不存在：{pp_json_abs}")
        gid = _register_pp_json(api, pp_json_abs, f"dave_grid_reuse_{Path(pp_json_abs).stem}")
        print(f"[PF ] Registered pandapower grid_id = {gid}")
        if not args.no_probe:
            try:
                buses = _probe_buses(api, gid)
                print(f"[PF ] Probe buses: total={len(buses)}, first 12={buses[:12]}")
            except Exception as e:
                print(f"[PF ] Probe skipped: {e}")
        if args.run24:
            out_dir = OUT_ROOT / time.strftime("dave_export_%Y%m%d_%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)
            _run_timeseries(api, gid, args.buses, args.date, args.hours, args.vmin, args.vmax, out_dir)
        else:
            print("\n=== REGISTER-ONLY DONE (reuse) ===")
            print(f"grid_id      : {gid}")
            print(f"pp_json_path : {pp_json_abs}\n")
        return

    # 正常 DAVE→PP→注册
    try:
        res = run_register_only(
            api=api,
            selector=args.selector,
            geodata=[g.lower() for g in args.geodata],
            power_levels=[p.lower() for p in args.power_levels],
            own_area_geojson=args.own_area_geojson,
            town=args.town,
            postalcode=args.postalcode,
            nuts=args.nuts,
            nuts_year=args.nuts_year,
        )
        if not args.no_probe:
            try:
                buses = _probe_buses(api, res["grid_id"])
                print(f"[PF ] Probe buses: total={len(buses)}, first 12={buses[:12]}")
            except Exception as e:
                print(f"[PF ] Probe skipped: {e}")
        if args.run24:
            _run_timeseries(api, res["grid_id"], args.buses, args.date, args.hours, args.vmin, args.vmax, Path(res["export_dir"]))
    except BaseException as e:
        try:
            export_root = OUT_ROOT
            subs = [p for p in export_root.glob("dave_export_*") if p.is_dir()]
            out_dir = max(subs, key=lambda p: p.stat().st_mtime) if subs else export_root
        except Exception:
            out_dir = OUT_ROOT
        _write_error_log(out_dir, e, {"stage": "main"})
        print(f"DAVE 生成失败：{type(e).__name__}: {e}")
        sys.exit(2)


if __name__ == "__main__":
    os.environ.setdefault("OSMNX_SETTINGS_LOG_CONSOLE", "false")
    os.environ.setdefault("OSMNX_SETTINGS_TIMEOUT", "180")
    os.environ.setdefault("OSMNX_SETTINGS_RETRY_ON_EMPTY", "True")
    os.environ.setdefault("OSMNX_SETTINGS_RETRY_ON_RATE_LIMIT", "True")
    main()
