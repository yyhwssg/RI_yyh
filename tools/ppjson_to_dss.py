# tools/ppjson_to_dss.py
# -*- coding: utf-8 -*-
"""
Convert pandapower JSON (DAVE output) -> OpenDSS master.dss (3相、MV为主)
覆盖要点：
- ext_grid -> Vsource
- line     -> Line（按每公里参数+Length=长度[km]）
- trafo    -> Transformer（用 vk_percent/vkr_percent 还原 xhl/%r）
- load     -> Load（p_mw/q_mvar 转 kW/kvar；kV = vn_kv/√3）
假设：频率50Hz；连接方式默认三相Y。
"""
import os, math, argparse
import pandapower as pp
import pandas as pd

def _name(net, idx):
    if "name" in net.bus.columns:
        n = str(net.bus.at[idx, "name"])
        if n and n.strip():
            return _sanitize(n)
    return str(int(idx))

def _sanitize(s: str) -> str:
    return str(s).replace(" ", "_").replace(".", "_").replace("-", "_")

def convert(pp_json: str, out_dir: str):
    net = pp.from_json(pp_json)
    os.makedirs(out_dir, exist_ok=True)
    dss_path = os.path.join(out_dir, "master.dss")

    lines = []
    lines.append("clear")
    lines.append("set defaultbasefrequency=50")

    # Vsource（取 ext_grid 的第一个；basekv 用母线 LL 电压）
    slack_bus = None
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        slack_bus = int(net.ext_grid.iloc[0]["bus"])
    else:
        slack_bus = int(net.bus.index[0])
    slack_name = _name(net, slack_bus)
    slack_kv_ll = float(net.bus.at[slack_bus, "vn_kv"])
    lines.append(f'New Vsource.Source bus1={slack_name} phases=3 basekv={slack_kv_ll} pu=1.0')

    # Lines（按每公里参数，Length=length_km, units=km）
    if hasattr(net, "line") and len(net.line):
        for i in net.line.index:
            r1 = float(net.line.at[i, "r_ohm_per_km"])
            x1 = float(net.line.at[i, "x_ohm_per_km"])
            c1 = float(net.line.at[i, "c_nf_per_km"]) if "c_nf_per_km" in net.line.columns else 0.0
            length_km = float(net.line.at[i, "length_km"])
            fb = _name(net, int(net.line.at[i, "from_bus"]))
            tb = _name(net, int(net.line.at[i, "to_bus"]))
            lname = _sanitize(f"L_{i}")
            # phases 默认3相；r1/x1/c1 为“每单位长度”参数；长度在 Length 指定
            lines.append(
                f"New Line.{lname} bus1={fb} bus2={tb} phases=3 length={length_km} units=km "
                f"r1={r1} x1={x1} c1={c1}"
            )

    # Transformers（用 vk_percent/vkr_percent）
    if hasattr(net, "trafo") and len(net.trafo):
        for i in net.trafo.index:
            hvb = _name(net, int(net.trafo.at[i, "hv_bus"]))
            lvb = _name(net, int(net.trafo.at[i, "lv_bus"]))
            sn_kva = float(net.trafo.at[i, "sn_mva"]) * 1000.0
            vk = float(net.trafo.at[i, "vk_percent"])    if "vk_percent"  in net.trafo.columns else 6.0
            vkr = float(net.trafo.at[i, "vkr_percent"])  if "vkr_percent" in net.trafo.columns else 0.5
            xhl = max(0.0, (vk**2 - vkr**2))**0.5
            r_pct = vkr
            hv_kv = float(net.trafo.at[i, "vn_hv_kv"])
            lv_kv = float(net.trafo.at[i, "vn_lv_kv"])
            tname = _sanitize(f"T_{i}")
            lines.append(
                f"New Transformer.{tname} phases=3 windings=2 xhl={xhl:.6g} %r={r_pct:.6g}"
            )
            lines.append(
                f"~ wdg=1 bus={hvb} conn=wye kv={hv_kv} kva={sn_kva:.6g}"
            )
            lines.append(
                f"~ wdg=2 bus={lvb} conn=wye kv={lv_kv} kva={sn_kva:.6g}"
            )

    # Loads（把现有静态负荷也写入，便于与PP一致）
    if hasattr(net, "load") and len(net.load):
        for i in net.load.index:
            b = int(net.load.at[i, "bus"])
            p_kw = float(net.load.at[i, "p_mw"]) * 1000.0
            q_kvar = float(net.load.at[i, "q_mvar"]) * 1000.0
            kv_ln = float(net.bus.at[b, "vn_kv"]) / math.sqrt(3.0)
            lname = _sanitize(f"LD_{i}")
            lines.append(
                f"New Load.{lname} bus1={_name(net,b)} phases=3 conn=wye kV={kv_ln:.6g} kW={p_kw:.6g} kvar={q_kvar:.6g}"
            )

    with open(dss_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 额外给出一个 bus 列表，便于你选注入母线时参考
    busmap = os.path.join(out_dir, "busmap.csv")
    pd.DataFrame({
        "bus_index": [int(i) for i in net.bus.index],
        "bus_name":  [_name(net, int(i)) for i in net.bus.index],
        "vn_kv":     [float(net.bus.at[i, "vn_kv"]) for i in net.bus.index],
    }).to_csv(busmap, index=False, encoding="utf-8-sig")

    print("[OK] DSS saved ->", dss_path)
    print("[OK] Bus map   ->", busmap)
    return dss_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp_json", required=True)
    ap.add_argument("--out_dir", default="data/dss/dave_case")
    args = ap.parse_args()
    convert(args.pp_json, args.out_dir)
