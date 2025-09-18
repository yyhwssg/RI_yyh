# tools/ppjson_to_dss.py
# -*- coding: utf-8 -*-
"""
Convert pandapower JSON (DAVE output) -> OpenDSS master.dss (三相、以 MV 为主)

修正点（相对你当前版本）：
- 统一频率：在 Circuit/Vsource/Line/Transformer/Load 全部指定 basefreq/freq=50
- 头部采用：New Circuit.auto ... + set defaultbasefrequency=50 + Edit Vsource.Source ...
  （避免重复 New Vsource 导致的 Duplicate new elem 报错）
- Line 显式写 basefreq=50（修复 PMD 警告）
- 仍沿用：Line 按每公里参数 + units=km；Transformer 用 vk%、vkr% 还原 xhl/%r；
  Load 使用 kV=vn_kv/sqrt(3)（三相 wye）

生成文件：
- <out_dir>/master.dss
- <out_dir>/busmap.csv
"""
import os, math, argparse
import pandapower as pp
import pandas as pd

def _sanitize(s: str) -> str:
    return str(s).replace(" ", "_").replace(".", "_").replace("-", "_")

def _name(net, idx):
    if "name" in net.bus.columns:
        n = str(net.bus.at[idx, "name"])
        if n and n.strip():
            return _sanitize(n)
    return str(int(idx))

def convert(pp_json: str, out_dir: str):
    net = pp.from_json(pp_json)
    os.makedirs(out_dir, exist_ok=True)
    dss_path = os.path.join(out_dir, "master.dss")

    lines = []
    lines.append("clear")

    # —— 选参考母线（若无 ext_grid，则取最高电压母线）——
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        slack_bus = int(net.ext_grid.iloc[0]["bus"])
    else:
        # 与服务中 fix_slack 逻辑一致：取最高电压母线（并发 tie 取最小索引）
        vn = net.bus["vn_kv"].astype(float)
        max_vn = float(vn.max())
        cand = vn.index[vn == max_vn].tolist()
        slack_bus = int(min(cand)) if cand else int(net.bus.index[0])

    slack_name = _name(net, slack_bus)
    slack_kv_ll = float(net.bus.at[slack_bus, "vn_kv"])

    # —— 统一 50 Hz：在 Circuit 指定 basefreq，随后再 set defaultbasefrequency=50 —— 
    lines.append(f"New Circuit.auto basekv={slack_kv_ll} pu=1.0 phases=3 bus1={slack_name} basefreq=50")
    lines.append("set defaultbasefrequency=50")

    # —— 只 Edit Vsource.Source（Circuit 已创建 Vsource）——
    lines.append(f"Edit Vsource.Source bus1={slack_name} phases=3 basekv={slack_kv_ll} pu=1.0 freq=50")

    # —— Lines（每公里参数 + basefreq=50 + units=km）——
    if hasattr(net, "line") and len(net.line):
        for i in net.line.index:
            r1 = float(net.line.at[i, "r_ohm_per_km"])
            x1 = float(net.line.at[i, "x_ohm_per_km"])
            c1 = float(net.line.at[i, "c_nf_per_km"]) if "c_nf_per_km" in net.line.columns else 0.0
            length_km = float(net.line.at[i, "length_km"])
            fb = _name(net, int(net.line.at[i, "from_bus"]))
            tb = _name(net, int(net.line.at[i, "to_bus"]))
            lname = _sanitize(f"L_{i}")
            lines.append(
                "New Line.{lname} bus1={fb} bus2={tb} phases=3 length={L} units=km "
                "r1={r1} x1={x1} c1={c1} basefreq=50"
                .format(lname=lname, fb=fb, tb=tb, L=length_km, r1=r1, x1=x1, c1=c1)
            )

    # —— Transformers（用 vk%、vkr% 恢复 xhl/%r，并设置 basefreq=50）——
    if hasattr(net, "trafo") and len(net.trafo):
        for i in net.trafo.index:
            hvb = _name(net, int(net.trafo.at[i, "hv_bus"]))
            lvb = _name(net, int(net.trafo.at[i, "lv_bus"]))
            sn_kva = float(net.trafo.at[i, "sn_mva"]) * 1000.0
            vk  = float(net.trafo.at[i, "vk_percent"])   if "vk_percent"  in net.trafo.columns else 6.0
            vkr = float(net.trafo.at[i, "vkr_percent"])  if "vkr_percent" in net.trafo.columns else 0.5
            xhl = max(0.0, (vk**2 - vkr**2))**0.5
            r_pct = vkr
            hv_kv = float(net.trafo.at[i, "vn_hv_kv"])
            lv_kv = float(net.trafo.at[i, "vn_lv_kv"])
            tname = _sanitize(f"T_{i}")
            lines.append(f"New Transformer.{tname} phases=3 windings=2 xhl={xhl:.6g} %r={r_pct:.6g} basefreq=50")
            lines.append(f"~ wdg=1 bus={hvb} conn=wye kv={hv_kv} kva={sn_kva:.6g}")
            lines.append(f"~ wdg=2 bus={lvb} conn=wye kv={lv_kv} kva={sn_kva:.6g}")

    # —— Loads（三相 wye，kV=LL/√3，basefreq=50）——
    if hasattr(net, "load") and len(net.load):
        for i in net.load.index:
            b = int(net.load.at[i, "bus"])
            p_kw = float(net.load.at[i, "p_mw"]) * 1000.0
            q_kvar = float(net.load.at[i, "q_mvar"]) * 1000.0
            kv_ln = float(net.bus.at[b, "vn_kv"]) / math.sqrt(3.0)
            lname = _sanitize(f"LD_{i}")
            lines.append(
                "New Load.{lname} bus1={bus} phases=3 conn=wye kV={kv:.6g} kW={p:.6g} kvar={q:.6g} basefreq=50"
                .format(lname=lname, bus=_name(net,b), kv=kv_ln, p=p_kw, q=q_kvar)
            )

    with open(dss_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # —— bus 列表（便于注入映射）——
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
