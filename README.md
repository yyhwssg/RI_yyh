# PF_CLEAN

## 1) Python 环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

## 2) Julia 环境
julia -e "using Pkg; Pkg.activate(joinpath(raw\"$(@__DIR__)\scripts\")); Pkg.instantiate()"

（或）
cd scripts
julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate()"

## 3) 启动服务
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

## 4) 典型流程
- DAVE 生成：python tools\dave_generate_and_register.py --selector own_area --own_area_geojson data\aoi\micro_bremen.geojson --geodata roads --power-levels mv --no_probe
- 注册后用 /pf/run（pandapower）
- OpenDSS/PMD 直接用 /pf/run 或 /pf/run_timeseries（带 dss_master_path）
