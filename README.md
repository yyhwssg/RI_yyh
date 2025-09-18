RI_yyh — Reproducible Quick-Run Guide (README.txt)

GOAL
-----
From a clean machine to a working demo that:
1) Generates a 1 km² MV network (DAVE),
2) Registers it to the local API,
3) Converts to OpenDSS,
4) Runs a FAST PMD snapshot check (no 24h run, so it’s quick).

This guide avoids machine-specific paths and does not require pushing large temporary files to GitHub.


ONE-TIME SETUP
--------------
1) Install: Python 3.11+ and Julia 1.10+ (1.11 recommended).

2) Clone the repo (official URL):
   - Windows PowerShell:
       git clone https://github.com/yyhwssg/RI_yyh.git
       cd RI_yyh
   - macOS/Linux (bash):
       git clone https://github.com/yyhwssg/RI_yyh.git
       cd RI_yyh

3) Create venv and install Python deps:
   - Windows PowerShell:
       python -m venv .venv
       .\.venv\Scripts\Activate.ps1
       pip install -U pip wheel
       pip install -r requirements.txt
   - macOS/Linux (bash):
       python3 -m venv .venv
       source .venv/bin/activate
       pip install -U pip wheel
       pip install -r requirements.txt

4) Start the API (keep it running in this terminal):
   - All:
       uvicorn api.main:app --host 127.0.0.1 --port 8000
     (Open a second terminal for the steps below.)


PREPARE JULIA DEPENDENCIES (first run only)
-------------------------------------------
Open a new terminal at the repo root.

- Windows PowerShell:
    # optional if Julia is not in PATH:
    # $env:JULIA_EXE="C:\Program Files\Julia-1.11.6\bin\julia.exe"
    julia --project=./scripts ./scripts/_check_env.jl

- macOS/Linux (bash):
    julia --project=./scripts ./scripts/_check_env.jl

This script installs/repairs the required Julia packages (OpenDSSDirect, PowerModelsDistribution, Ipopt).
If the registry is flaky, it falls back to direct Git URLs.


DAVE DATAPOOL HOTFIX (REQUIRED ON SOME OLDER BUILDS)
----------------------------------------------------
Older DAVE releases ship a broken datapool link that causes HTML/503 responses. Patch the installed package once
before the first run; it will download a small bundle and cache it locally.

- Windows PowerShell (run inside the venv):
    $rd = (python -c "import dave_core, os; print(os.path.join(os.path.dirname(dave_core.__file__), 'datapool', 'read_data.py'))").Trim()
    Copy-Item $rd "$($rd).bak" -Force
    $pattern = '^\s*url\s*=.*$'
    $replacement = "    url = f'https://owncloud.fraunhofer.de/index.php/s/Y5J1lBxeau3N48p/download?path=%2F&files={filename}'"
    $content = Get-Content -Raw -Encoding UTF8 $rd
    $content = [System.Text.RegularExpressions.Regex]::Replace($content, $pattern, $replacement, 'Multiline')
    Set-Content -Path $rd -Value $content -Encoding UTF8
    Select-String -Path $rd -Pattern '^\s*url\s*='

- macOS/Linux (bash; run inside the venv):
    RD="$(python - <<'PY'
import dave_core, os
print(os.path.join(os.path.dirname(dave_core.__file__), "datapool", "read_data.py"))
PY
)"
    cp "$RD" "$RD.bak"
    python - <<'PY'
import os, io, re
p = os.environ["RD"]
txt = io.open(p, "r", encoding="utf8").read()
txt = re.sub(r'(?m)^\s*url\s*=.*$', "    url = f'https://owncloud.fraunhofer.de/index.php/s/Y5J1lBxeau3N48p/download?path=%2F&files={filename}'", txt)
io.open(p, "w", encoding="utf8").write(txt)
print("Patched:", p)
PY
    grep -n "url =" "$RD"

(Optional) Clear prior HTTP cache if you already saw 503/HTML:
- Windows: Remove-Item .\intermediate\dave_cache\http_cache.sqlite -ErrorAction SilentlyContinue
- macOS/Linux: rm -f ./intermediate/dave_cache/http_cache.sqlite


ONE-SHOT QUICK RUN (fast, no 24h)
---------------------------------

WINDOWS POWERSHELL — copy ALL below and press Enter:
----------------------------------------------------
# Activate venv
.\.venv\Scripts\Activate.ps1

# Ensure localhost is NOT proxied
Remove-Item Env:\HTTP_PROXY  -ErrorAction SilentlyContinue
Remove-Item Env:\HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:\http_proxy  -ErrorAction SilentlyContinue
Remove-Item Env:\https_proxy -ErrorAction SilentlyContinue

# Robust Overpass mirror + longer timeout for OSMnx
$env:OVERPASS_API_URL="https://lz4.overpass-api.de/api/interpreter"
$env:OSMNX_OVERPASS_ENDPOINT=$env:OVERPASS_API_URL
$env:OSMNX_REQUESTS_TIMEOUT="180"

# 1) Generate & register (AOI = data\aoi\micro_bremen.geojson)
python tools\dave_generate_and_register.py `
  --selector own_area `
  --own_area_geojson data\aoi\micro_bremen.geojson `
  --geodata roads `
  --power-levels mv `
  --no_probe

# 2) Pick newest net_power.json and register again (stable grid_id)
$PPJSON = Get-ChildItem ".\intermediate\dave_export\dave_export_*" -Filter net_power.json -Recurse |
  Sort-Object LastWriteTime -Desc | Select-Object -Expand FullName -First 1
$regBody = @{ name=("quick_" + (Get-Date -Format "yyyyMMdd_HHmmss")); pp_json_path=$PPJSON } | ConvertTo-Json
$reg = Invoke-RestMethod -Method Post "http://127.0.0.1:8000/grid/register" -Body $regBody -ContentType "application/json" -TimeoutSec 120
$GRID = $reg.grid_id
"GRID = $GRID"

# 3) Convert PP JSON → OpenDSS (master.dss + busmap.csv)
$OUT_DSS = Join-Path "." ("intermediate\dss_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
python tools\ppjson_to_dss.py --pp_json "$PPJSON" --out_dir "$OUT_DSS"
$MASTER = Join-Path $OUT_DSS "master.dss"
$BUSMAP = Join-Path $OUT_DSS "busmap.csv"
"DSS = $MASTER"
"BUSMAP = $BUSMAP"

# 4) Attach DSS to the grid
$DSS_UNIX = $MASTER -replace '\\','/'
$attachBody = @{ grid_id=$GRID; dss_master_path=$DSS_UNIX } | ConvertTo-Json
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/grid/attach_dss" -Body $attachBody -ContentType "application/json" | Out-Host

# 5) PMD snapshot — empty injection
$req_empty = @{
  grid_id         = $GRID
  engine          = "pmd"
  mode            = "summary"
  vmin            = 0.95
  vmax            = 1.05
  dss_master_path = $DSS_UNIX
  injections      = @{}
} | ConvertTo-Json -Depth 10
$r_empty = Invoke-RestMethod -Method Post "http://127.0.0.1:8000/pf/run" -Body $req_empty -ContentType "application/json" -TimeoutSec 900
"PMD(empty) → termination = $($r_empty.result.summary.termination_status)"

# 6) PMD snapshot — single-bus injection (5 kW @ pf=0.95)
$bus = (Import-Csv $BUSMAP)[0].bus_name
$pf  = 0.95
$qratio = [math]::Tan([math]::Acos($pf))
$inj = @{}; $inj[$bus] = @([double]5.0, [double](5.0*$qratio))
$req_inj = @{
  grid_id         = $GRID
  engine          = "pmd"
  mode            = "summary"
  vmin            = 0.95
  vmax            = 1.05
  dss_master_path = $DSS_UNIX
  injections      = $inj
} | ConvertTo-Json -Depth 10
$r_inj = Invoke-RestMethod -Method Post "http://127.0.0.1:8000/pf/run" -Body $req_inj -ContentType "application/json" -TimeoutSec 900
"PMD(injection) → termination = $($r_inj.result.summary.termination_status)"

# 7) Save JSONs (optional)
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$SAVE  = Join-Path "." ("intermediate\quickcheck_" + $STAMP)
New-Item -ItemType Directory -Force -Path $SAVE | Out-Null
$r_empty | ConvertTo-Json -Depth 50 | Out-File -Encoding utf8 (Join-Path $SAVE "pmd_empty_summary.json")
$r_inj   | ConvertTo-Json -Depth 50 | Out-File -Encoding utf8 (Join-Path $SAVE "pmd_injection_summary.json")
"Saved to $SAVE"


MACOS / LINUX (bash) — copy ALL below and run:
----------------------------------------------
# Activate venv
source .venv/bin/activate

# Avoid proxying localhost
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# Overpass mirror + timeout
export OVERPASS_API_URL="https://lz4.overpass-api.de/api/interpreter"
export OSMNX_OVERPASS_ENDPOINT="$OVERPASS_API_URL"
export OSMNX_REQUESTS_TIMEOUT="180"

# 1) Generate & register
python tools/dave_generate_and_register.py \
  --selector own_area \
  --own_area_geojson data/aoi/micro_bremen.geojson \
  --geodata roads \
  --power-levels mv \
  --no_probe

# 2) Latest PP JSON + re-register
PPJSON=$(ls -1t intermediate/dave_export/*/net_power.json | head -n1)
GRID=$(curl -s -X POST http://127.0.0.1:8000/grid/register \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"quick_$(date +%Y%m%d_%H%M%S)\",\"pp_json_path\":\"$PPJSON\"}" \
  | python -c 'import sys,json;print(json.load(sys.stdin)["grid_id"])')
echo "GRID=$GRID"

# 3) Convert to DSS
OUT_DSS="intermediate/dss_$(date +%Y%m%d_%H%M%S)"
python tools/ppjson_to_dss.py --pp_json "$PPJSON" --out_dir "$OUT_DSS"
MASTER="$OUT_DSS/master.dss"
BUSMAP="$OUT_DSS/busmap.csv"

# 4) Attach DSS
curl -s -X POST http://127.0.0.1:8000/grid/attach_dss \
  -H 'Content-Type: application/json' \
  -d "{\"grid_id\":\"$GRID\",\"dss_master_path\":\"$MASTER\"}"

# 5) PMD snapshot — empty
REQ_EMPTY="{\"grid_id\":\"$GRID\",\"engine\":\"pmd\",\"mode\":\"summary\",\"vmin\":0.95,\"vmax\":1.05,\"dss_master_path\":\"$MASTER\",\"injections\":{}}"
curl -s -X POST http://127.0.0.1:8000/pf/run -H 'Content-Type: application/json' -d "$REQ_EMPTY"

# 6) PMD snapshot — single-bus injection (5 kW @ pf=0.95)
BUS=$(python - <<'PY'
import csv,sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
print(rows[0].get('bus_name') or rows[0].get('name') or '')
PY
"$BUSMAP")
# pf=0.95 → q = p * tan(acos(pf)) ≈ 1.643421 for p=5 kW
REQ_INJ="{\"grid_id\":\"$GRID\",\"engine\":\"pmd\",\"mode\":\"summary\",\"vmin\":0.95,\"vmax\":1.05,\"dss_master_path\":\"$MASTER\",\"injections\":{\"$BUS\":[5.0,1.643421]}}"
curl -s -X POST http://127.0.0.1:8000/pf/run -H 'Content-Type: application/json' -d "$REQ_INJ"


EXPECTED OUTPUT (quick check)
-----------------------------
- Console shows something like:
    PMD(empty) → termination = LOCALLY_SOLVED
    PMD(injection) → termination = LOCALLY_SOLVED
- Optional JSONs saved under:
    intermediate/quickcheck_YYYYMMDD_HHMMSS/


.GITIGNORE (recommended)
------------------------
.venv/
**/__pycache__/
*.pyc

intermediate/
saved_json/grids/
run.txt
run_logs/

.vscode/
.DS_Store
Thumbs.db

scripts/Project.toml.bak
scripts/Manifest.toml
.julia/
.ipynb_checkpoints/


WHAT EACH STEP DOES (plain English)
-----------------------------------
• Create venv & install requirements.txt
  Sets up an isolated Python environment so the demo doesn’t depend on global packages.

• Start API (uvicorn api.main:app)
  Launches the local FastAPI service exposing /grid/register, /pf/run, etc. All later steps talk to http://127.0.0.1:8000.

• Julia _check_env.jl
  Installs and verifies Julia packages (OpenDSSDirect, PowerModelsDistribution, Ipopt). Falls back to Git URLs if the public registry is slow.

• DAVE generate & register
  Builds a synthetic MV network from OpenStreetMap for the AOI (micro_bremen.geojson). Saves to intermediate/dave_export/.../net_power.json and registers once automatically.

• Re-register latest net_power.json
  Registers the newest PP JSON again to get a stable grid_id for subsequent calls.

• Convert PP JSON to OpenDSS (ppjson_to_dss.py)
  Produces master.dss and busmap.csv — inputs for OpenDSSDirect and PMD.

• Attach DSS path to grid
  Stores the DSS master path in the registry so the API knows where the grid’s DSS lives (forward slashes work cross-platform).

• PMD snapshot (empty and single-bus injection)
  Runs fast optimization-based PF with no/with small injection to sanity-check feasibility and the full pipeline.

• Save outputs (optional)
  Dumps returned JSONs to intermediate/quickcheck_* for sharing or debugging.
