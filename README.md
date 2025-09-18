Below is a single, copy-paste friendly **README.txt** you can drop into your repo root. It’s fully in English, includes a project tree, one-shot commands for Windows (PowerShell) and macOS/Linux (bash), and ends with a plain-English explanation of what each step does.

---

PF\_CLEAN — Reproducible Quick-Run Guide (README.txt)

## Goal

From a clean machine to a working demo that:

1. Generates a 1 km² MV network (DAVE),
2. Registers it to the local API,
3. Converts to OpenDSS,
4. Runs a FAST PMD snapshot check (no 24h run, so it’s quick).

This guide avoids machine-specific paths and does not require pushing large temporary files to GitHub.

## Project Structure (suggested)

PF\_CLEAN/
api/
main.py                     # FastAPI app (uvicorn entry)
data/
aoi/
micro\_bremen.geojson      # Example AOI polygon (1 km² demo)
dss/                        # (optional) place for fixed DSS cases
intermediate/                 # runtime outputs (ignored by git)
saved\_json/
grid\_registry.json          # registry DB for grids (created/updated at runtime)
grids/                      # per-grid snapshots (ignored by git)
scripts/
Project.toml                # Julia env for the runner
run\_pf\_dss.jl               # Julia runner for OpenDSS & PMD (your fixed version)
\_check\_env.jl               # first-time Julia deps check/install
tools/
dave\_generate\_and\_register.py
ppjson\_to\_dss.py
requirements.txt              # Python deps (pip)
.gitignore                    # ignore venv, intermediates, caches, etc. (see bottom)
README.txt                    # this file

## One-time Setup

1. Install Python 3.11+ and Julia 1.10+ (1.11 recommended).
2. Clone repo:

   * Windows PowerShell:
     git clone [https://github.com/](https://github.com/)<your-org-or-name>/PF\_CLEAN.git
     cd PF\_CLEAN
   * macOS/Linux (bash):
     git clone [https://github.com/](https://github.com/)<your-org-or-name>/PF\_CLEAN.git
     cd PF\_CLEAN
3. Create venv and install Python deps:

   * Windows PowerShell:
     python -m venv .venv
     ..venv\Scripts\Activate.ps1
     pip install -U pip wheel
     pip install -r requirements.txt
   * macOS/Linux (bash):
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -U pip wheel
     pip install -r requirements.txt
4. Start the API (keep it running in this terminal):

   * All:
     uvicorn api.main\:app --host 127.0.0.1 --port 8000
     (Open a second terminal for the following steps.)

## Prepare Julia Dependencies (first run only)

In a new terminal at repo root:

* Windows PowerShell:

  # optional if Julia is not in PATH:

  # \$env\:JULIA\_EXE="C:\Program Files\Julia-1.11.6\bin\julia.exe"

  julia --project=./scripts ./scripts/\_check\_env.jl

* macOS/Linux (bash):
  julia --project=./scripts ./scripts/\_check\_env.jl

This script installs/repairs the required Julia packages (OpenDSSDirect, PowerModelsDistribution, Ipopt). If the registry is flaky, it falls back to direct Git URLs.

## ONE-SHOT QUICK RUN (fast, no 24h)

## Windows PowerShell (copy all, paste, Enter):

# Activate venv

..venv\Scripts\Activate.ps1

# Ensure localhost is NOT proxied

Remove-Item Env:\HTTP\_PROXY  -ErrorAction SilentlyContinue
Remove-Item Env:\HTTPS\_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:\http\_proxy  -ErrorAction SilentlyContinue
Remove-Item Env:\https\_proxy -ErrorAction SilentlyContinue

# Robust Overpass mirror + longer timeout for OSMnx

\$env\:OVERPASS\_API\_URL="[https://lz4.overpass-api.de/api/interpreter](https://lz4.overpass-api.de/api/interpreter)"
\$env\:OSMNX\_OVERPASS\_ENDPOINT=\$env\:OVERPASS\_API\_URL
\$env\:OSMNX\_REQUESTS\_TIMEOUT="180"

# 1) Generate & register (AOI = data\aoi\micro\_bremen.geojson)

python tools\dave\_generate\_and\_register.py `  --selector own_area`
\--own\_area\_geojson data\aoi\micro\_bremen.geojson `  --geodata roads`
\--power-levels mv \`
\--no\_probe

# 2) Pick newest net\_power.json and register again (stable grid\_id)

\$PPJSON = Get-ChildItem ".\intermediate\dave\_export\dave\_export\_\*" -Filter net\_power.json -Recurse |
Sort-Object LastWriteTime -Desc | Select-Object -Expand FullName -First 1
\$regBody = @{ name=("quick\_" + (Get-Date -Format "yyyyMMdd\_HHmmss")); pp\_json\_path=\$PPJSON } | ConvertTo-Json
\$reg = Invoke-RestMethod -Method Post "[http://127.0.0.1:8000/grid/register](http://127.0.0.1:8000/grid/register)" -Body \$regBody -ContentType "application/json" -TimeoutSec 120
\$GRID = \$reg.grid\_id
"GRID = \$GRID"

# 3) Convert PP JSON → OpenDSS (master.dss + busmap.csv)

\$OUT\_DSS = Join-Path "." ("intermediate\dss\_" + (Get-Date -Format "yyyyMMdd\_HHmmss"))
python tools\ppjson\_to\_dss.py --pp\_json "\$PPJSON" --out\_dir "\$OUT\_DSS"
\$MASTER = Join-Path \$OUT\_DSS "master.dss"
\$BUSMAP = Join-Path \$OUT\_DSS "busmap.csv"
"DSS = \$MASTER"
"BUSMAP = \$BUSMAP"

# 4) Attach DSS to the grid

\$DSS\_UNIX = \$MASTER -replace '\\','/'
\$attachBody = @{ grid\_id=\$GRID; dss\_master\_path=\$DSS\_UNIX } | ConvertTo-Json
Invoke-RestMethod -Method Post "[http://127.0.0.1:8000/grid/attach\_dss](http://127.0.0.1:8000/grid/attach_dss)" -Body \$attachBody -ContentType "application/json" | Out-Host

# 5) PMD snapshot — empty injection

\$req\_empty = @{
grid\_id         = \$GRID
engine          = "pmd"
mode            = "summary"
vmin            = 0.95
vmax            = 1.05
dss\_master\_path = \$DSS\_UNIX
injections      = @{}
} | ConvertTo-Json -Depth 10
\$r\_empty = Invoke-RestMethod -Method Post "[http://127.0.0.1:8000/pf/run](http://127.0.0.1:8000/pf/run)" -Body \$req\_empty -ContentType "application/json" -TimeoutSec 900
"PMD(empty) → termination = \$(\$r\_empty.result.summary.termination\_status)"

# 6) PMD snapshot — single-bus injection (5 kW @ pf=0.95)

\$bus = (Import-Csv \$BUSMAP)\[0].bus\_name
\$pf  = 0.95
\$qratio = \[math]::Tan(\[math]::Acos(\$pf))
\$inj = @{}; \$inj\[\$bus] = @(\[double]5.0, [double](5.0*$qratio))
\$req\_inj = @{
grid\_id         = \$GRID
engine          = "pmd"
mode            = "summary"
vmin            = 0.95
vmax            = 1.05
dss\_master\_path = \$DSS\_UNIX
injections      = \$inj
} | ConvertTo-Json -Depth 10
\$r\_inj = Invoke-RestMethod -Method Post "[http://127.0.0.1:8000/pf/run](http://127.0.0.1:8000/pf/run)" -Body \$req\_inj -ContentType "application/json" -TimeoutSec 900
"PMD(injection) → termination = \$(\$r\_inj.result.summary.termination\_status)"

# 7) Save JSONs (optional)

\$STAMP = Get-Date -Format "yyyyMMdd\_HHmmss"
\$SAVE  = Join-Path "." ("intermediate\quickcheck\_" + \$STAMP)
New-Item -ItemType Directory -Force -Path \$SAVE | Out-Null
\$r\_empty | ConvertTo-Json -Depth 50 | Out-File -Encoding utf8 (Join-Path \$SAVE "pmd\_empty\_summary.json")
\$r\_inj   | ConvertTo-Json -Depth 50 | Out-File -Encoding utf8 (Join-Path \$SAVE "pmd\_injection\_summary.json")
"Saved to \$SAVE"
-----------------

## macOS / Linux (bash) quick run (no 24h):

# Activate venv

source .venv/bin/activate

# Avoid proxying localhost

unset HTTP\_PROXY HTTPS\_PROXY http\_proxy https\_proxy

# Overpass mirror + timeout

export OVERPASS\_API\_URL="[https://lz4.overpass-api.de/api/interpreter](https://lz4.overpass-api.de/api/interpreter)"
export OSMNX\_OVERPASS\_ENDPOINT="\$OVERPASS\_API\_URL"
export OSMNX\_REQUESTS\_TIMEOUT="180"

# 1) Generate & register

python tools/dave\_generate\_and\_register.py&#x20;
\--selector own\_area&#x20;
\--own\_area\_geojson data/aoi/micro\_bremen.geojson&#x20;
\--geodata roads&#x20;
\--power-levels mv&#x20;
\--no\_probe

# 2) Latest PP JSON + re-register

PPJSON=\$(ls -1t intermediate/dave\_export/\*/net\_power.json | head -n1)
GRID=\$(curl -s -X POST [http://127.0.0.1:8000/grid/register](http://127.0.0.1:8000/grid/register)&#x20;
-H 'Content-Type: application/json'&#x20;
-d "{"name":"quick\_\$(date +%Y%m%d\_%H%M%S)","pp\_json\_path":"\$PPJSON"}"&#x20;
\| python -c 'import sys,json;print(json.load(sys.stdin)\["grid\_id"])')
echo "GRID=\$GRID"

# 3) Convert to DSS

OUT\_DSS="intermediate/dss\_\$(date +%Y%m%d\_%H%M%S)"
python tools/ppjson\_to\_dss.py --pp\_json "\$PPJSON" --out\_dir "\$OUT\_DSS"
MASTER="\$OUT\_DSS/master.dss"
BUSMAP="\$OUT\_DSS/busmap.csv"

# 4) Attach DSS

curl -s -X POST [http://127.0.0.1:8000/grid/attach\_dss](http://127.0.0.1:8000/grid/attach_dss)&#x20;
-H 'Content-Type: application/json'&#x20;
-d "{"grid\_id":"\$GRID","dss\_master\_path":"\$MASTER"}"

# 5) PMD snapshot — empty

REQ\_EMPTY="{"grid\_id":"\$GRID","engine":"pmd","mode":"summary","vmin":0.95,"vmax":1.05,"dss\_master\_path":"\$MASTER","injections":{}}"
curl -s -X POST [http://127.0.0.1:8000/pf/run](http://127.0.0.1:8000/pf/run) -H 'Content-Type: application/json' -d "\$REQ\_EMPTY"

# 6) PMD snapshot — single-bus injection

BUS=\$(python - <<'PY'
import csv,sys
with open(sys.argv\[1], newline='', encoding='utf-8') as f:
rows=list(csv.DictReader(f))
print(rows\[0].get('bus\_name') or rows\[0].get('name') or '')
PY
"\$BUSMAP")
REQ\_INJ="{"grid\_id":"\$GRID","engine":"pmd","mode":"summary","vmin":0.95,"vmax":1.05,"dss\_master\_path":"\$MASTER","injections":{"\$BUS":\[5.0,1.643421]}}"
curl -s -X POST [http://127.0.0.1:8000/pf/run](http://127.0.0.1:8000/pf/run) -H 'Content-Type: application/json' -d "\$REQ\_INJ"
--------------------------------------------------------------------------------------------------------------------------------

## Expected Output (quick check)

* Console shows something like:
  PMD(empty) → termination = LOCALLY\_SOLVED
  PMD(injection) → termination = LOCALLY\_SOLVED
* Optional JSONs saved under intermediate/quickcheck\_YYYYMMDD\_HHMMSS/

## .gitignore (recommended)

.venv/
**pycache**/
\*.pyc

intermediate/
saved\_json/grids/
run.txt
run\_logs/

.vscode/
.DS\_Store
Thumbs.db

scripts/Project.toml.bak
scripts/Manifest.toml
.julia/
.ipynb\_checkpoints/

## What each step does (plain English)

• Create venv & install requirements.txt
Sets up an isolated Python environment so the demo doesn’t depend on global packages.

• Start API (uvicorn api.main\:app)
Launches the local FastAPI service that exposes endpoints like /grid/register, /pf/run, etc. All later steps talk to this server at [http://127.0.0.1:8000](http://127.0.0.1:8000).

• Julia \_check\_env.jl
Installs and verifies Julia packages (OpenDSSDirect, PowerModelsDistribution, Ipopt). If the official registry is slow, it falls back to Git URLs.

• DAVE generate & register
Builds a synthetic MV network from OpenStreetMap for the provided AOI polygon (micro\_bremen.geojson). Saves to intermediate/dave\_export/.../net\_power.json. Also registers the grid once automatically.

• Re-register latest net\_power.json
Explicitly registers the latest PP JSON again to get a stable grid\_id (used by subsequent calls). This avoids ambiguity when multiple exports exist.

• Convert PP JSON to OpenDSS (ppjson\_to\_dss.py)
Produces master.dss and busmap.csv. The DSS file is the input for both OpenDSSDirect and PMD (via PMD’s DSS parser).

• Attach DSS path to grid
Stores the DSS master path in the registry so the API knows where that grid’s DSS lives (use forward slashes for cross-platform compatibility).

• PMD snapshot (empty)
Runs a single optimization-based power flow without any extra injections. We check feasibility (termination\_status, model, solver\_time\_sec). This is fast, safe, and does not rely on voltage scaling assumptions.

• PMD snapshot (single-bus injection)
Picks the first bus from busmap.csv and injects 5 kW at pf=0.95. This verifies the pipeline also works with a small disturbance.

• Save outputs (optional)
Dumps the returned JSONs to intermediate/quickcheck\_\*/ for grading, sharing, or debugging.

## Notes / Tips

* Overpass can be flaky; we default to a stable mirror and longer timeouts. If it still fails, rerun the generation step or switch networks.
* Do NOT commit .venv, intermediate, or saved\_json/grids/\*\* to Git. They can be large and machine-specific.
* If you later want time-series runs: the API supports /profiles/build and /pf/run\_timeseries, but they are intentionally excluded here to keep the demo quick and consistent.

---
