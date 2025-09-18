# --- config ---
$API   = "http://127.0.0.1:8000"
$ROOT  = (Get-Location).Path
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"

function Assert-True($cond, $msg){
  if(-not $cond){ throw "ASSERT FAIL: $msg" }
}

function Show-Head($obj, $title){
  Write-Host "`n==== $title ====" -ForegroundColor Cyan
  ($obj | ConvertTo-Json -Depth 6) | Write-Host
}

Write-Host "API = $API"

# 0) health/version
$h = Invoke-RestMethod "$API/health" -TimeoutSec 10
Assert-True ($h.status -eq "ok") "/health"
$deps = Invoke-RestMethod "$API/health/deps" -TimeoutSec 10
$ver  = Invoke-RestMethod "$API/version" -TimeoutSec 20
Show-Head $ver "version"

# 1) 准备 pandapower 源
#   优先找 DAVE 导出的 net_power.json；若没有，则回退到预置 pickle
$PPJSON = Get-ChildItem "$ROOT\intermediate\dave_export\dave_export_*" -Filter net_power.json -Recurse -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime -Desc | Select-Object -Expand FullName -First 1
if(-not $PPJSON){
  $PPPKL = Join-Path $ROOT "saved_json\grids\3b86aa918529\net.p"
  if(Test-Path $PPPKL){
    $regBody = @{ name=("validate_pp_" + $STAMP); pp_pickle_path="$PPPKL" } | ConvertTo-Json
  } else {
    throw "找不到 net_power.json，且无 saved_json\grids\3b86aa918529\net.p；请先跑一次 DAVE 或放入一个 .p/.json"
  }
}else{
  $regBody = @{ name=("validate_pp_" + $STAMP); pp_json_path="$PPJSON" } | ConvertTo-Json
}

# 2) 注册网架
$reg = Invoke-RestMethod -Method Post "$API/grid/register" -Body $regBody -ContentType "application/json" -TimeoutSec 180
$GRID = $reg.grid_id
Assert-True $GRID "grid/register 返回了 grid_id"
Show-Head $reg "grid/register"

# 3) 绑定 master.dss（用项目内示例/或你自己近期生成的）
$DSS = Join-Path $ROOT "data\dss\dave_case\master.dss"
if(-not (Test-Path $DSS)){ throw "缺少示例 DSS: $DSS" }
$attach = Invoke-RestMethod -Method Post "$API/grid/attach_dss" -Body (@{grid_id=$GRID; dss_master_path=$DSS} | ConvertTo-Json) -ContentType "application/json"
Assert-True ($attach.status -eq "ok") "/grid/attach_dss ok"
Show-Head $attach "grid/attach_dss"

# 4) 列出现有网架 & 获取单个
$lst = Invoke-RestMethod "$API/grid/list"
Assert-True ($lst.count -ge 1) "/grid/list count>=1"
$one = Invoke-RestMethod "$API/grid/get?grid_id=$GRID"
Assert-True ($one.grid_id -eq $GRID) "/grid/get grid_id match"
Show-Head $one "grid/get"

# 5) 列 DSS 母线（走 grid_id 自动取绑定路径）
$dssbus = Invoke-RestMethod "$API/dss/bus/list?grid_id=$GRID"
Assert-True ($dssbus.bus_count -ge 1) "/dss/bus/list 有母线"
$BUS = $dssbus.buses[0]
Write-Host "Use BUS = $BUS"

# 6) 生成 24h 曲线（BUS:annual_kWh:pf）
$buildReq = @{
  hours = 24
  date  = (Get-Date -Format "yyyy-MM-dd")
  code  = "H0"
  specs = @("$BUS:3500:0.95")
} | ConvertTo-Json
$prof = Invoke-RestMethod -Method Post "$API/profiles/build" -Body $buildReq -ContentType "application/json"
Assert-True ($prof.status -eq "ok") "/profiles/build ok"
$p0 = $prof.profiles.$BUS.P_kW[0]; $q0 = $prof.profiles.$BUS.Q_kVAr[0]
Write-Host "P0=$p0, Q0=$q0"

# 7) /pf/run - pandapower
$ppRunReq = @{
  grid_id = $GRID
  engine  = "pandapower"
  mode    = "summary"
  vmin    = 0.94
  vmax    = 1.06
  injections = @{}
} | ConvertTo-Json
$ppRun = Invoke-RestMethod -Method Post "$API/pf/run" -Body $ppRunReq -ContentType "application/json"
Assert-True ($ppRun.status -eq "ok") "/pf/run pandapower ok"
Assert-True ($ppRun.result.summary.limits.vmin -eq 0.94 -and $ppRun.result.summary.limits.vmax -eq 1.06) "PP limits 回显一致"
Assert-True ($ppRun.result.aggregate.min_vpu -ne $null) "PP aggregate.min_vpu 存在"
Show-Head $ppRun "pf/run (pandapower)"

# 8) /pf/run - opendss（使用绑定的 DSS，单点注入）
$inj = @{ $BUS = @([double]$p0, [double]$q0) }
$dssRunReq = @{
  grid_id = $GRID
  engine  = "opendss"
  mode    = "summary"
  vmin    = 0.95
  vmax    = 1.05
  injections = $inj
} | ConvertTo-Json -Depth 6
$dssRun = Invoke-RestMethod -Method Post "$API/pf/run" -Body $dssRunReq -ContentType "application/json"
Assert-True ($dssRun.status -eq "ok") "/pf/run opendss ok"
Assert-True ($dssRun.result.summary.limits.vmin -eq 0.95 -and $dssRun.result.summary.limits.vmax -eq 1.05) "OpenDSS limits 回显一致"
Assert-True ($dssRun.result.aggregate.min_vpu -ne $null) "OpenDSS aggregate.min_vpu 存在"
Show-Head $dssRun "pf/run (opendss)"

# 9) /pf/run - pmd（同一注入）
$pmdRunReq = @{
  grid_id = $GRID
  engine  = "pmd"
  mode    = "summary"
  vmin    = 0.95
  vmax    = 1.05
  injections = $inj
} | ConvertTo-Json -Depth 6
$pmdRun = Invoke-RestMethod -Method Post "$API/pf/run" -Body $pmdRunReq -ContentType "application/json" -TimeoutSec 900
Assert-True ($pmdRun.status -eq "ok") "/pf/run pmd ok"
Assert-True ($pmdRun.result.summary.limits.vmin -eq 0.95 -and $pmdRun.result.summary.limits.vmax -eq 1.05) "PMD limits 回显一致"
if($pmdRun.result.aggregate){
  Assert-True ($pmdRun.result.aggregate.min_vpu -ne $null) "PMD aggregate.min_vpu 存在"
}
Show-Head $pmdRun "pf/run (pmd)"

# 10) /pf/run_timeseries - pandapower（用 profiles/build 的结果）
$ppTSReq = @{
  grid_id = $GRID
  engine  = "pandapower"
  mode    = "summary"
  hours   = 24
  vmin    = 0.95
  vmax    = 1.05
  profiles= $prof.profiles
} | ConvertTo-Json -Depth 10
$ppTS = Invoke-RestMethod -Method Post "$API/pf/run_timeseries" -Body $ppTSReq -ContentType "application/json" -TimeoutSec 1200
Assert-True ($ppTS.status -eq "ok") "/pf/run_timeseries pandapower ok"
Assert-True ($ppTS.series.min_vpu.Count -eq 24) "PP TS 长度 24"
Show-Head $ppTS.summary "pf/run_timeseries (pandapower) summary"

# 11) /pf/run_timeseries - pmd（同 profiles）
$pmdTSReq = @{
  grid_id = $GRID
  engine  = "pmd"
  mode    = "summary"
  hours   = 24
  vmin    = 0.95
  vmax    = 1.05
  profiles= $prof.profiles
} | ConvertTo-Json -Depth 10
$pmdTS = Invoke-RestMethod -Method Post "$API/pf/run_timeseries" -Body $pmdTSReq -ContentType "application/json" -TimeoutSec 1800
Assert-True ($pmdTS.status -eq "ok") "/pf/run_timeseries pmd ok"
if($pmdTS.series.min_vpu -and $pmdTS.series.min_vpu.Count){
  Assert-True ($pmdTS.series.min_vpu.Count -eq 24) "PMD TS 长度 24（若返回了电压摘要）"
}
Show-Head $pmdTS.summary "pf/run_timeseries (pmd) summary"

Write-Host "`nALL CHECKS PASSED" -ForegroundColor Green
