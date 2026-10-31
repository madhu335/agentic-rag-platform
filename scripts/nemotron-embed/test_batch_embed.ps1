<#
  32-parallel-query batching test for Triton model nemotron_embed.
  Requires PowerShell 7+ (ForEach-Object -Parallel).

  Checks:
    1. All 32 requests succeed, each returns 2048 dims with L2 norm ~= 1.0
    2. Dynamic batching actually happened (inference_count vs execution_count delta from /stats)
    3. Batched result == solo result for the same text (cosine ~= 1.0)
    4. With -MixDocuments, every 4th request is a "document" so the per-type grouping is exercised

  Usage:
    .\scripts\nemotron-embed\test_batch_embed.ps1
    .\scripts\nemotron-embed\test_batch_embed.ps1 -MixDocuments
    .\scripts\nemotron-embed\test_batch_embed.ps1 -TritonUrl http://localhost:8000 -Count 64
#>
param(
    [string]$TritonUrl   = "http://localhost:8000",
    [string]$ModelName   = "nemotron_embed",
    [int]   $Count       = 32,
    [int]   $ExpectedDim = 2048,
    [switch]$MixDocuments
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "PowerShell 7+ required (pwsh)." }

$InferUrl = "$TritonUrl/v2/models/$ModelName/infer"
$StatsUrl = "$TritonUrl/v2/models/$ModelName/stats"

function New-Body([string]$text, [string]$type) {
    @{
        inputs = @(
            @{ name = "TEXT";       shape = @(1, 1); datatype = "BYTES"; data = @($text) },
            @{ name = "INPUT_TYPE"; shape = @(1, 1); datatype = "BYTES"; data = @($type) }
        )
        outputs = @(@{ name = "EMBEDDING" })
    } | ConvertTo-Json -Depth 6 -Compress
}

function Get-Stats {
    $s = (Invoke-RestMethod -Uri $StatsUrl -Method Get).model_stats[0]
    [pscustomobject]@{ Inference = [long]$s.inference_count; Execution = [long]$s.execution_count }
}

function Get-Cosine($a, $b) {
    $dot = 0.0; $na = 0.0; $nb = 0.0
    for ($i = 0; $i -lt $a.Count; $i++) { $dot += $a[$i]*$b[$i]; $na += $a[$i]*$a[$i]; $nb += $b[$i]*$b[$i] }
    $dot / ([math]::Sqrt($na) * [math]::Sqrt($nb))
}

$topics = @(
    "fuel economy of the 2024 Toyota Camry hybrid", "best towing capacity midsize truck",
    "Tesla Model Y range in cold weather", "Honda CR-V maintenance schedule 60k miles",
    "safest SUVs under 40000 dollars", "horsepower over 400 sports sedans",
    "Ford F-150 Lightning charging time", "Subaru Outback ground clearance",
    "cheapest EV to insure", "Mazda CX-5 vs Toyota RAV4 reliability",
    "BMW M3 0-60 time", "Jeep Wrangler resale value"
)

$jobs = for ($i = 0; $i -lt $Count; $i++) {
    $type = if ($MixDocuments -and ($i % 4 -eq 3)) { "document" } else { "query" }
    $text = if ($type -eq "document") {
        "Review: the $($topics[$i % $topics.Count]) is covered in detail here. (doc $i)"
    } else {
        "$($topics[$i % $topics.Count]) (q$i)"
    }
    [pscustomobject]@{ Id = $i; Type = $type; Text = $text; Body = (New-Body $text $type) }
}

# ---- 0. Warm + readiness ----
Write-Host "Checking model readiness..." -ForegroundColor Cyan
Invoke-RestMethod -Uri "$TritonUrl/v2/models/$ModelName/ready" -Method Get | Out-Null
Invoke-RestMethod -Uri $InferUrl -Method Post -ContentType "application/json" -Body (New-Body "warmup" "query") | Out-Null

# ---- 1. Solo baseline for request 0 (for equivalence check) ----
$solo = (Invoke-RestMethod -Uri $InferUrl -Method Post -ContentType "application/json" -Body $jobs[0].Body).outputs[0].data

# ---- 2. Fire $Count requests in parallel ----
$before = Get-Stats
Write-Host "Firing $Count parallel requests (MixDocuments=$MixDocuments)..." -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$results = $jobs | ForEach-Object -ThrottleLimit $Count -Parallel {
    $job = $_
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-RestMethod -Uri $using:InferUrl -Method Post -ContentType "application/json" -Body $job.Body -TimeoutSec 60
        $t.Stop()
        $vec = $r.outputs[0].data
        $norm = [math]::Sqrt(($vec | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
        [pscustomobject]@{ Id = $job.Id; Type = $job.Type; Ok = $true; Dim = $vec.Count; Norm = $norm; Ms = $t.ElapsedMilliseconds; Vec = $vec; Err = $null }
    } catch {
        $t.Stop()
        [pscustomobject]@{ Id = $job.Id; Type = $job.Type; Ok = $false; Dim = 0; Norm = 0; Ms = $t.ElapsedMilliseconds; Vec = $null; Err = $_.Exception.Message }
    }
}

$sw.Stop()
$after = Get-Stats

# ---- 3. Report ----
$ok     = @($results | Where-Object Ok)
$failed = @($results | Where-Object { -not $_.Ok })
$badDim = @($ok | Where-Object { $_.Dim -ne $ExpectedDim })
$badNrm = @($ok | Where-Object { [math]::Abs($_.Norm - 1.0) -gt 1e-3 })

$dInf  = $after.Inference - $before.Inference
$dExec = $after.Execution - $before.Execution
$avgBatch = if ($dExec -gt 0) { [math]::Round($dInf / $dExec, 2) } else { 0 }

$lat = $ok | Measure-Object -Property Ms -Average -Maximum -Minimum
$batched0 = ($results | Where-Object Id -eq 0).Vec
$cos = if ($batched0) { Get-Cosine $solo $batched0 } else { [double]::NaN }

Write-Host ""
Write-Host "==== nemotron_embed batching test ====" -ForegroundColor Yellow
"{0,-28} {1}" -f "Requests ok / total:", "$($ok.Count) / $Count"
"{0,-28} {1}" -f "By type:", (($results | Group-Object Type | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ", ")
"{0,-28} {1} ms" -f "Wall clock:", $sw.ElapsedMilliseconds
"{0,-28} {1:N0} / {2:N0} / {3:N0} ms" -f "Latency min/avg/max:", $lat.Minimum, $lat.Average, $lat.Maximum
"{0,-28} {1}" -f "Throughput:", ("{0:N1} req/s" -f ($Count / ($sw.ElapsedMilliseconds / 1000.0)))
"{0,-28} {1}" -f "Triton inference_count +", $dInf
"{0,-28} {1}" -f "Triton execution_count +", $dExec
"{0,-28} {1}" -f "Avg rows per execute():", $avgBatch
"{0,-28} {1}" -f "Solo vs batched cosine:", ("{0:N6}" -f $cos)

Write-Host ""
$pass = $true
if ($failed.Count) { $pass = $false; Write-Host "FAIL: $($failed.Count) request(s) errored" -ForegroundColor Red; $failed | Select-Object -First 5 Id, Type, Err | Format-Table -AutoSize }
if ($badDim.Count) { $pass = $false; Write-Host "FAIL: $($badDim.Count) wrong dim (expected $ExpectedDim)" -ForegroundColor Red }
if ($badNrm.Count) { $pass = $false; Write-Host "FAIL: $($badNrm.Count) not unit-norm" -ForegroundColor Red }
if ($cos -lt 0.999) { $pass = $false; Write-Host "FAIL: batched result differs from solo (cos=$cos)" -ForegroundColor Red }
if ($avgBatch -le 1.0) { $pass = $false; Write-Host "FAIL: no batching observed (avg rows/execute = $avgBatch)" -ForegroundColor Red }
elseif ($avgBatch -lt 8) { Write-Host "WARN: avg batch $avgBatch < 8 -- client may not be truly concurrent; try -Count 64" -ForegroundColor DarkYellow }

if ($pass) { Write-Host "PASS" -ForegroundColor Green } else { exit 1 }
