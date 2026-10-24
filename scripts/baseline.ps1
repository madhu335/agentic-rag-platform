<#
  Inference baseline: judge latency/verdicts, M3 /ask, recall eval.
  Run the same script on each judge machine and compare the folders.

  Usage (from C:\dev\ai-rag-assistant, with the app running on 8080):
    powershell -ExecutionPolicy Bypass -File scripts\baseline.ps1 -Label halo
    powershell -ExecutionPolicy Bypass -File scripts\baseline.ps1 -Label gn100

  Output: baseline\<label>-<yyyyMMdd-HHmm>\  (JSON per call + judge-summary.csv + summary.txt)
  Note: completionTokens is only in the app log ("Halo judge: ..." lines), not in the API response.
#>
param(
    [string]$Label = "halo",
    [string]$BaseUrl = "http://localhost:8080",
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root ("baseline\{0}-{1}" -f $Label, (Get-Date -Format "yyyyMMdd-HHmm"))
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Post-Json($url, $body) {
    $json = $body | ConvertTo-Json -Depth 10
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -Body $json -TimeoutSec 300
    $sw.Stop()
    return @{ resp = $resp; wallMs = $sw.ElapsedMilliseconds }
}

# ── Metadata ─────────────────────────────────────────────────────────────
$commit = (git -C $root rev-parse --short HEAD) 2>$null
$dirty = (git -C $root status --porcelain) 2>$null
@(
    "label:   $Label"
    "time:    $(Get-Date -Format o)"
    "commit:  $commit"
    "dirty:   $(if ($dirty) { 'YES - uncommitted changes' } else { 'no' })"
    "baseUrl: $BaseUrl"
) | Set-Content (Join-Path $out "meta.txt")

# ── Judge tests (same cases as halo-judge.http requests 2-4) ─────────────
$m3Chunk = "[bmw-m3-2025-competition:1] The 2025 BMW M3 Competition uses a twin-turbo 3.0L inline-six producing 503 hp and 479 lb-ft of torque."
$cases = @(
    @{ name = "2-correct"; expect = "grounded=true, score>=0.8"; body = @{
        question = "What is the horsepower of the 2025 BMW M3 Competition?"
        answer   = "The 2025 BMW M3 Competition produces 503 horsepower."
        context  = @($m3Chunk) } },
    @{ name = "3-hallucinated"; expect = "grounded=false, low score"; body = @{
        question = "What is the horsepower of the 2025 BMW M3 Competition?"
        answer   = "The 2025 BMW M3 Competition produces 620 horsepower and has a top speed of 210 mph."
        context  = @($m3Chunk) } },
    @{ name = "4-partial"; expect = "grounded=false, reason names Honda CR-V"; body = @{
        question = "Which vehicles have all-wheel drive?"
        answer   = "The Tesla Model 3 Long Range, Rivian R1T and Honda CR-V Sport Hybrid all offer AWD."
        context  = @(
            "[tesla-model3-2025-long-range:1] Dual-motor all-wheel drive, 0-60 mph in 4.2 seconds.",
            "[rivian-r1t-2025-adventure:1] Quad-motor AWD with adjustable air suspension.") } }
)

$rows = @()
foreach ($c in $cases) {
    for ($i = 1; $i -le $Runs; $i++) {
        Write-Host ("judge {0} run {1}/{2}..." -f $c.name, $i, $Runs)
        $r = Post-Json "$BaseUrl/api/judge/test" $c.body
        $r.resp | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $out ("judge-{0}-run{1}.json" -f $c.name, $i))
        $rows += [pscustomobject]@{
            case      = $c.name
            run       = $i
            latencyMs = $r.resp.latencyMs
            wallMs    = $r.wallMs
            grounded  = $r.resp.result.grounded
            correct   = $r.resp.result.correct
            complete  = $r.resp.result.complete
            score     = $r.resp.result.score
            reason    = $r.resp.result.reason
            expect    = $c.expect
        }
    }
}
$rows | Export-Csv (Join-Path $out "judge-summary.csv") -NoTypeInformation

# ── M3 /ask (full pipeline: retrieval + Qwen + citation check + judge) ───
Write-Host "M3 /ask..."
$ask = Post-Json "$BaseUrl/ask" @{
    docType = "vehicle"; docId = "vehicles"; topK = 5
    question = "How much horsepower does the BMW M3 Competition have?" }
$ask.resp | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $out "ask-m3.json")

# ── Recall eval ──────────────────────────────────────────────────────────
Write-Host "recall eval..."
$sw = [Diagnostics.Stopwatch]::StartNew()
$eval = Invoke-RestMethod -Uri "$BaseUrl/api/eval/vehicles/recall/report" -TimeoutSec 600
$sw.Stop()
$eval | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $out "eval.json")

# ── Summary ──────────────────────────────────────────────────────────────
$summary = @()
$summary += "=== Judge (latencyMs per run) ==="
foreach ($g in ($rows | Group-Object case)) {
    $lat = $g.Group.latencyMs
    $summary += ("{0,-15} runs={1}  min={2}  avg={3:N0}  max={4}  scores={5}" -f `
        $g.Name, $g.Count, ($lat | Measure-Object -Minimum).Minimum, ($lat | Measure-Object -Average).Average,
        ($lat | Measure-Object -Maximum).Maximum, (($g.Group.score) -join ","))
}
$summary += ""
$summary += "=== M3 /ask ==="
$summary += ("wallMs={0}  judgeScore={1}  cited={2}" -f $ask.wallMs, $ask.resp.judge.score, (($ask.resp.citedChunkIds) -join ","))
$summary += ("answer: {0}" -f $ask.resp.answer)
$summary += ""
$summary += "=== Recall eval ==="
$summary += ("recall={0:N3}  precision={1:N3}  passed={2}/{3}  wallMs={4}" -f `
    $eval.overallRecall, $eval.overallPrecision, $eval.totalPassed, $eval.totalEntries, $sw.ElapsedMilliseconds)
$summary += ("failed: {0}" -f ((@($eval.failedEntries) | ForEach-Object { $_.entryId }) -join ", "))

$summary | Set-Content (Join-Path $out "summary.txt")
Write-Host ""
$summary | ForEach-Object { Write-Host $_ }
Write-Host ""
$rows | Format-Table case, run, latencyMs, grounded, correct, complete, score, reason -AutoSize
Write-Host "Saved to $out"
