<#
  Ingestion baseline for one large vehicle payload.
  Measures end-to-end ingest time, chunks written, and embedding time (from Triton metrics).

  Usage (from C:\dev\ai-rag-assistant, app running on 8080, Triton on 8000/8002):
    powershell -ExecutionPolicy Bypass -File scripts\ingest-baseline.ps1 -Label 5090-embed

  The payload uses vehicleId "baseline-test-vehicle" so real data and the golden set are untouched.
  By default the test vehicle is deleted at the end (chunks + summary). Use -KeepData to keep it.
  Output: baseline\ingest-<label>-<yyyyMMdd-HHmm>\
#>
param(
    [string]$Label = "5090-embed",
    [string]$BaseUrl = "http://localhost:8080",
    [string]$TritonMetricsUrl = "http://localhost:8002/metrics",
    [string]$EmbeddingModel = "text_embedding",
    [string]$Payload = "",
    [int]$Runs = 3,
    [switch]$KeepData
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Payload) { $Payload = Join-Path $root "baseline\payloads\rich-vehicle-large.json" }
$out = Join-Path $root ("baseline\ingest-{0}-{1}" -f $Label, (Get-Date -Format "yyyyMMdd-HHmm"))
New-Item -ItemType Directory -Force -Path $out | Out-Null

$body = Get-Content $Payload -Raw
$vehicleId = ($body | ConvertFrom-Json).vehicleId
if (-not $vehicleId -or -not $vehicleId.StartsWith("baseline-")) {
    throw "Payload vehicleId must start with 'baseline-' (got '$vehicleId') so cleanup can never delete real data."
}

function Psql($sql) {
    $r = docker exec ai-rag-postgres psql -U postgres -d ai_rag_assistant -t -A -c $sql
    return ($r | Out-String).Trim()
}

# Sum a Triton counter across all label sets for the embedding model
function Triton-Metric($text, $name) {
    $sum = 0.0
    foreach ($line in ($text -split "`n")) {
        if ($line -match "^$name\{[^}]*model=`"$EmbeddingModel`"[^}]*\}\s+([0-9.eE+-]+)") {
            $sum += [double]$Matches[1]
        }
    }
    return $sum
}

function Triton-Snapshot {
    try {
        $t = (Invoke-WebRequest -Uri $TritonMetricsUrl -UseBasicParsing -TimeoutSec 10).Content
        return @{
            ok        = $true
            requests  = Triton-Metric $t "nv_inference_request_success"
            items     = Triton-Metric $t "nv_inference_count"
            execs     = Triton-Metric $t "nv_inference_exec_count"
            requestUs = Triton-Metric $t "nv_inference_request_duration_us"
            computeUs = Triton-Metric $t "nv_inference_compute_infer_duration_us"
        }
    } catch {
        return @{ ok = $false }
    }
}

@(
    "label:     $Label"
    "time:      $(Get-Date -Format o)"
    "commit:    $((git -C $root rev-parse --short HEAD) 2>$null)"
    "payload:   $Payload ($((Get-Item $Payload).Length) bytes)"
    "vehicleId: $vehicleId"
    "embedding: Triton model '$EmbeddingModel' via $TritonMetricsUrl"
) | Set-Content (Join-Path $out "meta.txt")

$rows = @()
for ($i = 1; $i -le $Runs; $i++) {
    Write-Host "ingest run $i/$Runs..."
    $before = Triton-Snapshot
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/vehicles/ingest/rich" `
        -ContentType "application/json" -Body $body -TimeoutSec 300
    $sw.Stop()
    $after = Triton-Snapshot

    $resp | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $out "ingest-run$i.json")

    # Count across all partitions and show which doc_type the chunks got.
    # Fleet search only reads doc_type='vehicle' (chunks_vehicle); anything else is invisible to it.
    $chunks = Psql "SELECT count(*) FROM document_chunks WHERE doc_id = '$vehicleId';"
    $chunkTypes = Psql "SELECT string_agg(doc_type || ':' || n, ',') FROM (SELECT doc_type, count(*) n FROM document_chunks WHERE doc_id = '$vehicleId' GROUP BY doc_type) t;"
    $summaryRows = Psql "SELECT count(*) FROM vehicle_summaries WHERE vehicle_id = '$vehicleId';"

    $row = [ordered]@{ run = $i; wallMs = $sw.ElapsedMilliseconds; chunks = $chunks; chunkDocTypes = $chunkTypes; summaryRows = $summaryRows }
    if ($before.ok -and $after.ok) {
        $embedMs = ($after.requestUs - $before.requestUs) / 1000.0
        $row.embedRequests = [int]($after.requests - $before.requests)
        $row.embedItems    = [int]($after.items - $before.items)
        $row.embedExecs    = [int]($after.execs - $before.execs)
        $row.embedMs       = [math]::Round($embedMs, 1)
        $row.embedComputeMs = [math]::Round(($after.computeUs - $before.computeUs) / 1000.0, 1)
        $row.embedSharePct = if ($sw.ElapsedMilliseconds -gt 0) { [math]::Round(100 * $embedMs / $sw.ElapsedMilliseconds, 1) } else { 0 }
    } else {
        $row.embedRequests = "n/a"; $row.embedItems = "n/a"; $row.embedExecs = "n/a"
        $row.embedMs = "n/a"; $row.embedComputeMs = "n/a"; $row.embedSharePct = "n/a"
    }
    $rows += [pscustomobject]$row
}
$rows | Export-Csv (Join-Path $out "ingest-summary.csv") -NoTypeInformation

$wall = $rows.wallMs
$summary = @(
    "=== Ingest: $vehicleId x $Runs runs ==="
    ("wallMs  min={0}  avg={1:N0}  max={2}" -f ($wall | Measure-Object -Minimum).Minimum,
        ($wall | Measure-Object -Average).Average, ($wall | Measure-Object -Maximum).Maximum)
    ("chunks after each run: {0}   (same number every run = upsert, no duplicates)" -f (($rows.chunks) -join ","))
    ("chunk doc_type: {0}   (must be vehicle:N, otherwise fleet search cannot see them)" -f (($rows.chunkDocTypes | Select-Object -Last 1)))
    ("summary rows: {0}" -f (($rows.summaryRows) -join ","))
    ("embedding: requests/run={0}  items/run={1}  ms/run={2}  share of ingest={3}%" -f `
        (($rows.embedRequests) -join ","), (($rows.embedItems) -join ","), (($rows.embedMs) -join ","), (($rows.embedSharePct) -join ","))
    "Note: run 1 can include warm-up; compare runs 2-3."
)
$summary | Set-Content (Join-Path $out "summary.txt")
$summary | ForEach-Object { Write-Host $_ }
$rows | Format-Table -AutoSize

if (-not $KeepData) {
    Write-Host "Cleaning up test vehicle '$vehicleId' (chunks + summary)..."
    Psql "DELETE FROM document_chunks WHERE doc_id = '$vehicleId';" | Out-Null
    Psql "DELETE FROM vehicle_summaries WHERE vehicle_id = '$vehicleId';" | Out-Null
    $left = Psql "SELECT (SELECT count(*) FROM document_chunks WHERE doc_id = '$vehicleId') + (SELECT count(*) FROM vehicle_summaries WHERE vehicle_id = '$vehicleId');"
    Write-Host "Rows left for test vehicle: $left"
    "cleanup: deleted test vehicle rows (left=$left)" | Add-Content (Join-Path $out "meta.txt")
}
Write-Host "Saved to $out"
