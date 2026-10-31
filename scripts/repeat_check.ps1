<#
  Repeat-answer check for #15: runs the M3 horsepower /ask N times and counts
  multi-sentence answers and judge scores below 1.0.
  Usage: powershell -ExecutionPolicy Bypass -File scripts\repeat_check.ps1 -Label A -Runs 20
#>
param([string]$Label = "run", [int]$Runs = 20,
      [string]$Question = "How much horsepower does the BMW M3 Competition have?")
$body = @{ docType="vehicle"; docId="vehicles"; topK=5; question=$Question } | ConvertTo-Json
$multi = 0; $lowJudge = 0
foreach ($i in 1..$Runs) {
  $r = Invoke-RestMethod -Method Post -Uri http://localhost:8080/ask -ContentType 'application/json' -Body $body -TimeoutSec 120
  $t = ($r.answer -split "Sources:")[0].Trim()
  $n = ($t -split "(?<=[.!?])\s+" | Where-Object { $_.Trim() }).Count
  if ($n -gt 1) { $multi++ }
  if ($r.judge.score -lt 1.0) { $lowJudge++ }
  $flat = $t -replace "\s+", " "
  "{0,2} s={1} judge={2}  {3}" -f $i, $n, $r.judge.score, $flat.Substring(0, [Math]::Min(95, $flat.Length))
}
"==> [$Label] multi-sentence: $multi/$Runs   judge<1.0: $lowJudge/$Runs"
