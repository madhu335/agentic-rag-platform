Set-Location C:\dev\ai-rag-assistant

docker compose up -d postgres triton

Write-Host "Waiting for Triton..."
do {
    Start-Sleep -Seconds 5
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/v2/health/ready" -Method Get -UseBasicParsing
        $ready = ($resp.StatusCode -eq 200)
    } catch {
        $ready = $false
    }
} until ($ready)

Write-Host "Triton is ready. Starting vLLM..."
docker compose up -d vllm

Write-Host "Done."