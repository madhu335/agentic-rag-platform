Set-Location C:\dev\ai-rag-assistant

# Single Triton serves embedding, reranker and the Nemotron generator.
docker compose up -d postgres triton

Write-Host "Waiting for Triton (all models, ~3 min)..."
do {
    Start-Sleep -Seconds 10
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/v2/health/ready" -Method Get -UseBasicParsing
        $ready = ($resp.StatusCode -eq 200)
    } catch {
        $ready = $false
    }
} until ($ready)

Write-Host "Triton is ready (text_embedding, cross_reranker, nemotron)."
Write-Host "Done."
