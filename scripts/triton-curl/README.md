# Triton + vehicle API curl commands

Run from this folder: `cd C:\dev\ai-rag-assistant\scripts\triton-curl`

Request bodies are in the `.json` files here and are passed with `-d "@file.json"`. Windows PowerShell
breaks inline JSON that contains double quotes, so keep bodies in files.

- **Port 8000** = Triton directly (one model per call). The Spring app does not need to be running.
- **Port 8080** = the Spring app (full pipeline). Start the app first.

Readable output: append `| ConvertFrom-Json | ConvertTo-Json -Depth 6` to any command.

---

## 1. Triton direct (port 8000)

### List / check models
```powershell
curl.exe -s -X POST http://localhost:8000/v2/repository/index
curl.exe -s http://localhost:8000/v2/models/nemotron
curl.exe -s http://localhost:8000/v2/models/nemotron_embed/stats
```

### Ingest: embed passages (`nemotron_embed`, INPUT_TYPE=passage)
```powershell
curl.exe -s -X POST http://localhost:8000/v2/models/nemotron_embed/infer -H "Content-Type: application/json" -d "@ingest_embed_passages.json" -o passages_out.json
(Get-Content passages_out.json | ConvertFrom-Json).outputs[0].shape    # expect 3, 2048
```

### Retrieve step 1: embed the query (`nemotron_embed`, INPUT_TYPE=query)
```powershell
curl.exe -s -X POST http://localhost:8000/v2/models/nemotron_embed/infer -H "Content-Type: application/json" -d "@retrieve_embed_query.json" -o query_out.json
(Get-Content query_out.json | ConvertFrom-Json).outputs[0].shape       # expect 1, 2048
```

### Retrieve step 2: rerank candidates (`cross_reranker`)
```powershell
curl.exe -s -X POST http://localhost:8000/v2/models/cross_reranker/infer -H "Content-Type: application/json" -d "@retrieve_rerank.json"
```
One SCORE per (query, document) pair; the BMW chunk should score highest. App gate: `min-rerank-score: 0.01`.

### Retrieve step 3: generate the answer (`nemotron`)
```powershell
curl.exe -s -X POST http://localhost:8000/v2/models/nemotron/generate -H "Content-Type: application/json" -d "@retrieve_generate.json"
```

**Editing request files:** change the `data` lists. If you change how many texts there are, set `shape` to
match, e.g. `[5, 1]` for 5 texts (QUERY and DOCUMENT must have the same count in the rerank file).

---

## 2. Spring app (port 8080)

### Ingest
```powershell
# One rich vehicle -> chunks -> nemotron_embed (passage) -> pgvector
curl.exe -s -X POST http://localhost:8080/vehicles/ingest/rich -H "Content-Type: application/json" -d "@app_ingest_rich.json"

# Bulk list of rich vehicles (pages of 50 -> batch embed -> upsert -> summaries)
curl.exe -s -X POST http://localhost:8080/vehicles/ingest/bulk -H "Content-Type: application/json" -d "@app_ingest_bulk.json"

# Check Tier-1 summaries / rebuild structured attributes
curl.exe -s http://localhost:8080/vehicles/admin/summaries
curl.exe -s -X POST http://localhost:8080/vehicles/admin/attributes/backfill
```

### Retrieve
```powershell
# Single vehicle: retrieval + Nemotron answer (mode: VECTOR | BM25 | HYBRID | HYBRID_RERANK)
curl.exe -s -X POST http://localhost:8080/vehicles/jaguar-f-type-2024-r-awd-coupe-5-0l-8-cylinder-supercharged-epa46444/ask -H "Content-Type: application/json" -d "@app_ask_vehicle.json"

# Fleet search across all vehicles (vector + BM25, RRF, relevance gate) -> ranked chunks, no LLM answer
curl.exe -s -X POST http://localhost:8080/vehicles/ask -H "Content-Type: application/json" -d "@app_ask_fleet.json"

# Two-tier: summaries first, then detail chunks (returns tier1Ms / totalMs)
curl.exe -s -X POST http://localhost:8080/vehicles/ask/fleet -H "Content-Type: application/json" -d "@app_ask_fleet.json"

# ParadeDB hybrid in one SQL query (vectorWeight 0 = BM25 only, 1 = vector only)
curl.exe -s -X POST http://localhost:8080/vehicles/ask/hybrid -H "Content-Type: application/json" -d "@app_ask_hybrid.json"
```

### Eval
```powershell
curl.exe -s http://localhost:8080/api/eval/vehicles/recall/report
```

---

## 3. GN100 judge — gpt-oss-120b (llama.cpp llama-server, OpenAI-compatible)

Base URL from `judge.halo.base-url`: `http://192.168.50.20:18002/v1` (10G direct link).
If the server was started with `--api-key`, add `-H "Authorization: Bearer <key>"` to each call.

### Health / model list
```powershell
curl.exe -s http://192.168.50.20:18002/health
curl.exe -s http://192.168.50.20:18002/v1/models
curl.exe -s http://192.168.50.20:18002/props          # llama-server settings: ctx size, slots, chat template
```

### Plain chat (`Reasoning: low` system line, like the app)
```powershell
curl.exe -s -X POST http://192.168.50.20:18002/v1/chat/completions -H "Content-Type: application/json" -d "@gn100_chat.json"
```
Answer: `choices[0].message.content`. Reasoning (if the server splits it out): `choices[0].message.reasoning_content`.
Token counts and speed: `usage`, and llama-server's `timings` (prompt_per_second / predicted_per_second).

### Streaming chat
```powershell
curl.exe -s -N -X POST http://192.168.50.20:18002/v1/chat/completions -H "Content-Type: application/json" -d "@gn100_chat_stream.json"
```
`-N` turns off buffering so `data: {...}` chunks print as they arrive; ends with `data: [DONE]`.

### Judge call, exactly as HaloJudgeClient + JudgeService send it
```powershell
curl.exe -s -X POST http://192.168.50.20:18002/v1/chat/completions -H "Content-Type: application/json" -d "@gn100_judge.json"
```
The sample answer adds an unsupported claim ("0-60 in 3.0 seconds"), so expect `"grounded": false` or a reduced
score. If `finish_reason` is `length` with empty content, reasoning used up `max_tokens` (app uses 1024).

### Judge through the app (port 8080, `judge.debug.enabled=true`)
```powershell
curl.exe -s -X POST http://localhost:8080/api/judge/test -H "Content-Type: application/json" -d "@app_judge_test.json"
```
Same judge, plus the app's JSON extraction, score normalization and retry. Returns provider, latencyMs, result.

### GN100 logs (from the workstation, over SSH)
The judge is **not** a container: llama-server runs on the GN100 as systemd service `llama-judge.service`,
so it does not appear in the workstation's Docker Desktop. Read its logs from the GN100 journal.

```powershell
# Follow live (leave running; fire a judge curl in another window). -t lets Ctrl+C stop cleanly.
ssh -t madhu@192.168.50.20 "journalctl -u llama-judge -f -n 50"

# Last 200 lines, no follow
ssh madhu@192.168.50.20 "journalctl -u llama-judge -n 200 --no-pager"

# Errors and warnings only, last 24 hours
ssh madhu@192.168.50.20 "journalctl -u llama-judge -p warning --since '24 hours ago' --no-pager"

# Status, uptime, memory and the exact llama-server command line
ssh madhu@192.168.50.20 "systemctl status llama-judge --no-pager -l"

# Find the service again if the name changes
ssh madhu@192.168.50.20 "systemctl list-units --type=service --all | grep -i -E 'llama|gpt|judge'"
```
Per request, llama-server logs prompt-processing and generation timings (tokens/sec).
If journalctl shows nothing or "No journal files were found", prefix it with `sudo` (keep `ssh -t` so it can prompt).

**Skip the password prompt (one-time SSH key setup):**
```powershell
ssh-keygen          # accept defaults
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh madhu@192.168.50.20 "cat >> ~/.ssh/authorized_keys"
```

---

## Files

| File | Endpoint | Body |
|---|---|---|
| `ingest_embed_passages.json` | Triton `nemotron_embed/infer` | 3 vehicle texts, INPUT_TYPE=passage |
| `retrieve_embed_query.json` | Triton `nemotron_embed/infer` | 1 question, INPUT_TYPE=query |
| `retrieve_rerank.json` | Triton `cross_reranker/infer` | 3 (query, document) pairs |
| `retrieve_generate.json` | Triton `nemotron/generate` | ChatML prompt with 1 context chunk, thinking off |
| `app_ingest_rich.json` | app `/vehicles/ingest/rich` | 1 RichVehicleRecord (from data/epa/vehicles_enriched.jsonl) |
| `app_ingest_bulk.json` | app `/vehicles/ingest/bulk` | list of 2 RichVehicleRecords |
| `app_ask_vehicle.json` | app `/vehicles/{id}/ask` | question, topK, mode |
| `app_ask_fleet.json` | app `/vehicles/ask`, `/vehicles/ask/fleet` | question, topK |
| `app_ask_hybrid.json` | app `/vehicles/ask/hybrid` | question, topK, vectorWeight |
| `compare_query_vs_passage.json` | Triton `nemotron_embed/infer` | same sentence as query and passage |
| `gn100_chat.json` | GN100 `/v1/chat/completions` | plain chat, Reasoning: low |
| `gn100_chat_stream.json` | GN100 `/v1/chat/completions` | same, stream=true |
| `gn100_judge.json` | GN100 `/v1/chat/completions` | exact judge system + user prompt |
| `app_judge_test.json` | app `/api/judge/test` | question, answer, context |

Output files (`passages_out.json`, `query_out.json`) are created when you run the embed commands.
