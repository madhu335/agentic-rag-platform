-- Rebuild vector indexes on the 1024-dim halfvec columns (run after reembed.py).
-- Same index types and settings as before. IVFFlat lists ~ rows/1000 (13,670 vehicle chunks -> 14).

SELECT 'NULL embeddings left' AS check, count(*) FROM document_chunks WHERE embedding IS NULL
UNION ALL
SELECT 'NULL summary embeddings left', count(*) FROM vehicle_summaries WHERE embedding IS NULL;

CREATE INDEX IF NOT EXISTS idx_vehicle_ivfflat_half
    ON chunks_vehicle USING ivfflat (embedding_half halfvec_cosine_ops) WITH (lists = 14);

CREATE INDEX IF NOT EXISTS idx_article_hnsw_half
    ON chunks_article USING hnsw (embedding_half halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_pdf_hnsw_half
    ON chunks_pdf USING hnsw (embedding_half halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_summary_hnsw_half
    ON vehicle_summaries USING hnsw (embedding_half halfvec_cosine_ops) WITH (m = 24, ef_construction = 128);

ANALYZE document_chunks;
ANALYZE vehicle_summaries;

SELECT tablename, indexname FROM pg_indexes WHERE indexdef ~* '(ivfflat|hnsw)' ORDER BY 1;
