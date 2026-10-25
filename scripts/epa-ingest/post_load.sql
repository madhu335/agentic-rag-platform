-- post_load.sql — run AFTER a large vehicle ingestion.
--
-- Usage (PowerShell, from C:\dev\ai-rag-assistant):
--   Get-Content scripts\epa-ingest\post_load.sql | docker exec -i ai-rag-postgres psql -U postgres -d ai_rag_assistant
--
-- Why this matters:
--   The IVFFlat index on chunks_vehicle computes its cluster centroids ONCE, at
--   CREATE INDEX time, from whatever rows existed then (about 10 vehicles).
--   New rows are assigned to those stale centroids, so after loading thousands
--   of vehicles most clusters are badly unbalanced and recall drops. It's the
--   same failure mode as keeping a 3-node partition map after scaling to 300
--   nodes. Rebuilding recomputes centroids from the real data distribution.

\timing on

-- ─── 1. Sanity checks ────────────────────────────────────────────────────

\echo '== Rows per partition =='
SELECT tableoid::regclass AS partition, COUNT(*) AS chunks, COUNT(DISTINCT doc_id) AS docs
FROM document_chunks
GROUP BY 1 ORDER BY 1;

\echo '== EPA vehicles that leaked into a non-vehicle partition (must be 0) =='
SELECT doc_type, COUNT(DISTINCT doc_id) AS leaked_docs
FROM document_chunks
WHERE doc_id ~ '-epa[0-9a-z]+$' AND doc_type <> 'vehicle'
GROUP BY doc_type;

\echo '== Vehicle chunks missing halfvec or tsvector (must be 0) =='
SELECT COUNT(*) FILTER (WHERE embedding_half IS NULL) AS missing_halfvec,
       COUNT(*) FILTER (WHERE content_tsv IS NULL)    AS missing_tsv
FROM chunks_vehicle;

\echo '== Vehicles with chunks but no Tier-1 summary (must be 0) =='
SELECT COUNT(*) AS vehicles_without_summary
FROM (SELECT DISTINCT doc_id FROM chunks_vehicle) c
LEFT JOIN vehicle_summaries s ON s.vehicle_id = c.doc_id
WHERE s.vehicle_id IS NULL;

\echo '== Chunk type mix (chunk_index buckets) =='
SELECT CASE
         WHEN chunk_index = 1 THEN '1 identity'
         WHEN chunk_index = 2 THEN '2 performance'
         WHEN chunk_index = 3 THEN '3 ownership'
         WHEN chunk_index = 4 THEN '4 rankings'
         WHEN chunk_index = 5 THEN '5 safety'
         WHEN chunk_index = 6 THEN '6 features'
         WHEN chunk_index = 7 THEN '7 reviews'
         WHEN chunk_index BETWEEN 10 AND 19 THEN '10+ maintenance'
         ELSE '20+ recalls'
       END AS chunk_type,
       COUNT(*)
FROM chunks_vehicle GROUP BY 1 ORDER BY 1;

-- ─── 2. Rebuild IVFFlat with lists sized to the data ─────────────────────
-- pgvector guidance: lists = rows / 1000 up to 1M rows, sqrt(rows) beyond.
-- Floor of 10 so tiny tables still get a usable index.

SET maintenance_work_mem = '2GB';

DO $$
DECLARE
    n      BIGINT;
    lists  INT;
BEGIN
    SELECT COUNT(*) INTO n FROM chunks_vehicle WHERE embedding_half IS NOT NULL;
    IF n <= 1000000 THEN
        lists := GREATEST(10, (n / 1000)::INT);
    ELSE
        lists := CEIL(SQRT(n))::INT;
    END IF;
    RAISE NOTICE 'chunks_vehicle rows with embeddings: %, rebuilding IVFFlat with lists = %', n, lists;

    EXECUTE 'DROP INDEX IF EXISTS idx_vehicle_ivfflat_half';
    EXECUTE format(
        'CREATE INDEX idx_vehicle_ivfflat_half ON chunks_vehicle '
        'USING ivfflat (embedding_half halfvec_cosine_ops) WITH (lists = %s)', lists);
END $$;

-- ─── 3. Refresh planner statistics ──────────────────────────────────────

ANALYZE chunks_vehicle;
ANALYZE vehicle_summaries;

-- ─── 4. Report ──────────────────────────────────────────────────────────

\echo '== Index sizes =='
SELECT indexrelname AS index, pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname IN ('chunks_vehicle', 'vehicle_summaries')
ORDER BY pg_relation_size(indexrelid) DESC;

\echo '== Suggested ivfflat.probes (about sqrt(lists)); current app setting is 10 in application.yml =='
SELECT (regexp_match(indexdef, 'lists=''?(\d+)'))[1]::INT AS lists,
       CEIL(SQRT((regexp_match(indexdef, 'lists=''?(\d+)'))[1]::INT))::INT AS suggested_probes
FROM pg_indexes WHERE indexname = 'idx_vehicle_ivfflat_half';
