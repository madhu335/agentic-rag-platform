-- ═══════════════════════════════════════════════════════════════════════
-- Embedding migration: bge-base-en-v1.5 (768) -> Nemotron 3.5 Embed 1B (1024)
-- ═══════════════════════════════════════════════════════════════════════
-- Vectors from different models are not comparable, so all embeddings are
-- cleared (set to NULL) and re-computed by reembed.py from the stored text.
-- Run post_reembed_indexes.sql afterwards (IVFFlat needs data to build lists).
--
-- Legacy tables chunks, document_chunks_old and data_llamaindex_chunks are untouched.
-- One transaction: any error rolls everything back.

BEGIN;

-- 1. Vector indexes (rebuilt after re-embedding)
DROP INDEX IF EXISTS idx_vehicle_ivfflat_half;
DROP INDEX IF EXISTS idx_article_hnsw_half;
DROP INDEX IF EXISTS idx_pdf_hnsw_half;
DROP INDEX IF EXISTS idx_summary_hnsw_half;

-- 2a. Triggers that reference "embedding" block ALTER TYPE; drop them (recreated in step 3).
--     Dropping the parent trigger also drops its clones on the partitions.
DROP TRIGGER IF EXISTS document_chunks_halfvec_trigger ON document_chunks;
DROP TRIGGER IF EXISTS vehicle_summaries_halfvec_trigger ON vehicle_summaries;

-- 2b. Column sizes 768 -> 1024 (partitioned parent propagates to chunks_vehicle/article/pdf)
ALTER TABLE document_chunks
    ALTER COLUMN embedding      TYPE vector(1024)  USING NULL,
    ALTER COLUMN embedding_half TYPE halfvec(1024) USING NULL;

ALTER TABLE vehicle_summaries
    ALTER COLUMN embedding      TYPE vector(1024)  USING NULL,
    ALTER COLUMN embedding_half TYPE halfvec(1024) USING NULL;

-- 3. halfvec trigger functions cast to the new size; recreate the triggers
CREATE OR REPLACE FUNCTION document_chunks_halfvec_fn()
RETURNS trigger AS $$
BEGIN
    IF NEW.embedding IS NOT NULL THEN
        NEW.embedding_half := NEW.embedding::halfvec(1024);
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION vehicle_summaries_halfvec_fn()
RETURNS trigger AS $$
BEGIN
    IF NEW.embedding IS NOT NULL THEN
        NEW.embedding_half := NEW.embedding::halfvec(1024);
    END IF;
    NEW.updated_at := NOW();
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER document_chunks_halfvec_trigger
    BEFORE INSERT OR UPDATE OF embedding
    ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION document_chunks_halfvec_fn();

CREATE TRIGGER vehicle_summaries_halfvec_trigger
    BEFORE INSERT OR UPDATE OF embedding
    ON vehicle_summaries
    FOR EACH ROW
    EXECUTE FUNCTION vehicle_summaries_halfvec_fn();

COMMIT;

-- Check triggers (expect the halfvec trigger on document_chunks, the 3 partitions and vehicle_summaries)
SELECT t.tgname, t.tgrelid::regclass AS on_table
FROM pg_trigger t WHERE NOT t.tgisinternal AND t.tgname LIKE '%halfvec%' ORDER BY 2;

-- Check
SELECT c.relname AS table, a.attname AS column, format_type(a.atttypid, a.atttypmod) AS type
FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
WHERE c.relname IN ('document_chunks', 'chunks_vehicle', 'chunks_article', 'chunks_pdf', 'vehicle_summaries')
  AND a.attname IN ('embedding', 'embedding_half')
ORDER BY 1, 2;
