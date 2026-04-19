-- ═══════════════════════════════════════════════════════════════════════
-- Migration: Partition + Two-Tier Retrieval + ParadeDB + halfvec
-- ═══════════════════════════════════════════════════════════════════════
--
-- Run this AFTER backing up your data. This migration:
--   1. Creates partitioned table structure
--   2. Migrates existing data from document_chunks → partitions
--   3. Creates vehicle_summaries table for two-tier retrieval
--   4. Sets up ParadeDB pg_search for hybrid search
--   5. Adds IVFFlat indexes on vehicle partition
--
-- Prerequisites:
--   - pgvector 0.7+ (for halfvec support)
--   - ParadeDB pg_search extension installed in Docker image
--
-- IMPORTANT: Run each section in order. If a section fails, fix
-- the issue before proceeding — later sections depend on earlier ones.
-- ═══════════════════════════════════════════════════════════════════════


-- ─── Section 1: Enable extensions ─────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;   -- ParadeDB


-- ─── Section 2: Create partitioned table ──────────────────────────────
--
-- We create a new partitioned table, migrate data, then swap names.
-- This avoids downtime — the old table stays readable until the swap.

-- 2a. Create the new partitioned table
CREATE TABLE IF NOT EXISTS document_chunks_partitioned (
    id              BIGSERIAL,
    doc_id          TEXT        NOT NULL,
    chunk_index     INT         NOT NULL,
    content         TEXT,
    embedding       vector(768),
    embedding_half  halfvec(768),       -- float16 copy for memory-efficient indexes
    content_tsv     tsvector,
    doc_type        VARCHAR(50) NOT NULL DEFAULT 'pdf',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_part_doc_chunk UNIQUE (doc_id, chunk_index, doc_type),
    PRIMARY KEY (id, doc_type)
) PARTITION BY LIST (doc_type);

-- 2b. Create partitions
CREATE TABLE IF NOT EXISTS chunks_vehicle
    PARTITION OF document_chunks_partitioned FOR VALUES IN ('vehicle');

CREATE TABLE IF NOT EXISTS chunks_article
    PARTITION OF document_chunks_partitioned FOR VALUES IN ('article');

CREATE TABLE IF NOT EXISTS chunks_pdf
    PARTITION OF document_chunks_partitioned FOR VALUES IN ('pdf');

-- 2c. Migrate existing data (skip if fresh install)
INSERT INTO document_chunks_partitioned
    (doc_id, chunk_index, content, embedding, content_tsv, doc_type)
SELECT
    doc_id,
    chunk_index,
    content,
    embedding,
    content_tsv,
    COALESCE(doc_type, 'pdf')
FROM document_chunks
ON CONFLICT (doc_id, chunk_index, doc_type) DO NOTHING;

-- 2d. Populate halfvec column from full-precision embeddings
UPDATE document_chunks_partitioned
SET embedding_half = embedding::halfvec(768)
WHERE embedding IS NOT NULL AND embedding_half IS NULL;

-- 2e. Rename tables (atomic swap)
ALTER TABLE document_chunks RENAME TO document_chunks_old;
ALTER TABLE document_chunks_partitioned RENAME TO document_chunks;

-- 2f. tsvector trigger on new table
CREATE OR REPLACE FUNCTION document_chunks_tsv_trigger_fn()
RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', coalesce(NEW.content, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS document_chunks_tsv_trigger ON document_chunks;

CREATE TRIGGER document_chunks_tsv_trigger
    BEFORE INSERT OR UPDATE OF content
    ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION document_chunks_tsv_trigger_fn();

-- 2g. Auto-populate halfvec on insert/update
CREATE OR REPLACE FUNCTION document_chunks_halfvec_fn()
RETURNS trigger AS $$
BEGIN
    IF NEW.embedding IS NOT NULL THEN
        NEW.embedding_half := NEW.embedding::halfvec(768);
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER document_chunks_halfvec_trigger
    BEFORE INSERT OR UPDATE OF embedding
    ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION document_chunks_halfvec_fn();


-- ─── Section 3: Indexes on partitions ─────────────────────────────────
--
-- Each partition gets its own indexes — Postgres uses partition pruning
-- to skip irrelevant partitions during queries.

-- Vehicle partition: IVFFlat on halfvec (memory-efficient, fast filtered search)
-- lists = sqrt(N) is a good starting point. For 9M rows: ~3000.
-- Start with 100 for current scale, increase when you hit 90K vehicles.
CREATE INDEX IF NOT EXISTS idx_vehicle_ivfflat_half
    ON chunks_vehicle
    USING ivfflat (embedding_half halfvec_cosine_ops)
    WITH (lists = 100);

-- Vehicle: GIN index for keyword search
CREATE INDEX IF NOT EXISTS idx_vehicle_tsv
    ON chunks_vehicle USING gin (content_tsv);

-- Vehicle: btree on doc_id for exact-match lookups
CREATE INDEX IF NOT EXISTS idx_vehicle_doc_id
    ON chunks_vehicle (doc_id);

-- Article partition: HNSW on halfvec (smaller corpus, recall matters more)
CREATE INDEX IF NOT EXISTS idx_article_hnsw_half
    ON chunks_article
    USING hnsw (embedding_half halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Article: GIN for keyword search
CREATE INDEX IF NOT EXISTS idx_article_tsv
    ON chunks_article USING gin (content_tsv);

-- Article: btree on doc_id
CREATE INDEX IF NOT EXISTS idx_article_doc_id
    ON chunks_article (doc_id);

-- PDF partition: HNSW (smallest corpus)
CREATE INDEX IF NOT EXISTS idx_pdf_hnsw_half
    ON chunks_pdf
    USING hnsw (embedding_half halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- PDF: GIN for keyword search
CREATE INDEX IF NOT EXISTS idx_pdf_tsv
    ON chunks_pdf USING gin (content_tsv);


-- ─── Section 4: Vehicle summaries (Tier 1 of two-tier retrieval) ──────
--
-- One row per vehicle. The embedding represents the vehicle's identity +
-- key specs. Fleet search queries this table first (90K rows) to find
-- candidate vehicles, then does detail retrieval in chunks_vehicle.

CREATE TABLE IF NOT EXISTS vehicle_summaries (
    vehicle_id      TEXT        PRIMARY KEY,
    make            VARCHAR(100),
    model           VARCHAR(100),
    year            INT,
    trim            VARCHAR(100),
    summary_text    TEXT        NOT NULL,
    embedding       vector(768),
    embedding_half  halfvec(768),
    chunk_count     INT         NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW on summaries — only 90K rows, recall matters most here
CREATE INDEX IF NOT EXISTS idx_summary_hnsw_half
    ON vehicle_summaries
    USING hnsw (embedding_half halfvec_cosine_ops)
    WITH (m = 24, ef_construction = 128);

-- Auto-populate halfvec on summaries
CREATE OR REPLACE FUNCTION vehicle_summaries_halfvec_fn()
RETURNS trigger AS $$
BEGIN
    IF NEW.embedding IS NOT NULL THEN
        NEW.embedding_half := NEW.embedding::halfvec(768);
    END IF;
    NEW.updated_at := NOW();
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER vehicle_summaries_halfvec_trigger
    BEFORE INSERT OR UPDATE OF embedding
    ON vehicle_summaries
    FOR EACH ROW
    EXECUTE FUNCTION vehicle_summaries_halfvec_fn();


-- ─── Section 5: ParadeDB BM25 indexes ─────────────────────────────────
--
-- ParadeDB's pg_search replaces ts_rank_cd with Tantivy-based BM25.
-- Faster and more accurate than PostgreSQL's built-in text search.

-- BM25 index on vehicle chunks
CALL paradedb.create_bm25(
    index_name => 'idx_vehicle_bm25',
    table_name => 'chunks_vehicle',
    key_field  => 'id',
    text_fields => paradedb.field('content')
);

-- BM25 index on article chunks
CALL paradedb.create_bm25(
    index_name => 'idx_article_bm25',
    table_name => 'chunks_article',
    key_field  => 'id',
    text_fields => paradedb.field('content')
);


-- ─── Section 6: IVFFlat tuning helper ─────────────────────────────────
--
-- Set probes at query time for recall/speed balance.
-- Higher probes = better recall, slower queries.
-- At 100 lists: probes=10 gives ~95% recall, probes=30 gives ~99%.

-- Set this in your application's connection init or per-session:
-- SET ivfflat.probes = 10;

-- For your Spring Boot app, add to application.properties:
-- spring.datasource.hikari.connection-init-sql=SET ivfflat.probes = 10


-- ─── Section 7: Verification queries ──────────────────────────────────

-- Check partition sizes
SELECT
    tableoid::regclass AS partition_name,
    COUNT(*) AS row_count
FROM document_chunks
GROUP BY tableoid
ORDER BY partition_name;

-- Check index sizes
SELECT
    indexrelname AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND (indexrelname LIKE 'idx_vehicle%'
    OR indexrelname LIKE 'idx_article%'
    OR indexrelname LIKE 'idx_pdf%'
    OR indexrelname LIKE 'idx_summary%')
ORDER BY pg_relation_size(indexrelid) DESC;

-- Check halfvec population
SELECT doc_type, COUNT(*) AS total,
       COUNT(embedding_half) AS has_halfvec,
       COUNT(*) - COUNT(embedding_half) AS missing_halfvec
FROM document_chunks
GROUP BY doc_type;
