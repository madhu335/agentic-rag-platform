ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS doc_type VARCHAR(50);

-- Articles — all start with "motortrend-"
UPDATE document_chunks
SET doc_type = 'article'
WHERE doc_id LIKE 'motortrend-%';

-- Vehicles — your known vehicle IDs
UPDATE document_chunks
SET doc_type = 'vehicle'
WHERE doc_id IN (
                 'bmw-m3-2025-competition',
                 'tesla-model3-2025-long-range',
                 'chevrolet-corvette-2025-z06',
                 'rivian-r1t-2025-adventure',
                 'ford-f150-2025-lariat',
                 'toyota-camry-2025-xse-hybrid',
                 'honda-crv-2025-sport-hybrid',
                 'porsche-911-2025-carrera-s'
    );

UPDATE document_chunks
SET doc_type = 'pdf'
WHERE doc_type IS NULL;


CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_type
    ON document_chunks(doc_type);

-- Step 5: Verify backfill
SELECT doc_type, COUNT(*) as chunk_count
FROM document_chunks
GROUP BY doc_type
ORDER BY doc_type;
