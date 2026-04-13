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