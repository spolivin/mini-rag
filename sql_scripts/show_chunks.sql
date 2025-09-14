.headers on
.mode column

SELECT chunk_id, doc_id, chunk_text
FROM (
    SELECT 
        c.*,
        ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY chunk_id) AS rn
    FROM chunks c
)
WHERE rn <= 5;
