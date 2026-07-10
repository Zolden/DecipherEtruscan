# Legacy TexRelations query list

This directory preserves the 242 distinct Trismegistos identifiers that were
queried during an exploratory TexRelations check on 2026-07-09.

The original file was malformed JSONL: each identifier was followed by an
empty `resp` field, and no response body, HTTP status, exact source parameter,
or endpoint revision was retained. On 2026-07-10 it was mechanically repaired
into valid JSONL without inventing results:

```json
{"tm":"247797","response":null,"status":"legacy_response_not_preserved"}
```

Consequently, this is a reproducible **query-id list**, not evidence that the
service returned zero relations. Any future refresh must use a documented
endpoint/source parameter, retain retrieval time and HTTP status/body, and
observe the current Trismegistos data-service terms.

Integrity hashes are in `SHA256SUMS`.
