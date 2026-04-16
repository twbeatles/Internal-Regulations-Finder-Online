# Implementation Review Status (2026-04-16)

## Summary
- This document records the implementation review items raised on 2026-04-16 and the fact that they have been addressed in the current branch.
- Primary references kept in sync during this pass:
  - `README.md`
  - `CLAUDE.md`
  - `GEMINI.md`
  - `BUILD.md`
  - `OFFLINE_SETUP.md`

## Resolved items

### Search/cache correctness
- Search cache keys now include `filter_file` and `filter_file_id`.
- Filtered searches no longer reuse cached unfiltered results.
- Search history is recorded only for successful searches.

### Index consistency
- File deletion now rebuilds both BM25 and vector index state from the current in-memory source of truth.
- Shared index cleanup helpers were introduced:
  - `RegulationQASystem.clear_index()`
  - `RegulationQASystem.remove_file_from_index()`
- Duplicate-basename sync cache collisions were removed by switching cache metadata keys to folder-relative paths.

### Revision and frontend behavior
- Revision version numbering now continues across both legacy filename keys and current `file_id` keys.
- Reader mode preview now prefers by-id preview routes when `file_id` is available.

### Packaging/spec audit
- All `.spec` files were re-validated with:
  - `Get-ChildItem -Name *.spec | ForEach-Object { python -m py_compile $_ }`
- `internal_regulations_lite.spec` was checked specifically.
- Result: no additional spec changes were required for the 2026-04-16 backend/frontend fixes.

## Verification
- `python -m pytest -q` -> `70 passed`
- `python -m py_compile app/services/search.py app/routes/api_search.py app/routes/api_files.py app/services/file_manager.py` -> passed
- `node --check static/app.js` -> passed

## Notes
- This file is an audit/status artifact, not a replacement for the main product docs.
- Canonical operational context remains in `README.md`, `CLAUDE.md`, `GEMINI.md`, `BUILD.md`, and `OFFLINE_SETUP.md`.
