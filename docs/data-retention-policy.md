# Data Retention Policy

## Profiles
- `minimal`: metrics only, no long-term pose/video artifacts.
- `standard`: metrics + artifacts for routine user workflows.
- `full`: extended retention for debugging/research with stricter controls.

## Environment Controls
- `RETENTION_DAYS_MINIMAL`
- `RETENTION_DAYS_STANDARD`
- `RETENTION_DAYS_FULL`
- `RETENTION_PURGE_INTERVAL_SEC`

## Purge Behaviour
- Scheduled purge removes expired rows from `analyses`.
- Related files are removed from `results/`:
  - `<analysis_id>_poses.json`
  - `<analysis_id>_annotated.mp4`
  - `<analysis_id>_result.json`

## User-Initiated Deletion
- `DELETE /analyses/{id}` for direct delete.
- `POST /dsar/delete` for DSAR-tracked deletion requests.
