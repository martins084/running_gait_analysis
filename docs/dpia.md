# Data Protection Impact Assessment (DPIA)

## Processing Description
- Upload and analysis of running videos to compute gait metrics.
- Optional generation and storage of pose trajectories and annotated videos.
- User accounts for ownership-bound access.

## Data Categories
- Identification: account email.
- Sensitive/biometric-adjacent: movement/pose data from video.
- Operational metadata: timestamps, retention profile, DSAR records.

## Necessity and Proportionality
- Processing is necessary to deliver requested gait analysis output.
- Minimal profile defaults to reduced retention and reduced data footprint.
- Consent captured at upload before processing starts.

## Risk Assessment
- Re-identification risk from video/pose artifacts.
- Unauthorized access to another user's analysis.
- Over-retention beyond stated purpose.
- Incomplete DSAR fulfillment.

## Mitigations
- Auth + per-user authorization checks.
- Storage minimization profile with minimal default.
- Scheduled retention purge + explicit delete endpoints.
- DSAR endpoints with request state records.
- Deletion audit logging.

## Residual Risks and Actions
- Add stronger token system (JWT with rotation/issuer controls) for production hardening.
- Add external key management and transport/security controls in infrastructure.
- Perform legal review of consent/legal basis text before launch.
