# Privacy Policy

## Controller and Scope
- This application processes user-uploaded running videos and derived gait metrics.
- For SaaS deployments, the deployer is the data controller.

## Data Categories
- Account data: email, password hash.
- Analysis data: derived gait features.
- Optional artifacts (profile-dependent): pose JSON and annotated video.
- DSAR and deletion audit records.

## Legal Basis
- Explicit consent captured before upload.
- Contractual necessity for delivering requested analysis features.

## Data Subject Rights
- Access/export: available through DSAR export request.
- Erasure: available through direct analysis deletion and DSAR delete request.
- Restriction/objection: handled through support workflow.

## Retention
- Minimal profile: shortest retention.
- Standard/full profile: longer retention for operational needs.
- Automatic purge runs on schedule and removes expired records/artifacts.

## Security Controls
- Authenticated endpoints with bearer token.
- Per-user ownership checks for all analysis data reads/downloads.
- Passwords stored as salted PBKDF2 hashes.

## Contact
- Define controller support email and postal contact before production use.
