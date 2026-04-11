# GDPR Compliance Description

## 1. Purpose and Scope

This document provides a single, detailed description of how GDPR compliance is implemented and operated for the Running Gait Analysis system.

The system processes user-uploaded running videos and generates biomechanical analysis outputs (pose-derived metrics, optional pose JSON, optional annotated video). In SaaS use, uploaded media and derived movement data can be personal data and may become biometric-adjacent data depending on use context, so GDPR controls are applied across the full lifecycle.

This document is intended to be the central compliance reference for:
- product owners,
- developers,
- security reviewers,
- legal/privacy reviewers,
- operations/support teams.

## 2. Roles and GDPR Positioning

- **Controller**: the deployed SaaS operator (organization that decides purposes and means of processing).
- **Processor(s)**: hosting/infrastructure vendors and subprocessors used by the controller.
- **Data subject**: the end user (or person whose running video is uploaded).

If deployed for public or organizational use, GDPR obligations apply in full, including transparency, lawful basis, rights handling, retention limits, and accountability.

## 3. Data Categories Processed

### 3.1 Account and identity data
- Email address.
- Password hash (salted PBKDF2 hash; no plaintext password storage).
- Internal user ID.

### 3.2 User-provided content
- Uploaded running video file.

### 3.3 Derived analysis data
- Feature-level gait metrics (stride, cadence, symmetry, joint-angle features, etc.).
- Optional pose landmarks JSON.
- Optional annotated MP4 output.
- Analysis metadata (analysis ID, status, timestamps, selected storage profile).

### 3.4 Compliance and audit records
- Consent metadata (version, timestamp, locale).
- DSAR request records (request type, status, payload, timestamps).
- Deletion audit records.

## 4. Lawful Basis and Transparency

### 4.1 Lawful basis model
- **Primary**: explicit user consent before upload/processing.
- **Supplementary**: performance of requested service (analysis execution and result delivery).

### 4.2 Transparency implementation
- Privacy notice page is available in the frontend.
- Consent gate is presented before upload.
- User-facing rights actions (export/delete request) are available in the results workflow.

### 4.3 Consent evidence
Consent metadata is stored with each analysis event:
- consent version,
- consent timestamp,
- notice locale.

## 5. Access Control and Authorization

### 5.1 Authentication
- User registration and login endpoints issue bearer tokens.
- Authenticated identity is required for sensitive API operations.

### 5.2 Ownership model
- Analysis records are bound to `user_id`.
- Reads/downloads are scoped to owner-only access.
- Cross-user access attempts are denied (resource isolation).

### 5.3 Protected resources
Owner authorization is enforced for:
- results retrieval,
- pose JSON retrieval,
- annotated video download,
- deletion operations,
- DSAR request/status access.

## 6. Data Minimization and Storage Profiles

The platform uses profile-based storage minimization to reduce unnecessary retention:

- **minimal** (default): stores essential derived outputs with reduced artifact retention.
- **standard**: stores commonly needed artifacts for routine user workflows.
- **full**: extended retention for research/debug use cases with stricter governance.

The default profile is designed to support privacy-by-default and storage limitation principles.

## 7. Retention and Deletion

### 7.1 Retention controls
Retention windows are configurable via environment variables (days by profile). A scheduled purge process removes expired data.

### 7.2 Automatic purge
The purge process removes:
- expired DB records,
- associated result artifacts (`_result.json`, `_poses.json`, `_annotated.mp4`).

### 7.3 User-initiated deletion
Two deletion paths exist:
- direct analysis deletion endpoint,
- DSAR deletion workflow.

Both are tracked with deletion audit records for accountability.

## 8. Data Subject Rights (DSAR)

### 8.1 Supported rights workflow
- **Export** request: generate machine-readable export package for the requesting user.
- **Deletion** request: remove user-owned records/artifacts per request scope.
- **Status** endpoint: request state visibility and traceability.

### 8.2 Safeguards
- DSAR requests are user-authenticated.
- DSAR results are user-scoped.
- Request metadata is persisted for traceability and SLA monitoring.

## 9. Security Controls (Technical and Organizational)

### 9.1 Technical controls
- Authenticated API access for sensitive endpoints.
- Owner-scoped authorization checks.
- Password hashing with salt and PBKDF2.
- Configurable CORS restrictions.
- Artifact deletion and audit trails.

### 9.2 Organizational controls
- DSAR runbook for support/operations.
- Retention policy documentation.
- Privacy policy and DPIA documentation.
- Release gate requirement for GDPR-critical controls before production deployment.

## 10. DPIA Summary

The DPIA identifies and mitigates key risks:

- **Risk**: re-identification from video/pose artifacts.
  - **Mitigation**: minimization profiles, restricted access, retention limits.

- **Risk**: unauthorized cross-user data access.
  - **Mitigation**: auth + per-user ownership checks on all sensitive reads/downloads.

- **Risk**: over-retention.
  - **Mitigation**: scheduled purge + profile-based retention configuration.

- **Risk**: incomplete rights fulfillment.
  - **Mitigation**: explicit DSAR endpoints + tracked request states + runbook.

Residual risk is managed through periodic review, hardening, and legal validation for deployment-specific terms.

## 11. API Compliance Surface (High-Level)

- Auth:
  - register,
  - login,
  - current user check.
- Analysis:
  - analyze with consent metadata and storage profile.
- Protected retrieval:
  - results,
  - pose JSON,
  - annotated download.
- Deletion:
  - direct analysis deletion.
- DSAR:
  - export request,
  - delete request,
  - request status,
  - export artifact retrieval.
- Retention operations:
  - scheduled purge,
  - admin-trigger purge endpoint.

## 12. Logging, Evidence, and Accountability

Compliance evidence is maintained through:
- DSAR request records,
- deletion audit records,
- consent metadata on analysis rows,
- retention policy configuration and purge behavior.

For production use, log retention and access-to-logs policy should be formally defined and reviewed with legal/security stakeholders.

## 13. Operational Checklist for Production Readiness

Before go-live, confirm:
- [ ] controller identity and contact details are finalized in privacy notice.
- [ ] lawful basis text is legally reviewed for jurisdiction and use case.
- [ ] retention windows approved by legal/security and set in environment.
- [ ] DSAR SLA and support ownership are documented and staffed.
- [ ] admin access paths are secured and monitored.
- [ ] backup/restore procedures align with deletion and retention obligations.
- [ ] incident response process includes personal data breach handling.

## 14. Limitations and Recommended Next Hardening

To increase compliance maturity further:
- replace basic token model with hardened JWT/IdP-based auth (key rotation, issuer/audience claims),
- add formal consent versioning governance and change log,
- add cryptographic at-rest protection controls at infrastructure level,
- add immutable audit log pipeline for critical privacy events,
- implement formal Records of Processing Activities (RoPA),
- run scheduled DPIA re-assessment when processing scope changes.

## 15. Related Documents

- `docs/privacy-policy.md`
- `docs/data-retention-policy.md`
- `docs/dsar-runbook.md`
- `docs/dpia.md`

---

This document is the single consolidated GDPR compliance description for the project and should be updated whenever processing scope, legal basis, retention logic, or rights workflows materially change.
