# DSAR Runbook

## Supported Requests
- Export: `POST /dsar/export`
- Delete: `POST /dsar/delete`
- Status: `GET /dsar/{request_id}`

## Operational Steps
1. Verify requester identity through authenticated account session.
2. Submit DSAR request endpoint.
3. Track completion state via DSAR status endpoint.
4. Deliver export file or deletion confirmation to requester.
5. Record exception handling (if any) in support ticketing.

## SLA Guidance
- Target completion: within 30 days.
- For complex requests, document extension reason and notify user.

## Audit Requirements
- Keep `dsar_requests` rows.
- Keep `delete_audit` rows for deletion evidence.
