# Testing and API

## Browser

```powershell
python server/flask_app.py
```

Open <http://127.0.0.1:5000>. Option 1, pasted subject/body, is above Option 2, the
full-width EML uploader. When a file is selected, it becomes the active source and a
Remove action restores pasted-text input without deleting it.

Validation is announced beside the relevant input. Loading uses `aria-busy`; keyboard
focus is visible; result updates use a live region. The result is advisory and includes
the score, threshold marker, model version, and independent-verification guidance.

## Prediction endpoint

`POST /api/v1/predict` accepts either JSON:

```json
{
  "subject": "Optional subject",
  "text": "Required body"
}
```

or a raw RFC 822 message with `Content-Type: message/rfc822`.

Response fields remain:

```json
{
  "label": "phishing",
  "is_phishing": true,
  "probability": 0.93,
  "threshold": 0.39,
  "model_version": "YYYYMMDDTHHMMSSZ"
}
```

PowerShell:

```powershell
$body = @{
    subject = "Account notice"
    text = "Verify your password immediately"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/api/v1/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

EML:

```powershell
Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/api/v1/predict" `
    -Method Post `
    -ContentType "message/rfc822" `
    -InFile "message.eml"
```

`GET /health` returns `200` with the loaded model version or `503` when no valid artifact
can be loaded. Empty input returns `400`; oversized requests return `413`.

## Automated checks

```powershell
ruff check src tests main.py server/flask_app.py
python -m pytest
python -m pip check
```

Tests cover provenance verification and overwrite refusal, privacy canonicalization,
MinHash grouping and exact verification, grouped split isolation, nested grouped CV,
calibration metrics, SpaPhish adaptation, monitoring alerts, promotion rollback, release
contents, stable JSON/EML inference, UI order and source precedence, and generated report
charts.

## Release commands

```powershell
python -m spamandphishingdetection generalization-report --artifact <run-id>
python -m spamandphishingdetection promote --artifact <run-id>
python -m spamandphishingdetection promote --rollback
python -m spamandphishingdetection package-release --artifact <run-id>
```

`generalization-report` refuses a second external evaluation for the same run. Promotion
requires schema 2, matching model checksums, privacy clearance, and complete internal and
external reports. Packaging rejects raw CSV or EML files and verifies both internal and
outer `SHA256SUMS`.
