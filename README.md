# Spam and Phishing Email Detection

A reproducible email-risk classifier with grouped leakage controls, calibrated
probabilities, external validation, privacy-safe monitoring, and a browser review tool.
The model is advisory: it is not suitable for automatic blocking without validation on
the deployment population and human review of errors.

Repository: <https://github.com/Koon-Kiat/Spam-And-Phishing-Detection-Using-Machine-Learning>

## What is released

- MIT-licensed source code and documentation.
- A separately licensed, privacy-scanned model bundle on the GitHub `v1.0.0` release.
- Download instructions and immutable hashes for the source datasets.

Raw CSV and EML content is never committed or attached to a release. Automated review
found thousands of address-, URL-, and phone-like values in the source material, so
redistributing raw rows would create unnecessary privacy and provenance risk.

## Release result

The selected word/character TF-IDF linear SVM scored 99.45% macro F1 on the 8,470-row
untouched familiar-source test, but only 48.87% macro F1 on blind SpaPhish v5. External
phishing recall was 96.99%, while the false-positive rate was 84.04% and ten-bin ECE
was 36.52%. This measured domain/calibration failure is why the model is released only
as an advisory research baseline—not as an automatic blocker. See the
[model card](MODEL_CARD.md) for the full interpretation.

## Quick start

Python 3.14.6 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Review [dataset attribution and terms](docs/DATASETS.md), then fetch the exact training
files. Add `--external` only when you are ready to perform the one-time blind SpaPhish
evaluation.

```powershell
python -m spamandphishingdetection fetch-data --accept-licenses
python -m spamandphishingdetection audit
python -m spamandphishingdetection train --skip-transformer
```

The full optional DistilBERT benchmark requires:

```powershell
python -m pip install -e ".[transformer,dev]"
python -m spamandphishingdetection train
```

After model and threshold selection are locked:

```powershell
python -m spamandphishingdetection fetch-data --accept-licenses --external
python -m spamandphishingdetection generalization-report --artifact <run-id>
python -m spamandphishingdetection promote --artifact <run-id>
python -m spamandphishingdetection package-release --artifact <run-id>
```

## Browser review

```powershell
python server/flask_app.py
```

Open <http://127.0.0.1:5000>. Paste an email or upload one `.eml` file. A selected file
is the active source until removed; pasted text is retained locally in the form but is
not sent while the file is active.

The result explains the phishing probability, decision threshold, model version, and
recommended verification steps. Submitted content is processed in memory and is not
added to training data or reports.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"subject\":\"Account notice\",\"text\":\"Verify your password immediately\"}"
```

Stable endpoints:

- `POST /api/v1/predict`
- `GET /health`

Raw RFC 822 requests use `Content-Type: message/rfc822`. The removed Outlook add-in
routes `/evaluateEmail`, `/taskpane.html`, and `/commands.html` return `404`.

## Generalization controls

The training and release workflow:

1. normalizes and privacy-canonicalizes volatile identifiers;
2. removes exact duplicates and conflicting-label copies;
3. groups near duplicates with 128-permutation MinHash/LSH and exact Jaccard `>= 0.85`;
4. assigns groups across 20 folds, using 14/3/3 for train/validation/test;
5. selects classical candidates using three repetitions of five-fold grouped CV;
6. keeps threshold and SVM calibration groups disjoint;
7. reports Brier score, log loss, ECE, reliability, source, domain, campaign, and
   robustness slices;
8. evaluates SpaPhish v5 once after selection, preserving all 1,395 rows and its observed
   prevalence;
9. packages only a privacy-scanned model, manifest, reports, notices, and checksums.

See [ML workflow](docs/ML_WORKFLOW.md) and the generated model
[card](MODEL_CARD.md) for measured results and limitations.

## Monitoring

`monitor` accepts content-free JSONL. Required fields are `timestamp`, `probability`, and
`predicted_label`; `true_label` and a `slices` object are optional.

```powershell
python -m spamandphishingdetection monitor --input batch.jsonl
```

Message-content fields are rejected. Domain values are hashed before aggregation. See
[monitoring and retraining](docs/MONITORING_AND_ROLLBACK.md).

## Development checks

```powershell
ruff check src tests main.py server/flask_app.py
python -m pytest
python -m pip check
```

## Licenses and security

Source code is [MIT licensed](LICENSE). The model and datasets have separate terms; see
[MODEL_LICENSE.md](MODEL_LICENSE.md) and [DATASET_NOTICES.md](DATASET_NOTICES.md).
Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
