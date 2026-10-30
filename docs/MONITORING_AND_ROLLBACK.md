# Monitoring, Retraining, and Rollback

## Batch contract

One JSON object per line:

```json
{
  "timestamp": "2026-07-25T12:00:00Z",
  "probability": 0.73,
  "predicted_label": "phishing",
  "true_label": 1,
  "slices": {
    "source": "review_queue",
    "language_slice": "spanish",
    "domain_group": "domain_abcd1234"
  }
}
```

Each line is at most 8,192 characters. Only these top-level fields are accepted:

| Field | Rule |
|---|---|
| `timestamp` | Required ISO 8601 string, at most 64 characters |
| `probability` | Required JSON number in `[0, 1]` |
| `predicted_label` | Required `"safe"`, `"phishing"`, `0`, `1`, `"0"`, or `"1"`; must equal `probability >= threshold` for the monitored artifact |
| `true_label` | Optional, same binary values; booleans and other values are rejected |
| `slices` | Optional object of allowlisted scalar values (below) |

Allowed `slices` dimensions:

| Dimension | Accepted values | Reported as |
|---|---|---|
| `length_slice` | `short`, `medium`, `long`, `very_long` | as given |
| `has_html`, `has_url`, `has_obfuscation` | JSON boolean, or `"true"`/`"false"` | `True`/`False` |
| `attachment_slice` | `has_attachment`, `no_attachment`, `unknown` | as given |
| `language_slice` | `spanish`, `other_or_unknown` | as given |
| `source` | 1–64 characters of letters, digits, `_`, `.`, `-` | as given |
| `domain_group` | `domain_` plus 16 lowercase hex characters, or `domain_none` | as given |
| `campaign_group` | `campaign_` plus 16 lowercase hex characters | as given |
| `domain`, `sender_domain`, `url_domain` | a hostname | hashed registrable-domain ID |
| `campaign` | 1–128 printable characters | hashed `campaign_` ID |

Raw hostnames are reduced to their registrable domain and hashed with the same
`domain_` scheme used for training-time `domain_group`, so monitoring IDs match the
baseline. Campaign names are case-folded and hashed with the `campaign_` namespace.

Everything else is rejected, including unknown fields and dimensions, nested objects,
arrays, `null`, numbers, free text with spaces, and over-length values.
Content-like keys such as `subject`, `body`, `message`, `text`, `sender`, `url`,
`headers`, or `attachments` are rejected at any depth. Error messages give the line
number and the rule that failed, never the submitted value. A report can therefore
contain only counts, scores, allowlisted categories, and hashed IDs, which is what
`content_retained: false` asserts.

```powershell
python -m spamandphishingdetection monitor --input batch.jsonl
```

## Alerts and retraining triggers

Open a review when any default threshold is reached:

| Signal | Trigger |
|---|---:|
| Score population stability index | `>= 0.20` |
| Predicted phishing prevalence absolute change | `>= 0.10` |
| Slice fraction absolute change | `>= 0.15` |
| Labeled macro-F1 drop from release test | `>= 0.05` |

Also retrain after a material input-policy change, new language/source integration,
confirmed sustained campaign miss, calibration failure, or security dependency change
that affects serialization or inference.

Monitoring is an alert, not an automatic retraining command. Review labeling quality and
collection rights first. New external evidence must not be silently folded into
training; create a new versioned release protocol.

## Promotion

```powershell
python -m spamandphishingdetection promote --artifact <run-id>
```

Promotion requires manifest schema 2, matching model checksum, privacy clearance, full
reports, and locked external evaluation. `latest.json` is replaced atomically with the
validated `manifest_sha256`, and the same record is appended to
`promotion_history.json`. Loading verifies the manifest and payload against that record
before deserializing anything; see the trust model in [SECURITY.md](../SECURITY.md).

## Rollback

```powershell
python -m spamandphishingdetection promote --rollback
```

Rollback revalidates the previous artifact and refuses it if its manifest changed after
that promotion was recorded, before atomically restoring it. After a rollback:

1. confirm `/health` reports the expected model version;
2. run a known safe and known phishing smoke case;
3. preserve the alert batch without message content;
4. open a private security report when compromise is suspected;
5. do not delete the failed artifact until the incident review is complete.
