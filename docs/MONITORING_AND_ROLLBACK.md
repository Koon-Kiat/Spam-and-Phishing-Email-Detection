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

`true_label` and `slices` are optional. Do not include subject, body, message, raw email,
or other content fields; the command rejects them. Use privacy-hashed domain/campaign
IDs. If a raw domain key is supplied, it is hashed before aggregation.

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
reports, and locked external evaluation. `latest.json` is replaced atomically and the
event is recorded in `promotion_history.json`.

## Rollback

```powershell
python -m spamandphishingdetection promote --rollback
```

Rollback revalidates the previous artifact before atomically restoring it. After a
rollback:

1. confirm `/health` reports the expected model version;
2. run a known safe and known phishing smoke case;
3. preserve the alert batch without message content;
4. open a private security report when compromise is suspected;
5. do not delete the failed artifact until the incident review is complete.
