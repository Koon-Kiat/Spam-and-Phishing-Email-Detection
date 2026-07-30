# Model Card: Spam and Phishing Email Risk Classifier

## Model details

- Project: Spam and Phishing Email Detection
- Version: `v1.0.0`
- Task: binary safe/phishing email-text classification
- Intended use: advisory review support
- Prohibited release claim: suitable for automatic blocking without deployment-specific
  validation and human escalation

The release bundle's `manifest.json` is authoritative for the selected architecture,
threshold, exact run ID, dependency versions, hashes, and measured metrics.

## Training and evaluation

Training uses the two version-1 datasets recorded in `config/datasets.v1.json`. Raw data
is not redistributed. Exact and near-duplicate groups are isolated using masked word
trigrams, 128-permutation MinHash/LSH, exact Jaccard `>= 0.85`, and a 20-fold grouped
14/3/3 partition.

Classical model choice uses three repetitions of five-fold grouped CV with a
group-disjoint inner threshold split. SVM probability calibration is group-aware.
Candidate, threshold, and calibration decisions do not access the untouched test or
SpaPhish v5.

SpaPhish v5 is evaluated once after the model is locked. All 1,395 records and observed
52.4% phishing prevalence are preserved.

SpaPhish was selected to test cross-language and cross-domain transfer that the
English-heavy familiar-source split cannot measure. It contains Spanish-native rather
than translated or synthetic messages, has versioned provenance and complete binary
labels, and supports time and attachment slices. It is external evidence only: its
psychological annotations are not features, and no SpaPhish row influences training,
model selection, threshold selection, or calibration.

## Measured release results

Run `20260725T101318Z` selected the word/character TF-IDF linear SVM at threshold
`0.525`. The optional DistilBERT benchmark ran but did not replace it.

| Evaluation | Rows/folds | Macro F1 | Phishing precision | Phishing recall |
|---|---:|---:|---:|---:|
| Repeated grouped CV mean | 15 folds | 99.05% | — | 98.83% |
| Repeated grouped CV worst fold | 1 fold | 97.77% | — | — |
| Locked validation | 8,471 | 99.46% | 99.49% | 99.43% |
| Untouched familiar-source test | 8,470 | 99.45% | 99.25% | 99.65% |
| Blind SpaPhish v5 | 1,395 | 48.87% | 55.96% | 96.99% |

The untouched test Brier score is `0.0072` and ten-bin ECE is `0.0203`. On SpaPhish
they worsen to `0.3621` and `0.3652`; its false-positive rate is `84.04%`. The model
therefore does **not** generalize adequately to SpaPhish-like deployment traffic without
new representative training data, threshold work, and calibration performed under a
new training-only protocol. High external recall does not offset the operational harm
from that false-positive rate.

The 1,032-row Spanish slice has 47.14% macro F1 and an 84.75% false-positive rate.
Time slices are also unstable: macro F1 is 23.17% for 2014–2023, 55.16% for 2024, and
60.44% for 2025. Another 820 rows have missing or out-of-range dates and are reported
separately rather than silently assigned to a year.

The complete measured evidence is generated in:

- `metrics.json`
- `grouped_cv_summary.csv`
- `generalization_report.json`
- `reliability_diagram.svg`

No earlier random-split 99.49% claim applies to this release. The strong familiar-source
result and weak external result must always be presented together.

## Inputs and outputs

Input is pasted subject/body text or an RFC 822 EML message. Attachments are ignored.
Output includes label, phishing probability, decision threshold, and model version.

Email local parts, phone/account-like identifiers, and URL query values are masked before
feature extraction. This reduces volatile-identifier memorization but does not guarantee
anonymity of every possible input.

## Limitations and risks

- Training sources can contain collection-specific shortcuts.
- English-heavy training may not represent multilingual inboxes; SpaPhish measures one
  Spanish corpus, not all Spanish email.
- Phishing tactics and legitimate workflows drift over time.
- Missing attachment content and incomplete header context limit detection.
- A calibrated score is not a guarantee and can be wrong confidently.
- External prevalence differs from many production inboxes, affecting apparent
  precision and operational workload.
- The blind external result demonstrates severe cross-domain false-positive and
  calibration drift; this release is a reproducible research baseline, not a
  production-ready blocking model.
- Training timestamps are incomplete, so chronological training validation is limited;
  external year slices provide time-separated evidence only.

Users should verify requests through trusted channels, navigate directly to known
websites, and never rely on this model as the sole security control.

## Monitoring and retraining

Use content-free monitoring and the thresholds in
`docs/MONITORING_AND_ROLLBACK.md`. Retraining requires a new provenance review,
training-only selection process, untouched test, external protocol, model card, and
versioned release.

## Ethical and privacy considerations

Raw emails may contain personal, sensitive, explicit, or fraudulent content. This
repository distributes no raw rows. Release artifacts undergo a full-email and long
numeric identifier scan. Domain/campaign diagnostics use hashed IDs.

## License

The trained model bundle is provided under the terms in `MODEL_LICENSE.md`. The source
code's MIT license does not relicense dataset content or the model bundle.
