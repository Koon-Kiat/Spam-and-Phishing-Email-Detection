# v1.0.0 — Leakage-aware public research release

This initial release provides the source, browser review interface, reproducible
training/generalization workflow, and a separately licensed model bundle.

## Model evidence

- Selected model: word/character TF-IDF linear SVM
- Locked threshold: `0.525`
- Repeated grouped-CV macro F1: `99.05%` mean, `97.77%` worst fold
- Untouched familiar-source macro F1: `99.45%`
- Blind SpaPhish v5 macro F1: `48.87%`
- Blind SpaPhish phishing recall: `96.99%`
- Blind SpaPhish false-positive rate: `84.04%`

The external result demonstrates severe domain and calibration drift. This model is an
advisory research baseline and is not suitable as an automatic email blocker.

## Release boundaries

- Raw CSV and EML data are not distributed.
- The model bundle passed the release privacy scan.
- Source code is MIT licensed; the model bundle and datasets have separate terms.
- The ZIP contains internal `SHA256SUMS`.
- Outer asset SHA-256:
  `497d12daca39312e470d6e5e1d70a74a6f032439951a1fcfe4a26a603c1f8220`

See `MODEL_CARD.md`, `DATASET_NOTICES.md`, and `MODEL_LICENSE.md` inside the model
bundle before reuse.
