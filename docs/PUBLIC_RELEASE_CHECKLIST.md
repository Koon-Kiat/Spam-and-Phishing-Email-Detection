# Public Release Checklist

Evidence for the `v1.0.0` source and model release. Generated reports and raw datasets
remain untracked; their hashes are recorded in the model bundle.

## Dataset and privacy

- [x] Exact Kaggle version-1 source URLs, archive hashes, member hashes, licenses, and
      normalization are recorded in `config/datasets.v1.json`.
- [x] SpaPhish v5 DOI, license, official file ID, size, hash, row count, and prevalence
      are pinned.
- [x] Raw-message automated privacy review identified address-, URL-, and phone-like
      content.
- [x] Raw CSV and EML redistribution is prohibited by release policy and `.gitignore`.
- [x] `fetch-data --accept-licenses [--external]` verifies every download and refuses
      mismatched overwrites.
- [x] Model feature preparation masks volatile sensitive identifiers.
- [x] Release validation scans the serialized payload for full emails and long numeric
      identifiers.
- [x] Source attribution and license context are preserved in `DATASET_NOTICES.md`.

## Leakage and generalization

- [x] Exact duplicates and conflicting-label groups are handled before splitting.
- [x] 128-permutation MinHash/LSH candidates require exact Jaccard `>= 0.85`.
- [x] Similarity groups are indivisible across the 20-fold 14/3/3 split.
- [x] Classical selection uses three repetitions of five-fold grouped CV and worst-fold
      stability.
- [x] Inner threshold and SVM calibration folds are group-disjoint.
- [x] Untouched and external partitions are excluded from all selection.
- [x] Brier, log loss, ten-bin ECE, and reliability diagrams are generated.
- [x] Domain, campaign, language, length, HTML, URL, obfuscation, attachment, source,
      time-period, and perturbation diagnostics are generated.
- [x] SpaPhish v5 retains all 1,395 rows and observed prevalence.
- [x] Cross-domain limitations and the advisory-not-blocking policy are in the model
      card.

## Product and API

- [x] Outlook manifest, routes, templates, scripts, styles, icons, and setup document
      are removed.
- [x] `/evaluateEmail`, `/taskpane.html`, and `/commands.html` return `404`.
- [x] `/api/v1/predict`, `/health`, JSON, EML, and response fields remain stable.
- [x] Browser inputs are stacked, file precedence/removal is explicit, and validation,
      loading, keyboard focus, risk scale, threshold, version, and verification guidance
      are accessible.
- [x] The UI uses ivory, ink, amber, rust, and moss with solid controls and no decorative
      grid, glow, gradient, status dot, or rounded status pill.

## Model operations

- [x] Manifest schema 2 records grouping, calibration, baseline, dependency, dataset,
      model, and report hashes.
- [x] Schema-1 artifacts remain loadable for inference.
- [x] Content-free monitoring rejects message fields and defines drift/retraining
      thresholds.
- [x] Promotion validates and updates atomically; rollback revalidates the prior model.
- [x] Release packaging contains model, manifest, metrics, model card, notices, and
      checksums, but no raw data.
- [x] The model uses a separate attribution/share-alike notice; MIT covers source only.
- [x] The exact dependency lock passed `pip check` and `pip-audit` with no known
      vulnerabilities; the vulnerable build-tool pin found during review was upgraded.

## Repository and community

- [x] The repository URL and project security contact route are documented without
      requiring personal-name attribution.
- [x] `SECURITY.md`, the model card, dataset notices, monitoring, retraining, and
      rollback instructions are present.
- [x] CI runs lint, tests, dependency checks, and raw-data/release-source guards.
- [x] Private GitHub CI passed for the final pre-public source
      ([run 30156164393](https://github.com/Koon-Kiat/Spam-and-Phishing-Email-Detection/actions/runs/30156164393)).
- [x] The GitHub [`v1.0.0` release](https://github.com/Koon-Kiat/Spam-and-Phishing-Email-Detection/releases/tag/v1.0.0)
      and model asset were published while private; a fresh download matched SHA-256
      `497d12daca39312e470d6e5e1d70a74a6f032439951a1fcfe4a26a603c1f8220`.
- [x] Ordering exception documented: GitHub limits private vulnerability reporting to
      public repositories. The authenticated private-repository attempt returned `404`;
      the feature was enabled immediately after visibility changed and verified as
      `enabled: true`. See
      [GitHub's eligibility documentation](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository).
- [x] Repository visibility changed from private to public only after private CI,
      release publication, and downloaded-asset checksum verification passed.

## Local acceptance commands

```powershell
ruff check src tests main.py server/flask_app.py
python -m pytest
python -m pip check
pip-audit -r requirements.lock --no-deps
python -m spamandphishingdetection fetch-data --accept-licenses --external
python -m spamandphishingdetection train
python -m spamandphishingdetection generalization-report --artifact <run-id>
python -m spamandphishingdetection report --artifact <run-id>
python -m spamandphishingdetection promote --artifact <run-id>
python -m spamandphishingdetection package-release --artifact <run-id>
```

Before every commit:

```powershell
git status --short
git diff --check
git ls-files | Select-String -Pattern '\.(csv|eml|log)$'
```
