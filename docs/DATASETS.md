# Dataset Provenance, Licensing, and Privacy

Raw datasets are inputs, not project source code. They are excluded from Git and every
release archive. The versioned machine-readable record is
[`config/datasets.v1.json`](../config/datasets.v1.json).

## Exact training inputs

| Local file | Source and version | License context | Bytes | SHA-256 |
|---|---|---|---:|---|
| `data/ceas_08.csv` | Kaggle `naserabdullahalam/phishing-email-dataset`, version 1 | CC BY-SA 4.0 collection | 67,902,056 | `22375e7d5f5a8229dbe987914ee9b3705656c590038662a7df6054629b376074` |
| `data/phishing_email.csv` | Kaggle `subhajournal/phishingemails`, version 1 | LGPL-3.0 listing | 52,015,953 | `41290f595b8b436cb7e292e6463fd5640024e7f2dc3a520a29756bde01ea02c4` |

The first archive is pinned by SHA-256
`3a21343518a3a5e762d964ef6eced697f3cbde3a6d0ab7c9cbe0c17dc93b98f8`.
The second is pinned by
`5d4d5cd13963aa8272ee771a64cae31aad9a4028534a9fc598f00fbf52b821be`.

The standalone archive stores `Phishing_Email.csv` with CRLF line endings. The fetch
command verifies its source hash, changes only CRLF to LF, and then verifies the local
hash above. This explains why the local hash differs from the archive-member hash.

## Fetching

Review the linked terms, then explicitly accept them:

```powershell
python -m spamandphishingdetection fetch-data --accept-licenses
```

The command:

- downloads the exact version-1 archives from Kaggle's public download endpoints;
- verifies archive byte counts and SHA-256 values before extraction;
- extracts only the required member;
- verifies the pre-normalization and final file hashes;
- accepts an already matching local file;
- refuses to overwrite any mismatched file.

The downloads are free to the repository owner. No paid storage is selected. Raw data
stays local; the public model uses a GitHub Release asset, for which GitHub documents a
2 GiB per-file limit and no total release size or bandwidth quota.

## External corpus

`--external` fetches only the primary SpaPhish v5 CSV:

- DOI: `10.17632/hz2d6gz7pc.5`
- License: CC BY 4.0
- File SHA-256: `656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e`
- Rows: 1,395
- Observed labels: 664 legitimate and 731 phishing

```powershell
python -m spamandphishingdetection fetch-data --accept-licenses --external
```

SpaPhish is used only for blind external evaluation after model, threshold, and
calibration are locked. It is never added to training during this release.

## Privacy review and publication decision

Automated scans of the training CSVs detected thousands of rows containing address-,
URL-, or phone-like strings. CEAS also carries sender, receiver, and message fields.
Automated pattern matching cannot establish consent or remove every personal identifier.

Decision: publish source links, versions, license context, normalization rules, and
checksums only. Do not redistribute the raw CSVs, EML files, row samples, or logs.

Feature preparation masks:

- email local parts while retaining domain context;
- phone-like and long account-like identifiers;
- URL query values and fragments.

Release packaging scans serialized model payloads for full email addresses and long
numeric identifiers and fails closed when either is detected.

## Attribution

Full attribution and license links are preserved in
[`DATASET_NOTICES.md`](../DATASET_NOTICES.md). A source page's license statement does not
guarantee that every upstream component was originally collected under identical terms.
The no-redistribution decision limits this unresolved upstream risk.
