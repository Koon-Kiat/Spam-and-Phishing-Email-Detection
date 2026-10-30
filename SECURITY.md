# Security Policy

## Supported version

Security fixes are provided for the latest GitHub release.

## Report a vulnerability

Use GitHub's private vulnerability reporting on the repository Security tab:

<https://github.com/Koon-Kiat/Spam-and-Phishing-Email-Detection/security/advisories/new>

Do not disclose exploitable details in a public issue. Include the affected version,
reproduction steps, impact, and any suggested mitigation. The project will acknowledge
a report as soon as practical and coordinate disclosure after a fix is available.

## Scope

Relevant reports include unsafe EML parsing, request-size bypass, model artifact
tampering, serialized-model risks, privacy leaks, dataset-fetch integrity failures,
release checksum bypass, and secret exposure.

Model false positives and false negatives are expected safety limitations rather than
software vulnerabilities unless they result from a reproducible security defect.

Only load model artifacts from this repository's checksum-verified GitHub Releases.
Joblib/pickle artifacts can execute code while loading and must be treated as executable
software.

## Model artifact trust model

Nothing inside an artifact directory is allowed to vouch for itself. Before any payload
is deserialized, the prediction service, browser host, and every CLI command that reads
an artifact (`predict`, `serve`, `report`, `generalization-report`, `monitor`) apply the
same checks:

1. The trusted manifest SHA-256 is taken from a record outside the artifact directory:
   `artifacts/latest.json` when it points at the artifact, otherwise the most recent
   matching entry in `artifacts/promotion_history.json`. `train` and `promote` write
   these records; `promote` writes them only after full release validation.
2. An artifact without such a record, including any directory outside `artifacts/`, is
   refused unless the operator passes `--manifest-sha256` with a value obtained out of
   band, such as the `manifest.json` line of a release's verified internal `SHA256SUMS`.
3. The manifest bytes are hashed before they are parsed and must match exactly.
4. `model.joblib` is read into memory, its SHA-256 must equal the manifest's
   `model_sha256`, and only those verified bytes are passed to `joblib.load`.
5. Transformer and ensemble payloads must match the manifest's `transformer_files`
   exactly: every recorded file present with the recorded size and SHA-256, and no
   unrecorded file, link, or special file anywhere under `transformer/`. Pickle-capable
   weight formats and Python files are refused. Any sharded-checkpoint index must map
   every tensor to a plain file name inside the verified set. Only then are the tokenizer
   and model loaded, from local files only, as safetensors, with remote code disabled.

Step 5 is also the project's mitigation for GHSA-4j2p-28q2-5m79 (CVE-2026-69112). The
advisory lists Accelerate through 1.14.0, but Accelerate 1.15.0 and transformers 5.19.0
still join `weight_map` entries onto the checkpoint folder without validation, and no
fixed release exists yet. The project does not call the affected Accelerate loaders, and
it never hands an unverified index to any loader.

Symlinks, non-regular files, oversized files, corrupted JSON, missing integrity fields,
checksum mismatches, and deserialization failures all fail closed with `ArtifactError`.
The API then reports the stable `Prediction model is unavailable` response; paths,
exception text, and backend details are written to operator logs only.

These checks detect corruption and tampering after a trust record was written. They do
not defend against an attacker who can already rewrite `latest.json` and
`promotion_history.json`; protect the artifacts directory with file-system permissions.

`latest.json` files written before this check carry no `manifest_sha256`. They remain
loadable when `promotion_history.json` recorded the same manifest hash at promotion time;
otherwise run `promote --artifact <run-id>` again to bind the pointer.
