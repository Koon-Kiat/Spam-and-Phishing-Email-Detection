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
