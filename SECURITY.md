# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's "Report a vulnerability" button on the Security tab of the repository.
Do not open a public issue. Expect an answer within a week.

## Check a release

Each release has a checksum file, an SBOM and a PyPI attestation for each distribution file. `docs/releasing.md` shows
how to check them, and what the release workflow checks before the upload.

## Data sent to the LLM provider

EBaMl sends the text of EBM graphs, feature names and, for local explanations, feature values of the explained instance
to the configured LLM provider. Do not use it with a provider you do not trust with that data. The default provider is
OpenRouter's free router, which may route to models that log prompts.
