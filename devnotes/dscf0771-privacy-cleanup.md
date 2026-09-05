# DSCF0771 privacy cleanup

## Publication policy

Only reviewed earring crops and crop-only comparison montages may be shared.
The RAF, TIFFs, reduced complete portraits, and full-frame montages remain
private benchmark evidence outside Git. Numerical results, source hashes,
and the existing runtime/corpus/model artifacts are unchanged.

Both the research and clean TGMR feature histories were sanitized in an
isolated repository, from a verified private recovery bundle. Removal uses
historical blob identities as well as paths, including the renamed identical
Markesteijn comparison. This is not merely a later deletion commit.

The publication update targets only the sanitized equivalent of the already
published research tip plus this privacy correction. It does not publish the
newer local research commits, the clean feature branch, or its bundled model.
The separate model redistribution gate remains in force.

## Verification

- Twenty distinct prohibited PNG payloads were removed from 21 historical
  paths, including two full-frame comparison montages.
- The history projection was checked commit-by-commit: only prohibited image
  entries changed. Code, training inputs, model data, crop bytes, and the paper
  were not changed by the rewrite.
- Public comparison generators now emit only the established earring crops.
  Crop manifests retain the original hashes and no longer advertise portraits.
- A standard-library privacy guard checks tracked files, exact crop identities,
  known prohibited blobs throughout reachable history, and embedded base64 copies.
  A regression test proves that a deletion commit alone cannot pass that audit.
- Changed opaque papers/reports require explicit image review. The TGMR paper's
  HTML embeds charts only; its six PDF raster figures match those chart shapes.
  The PDF, HTML, figure hashes, and paper-claim verification remain unchanged.
- Crop-writer and comparison tests, offline corpus/release tests on the clean
  branch, and whitespace validation are run separately from history inspection.
  No native rebuild is needed: runtime, trainer, and model contents are unchanged.

Run `python3 tools/privacy/check_dscf0771.py --history` from the repository root.
The check is also registered in the DSCF0771 privacy GitHub Actions workflow.
It is an identity/publication guard, not an automatic face detector.

## Exposure that a branch rewrite cannot erase

At the pre-publication audit, the user's GitHub fork had no releases, no PRs,
and no listed downstream forks; no upstream PR referenced this research branch.
Other public heads/tags were checked for the prohibited histories. None is
silently rewritten or deleted as part of this two-branch operation.

Private recovery bundles, local reflogs, and Codex recovery refs may intentionally
retain original bytes. They must never be uploaded, merged back, or mirror-pushed.
GitHub may also retain old commit views or shared fork-network objects even after
the rewritten branch is verified. A support-request draft is retained privately;
GitHub-side removal and deletion from third-party clones cannot be guaranteed.

Fresh-clone verification and remote head identities are recorded in the private
execution receipt. Do not describe historical cache removal as complete merely
because the current public branch passes the audit.
