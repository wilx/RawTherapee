# X-Trans TGMR hard-case retraining report

## Purpose and experiment boundary

The production-v1 TGMR candidate improves the licensed 500-source diagnostic
population by 1.747301 dB over Markesteijn and lowers pooled p99 error, but it
has 65 source losses greater than 0.5 dB.  Inspection of the worst source found
a digitally sharp white/black framing edge around an otherwise ordinary
photograph.  TGMR reconstructs colored ringing at that edge.  Markesteijn has
some corner and boundary artifacts too, but remains substantially more neutral
on the exact failure patch.

This follow-up tests two deliberately separate hypotheses:

1. a small, balanced population of synthetic hard cases can teach the model to
   avoid sparse chromatic ringing without erasing its ordinary-photo gains;
2. an optically filtered, area-integrated sensor forward model is a better
   approximation of real CFA measurements than direct sampling of already
   rasterized RGB photographs.

The experiment is a 2x2 factorial: direct/physical natural-patch rendering,
with/without synthetic training replacements.  The existing 500-source test is
diagnostic from this point forward because its observed failure influenced the
design.  It cannot qualify a replacement release model.

## Permanent production-v1 baseline

No file under `/home/wilx/TGPC/release/tgmr-corpus-v1` was changed.  A canonical
inventory was written outside the release tree at
`/home/wilx/TGPC/work/tgmr-hard-case-v1/baseline-preservation/release-inventory.json`.
It contains 16,789 regular files totaling 10,040,548,375 bytes, in relative-path
order, with the size and SHA-256 of every file.  The inventory itself has
SHA-256
`06eb9550e378a8075f5ca67064a08e25aa69508a6036b6379645a989c633ee73`.

The two canonical artifacts remain byte-identical:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `official-a.tgmr` | 6,073,768 | `5707fbd67d1998ed3bac646ecce967297a2022776821d62944a24dbbb8615285` |
| `official-b.tgmr` | 6,073,768 | `5707fbd67d1998ed3bac646ecce967297a2022776821d62944a24dbbb8615285` |
| either companion manifest | 77,764 | `d753847f9389d5dd07a04d10013f81ffe85c4cfda1dd5120f90eb3e6e46d79af` |

These files are the permanent production-v1 baseline, even if a later
candidate is better.  All follow-up artifacts live under
`/home/wilx/TGPC/work/tgmr-hard-case-v1` and never use `--force` against the
release directory.  The earlier research model and its tracked DSCF0771 assets
also remain distinct.

## Production-v1 DSCF0771 rendering

The current production-v1 model was run through the real RawTherapee CLI using
the established neutral TGMR PP3.  The CLI authenticated the expected model,
reported no fallback, and produced a complete 7752x5178 RGB 16-bit TIFF:

| Measurement | Value |
| --- | ---: |
| Complete CLI time | 10.45 s |
| TGMR engine time | 6.866515 s |
| Active workers | 24 |
| Maximum RSS | 2,002,024 KiB |
| TIFF bytes | 240,862,232 |
| TIFF SHA-256 | `bc61bc7f7a71b0c973de0ee8d2964f3fe13e616fe5b6b0e8feefe349ebe68cb0` |

The TIFF remains external at
`/home/wilx/TGPC/work/tgmr-hard-case-v1/baseline-production-v1/DSCF0771/DSCF0771-tgmr-production-v1.tif`.
Its RTv4_sRGB profile has SHA-256
`17aebbdf8a88c39b07eb881dcd824eb1cf9828914d5e74a7324d7a31045e8871`.

Tracked comparison assets are:

- [one-third full frame](images/xtrans-neural/DSCF0771/DSCF0771-tgmr-production-v1-full-third.png);
- [500-percent nearest-neighbour earring crop](images/xtrans-neural/DSCF0771/DSCF0771-tgmr-production-v1-earring-500.png);
- [canonical asset manifest](images/xtrans-neural/DSCF0771/tgmr-production-v1-manifest.json).

They deliberately do not replace the earlier images bound to the research
model.  The new earring result remains plausible and still suppresses much of
Markesteijn's segmented false color, although a few isolated colored pixels are
visible on the hoop.  This is the visual baseline that later candidates must
not regress.

## Implementation status

| Stage | Status |
| --- | --- |
| Preserve and inventory corpus-v1 outputs | complete |
| Render current-model DSCF0771 assets | complete |
| Forward-model and synthetic generator tooling | pending |
| 1,000-source ratio screen | pending |
| Full 4,000-source factorial | pending |
| Old-test diagnostic and RAF/control qualification | pending |
| New untouched-test freeze, if warranted | pending |

The renderer and synthetic recipes are default-off.  The previously frozen
production-v1 pack path must remain byte-identical, including its configuration
digest and held-out validation/test bytes.
