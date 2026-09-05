# X-Trans TGMR hard-case retraining report

> Privacy update: DSCF0771 full-frame imagery is private and is not
> distributed. Only the reviewed earring crops may be shared. Historical
> full-frame asset references below describe private benchmark evidence,
> not current publication instructions.

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

- Full-frame image withheld for privacy (private benchmark only);
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
| Forward-model and synthetic generator tooling | complete |
| Independent physical/synthetic control support | complete |
| 1,000-source ratio screen | complete -- no candidate passed |
| Full 4,000-source factorial | stopped by the screen gate |
| Old-test diagnostic and RAF/control qualification | not opened; no winner |
| New untouched-test freeze, if warranted | not warranted |

The renderer and synthetic recipes are default-off.  The previously frozen
production-v1 pack path must remain byte-identical, including its configuration
digest and held-out validation/test bytes.

## Training and control generators

The standalone C++ trainer now has two opt-in natural-patch forward models:
`direct-v1` and `sensor-physical-v1`. The latter keeps the frozen 25-percent
identity population direct and renders every other training patch through a
deterministically selected scale (1.5x, 2x, or 2.5x), Gaussian PSF sigma (0.25,
0.50, or 0.75 output pixels), and one of sixteen quarter-pixel placements.
Gaussian filtering and square sensor-pixel integration are combined in a
separable normalized kernel. Constant-color area conservation, gradient
ordering, subpixel sensitivity, and batched/isolated rendering identity are
covered by native tests. Camera transformation, exposure, white balance,
clipping, and uint16 quantization are applied only after rendering.

Synthetic replacement ratios are restricted to 0, 0.25, 0.5, 1, 2, and 5
percent and change training records only. The schedule preserves exactly
1,024,000 records and emits the exact rounded population at every ratio. It
balances eight families—steps, thin lines, intersections, dots/stars,
saturated highlights, periodic detail, procedural strokes, and frame/matte
borders—with 25 percent sharp digital cases and 75 percent oversampled,
optically filtered cases. Independent frozen seeds distinguish training from
control generation, and every X-Trans phase placement is exercised.

Held-out physical corpora can be packed explicitly without changing the
ordinary validation/test path. The new `corpus synthetic-controls` command
emits a separately authenticated TGPC, grouped by synthetic family, and the
new `validate-external` command evaluates such a control without weakening the
normal model/corpus identity check. Default packing still uses the exact
legacy configuration string and record path; tests compare the public legacy
wrapper byte-for-byte with the default option path and prove that default-off
features do not change held-out records.

Split-only packing can emit just validation, test, or training records while
retaining their full-manifest source ordinals. This keeps authenticated
physical-control artifacts compact and avoids reopening the 4,000 unrelated
training sources for a validation-only render. The selected split is bound to
the TGPC configuration identity and cannot be confused with the ordinary
three-split corpus.

For tractable screening, a synthetic-only fast path accepts an authenticated
no-synthetic TGPC having the exact same manifest and forward-model
configuration. It reuses every natural record and replaces only the frozen
synthetic schedule. A native test requires this path to produce the exact same
bytes as full source decoding and repacking. Thus the direct production TGPC
can seed every direct ratio, and one newly rendered physical/no-synthetic TGPC
can seed every physical ratio.

The focused standalone build and both native/Python CTest cases pass. A full
default-option production pack was also regenerated after the implementation
change. It is byte-identical to the frozen `production-a.tgpc`: both are
442,368,256 bytes with SHA-256
`acf8483c21e4d8b6f01679e92663d4345fd13244beb853e94a2b799827fd3c35`.
This verifies the default-off compatibility promise using the complete corpus,
not only a small fixture. All six physical-ratio corpora retain the ordinary
validation payload digest
`76e1ef73ca5f7ac590f97c0846b13300bb070630a2cce81e4e79b11bef5babe9`
and test payload digest
`658576612d7bf5e07e550a57f2dd153adadf1568e79374d4ac3eab0f8346cc12`.

The frozen control generator was exercised with 512 cases per family (4,096
records total). The uncompressed TGPC is 1,573,120 bytes with file SHA-256
`8b4bc67ffff42a94c599c0373d4c9adb291d471c39e447cb9662346519e88d5e`
and payload SHA-256
`88bf5c1cc1d47c9393a643a814a8898fd6a5785dc511f49760c42542a17e930f`.
The permanent production-v1 baseline scores 20.107503 dB, p99 patch RMS
0.320029, worst patch RMS 0.508530, and 1.599094 dB phase spread on this
deliberately severe population. Per-family baseline PSNR is:

| Family | PSNR (dB) |
| --- | ---: |
| steps | 24.224616 |
| thin lines | 21.262420 |
| intersections | 20.142810 |
| dots/stars | 19.070781 |
| saturated highlights | 17.655576 |
| periodic detail | 19.940468 |
| procedural strokes | 19.981200 |
| frame/matte borders | 21.316265 |

These values are a before-retraining reference, not a release gate by
themselves. Candidate selection uses relative mean-MSE reduction plus the
ordinary and physical validation safety gates.

## Frozen evaluation baselines

The 1,000-source direct/no-synthetic model is the matched screen baseline. It
has model SHA-256
`6abd542542511678795c29013a22f459bc346d647d63f1e98a2b97bdee43f20c`
and the following results:

| Population | PSNR (dB) | p99 RMS | Worst RMS | Phase spread (dB) |
| --- | ---: | ---: | ---: | ---: |
| Ordinary validation | 34.021443 | 0.096168 | 0.266298 | 0.907544 |
| Physical validation | 35.295944 | 0.083258 | 0.265443 | 1.197705 |
| Synthetic controls | 20.032806 | 0.318933 | 0.483867 | 1.949071 |

The physical validation corpus contains 64,000 records. Its payload SHA-256 is
`e6004bbafe397260600b0ad609df65c31fda21f77b7521584526126f47fd5c72`.
The previously opened 500-source test remains a diagnostic baseline inside the
permanent release inventory. Its TGMR report has SHA-256
`53add2213fab02a47798db652bee756d7535cbd741d3f22f3a0912d7639c20ed`
and its TGMR/Markesteijn comparison has SHA-256
`9e2b799fa829d8cac33576a113594891caf0396a1f30037f6164474aff8a6dba`.
Neither was used to select or reject a screen candidate.

## 1,000-source ratio screen

All twelve candidates used the same K32/D51 fit: ten Gaussian iterations,
thirty Student-t iterations, all eighteen X-Trans phases, `nu=3`, covariance
floor `1e-6`, 4,096-record batches, and the optimized CPU backend. Each model,
checkpoint series, timing log, and validation report remains external under
`/home/wilx/TGPC/work/tgmr-hard-case-v1/screen-1000`.

The safety reference for ordinary and physical validation is the direct 0%
candidate above. Synthetic MSE reduction is relative to the 0% candidate using
the same natural-patch renderer. A candidate had to lose no more than 0.1 dB
ordinary PSNR, raise ordinary p99 by no more than 2%, raise ordinary worst RMS
by no more than 5%, and reduce held-out synthetic mean MSE by at least 50%.

| Renderer | Synthetic ratio | Ordinary PSNR | Ordinary p99 | Physical PSNR | Synthetic PSNR | Synthetic MSE reduction | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Direct | 0% | 34.021443 | 0.096168 | 35.295944 | 20.032806 | 0.00% | baseline |
| Direct | 0.25% | 34.061985 | 0.095963 | 35.327730 | 20.057969 | 0.58% | hard-case gate failed |
| Direct | 0.5% | 34.025036 | 0.096322 | 35.274802 | 20.092249 | 1.36% | hard-case gate failed |
| Direct | 1% | 34.045292 | 0.096302 | 35.325567 | 20.187451 | 3.50% | hard-case gate failed |
| Direct | 2% | 33.974593 | 0.096847 | 35.251104 | 20.491862 | 10.03% | hard-case gate failed |
| Direct | 5% | 33.983022 | 0.097442 | 35.273072 | **20.781363** | **15.83%** | hard-case gate failed |
| Physical | 0% | 33.839912 | 0.099246 | **35.461792** | 20.589720 | 0.00% | ordinary gate failed |
| Physical | 0.25% | 33.799284 | 0.099449 | 35.424806 | 20.603582 | 0.32% | ordinary and hard-case gates failed |
| Physical | 0.5% | 33.825099 | 0.098481 | 35.444928 | 20.606971 | 0.40% | ordinary and hard-case gates failed |
| Physical | 1% | 33.844822 | 0.099151 | 35.443570 | 20.681022 | 2.08% | ordinary and hard-case gates failed |
| Physical | 2% | 33.752116 | 0.099405 | 35.361730 | 20.762992 | 3.91% | ordinary and hard-case gates failed |
| Physical | 5% | 33.709163 | 0.100746 | 35.302377 | **20.992140** | **8.85%** | ordinary and hard-case gates failed |

The physical renderer alone reduces synthetic MSE by 12.04% relative to the
direct baseline, so its 5% candidate is 19.82% better than direct 0%. That is
still far below the 50% target, and the physical candidates lose 0.177--0.312
dB on ordinary validation. The physical renderer does improve its intended
physical controls by up to 0.166 dB, but that gain is a renderer-domain
tradeoff rather than a generally safer model.

The complete model identities are:

| Renderer/ratio | Model SHA-256 |
| --- | --- |
| direct/0 | `6abd542542511678795c29013a22f459bc346d647d63f1e98a2b97bdee43f20c` |
| direct/0.25 | `da40cf0c6a7d5f8a04e732a62165c63571bbdbd11a79712fc2f6819e653d3035` |
| direct/0.5 | `1fdc262210831384368c2bd7e5b85bf89725e820dc3be0e490e9356c0dc2229d` |
| direct/1 | `280c43d38ec6fc9311d39ce5c691b385704fb4bbcc93de1dfd80e7bdf9b29079` |
| direct/2 | `068afbc8efc4605dc987753d29c3342e005d42618cdb8bf1e8481999e68eb05e` |
| direct/5 | `56e5f6b5cb8b15855f912d22c69e4da5881ba20773c2edf1431c82e76250fbbf` |
| physical/0 | `499bef2dbd489fa83872762790b08bd0b3275407b9f9dd38985be430af450d95` |
| physical/0.25 | `6224bc159546f5db5b7cffd354e14ba2693014f67e070af7014728df9fa0bd79` |
| physical/0.5 | `9762c946bccdd1989979ecf4da717fbce8c063f740782837624a521cf7260ae0` |
| physical/1 | `acaca4e287d0ccdda135dfd45f98677fba449683a4f3bf0a6785ea9c7333adc9` |
| physical/2 | `6082be142c628ad405d19cf83aa5e19c0f94a681109e33909cde304c54848a1a` |
| physical/5 | `0ff74e3428f85ae66207817be373315f38c0f18f1699de6f699b63ebaad25256` |

The largest direct intervention does help the saturated-highlight family
substantially (17.770741 to 21.413801 dB), but improvements are small on
periodic detail and frame/matte borders, and worst synthetic RMS increases
from 0.483867 to 0.547944. Under the physical renderer, the 5% intervention
also regresses intersections, procedural strokes, and frame/matte borders
relative to physical 0%. The aggregate gain is therefore neither large enough
nor uniform enough to satisfy the hard-case requirement.

## Decision

**NO-GO at the 1,000-source screen.** No synthetic ratio under either natural
renderer passes the frozen selection gates. The direct candidates preserve
ordinary validation but recover at most 15.83% of the required synthetic MSE;
the physical candidates improve physical validation but fail ordinary-photo
safety and recover at most 8.85% relative to their matched baseline.

Consequently the 4,000-source factorial, old-test candidate comparison, RAF
qualification, and new untouched-test freeze were deliberately not run. This
is the gate behavior specified before screening and avoids spending a much
larger training budget on an already disqualified recipe. The current
production-v1 model, all current trained model files, the research model, and
their comparison images remain preserved. The result rejects these two
training-data modifications as a production-v1 replacement; it does not imply
that the existing TGMR implementation or its normal-photo output is unusable.
