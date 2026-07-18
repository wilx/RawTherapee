# Phase 12 PackedXTransNet experiment report

## Status

The developer-only experiment is implemented and passes its focused
`DSCF0771.RAF` visual, colour, phase-texture, runtime, and memory gates. It is
still blocked from packaging and GUI exposure by the upstream checkpoint's CC
BY-NC 4.0 license. Markesteijn remains the default and fallback.

## Identity

| Item | Reviewed value |
| --- | --- |
| Upstream revision | `9c3cc5ab841c9afd2ed0bb702468950481043d06` |
| Checkpoint | `weights/packed_5183_3208.pt` |
| Checkpoint bytes | 646,145 |
| Checkpoint SHA-256 | `1c78b888e3f885252f84c1b12f75dd0af179a62b48499c5808eeb773d1bfc161` |
| License | CC BY-NC 4.0 |
| Converted ONNX bytes | 1,673,648 |
| Converted ONNX SHA-256 | `ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c` |
| Conversion-manifest SHA-256 | `ebd978aef293d1cf35a5d15234ef185d0785ac222c9c10f6477903e74d4c338d` |
| RAW SHA-256 | `26106d7da2ba9a87caebffd4bd5e5fb0371cc8a2b6139ec724b59ef4112842c0` |

The checkpoint, ONNX, compiled MIGraphX programs, RAW and full-resolution TIFFs
are external and untracked.

## Conversion and parity

The authenticated state dictionary contains 39 dense finite CPU float32
tensors, 158,865 stored values, and 635,460 payload bytes. The deterministic
opset-18 model reproduces the architecture without importing upstream
executable code. On the seeded tile, ONNX Runtime versus the local PyTorch
reference measured maximum `8.94e-7`, RMS `1.46e-7`, and p99 `3.87e-7`.

MIGraphX FP32 versus ONNX Runtime on the rendered full RAF had maximum one
16-bit code value, RMS `1.14e-6`, p99 zero, and 99.44 percent exact samples.
Three repeated cached GPU exports were pixel-identical; TIFF container hashes
differ because RawTherapee writes run metadata.

## Boundary decision

| Margin | Tiles | Wrapper time | Difference from margin 60 |
| ---: | ---: | ---: | --- |
| 12 | 600 | 4.91 s CPU | max `0.005005`, RMS `3.12e-5`, p99 `1.37e-4` |
| 60 | 1,457 | 12.23 s CPU | reference |

Margin 12 fails the maximum and p99 limits. Production experimental execution
therefore uses margin 60 and stride 168.

## DSCF0771 quality

Crop `(3450,1750,700,500)` uses the Markesteijn-selected lowest-detail half of
globally aligned 6x6 blocks.

| Metric | PackedXTransNet | X-veon | Markesteijn |
| --- | ---: | ---: | ---: |
| Red phase RMS | 0.000540 | 0.000794 | 0.000554 |
| Green phase RMS | 0.000464 | 0.000706 | 0.000446 |
| Blue phase RMS | 0.000746 | 0.000675 | 0.000782 |

PackedXTransNet minus Markesteijn crop RGB means are
`[-0.000155, +0.000127, +0.000083]`. The common luminance delta is `1.82e-5`
and RGB-delta range is `0.000281`; both limits are `0.005`. No seam or repeating
CFA texture is visible. The earring reconstruction is at least competitive
with X-veon and avoids the Markesteijn colour segmentation that motivated the
experiment.

The PackedXTransNet earring comparison is visibly smoother than X-veon in
parts of the hair, skin and metal. This is present in the original 16-bit TIFF
data and is not introduced by the tracked crop: both 500-percent PNGs use the
same source rectangle `(3510,1930,140,160)` and nearest-neighbour scaling. Over
that source rectangle, PackedXTransNet has 7.4 percent lower first-difference
gradient RMS and 23.4 percent lower four-neighbour Laplacian RMS than X-veon.
The Laplacian reduction is approximately 28 percent in the selected hair area,
17 percent around the earring and 5 percent in the selected skin area. These
measurements establish a more conservative, smoother reconstruction, but do
not by themselves distinguish removed noise/CFA false detail from removed real
detail. The lower red and green CFA-phase RMS above suggests that at least part
of the reduction is useful suppression of periodic reconstruction texture;
PackedXTransNet's blue phase RMS is slightly higher than X-veon's. The visual
quality conclusion is therefore that PackedXTransNet is cleaner and more
restrained, while X-veon is slightly crisper and retains more local variation.

The tracked PNGs and their separate canonical manifest are under
`images/xtrans-neural/DSCF0771/`. The prior Gharbi/X-veon/Markesteijn manifest
is unchanged.

## Performance

| Backend | Measured full-RAF seconds | Median | Peak RSS |
| --- | --- | ---: | ---: |
| ONNX Runtime CPU FP32 | one run: 15.64 | 15.64 | 1,847,576 KiB |
| MIGraphX FP32, cached | 6.73, 7.06, 7.33 | 7.06 | 2,338,328 KiB |
| MIGraphX FP16, cached | 6.37, 6.29, 6.23 | 6.29 | 2,319,192 KiB |

The authenticated FP32 compiled program is approximately 13 MiB. Cold FP32
compile took 11.91 seconds; cache load took about 0.33 seconds. Global VRAM
sampling rose by approximately 308 MiB. ROCm listed the process but reported
its VRAM field as `UNKNOWN`, so exact attribution is unavailable.

FP16 is 10.9 percent faster than FP32 by median full export. Its normalized
rendered difference from CPU is maximum `0.01431`, RMS `0.000338`, p99
`0.001175`; it remains visually indistinguishable on the reviewed earring crop
and passes the same colour and phase gates. It remains opt-in, while FP32 is
the default and numerical reference.

## Analytical limitations and remaining matrix

The 96x96 analytical suite completed without fallback. PackedXTransNet is very
strong on constant and black fields, but it performs poorly on the synthetic
saturated-edge case (19.51 dB CPSNR versus Markesteijn 62.42 dB) and trails on
the impulse case (42.28 versus 43.96 dB). These adversarial results prevent a
broad claim that it supersedes Markesteijn or X-veon.

The upstream author's personal training/validation corpus, the ten external
Gharbi Caffe test scenes, and public generation I-IV RAF samples were not
available locally. No substitute numbers are reported. The current real-RAF
finding applies to the reviewed 40 MP Fujifilm X-T50 image only.

## Decision

The hidden experiment passes the explicitly measured `DSCF0771` quality,
runtime, RSS, and bounded-VRAM criteria. FP32 and FP16 MIGraphX remain optional
developer backends. The method stays absent from the GUI and distributions
because the non-commercial checkpoint license is incompatible with normal
RawTherapee packaging and because the wider camera/content matrix is incomplete.
