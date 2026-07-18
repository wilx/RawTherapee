# Phase 13 Apache TVM Vulkan report

## Decision

The portable Vulkan mechanism works.  The same weight-bearing Linux x86-64
module files execute through AMD RADV and Mesa Lavapipe without Python, ONNX
Runtime, MIGraphX, ROCm, or the TVM compiler at runtime.

The two model conclusions differ:

- **PackedXTransNet passes the local technical, numerical, image, speed, and
  memory gates.** It remains hidden and external because its CC BY-NC 4.0
  license is incompatible with normal RawTherapee distribution and commercial
  use.
- **X-veon fails the strict portable end-to-end speed gate.** Its measured
  median is 23.27 seconds against the 20.91-second limit.  It still passes the
  39.4-second Markesteijn-relative limit, numerical checks, portability checks,
  and image comparison.  The absent upstream license is an additional blocker.

No Phase 13 result authorizes GUI exposure or model packaging.

## Reviewed inputs

| Input | Size | SHA-256 / revision | License |
| --- | ---: | --- | --- |
| Apache TVM `apache-tvm-src-v0.25.0.tar.gz` | 80,764,445 | `ea7c3248e2a8ca91969fa487247ed94db22f1dbfadf7b9408fede76c8f16e56d`; commit `c7ba0735a4f346c67b761e1fde38a68a60be8adb` | Apache-2.0 |
| TVM-FFI source | — | `59da4c0b82af0d499dae34bd89ef010f64d3ff45` | Apache-2.0 |
| X-veon `xtrans.onnx` | 15,536,134 | `45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500`; revision `2e6b96c63559aa3909b0c7c1bc45dfd4b5dfe680` | no license at reviewed revision |
| PackedXTransNet converted ONNX | 1,673,648 | `ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c`; revision `9c3cc5ab841c9afd2ed0bb702468950481043d06` | CC BY-NC 4.0 |

The compiler used CMake 3.28, Ninja 1.11, LLVM 18.1.3, Python 3.12, Vulkan
headers/runtime, SPIR-V Tools, and the Python lock at
`tools/neural_demosaic/requirements-tvm-phase13.lock`.  The ordinary build
enables only LLVM and Vulkan in TVM; CUDA, ROCm, OpenCL, RPC, and unrelated
contrib runtimes are disabled.

The GitHub `v0.25.0` maintenance branch is not a safe release selector.  Both
clean builds used the authenticated official archive rather than the moving
branch name.

## Portable artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `xveon-tvm-vulkan-linux-x86_64.so` | 32,594,896 | `8648e3741a98345c8bc76b9e1c853a3b4ef58155b65ed726f9c2fda0226c206d` |
| X-veon canonical manifest | 19,273 | `ddd39d1c3e68fe01f24ead4955ee61f2aedbb9350b82ce2c4bdc75ec5f556baf` |
| `packedxtransnet-tvm-vulkan-linux-x86_64.so` | 3,097,488 | `2975665b5f36ffb29e9b0c9dec62f69ffc916605189f41ae11484e19d18cfc6e` |
| PackedXTransNet canonical manifest | 11,027 | `b84eece056d97ef0218bf0dcf705027709f4564f01db7d9a3fcbcb78a9d7842f` |

Two separately configured clean TVM builds produced byte-identical modules and
manifests in different directories.  The X-veon artifact contains 43 audited
SPIR-V units; PackedXTransNet contains 22.  Workgroups are either `128×1×1` or
`8×8×1`; no shader uses more than the four permitted storage bindings.  Every
unit assembles as SPIR-V 1.3 and validates for Vulkan 1.1.

X-veon's 46 FP16 initializers and two FP16 cast targets are promoted to FP32
after source authentication to satisfy the portable float32/int32 profile.
PackedXTransNet requires no source-graph promotion.

The tested runtime bundle was:

| Runtime library | Size | SHA-256 |
| --- | ---: | --- |
| `libtvm_ffi.so` | 2,097,360 | `d1a947a02bc8565b7929f1da661b83403d4264805add37aad5a78c75f9c69d27` |
| `libtvm_runtime.so` | 2,940,872 | `69697fc18fafd7eaee70e0bda72b2e0f6a446e25fc933257406ebbb4f2c054de` |
| `libtvm_runtime_vulkan.so` | 467,752 | `057c9ab1b86f540f25b574f0f5b1ef6d099f78fbfb1722efc28237911fc72a14` |

Those hashes identify this tested bundle, not a general claim that TVM runtime
libraries built in arbitrary absolute directories are reproducible.  The
weight-bearing model modules and canonical manifests are reproducible.

## Numerical parity

Native CTest generated deterministic 288×288 inputs and compared portable TVM
output with ONNX Runtime 1.27.0 float32 output.  The acceptance bounds were
maximum `0.005`, RMS `0.0005`, and p99 `0.001`; the tighter diagnostic was
`5e-6 + 1e-5 × abs(reference)`.

| Model / Vulkan driver | Maximum | RMS | p99 | Tight failures |
| --- | ---: | ---: | ---: | ---: |
| X-veon / RADV | 0.000243962 | 0.000106623 | 0.000239134 | 222,119 |
| X-veon / Lavapipe | 0.000243962 | 0.000106623 | 0.000239134 | 222,115 |
| PackedXTransNet / RADV | 0.000000715 | 0.000000120 | 0.000000358 | 0 |
| PackedXTransNet / Lavapipe | 0.000000715 | 0.000000129 | 0.000000358 | 0 |

X-veon's consistent offset is expected from converting its FP16 hidden graph to
portable FP32 computation.  It passes the aggregate gate with substantial
margin.  Direct RADV-versus-Lavapipe comparisons were much tighter: maximum
approximately `5.37e-7` for X-veon and `9.24e-7` for PackedXTransNet.

## Full-image comparison

The established `DSCF0771.RAF` profiles, tiling, CFA mapping, margins, and
rendering path were reused.  No new comparison PNG was committed because no
reviewed visible difference from the existing ONNX/MIGraphX references was
found.

Against the prior ONNX Runtime CPU TIFFs on crop `(3450,1750,700,500)`:

| Model | CPSNR | SSIM | Normalized RMS | p99 | Observed-sample RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| X-veon TVM vs X-veon CPU | 87.1871 dB | 0.999999366 | 0.00004372 | 0.00015259 | 0.00003456 |
| Packed TVM vs Packed CPU | 89.5843 dB | 0.999999895 | 0.00003317 | 0.00010681 | 0.00002395 |

Both comparisons have negligible mean-channel shifts and slightly lower, not
higher, 3×3 phase RMS in each channel.  The existing X-veon earring quality and
the smoother PackedXTransNet rendition are preserved.  No repeating texture,
orientation error, seam, non-finite value, or new metallic-earring regression
was observed.

## Timing and memory

Host: AMD CPU and Radeon RX 7800 XT (`RADV NAVI32`, Mesa driver 25.2.8).  Each
median uses one warm-up followed by three complete 40 MP TIFF exports.

| Method | Wrapper median | Full-export median | Peak RSS range | Gate |
| --- | ---: | ---: | ---: | --- |
| X-veon portable TVM | 19.695 s | 23.27 s | 2,143,184–2,157,064 KiB | **fail** 20.91 s strict limit; pass 39.4 s limit |
| Packed portable TVM | 8.025 s | 11.90 s | 2,032,860–2,046,360 KiB | **pass** 21.18 s and 39.4 s limits |
| Markesteijn baseline | — | 3.94 s | 1,803,460 KiB | reference |

Repeated measured exports were pixel-identical within each TVM method.  The
largest host-RSS additions over Markesteijn were about 345 MiB for X-veon and
237 MiB for PackedXTransNet, far below 4 GiB.  They are consistent with the
reported working-buffer estimates: 197,799,184 bytes for X-veon and 5,751,696
bytes for PackedXTransNet, plus RawTherapee and runtime allocation overhead.

A separate VRAM-sampling export polled the global ROCm counter roughly every
0.1 second.  X-veon increased observed VRAM use by 211,726,336 bytes
(201.92 MiB); PackedXTransNet increased it by 136,876,032 bytes (130.54 MiB).
The corresponding one-off full times were 21.91 and 9.18 seconds.  These are
bounded-memory confirmation runs, not replacements for the medians above.

## Portability and dependency audit

The same module hashes passed:

- AMD RADV on the RX 7800 XT;
- Mesa Lavapipe selected explicitly through `lvp_icd.json`;
- tiny/odd images, all CFA mappings, tile seams, repeated inference, and
  ONNX-reference tile parity in native CTest.

This proves software/physical-driver portability locally, but not physical
cross-vendor portability.  Intel or NVIDIA validation remains required before
making that stronger claim.

An installed staging tree produced by a fresh TVM-only Release build contains
RawTherapee, the three TVM runtime libraries, Apache LICENSE/NOTICE, and the
runtime identity.  RawTherapee and the two TVM libraries that need one have
only `$ORIGIN` RUNPATH.  With
`PYTHONPATH`, `LD_LIBRARY_PATH`, `TVM_LIBRARY_PATH`, ONNX paths, and compiler
paths absent, the executable loads.  `ldd` and `readelf` show no Python, LLVM,
`libtvm_compiler`, ONNX Runtime, ROCm, or MIGraphX dependency.  The generated
modules depend only on the TVM runtime/FFI and ordinary C/C++ libraries.

## Tests

- Python neural tooling: **201 passed, 7 skipped** with the reviewed RTNN,
  Packed checkpoint, and both ONNX inputs.
- Native Release CTest with all available reviewed artifacts and Lavapipe:
  **21 passed, 2 skipped**; only MIGraphX cases skipped in this build.
- Native TVM parity on RADV: **2 passed**.
- Native TVM parity on Lavapipe: **2 passed**.
- Native no-artifact suite: **13 passed, 10 expected skips**.
- Clang 18 combined ASan/UBSan build: succeeded.  The six Phase 13 loader,
  mock, and real-Lavapipe tests passed with LeakSanitizer enabled.  The slower
  raw-wrapper contract also passed in 122.24 seconds under its expanded
  300-second timeout.  GCC 13 itself ICEs while sanitizing unrelated
  `dcraw.cc`, so Clang was used for the sanitizer build.
- Default-off Release CLI build: verified separately; no TVM target or runtime
  dependency is present when `WITH_TVM_VULKAN=OFF`.
- `git diff --check`: clean.

## Open diagnostic work

The proposed AMD-device MetaSchedule comparison (`from_device=0`, 20,000
trials, 64 trials per task/iteration, fixed seed) was not promoted into the
runtime binding or used to claim a result.  It is not portable, cannot rescue a
portable-artifact gate failure, and is unnecessary to establish the result
above.  If run later, its database, target, module, and timing must be reported
separately and its module must remain untracked.

No checkpoint, ONNX file, TVM module, conversion manifest, RAF, TIFF, or tuning
database is tracked by this phase.

## Vulkan 1.2 / SPIR-V 1.5 diagnostic

The follow-up diagnostic changed only the Vulkan API and SPIR-V target
versions.  Float32/int32 precision, fusion depth one, four storage-buffer
descriptors, 128-thread limits, disabled subgroup/FP16/vendor features, model
graphs, host target, TVM release, and runtime remained unchanged.  The default
converter still emits the byte-identical reviewed Vulkan 1.1 modules and
manifests listed above.

### TVM 0.25.0 conformance correction

Unmodified TVM 0.25.0 could not emit a conforming SPIR-V 1.5 diagnostic.  Its
SPIR-V builder always wrote a 1.0 header and listed only built-in variables in
`OpEntryPoint`.  Reassembling the first shader as SPIR-V 1.5 and validating it
for Vulkan 1.2 correctly failed because used storage buffers were absent from
the entry-point interface.

The diagnostic compiler therefore applies the tracked Apache-2.0 patch
`tools/neural_demosaic/patches/apache-tvm-0.25.0-vulkan12-spirv15.patch`,
SHA-256
`6148e1cd97347d19dc566f46d631186f7e20f2fa2e31bc8504de6a44e6c5568d`.
It propagates the requested SPIR-V version into the binary header and includes
used module-scope variables in the SPIR-V 1.4-or-later entry-point interface.
The converter authenticates both the tracked patch and an explicit build
marker before accepting a diagnostic compiler.  Targets below SPIR-V 1.4 keep
TVM's historical emission unchanged, which was confirmed by regenerating both
Vulkan 1.1 artifacts byte for byte.

### Diagnostic artifacts and shaders

Two independently patched and configured clean compiler builds produced
byte-identical modules and manifests:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| X-veon Vulkan 1.2 module | 32,594,896 | `43a1af9a0171844befe6e6d7714049abeb80cb6c3820e3503ef0186408a033d9` |
| X-veon Vulkan 1.2 manifest | 19,457 | `c27bff81782e09736e96353a0c7176e88501237ce6bb45c743a62d0a1b1eb1b7` |
| PackedXTransNet Vulkan 1.2 module | 3,097,488 | `aa618b4d2b0cdcd4f7d054e0d6d778e5afa1cfda0fa5ee9786fe8c5e1d79bbd1` |
| PackedXTransNet Vulkan 1.2 manifest | 11,211 | `f87bffed467da589e7155198ba10d408ceaa144039ef5382d25142da0bb51e71` |

X-veon still has 43 kernels and PackedXTransNet 22.  Workgroups remain
`128×1×1` or `8×8×1`; the only capability remains `Shader`; descriptor and
scalar-type limits are unchanged.  The externally assembled shader payload
totals rose from 1,177,460 to 1,178,076 bytes for X-veon and from 221,436 to
221,700 bytes for PackedXTransNet because the entry-point interface operands
are now explicit.  After normalizing the SPIR-V version line and those required
interface operands, all 65 Vulkan 1.1/1.2 shader disassemblies are byte
identical.  Every diagnostic shader validates with `spirv-val --target-env
vulkan1.2`.

### Numerical and full-image comparison

Native repeated-inference tests passed on both RADV and Lavapipe.  Against ONNX
Runtime FP32, Vulkan 1.2 reproduced the Vulkan 1.1 diagnostic statistics:

| Model / driver | Maximum | RMS | p99 | Tight failures |
| --- | ---: | ---: | ---: | ---: |
| X-veon / RADV | 0.000243962 | 0.000106623 | 0.000239134 | 222,119 |
| X-veon / Lavapipe | 0.000243962 | 0.000106623 | 0.000239134 | 222,115 |
| PackedXTransNet / RADV | 0.000000715 | 0.000000120 | 0.000000358 | 0 |
| PackedXTransNet / Lavapipe | 0.000000715 | 0.000000129 | 0.000000358 | 0 |

The `DSCF0771.RAF` comparison used one warm-up per version followed by three
alternating measured exports.  All successful logs contained the authenticated
completion diagnostic and no fallback marker.  Raw TIFF file hashes vary with
export metadata, so the stable output identity is the SHA-256 of contiguous
16-bit RGB pixels:

| Model | Vulkan 1.1 and 1.2 pixel SHA-256 |
| --- | --- |
| X-veon | `c49834e0e324b7ba85d90be3fa79d22c03074a8dd2d3eecf8d38c0d6e9cff8b0` |
| PackedXTransNet | `64ebc970620b14661ca99a54861d2f09b508c696c43a3eac9048a89d7a75bf19` |

ImageMagick reported zero differing pixels between repeated exports and between
Vulkan 1.1 and 1.2 for both models.  The complete images, seams, CFA phases,
and established full-frame and earring regions are therefore exactly
unchanged; no new comparison image is warranted.

### Timing, memory, and decision

Timings are medians of three alternating measurements; MAD is the median
absolute deviation.  Wrapper time comes from RawTherapee's completion
diagnostic and full time from `/usr/bin/time -v`.

| Model / target | Wrapper median ± MAD | Full median ± MAD | Peak RSS range |
| --- | ---: | ---: | ---: |
| X-veon Vulkan 1.1 | 18.341750 ± 0.052770 s | 21.40 ± 0.19 s | 2,158,824–2,159,616 KiB |
| X-veon Vulkan 1.2 | 18.639822 ± 0.008700 s | 21.66 ± 0.02 s | 2,145,724–2,160,324 KiB |
| Packed Vulkan 1.1 | 5.956678 ± 0.012410 s | 8.89 ± 0.05 s | 2,047,728–2,048,128 KiB |
| Packed Vulkan 1.2 | 6.082817 ± 0.004469 s | 9.11 ± 0.02 s | 2,049,028–2,049,204 KiB |

Vulkan 1.2 was 1.21% slower end to end for X-veon and 2.47% slower for
PackedXTransNet.  It does not satisfy the required 5% improvement, and X-veon
also remains above its 20.91-second gate.  RSS and shader allocation geometry
did not materially change, so an additional VRAM sampling run was unnecessary.

**Decision: reject the Vulkan 1.2 artifacts and retain Vulkan 1.1 exclusively.**
There is no basis for a dual-artifact runtime selector.  The diagnostic modules,
manifests, TIFFs, timings, disassemblies, and temporary runtime build remain
external and untracked; the production runtime continues accepting only the
reviewed Vulkan 1.1 sizes and hashes.

Final verification comprised 211 passing Python tests with two expected skips;
the complete production native suite with reviewed artifacts passed 21 tests
with only the two disabled MIGraphX cases skipped; and the diagnostic RADV and
Lavapipe parity cases both passed for both models.  The default-off CLI build
contains neither a TVM target nor a TVM library dependency.  The strict GCC 13
source build again stops only at the pre-existing `dcraw.cc`
`-Werror=stringop-overflow` diagnostic recorded as outside Phase 6 and Phase 13.
`git diff --check` is clean.
