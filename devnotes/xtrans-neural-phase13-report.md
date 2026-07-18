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
