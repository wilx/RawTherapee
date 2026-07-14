# Using published X-Trans neural demosaicers in RawTherapee

## Conclusion

The published X-Trans checkpoints from Gharbi et al. and
Kokkinos--Lefkimmiatis can be used for a meaningful experimental RawTherapee
demosaicer. Retraining is **not** a technical prerequisite: both models are
fully convolutional, include an X-Trans model, and use the same canonical
6 x 6 CFA pattern RawTherapee already recognises.

They cannot be dropped directly into the C++ engine. Their .pth files are
PyTorch state dictionaries, not self-contained models, and RawTherapee has no
PyTorch, ONNX Runtime, OpenVINO, ncnn, TensorFlow, or OpenCV-DNN dependency.
The practical route is to use a checkpoint only during development, convert it
to a small checked binary model format, and implement the fixed inference graph
in C++.

Start with Gharbi's X-Trans network. It is simple enough to reproduce exactly,
has a 12-pixel receptive-field radius, and can be tiled safely. The
Kokkinos--Lefkimmiatis model is a valuable later comparison, but its twenty
unrolled iterations imply a much larger effective receptive field and around an
order of magnitude more work. It needs a separate GPU/runtime or tiling design
before it is plausible as an interactive RawTherapee method.

The critical caveat is image domain. Both models were trained on synthetic
mosaics, not RawTherapee's linear, black-subtracted camera samples. Gharbi's
pretrained X-Trans model in particular appears to expect an 8-bit,
gamma/sRGB-like input domain. Its real-RAF quality must be measured before
considering it useful. This calls for testing direct-linear and gamma-wrapped
input contracts; it does not require retraining before the initial experiment.

## Sources and artifacts inspected

| Project | Revision inspected | X-Trans artifact | License evidence | Initial role |
| --- | --- | --- | --- | --- |
| [mgharbi/demosaicnet](https://github.com/mgharbi/demosaicnet) | 959e9d1630976b421d5af5e35b2e2a01f5630e5c (2023-06-22) | demosaicnet/data/xtrans.pth | Repository LICENSE: MIT | Preferred first implementation |
| [mgharbi/demosaicnet_caffe](https://github.com/mgharbi/demosaicnet_caffe) | ba4e77c50e46619b85cb871d740c6a39f12c0f25 (2020-12-13) | pretrained_models/xtrans/weights.caffemodel | MIT notices in source/deploy files; no top-level LICENSE found | Reference/specification only |
| [cig-skoltech/deep_demosaick](https://github.com/cig-skoltech/deep_demosaick) | 3ba7b86837b34b20588a0de956cdff21b711e5c2 (2018-09-06) | pretrained_models/xtrans/model_best.pth | Repository LICENSE: MIT | Defer; possible later comparison |

Pinning the exact artifacts matters: a model filename is not a reproducible
model version.

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| Gharbi xtrans.pth | 1,644,547 bytes | 3759a13296fecebae83a36a8f0c3890d8a2f7d33e5f8149ed9b76a70f8285bc7 |
| Deep Demosaick model_best.pth | 1,530,208 bytes | 98608c712c7f6414d9c453828cf45a093d8c761fc2ac2d0450841eaf2224151d |
| Gharbi Caffe weights.caffemodel | 1,642,942 bytes | a1c2d98d12237f730cfd4e21677c035dd0d3796547abb5009c6684e54eff1043 |

Relevant publications:

* Michaël Gharbi, Gaurav Chaurasia, Sylvain Paris, and Frédo Durand, “Deep
  Joint Demosaicking and Denoising,” *ACM Transactions on Graphics*, 2016.
  See the [project page](https://groups.csail.mit.edu/graphics/demosaicnet/).
* Michail Kokkinos and Stamatios Lefkimmiatis, “Deep Image Demosaicking using
  a Cascade of Convolutional Residual Denoising Networks,” ECCV 2018,
  [arXiv:1807.06403](https://arxiv.org/abs/1807.06403).

## What the checkpoints are

### Gharbi / DemosaicNet

The file xtrans.pth is a legacy PyTorch pickle serialization containing an
OrderedDict of CPU float tensors. It is a **state dictionary**, not TorchScript
and not a general C++ model file. The architecture must be supplied separately.

The X-Trans model in
[demosaicnet/modules.py](https://github.com/mgharbi/demosaicnet/blob/master/demosaicnet/modules.py)
is XTransDemosaick(depth=11, width=64, pad=false):

1. A 3 x 3 convolution maps three sparse colour planes to 64 channels,
   followed by ReLU.
2. Ten further 3 x 3, 64-to-64 convolution-plus-ReLU layers follow.
3. The original sparse three-channel mosaic is cropped to the valid interior
   and concatenated with the 64 learned feature maps (67 channels total).
4. A 3 x 3, 67-to-64 convolution plus ReLU is applied.
5. A 1 x 1, 64-to-3 convolution produces RGB.

There are approximately 409,923 trainable parameters including biases. Valid
convolutions shrink the output by 24 pixels overall, so the required border is
12 pixels on each side. That is a well-defined tiling halo.

The input is three sparse RGB planes: the observed channel is nonzero at each
CFA site and the other two channels are zero. The demonstration code takes an
8-bit image and divides it by 255 before mosaicking. That is evidence that the
published X-Trans model targets a gamma/sRGB-image-domain synthetic mosaic,
rather than native linear raw input.

The current PyTorch repository does not implement the paper's noise-aware
configuration. Its X-Trans checkpoint is noiseless. The older Caffe package
has a noise-aware **Bayer** configuration, not a noise-aware X-Trans one.

The Caffe command-line interface defaults to sRGB processing. Its
linear_input option applies approximate pow(x, 1/2.2) before the network and
pow(x, 2.2) after it. This makes a gamma wrapper a concrete, reference-backed
experiment rather than an arbitrary processing choice. It also uses
valid-network cropping and preserves the period-six CFA phase when tiling.

### Kokkinos--Lefkimmiatis / Deep Demosaick

The file model_best.pth is also a legacy pickle state dictionary. Its tensors
retain CUDA storage tags (cuda:0), so offline inspection/conversion must load
it with map_location=cpu. It must not be runtime-loaded as a pickle.

The X-Trans arguments specify a noiseless model with depth=5 and max_iter=20.
The code in
[main_xtrans.py](https://github.com/cig-skoltech/deep_demosaick/blob/master/main_xtrans.py)
is marked “CURRENTLY DEPRECATED”; the README also says that its real-Fujifilm
application path has unresolved orientation/rotation/CFA issues.

Its denoiser has a 5 x 5 input convolution from 3 to 64 channels, five residual
blocks (each has two 3 x 3 64-channel convolutions with PReLU, reflection
padding, and weight normalization), and a 5 x 5 transposed output convolution
to RGB. An MMNet wrapper repeats the learned denoiser and its data-consistency
update twenty times, with learned alpha and momentum schedules.

The training path used synthetic X-Trans mosaics from
Dataset_LINEAR_without_noise/xtrans_panasonic, with 16-bit mosaics scaled to
0..255 and 8-bit linear-ish targets. It is closer to RawTherapee's linear
domain than Gharbi, but still is not trained on Fujifilm RAF samples,
RawTherapee sensor normalization, or realistic X-Trans noise.

## CFA compatibility and coordinates

The canonical 6 x 6 X-Trans mask used by Gharbi is:

    G B G G R G
    R G R B G B
    G B G G R G
    G R G G B G
    B G B R G R
    G R G G B G

It matches the CANONICAL_XTRANS pattern already used by the experimental
Rafinazari work. The published weights are structurally applicable to supported
Fujifilm X-Trans layouts without re-training.

The mask alone is insufficient. A decoded RAF can expose a translated, rotated,
or reflected orientation. The neural input must be transformed to canonical
phase, evaluated, and transformed back. Reuse or generalise the existing
Rafinazari findTransform logic and test every supported translation, rotation,
and reflection.

Tile origin is part of the model input because the CFA is period 6 x 6, not
2 x 2. Tiles must retain global CFA phase; using phase-aware origins or
period-six multiples prevents colour-periodic tile seams.

## Raw-domain compatibility

The raw buffer reaching demosaicing is linear camera-space data after the raw
decoder's black/white normalization, but before later white balance and colour
management. That is not the input domain demonstrated by Gharbi.

Evaluate two explicitly defined paths:

1. **Direct linear:** normalize samples to the model range (normally 0..1),
   scatter them into sparse RGB planes, and run the network.
2. **Gamma-wrapped:** apply pow(max(x, 0), 1/2.2) before inference and
   pow(max(y, 0), 2.2) after it, matching Caffe's linear_input convention.

Both paths must define handling of negative values, values above one, and
clipping. Do not silently clamp highlights before comparison: saturated channels
and highlight rolloff are likely failure sites for a domain-mismatched model.
Always reject or repair non-finite output before it re-enters the normal raw
pipeline.

Neither X-Trans checkpoint is noise-aware. Real RAF files contain shot/read
noise, clipping, sensor response differences, and camera-generation variation
absent from their synthetic training inputs. This is the main quality risk, not
the CFA geometry.

Gharbi's network does not explicitly reinject observed samples after its final
layer. That may be intentional because it performs joint demosaicking and
denoising. The first implementation should reproduce the reference network and
measure observed-sample error. Do not add undocumented sample reinjection as a
quiet raw-safety measure: that would be a distinct variant with separate
artefacts and benchmarks.

## Runtime and secure model packaging

The state dictionary lacks the architecture, input/output contract, safe
file-format ABI, and inference runtime. Loading legacy PyTorch pickle content
in an image editor is also an unnecessary security risk. The RawTherapee binary
should read only a purpose-built, bounds-checked tensor format, never a Python
pickle.

The relevant PyTorch documentation is:

* [Serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html)
* [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

Use weights_only=true when converting Gharbi's model offline. Use
weights_only=true and map_location=cpu for Deep Demosaick. Verify the pinned
SHA-256 before conversion.

| Deployment choice | Assessment |
| --- | --- |
| Link libtorch and load state dictionaries | Technically possible, but a very large dependency; still needs architecture code and retains legacy-loading concerns. Do not use. |
| Offline ONNX export plus optional ONNX Runtime | Feasible for Gharbi, but adds a sizeable dependency and needs exact cropping/export parity. The deep iterative model is harder. Possible future generic route. |
| Fixed C++ inference plus a versioned binary weight file | Straightforward for Gharbi. Needs a converter, loader, tiled/SIMD kernels. Preferred. |
| Python/PyTorch helper process | Useful for a research reference only, not an editor runtime dependency. |

The Gharbi graph needs only convolution, ReLU, crop, concatenation, and a final
convolution. A C++ implementation using aligned per-thread workspaces, OpenMP
tiles, and RawTherapee's established CPU/SIMD conventions is practical.

## Performance and memory

The Gharbi model needs approximately 16.4 trillion multiply-accumulate
operations for a 40 MP output. Depending on SIMD, threads, and cache behaviour,
that may mean tens of seconds or minutes on CPU. Its small 1.6 MB checkpoint
does not mean cheap execution, so benchmark actual RawTherapee targets before
presenting it as interactive.

It is tile-friendly:

* use at least a 12-pixel input halo;
* allocate feature maps only for a tile plus halo;
* discard each tile's invalid outer output and stitch only valid interiors;
* define outer-image boundaries (for example reflection); and
* test internal tile boundaries against an untiled reference.

A full-frame 64-channel float feature map at 40 MP is about 10 GB. Tiling is
mandatory even for Gharbi.

Deep Demosaick is much more expensive: about 378 thousand convolution MACs per
pixel per iteration, or 7.56 million per pixel over 20 iterations. At 40 MP
that is approximately 302 trillion MACs. A single iteration has a radius around
14 pixels, but repeated data-consistent iterations can propagate dependencies
to roughly 280 pixels. Correct CPU tiling would require an enormous halo or a
state-exchange design, with substantial seam and memory risk.

## Licensing and redistribution

The current Gharbi and Deep Demosaick repositories are MIT licensed, compatible
with RawTherapee's GPL distribution. Keep their copyright and license notices
with a converted model and any source translation.

The Caffe repository lacks a top-level license despite MIT headers in its source
and deploy files, so prefer the current PyTorch project for any shipping work.

The checkpoints have no separate explicit model-license statement. Before an
upstream distribution embeds or downloads converted weights, obtain author
confirmation that redistributing the checkpoint is intended under the repository
license and record training-data provenance. This is release/legal diligence,
not a technical reason to retrain for a local experiment.

## When to retrain

Do not retrain merely because cameras use different Fujifilm sensor generations
or pixel pitches. The CFA geometry is the relevant structural constraint, and
both models are fully convolutional.

Use the published weights first to answer whether a neural X-Trans method helps
on real RAF files and whether it is competitive with Markesteijn on the
structures of interest. They are valid experimental baselines.

Train a new model if controlled evaluation shows a persistent domain failure:
false colour on real sensor noise, bad highlights, unacceptable colour bias
after white balance, or poor results across several Fujifilm generations despite
correct CFA mapping and reference-parity tiling. A production neural raw model
would likely need camera-linear data, realistic noise/clipping, a specified
black/white normalization contract, and held-out real RAF validation.

## Staged recommendation

### 1. Freeze and reference the Gharbi artifact

1. Keep the pinned Gharbi checkpoint in a development-only location and verify
   its SHA-256.
2. Record upstream URL, revision, license text, checksum, converter version,
   and tensor order in generated-artifact metadata.
3. Do not put raw Python pickle data in the RawTherapee runtime path.

### 2. Establish a Python parity and domain harness

1. Build a pinned Python/PyTorch reference using the upstream architecture and
   checkpoint.
2. Generate golden output from known sparse mosaics, with optional per-layer
   diagnostics for C++ debugging.
3. Feed it sparse mosaics derived from RawTherapee RAF data and compare
   direct-linear with gamma-wrapped input.
4. Use DSCF0771.RAF, particularly the metallic earring crop that exposed the
   Rafinazari false-colour failure, as the first real-image visual test against
   Markesteijn three-pass.

### 3. Convert to a safe binary tensor file

The offline converter should output a deterministic, versioned file containing:

* magic value and format version;
* architecture identifier and fixed layer dimensions;
* tensor order, counts, and little-endian float32 data;
* source-checkpoint SHA-256 and converter version;
* checksums sufficient to reject truncated/incompatible data; and
* upstream copyright/license attribution metadata or a nearby notice file.

The C++ loader must validate lengths and dimensions before allocation. A missing
or corrupt model should safely select an existing X-Trans method, never crash
or emit partly initialized pixels.

### 4. Implement fixed, tiled C++ Gharbi inference

Add a separately selectable method such as demosaicnet-xtrans, shown as
“DemosaicNet X-Trans (experimental).” It should:

1. map the actual CFA orientation to the canonical mask;
2. normalize/scatter scalar raw values into sparse RGB input;
3. process phase-stable tiles with the required 12-pixel halo;
4. execute the upstream convolutions, crop, and concatenation exactly;
5. map canonical RGB output back to camera orientation; and
6. return only finite RGB to the normal RawTherapee pipeline.

Use OpenMP and aligned reusable workspaces, not full-frame feature buffers.
Initially add no false-colour suppression, sharpening, sample reinjection, or
other postprocessing: those would obscure whether a defect comes from the
published model, raw-domain wrapper, or integration.

### 5. Validate before judging image quality

First establish C++/Python parity. Then test:

* constants, gradients, impulses, frequency sweeps, saturated edges, flat
  black, tiny images, and maximum tile sizes;
* every accepted CFA phase, rotation, and reflection;
* boundaries and tile seams against an untiled reference;
* NaN, infinity, clipping, and observed-sample error;
* DSCF0771.RAF plus public RAF files from multiple Fujifilm generations; and
* runtime, peak memory, determinism, CPSNR/PSNR where a reference exists, SSIM,
  false colour, and moiré against Markesteijn.

Report direct-linear and gamma-wrapped Gharbi results separately. The selected
wrapper must be evidence-driven and documented.

### 6. Make the quality decision

If Gharbi is useful on real RAF files at acceptable preview/export speed, retain
it as a clearly experimental alternative and improve engineering matters such
as SIMD kernels and model packaging.

If it has systematic raw-domain artefacts after correct CFA mapping and
reference-parity validation, train an X-Trans model with the same simple
architecture but camera-linear input, realistic noise/clipping, and a specified
target pipeline. Do not attempt to compensate for a mismatched checkpoint by
tuning unrelated postprocessing.

Evaluate Deep Demosaick only after the Gharbi result and after a credible
compute plan, likely optional GPU acceleration or a carefully validated
large-halo tiled CPU design. It is a valuable research baseline, but not the
sensible first in-engine neural demosaicer.
