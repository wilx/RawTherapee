# Personal integration branch

This build combines the nine open RawTherapee PRs by wilx on upstream `dev`
at `64c2d90a3248c7ad41842d9162368858b61aac6a`.

| PR | Feature or fix | Source branch | Integrated head |
| --- | --- | --- | --- |
| [#7726](https://github.com/RawTherapee/RawTherapee/pull/7726) | Add .cube LUT support to Film Simulation | `cube-luts` | `341debd3b` |
| [#7765](https://github.com/RawTherapee/RawTherapee/pull/7765) | Add experimental TGMR demosaicing for X-Trans sensors | `codex/xtrans-tgmr-productization` | `a54232f2e` |
| [#7772](https://github.com/RawTherapee/RawTherapee/pull/7772) | Add HEIF/HEIC import and export with metadata support | `dev-heif` | `6d5b4402c` |
| [#7787](https://github.com/RawTherapee/RawTherapee/pull/7787) | Fix Hue-Chroma vectorscope in working profile mode | `issue-7699` | `5a4f2b957` |
| [#7790](https://github.com/RawTherapee/RawTherapee/pull/7790) | Reduce zoom-dependent DRC brightness differences | `issue-7568` | `956eaf0cd` |
| [#7791](https://github.com/RawTherapee/RawTherapee/pull/7791) | Use Olympus lens metadata handling for OM cameras | `issue-7557` | `5124b1996` |
| [#7792](https://github.com/RawTherapee/RawTherapee/pull/7792) | Fix white balance controls rolling back after edits | `issue-7515` | `882bf9f7d` |
| [#7793](https://github.com/RawTherapee/RawTherapee/pull/7793) | Fix linear DNG active-area cropping | `issue-7320` | `cd5b06acf` |
| [#7800](https://github.com/RawTherapee/RawTherapee/pull/7800) | Use macOS 15 for Apple Silicon CI | `ci/macos-15-arm64` | `241d2cb7f` |

## Build and use

Install the normal RawTherapee build dependencies and libheif >= 1.23.6 with
the libde265 decoder and x265 encoder. Set `PKG_CONFIG_PATH` to the directory
containing `libheif.pc` when using a local libheif installation. Explicitly
requiring HEIF avoids silently omitting it when the dependency is unavailable.

```sh
cmake --preset dev -B build/integration \
    -DWITH_HEIF=ON -DWITH_HEIF_EXPORT_TESTS=ON -DBUILD_TGMR_TRAINER=ON \
    -DWITH_WB_FEEDBACK_TESTS=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
nice -n 10 cmake --build build/integration -j4
OMP_NUM_THREADS=4 nice -n 10 xvfb-run -a ctest --test-dir build/integration --output-on-failure
python3 -m unittest discover -s tools/tests -p test_bake_cube_lut.py
cmake --install build/integration
```

Use the installed application so that its data files, including the TGMR
model, are available. Select **Student-t GMR (experimental)** in the X-Trans
demosaicing tool; Markesteijn remains the default. Select a `.cube` file in
Film Simulation, and use the HEIF output presets when saving or queueing an
export. See [the LUT baker guide](tools/bake_cube_lut-README.md) for technical
LUTs requiring transfer-function or gamut adaptation.

## Updating

Start from current upstream `dev`, then merge each open PR head into a
separate integration merge commit. Build and verify the new branch before
publishing it to the fork.
