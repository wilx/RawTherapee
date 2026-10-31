# Personal integration branch

`wilx/RawTherapee:integration` combines the following open PRs for use in one
build. The initial integration, prepared on 2026-11-01, starts from upstream
`dev` at `b2ee504c47e5e6d5e054395aa12bf1fe43bd1c3c`.

| PR | Feature or fix | Source branch | Integrated head |
| --- | --- | --- | --- |
| [#7726](https://github.com/RawTherapee/RawTherapee/pull/7726) | `.cube` LUT support in Film Simulation | `cube-luts` | `93c38179d` |
| [#7765](https://github.com/RawTherapee/RawTherapee/pull/7765) | Experimental TGMR X-Trans demosaicing | `codex/xtrans-tgmr-productization` | `a09165418` |
| [#7772](https://github.com/RawTherapee/RawTherapee/pull/7772) | HEIF/HEIC import and export with metadata | `dev-heif` | `f38ffea0c` |
| [#7773](https://github.com/RawTherapee/RawTherapee/pull/7773) | System KLT dependency wiring | `codex/fix-system-klt` | `4c7b4f735` |
| [#7780](https://github.com/RawTherapee/RawTherapee/pull/7780) | Reject unsupported RAW layouts to avoid a crash | `issue-7771` | `0356f36bd` |

The base already includes the thumbnail luminance-normalization crash fix
[#7775](https://github.com/RawTherapee/RawTherapee/pull/7775) and the RGB/XYZ
conversion-matrix corrections
[#7777](https://github.com/RawTherapee/RawTherapee/pull/7777).

## Build and use

Install the normal RawTherapee build dependencies and libheif >= 1.23.6 with
the libde265 decoder and x265 encoder. Set `PKG_CONFIG_PATH` to the directory
containing `libheif.pc` when using a local libheif installation. Explicitly
requiring HEIF avoids silently omitting it when the dependency is unavailable.

```sh
cmake --preset dev -B build/integration \
    -DWITH_HEIF=ON -DWITH_HEIF_EXPORT_TESTS=ON -DBUILD_TGMR_TRAINER=ON \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
nice -n 10 cmake --build build/integration -j4
OMP_NUM_THREADS=4 nice -n 10 ctest --test-dir build/integration --output-on-failure
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

Keep the individual PR branches focused on their own changes. Update this
branch by fetching upstream `dev` and the source branches, then merging each
updated branch into `integration` with a separate merge commit. Rebuild and
rerun the checks before pushing. This preserves the PR commits as ancestors
and permits later updates without rewriting the published integration history.
