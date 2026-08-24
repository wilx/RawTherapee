# X-Trans Student-t GMR model-reduction experiment

This development-only package measures whether the validated phase-conditioned
Student-t GMR can be made substantially cheaper without changing its
reconstruction problem or safety contract. It adds no RawTherapee engine,
method identifier, PP3 option, C++ runtime, or GUI entry.

The frozen reference is `TGMR64-FULL` from
`devnotes/xtrans-student-t-gmr-report.md`:

```text
support       7x7
phase banks   18
components    64 per phase
joint state   49 observed + 2 missing-center values
nu            3
tau           0.0003
temperature   4
DC            observed-rgb
```

## Implemented reduction probes

- genuine K=8/16/32 Student-t continuations from their corresponding Gaussian
  GMR checkpoints;
- post-hoc component removal by mixture weight and stored training posterior;
- factor-plus-diagonal covariance approximation at ranks 2--32;
- exact Woodbury and determinant-lemma Student-t inference;
- exact-posterior top-q truncation;
- observable S9/S25 Student-t marginal shortlisting;
- float32 factor and selected dense/shortlist inference;
- a standalone C++11/OpenMP phase-batched benchmark for the selected model.

The factor implementation never constructs or inverts a dense 49x49 matrix at
inference time. Per component it evaluates a diagonal quadratic, projects into
factor space, solves the precomputed small positive-definite system, and forms
the two missing center values from the same solved latent coordinate.

S9/S25 denote the component-likelihood dimensions. The immutable
`observed-rgb` DC normalization still uses all 49 physical observations; using
a different DC estimate would be a model change, not an inference reduction.

## Reproduction boundary

BSDS, Hubble/Hydra, decoded training vectors, and every fitted `.npz` model
remain external. Model continuation is performed with the existing
`tools.xtrans_tgmr.train` CLI. The validation-only screen is:

```sh
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m tools.xtrans_tgmr_reduce.screen \
    --output /tmp/xtrans-tgmr-reduce/screen.json \
    --full-model /tmp/xtrans-tgmr/models/nu3-i30-n200.npz \
    --k8 /tmp/xtrans-tgmr-reduce/models/k8-nu3-i30.npz \
    --k16 /tmp/xtrans-tgmr-reduce/models/k16-nu3-i30.npz \
    --k32 /tmp/xtrans-tgmr-reduce/models/k32-nu3-i30.npz
```

The screen reproduces the frozen K=64 output before evaluating reductions. It
uses only training statistics and validation reconstruction to select K, rank,
or shortlist geometry. Untouched BSDS, external chromatic crops, star fields,
and analytical scenes are reserved for the subsequently frozen candidate.

True Student-t factor-analyzer training and native C++ are trigger-gated. They
must not be implemented merely because their supporting code would be
interesting; the preceding validation stage must first demonstrate useful
headroom.

## Result

The validation screen selected genuine K32 with an S9/top-8 shortlist. It
retains 91.45% of validation gain and 90.81% on untouched BSDS, and it passes
all frozen safety gates. Factor covariance failed its trigger. The selected
model is numerically safe in float32, but provides only a 7.01x arithmetic
reduction and reaches about 0.98 MP/s in the research native executor on the
24-thread test host. The result is therefore **PARTIAL**, not a production
method.

Run the frozen safety confirmation with:

```sh
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m tools.xtrans_tgmr_reduce.final \
    --output /tmp/xtrans-tgmr-reduce/results.json \
    --screen /tmp/xtrans-tgmr-reduce/screen.json \
    --models /tmp/xtrans-tgmr-reduce/models \
    --full-model /tmp/xtrans-tgmr/models/nu3-i30-n200.npz
```

The model-bearing native binary is generated externally:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m tools.xtrans_tgmr_reduce.export_native \
    /tmp/xtrans-tgmr-reduce/models/k32-nu3-i30.npz \
    /tmp/xtrans-tgmr-reduce/tgmr32-native.bin

g++ -O3 -DNDEBUG -std=c++11 -fopenmp \
  -Wall -Wextra -Wpedantic -Werror \
  tools/xtrans_tgmr_reduce/native/tgmr_benchmark.cc \
  -o /tmp/xtrans-tgmr-reduce/tgmr_benchmark
```

See `devnotes/xtrans-tgmr-reduction-report.md` for the complete quality/cost
frontier and safety result.
