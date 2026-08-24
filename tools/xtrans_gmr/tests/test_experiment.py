from __future__ import annotations

import numpy as np

from tools.xtrans_gmm.dataset import PatchSample
from tools.xtrans_gmr.experiment import _oracle


def test_block_oracle_does_not_mix_unaligned_phase_component_ids() -> None:
    samples = []
    for index in range(9):
        samples.append(PatchSample(
            source_id="grid", group="test", vector=np.zeros(27),
            indices=np.arange(9), phase=index % 2, x=index % 3, y=index // 3,
        ))
    truth = np.zeros((9, 3), dtype=np.float64)
    predictions = np.ones((9, 2, 3), dtype=np.float64)
    # Phase zero's coherent expert is component 0; phase one's coherent expert
    # is component 1.  A phase-blind block vote cannot represent this contract.
    for index, sample in enumerate(samples):
        predictions[index, sample.phase, :] = 0.1
    _, selected = _oracle(predictions, truth, samples, 3)
    np.testing.assert_array_equal(
        selected, np.asarray([sample.phase for sample in samples])
    )
