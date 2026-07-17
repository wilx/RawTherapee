import numpy as np
import pytest

from tools.neural_demosaic.compare_gamma22_outputs import ComparisonError
from tools.neural_demosaic.compare_xveon_migraphx import analyze_arrays


def test_identical_images_pass_backend_metrics():
    image = np.arange(18 * 18 * 3, dtype=np.uint16).reshape(18, 18, 3)
    result = analyze_arrays(image, image.copy(), image, (0, 0, 18, 18))
    assert result["all_rgb_samples_equal"]
    assert result["cpsnr_db"] == float("inf")
    assert result["ssim"] == 1
    assert result["phase"]["pass"]


def test_known_channel_difference_is_measured():
    cpu = np.full((18, 18, 3), 1000, np.uint16)
    gpu = cpu.copy()
    gpu[..., 0] += 100
    result = analyze_arrays(cpu, gpu, cpu, (0, 0, 18, 18))
    assert result["maximum_absolute_error"] == pytest.approx(100 / 65535)
    assert result["channel_mean_delta_rgb"][0] == pytest.approx(100 / 65535)


def test_rejects_mismatched_images_and_crop():
    image = np.zeros((18, 18, 3), np.uint16)
    with pytest.raises(ComparisonError):
        analyze_arrays(image, image[:12], image, (0, 0, 12, 12))
    with pytest.raises(ComparisonError):
        analyze_arrays(image, image, image, (12, 12, 12, 12))
