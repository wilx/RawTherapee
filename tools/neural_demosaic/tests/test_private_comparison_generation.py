"""Public comparison writers must not recreate the complete portrait."""
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from tools.neural_demosaic import compare_packedxtrans_outputs as packed
from tools.neural_demosaic import compare_xveon_migraphx as migraphx
from tools.neural_demosaic import compare_xveon_outputs as xveon
from tools.xtrans_global import compare_outputs as global_compare
from tools.xtrans_mlri import compare_outputs as mlri
from tools.xtrans_triangulation import compare_outputs as triangulation


@pytest.mark.parametrize("method", ["mlri", "packed", "xveon", "migraphx", "global", "triangulation"])
def test_only_earring_assets_are_written(method, tmp_path):
    # Only the documented earring rectangle is relevant; no private input file.
    source = Image.new("RGB", (3650, 2090), (70, 80, 90))
    with patch("PIL.Image.open", return_value=source):
        if method == "mlri":
            mlri.save_assets(Path("private.tiff"), tmp_path, b"")
        elif method == "packed":
            packed.save_assets(Path("private.tiff"), tmp_path, b"")
        elif method == "xveon":
            xveon.save_png_assets(Path("private.tiff"), tmp_path, b"")
        elif method == "migraphx":
            migraphx.save_assets(Path("private.tiff"), tmp_path)
        else:
            module = global_compare if method == "global" else triangulation
            # Each writer opens several images; supply independently owned copies.
            with patch("PIL.Image.open", side_effect=lambda _: Image.new("RGB", (3650, 2090))):
                module.save_assets({name: Path("private.tiff") for name in module.METHODS}, tmp_path, b"")
    files = list(tmp_path.iterdir())
    assert files and all("earring-500" in path.name for path in files)
    for path in files:
        with Image.open(path) as image:
            assert image.height == 800
            assert image.width in (700, 2100, 2800)
