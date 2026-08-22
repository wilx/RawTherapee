"""Authenticated natural-image corpus for the X-Trans hybrid experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


CROP_SIZE = 168
CROPS_PER_SOURCE = 3

# Splits are frozen at the source-image level.  The two Middlebury motorcycle
# views deliberately share one group and one split because they depict the
# same scene.  License/provenance text is taken from scikit-image 0.26.0's
# bundled data docstrings; "not stated" is recorded rather than guessed.
NATURAL_SOURCES = (
    {
        "id": "astronaut", "filename": "astronaut.png", "split": "test",
        "group": "astronaut", "category": "portrait, fabric, saturated edges",
        "license": "public domain; no known copyright restrictions",
        "sha256": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    },
    {
        "id": "brick", "filename": "brick.png", "split": "test",
        "group": "brick", "category": "architecture-like periodic texture",
        "license": "CC0",
        "sha256": "7966caf324f6ba843118d98f7a07746d22f6a343430add0233eca5f6eaaa8fcf",
    },
    {
        "id": "camera", "filename": "camera.png", "split": "train",
        "group": "camera", "category": "portrait, equipment, grayscale edges",
        "license": "CC0; photographer Lav Varshney",
        "sha256": "b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a",
    },
    {
        "id": "cell", "filename": "cell.png", "split": "train",
        "group": "cell", "category": "low-contrast microscopy and banding",
        "license": "CC0",
        "sha256": "8d23a7fb81f7cc877cd09f330357fc7f595651306e84e17252f6e0a1b3f61515",
    },
    {
        "id": "chelsea", "filename": "chelsea.png", "split": "train",
        "group": "chelsea", "category": "fur, portrait, diagonal edges",
        "license": "CC0; photographer Stefan van der Walt",
        "sha256": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    },
    {
        "id": "clock", "filename": "clock_motion.png", "split": "train",
        "group": "clock", "category": "motion blur and smooth grayscale",
        "license": "public domain; photographer Stefan van der Walt",
        "sha256": "f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955",
    },
    {
        "id": "coffee", "filename": "coffee.png", "split": "validation",
        "group": "coffee", "category": "saturated objects and texture",
        "license": "CC0; photographer Rachel Michetti",
        "sha256": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    },
    {
        "id": "coins", "filename": "coins.png", "split": "train",
        "group": "coins", "category": "specular objects and grayscale edges",
        "license": "no known copyright restrictions; Brooklyn Museum",
        "sha256": "f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba",
    },
    {
        "id": "grass", "filename": "grass.png", "split": "validation",
        "group": "grass", "category": "foliage-like irregular texture",
        "license": "CC0",
        "sha256": "b6b6022426b38936c43a4ac09635cd78af074e90f42ffa8227ac8b7452d39f89",
    },
    {
        "id": "gravel", "filename": "gravel.png", "split": "train",
        "group": "gravel", "category": "irregular high-frequency texture",
        "license": "CC0",
        "sha256": "c48615b451bf1e606fbd72c0aa9f8cc0f068ab7111ef7d93bb9b0f2586440c12",
    },
    {
        "id": "horse", "filename": "horse.png", "split": "train",
        "group": "horse", "category": "binary silhouette and strong edges",
        "license": "CC0; Andreas Preuss",
        "sha256": "c7fb60789fe394c485f842291ea3b21e50d140f39d6dcb5fb9917cc178225455",
    },
    {
        "id": "hubble", "filename": "hubble_deep_field.jpg", "split": "test",
        "group": "hubble", "category": "night-like field and fine colored detail",
        "license": "NASA public domain",
        "sha256": "3a19c5dd8a927a9334bb1229a6d63711b1c0c767fb27e2286e7c84a3e2c2f5f4",
    },
    {
        "id": "ihc", "filename": "ihc.png", "split": "train",
        "group": "ihc", "category": "fine saturated biological texture",
        "license": "no known copyright restrictions; CMMI",
        "sha256": "f8dd1aa387ddd1f49d8ad13b50921b237df8e9b262606d258770687b0ef93cef",
    },
    {
        "id": "moon", "filename": "moon.png", "split": "train",
        "group": "moon", "category": "low-contrast natural texture",
        "license": "not stated in scikit-image 0.26.0 data docstring",
        "sha256": "78739619d11f7eb9c165bb5d2efd4772cee557812ec847532dbb1d92ef71f577",
    },
    {
        "id": "motorcycle-left", "filename": "motorcycle_left.png", "split": "validation",
        "group": "middlebury-motorcycle", "category": "vehicles, fabric, fine text",
        "license": "Middlebury 2014 stereo benchmark; terms not stated in local docstring",
        "sha256": "db18e9c4157617403c3537a6ba355dfeafe9a7eabb6b9b94cb33f6525dd49179",
    },
    {
        "id": "motorcycle-right", "filename": "motorcycle_right.png", "split": "validation",
        "group": "middlebury-motorcycle", "category": "vehicles, fabric, fine text",
        "license": "Middlebury 2014 stereo benchmark; terms not stated in local docstring",
        "sha256": "5fc913ae870e42a4b662314bc904d1786bcad8e2f0b9b67dba5a229406357797",
    },
    {
        "id": "page", "filename": "page.png", "split": "test",
        "group": "page", "category": "fine text and uneven illumination",
        "license": "not stated in scikit-image 0.26.0 data docstring",
        "sha256": "341a6f0a61557662b02734a9b6e56ec33a915b2c41886b97509dedf2a43b47a3",
    },
    {
        "id": "retina", "filename": "retina.jpg", "split": "train",
        "group": "retina", "category": "saturated biological structure",
        "license": "CC0 1.0; Mikael Haeggstroem",
        "sha256": "38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6",
    },
    {
        "id": "rocket", "filename": "rocket.jpg", "split": "train",
        "group": "rocket", "category": "smooth sky, architecture, fine detail",
        "license": "SpaceX public domain",
        "sha256": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
    },
    {
        "id": "text", "filename": "text.png", "split": "train",
        "group": "text", "category": "fine text and corners",
        "license": "public domain; no known copyright restrictions",
        "sha256": "bd84aa3a6e3c9887850d45d606c96b2e59433fbef50338570b63c319e668e6d1",
    },
)


def _data_directory() -> Path:
    try:
        import skimage
        import skimage.data
    except ImportError as error:
        raise RuntimeError("scikit-image 0.26.0 is required") from error
    if skimage.__version__ != "0.26.0":
        raise RuntimeError(f"expected scikit-image 0.26.0, found {skimage.__version__}")
    return Path(skimage.data.data_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def srgb_decode(values: np.ndarray) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    return np.where(
        checked <= 0.04045,
        checked / 12.92,
        ((checked + 0.055) / 1.055) ** 2.4,
    )


def crop_coordinates(width: int, height: int, digest: str) -> tuple[tuple[int, int], ...]:
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError("source is smaller than the fixed crop")
    maximum_x = width - CROP_SIZE
    maximum_y = height - CROP_SIZE
    coordinates = [(maximum_x // 2, maximum_y // 2)]
    seed = bytes.fromhex(digest)
    candidate = 0
    while len(coordinates) < CROPS_PER_SOURCE:
        expanded = hashlib.sha256(seed + candidate.to_bytes(4, "little")).digest()
        x_bits = int.from_bytes(expanded[:8], "little")
        y_bits = int.from_bytes(expanded[8:16], "little")
        coordinate = (
            x_bits % (maximum_x + 1),
            y_bits % (maximum_y + 1),
        )
        if coordinate not in coordinates:
            coordinates.append(coordinate)
        candidate += 1
        if candidate > 128:
            raise RuntimeError("could not derive unique deterministic crops")
    return tuple(coordinates)


def load_sources() -> list[dict[str, object]]:
    data_directory = _data_directory()
    result = []
    for binding in NATURAL_SOURCES:
        path = data_directory / str(binding["filename"])
        if not path.is_file():
            raise RuntimeError(f"missing pinned source {binding['filename']}")
        if _sha256(path) != binding["sha256"]:
            raise RuntimeError(f"digest mismatch for {binding['filename']}")
        with Image.open(path) as image:
            encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        linear = np.moveaxis(srgb_decode(encoded), -1, 0)
        source = dict(binding)
        source["height"] = int(linear.shape[1])
        source["width"] = int(linear.shape[2])
        source["rgb"] = linear
        source["crops"] = crop_coordinates(
            int(linear.shape[2]), int(linear.shape[1]), str(binding["sha256"])
        )
        result.append(source)
    return result


def iter_crops(sources: Iterable[dict[str, object]]):
    for source in sources:
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        for crop_index, (x, y) in enumerate(source["crops"]):
            yield {
                "category": source["category"],
                "crop": [x, y, CROP_SIZE, CROP_SIZE],
                "crop_id": f"{source['id']}-{crop_index}",
                "group": source["group"],
                "source_id": source["id"],
                "split": source["split"],
                "truth": rgb[:, y : y + CROP_SIZE, x : x + CROP_SIZE].copy(),
            }


def dataset_manifest(sources: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = []
    for source in sources:
        rows.append({
            key: source[key]
            for key in (
                "category", "crops", "filename", "group", "height", "id",
                "license", "sha256", "split", "width",
            )
        })
    return {
        "crop_count": sum(len(row["crops"]) for row in rows),
        "crop_shape_chw": [3, CROP_SIZE, CROP_SIZE],
        "crops_per_source": CROPS_PER_SOURCE,
        "format": "rawtherapee-xtrans-hybrid-dataset-v1",
        "image_source": "scikit-image 0.26.0 bundled sample data",
        "linearization": "IEC 61966-2-1 sRGB inverse EOTF applied to normalized samples",
        "source_count": len(rows),
        "source_split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "sources": rows,
        "split_policy": "frozen source-level split; related stereo views share one group",
    }
