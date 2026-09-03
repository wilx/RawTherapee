from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.tgmr_trainer import prepare_corpus, reconstruct_corpus


def expanded_classification() -> dict[str, object]:
    return {
        "channel_means": [0.1, 0.2, 0.3],
        "chroma_ratio_mean": 0.1,
        "clipped_black_fraction": 0.0,
        "clipped_white_fraction": 0.0,
        "dhash": "0123456789abcdef",
        "gradient_rms": 0.01,
        "hue_degrees": 30.0,
        "hue_histogram": [1] * 12,
        "jpeg_blockiness": 0.0,
        "laplacian_rms": 0.01,
        "local_contrast": 0.02,
        "luminance_histogram": [1] * 16,
        "luminance_mean": 0.2,
        "luminance_p01": 0.01,
        "luminance_p99": 0.9,
        "luminance_stddev": 0.1,
        "perceptual_hash": "0123456789abcdef",
        "phash": "fedcba9876543210",
        "saturation_histogram": [1] * 8,
        "saturation_mean": 0.2,
    }


class PreparationTests(unittest.TestCase):
    def test_license_filter_accepts_only_frozen_permissive_set(self) -> None:
        self.assertEqual(
            prepare_corpus.license_id("https://creativecommons.org/licenses/by/4.0/"),
            "CC-BY-4.0",
        )
        self.assertEqual(prepare_corpus.license_id("CC0"), "CC0-1.0")
        self.assertIsNone(prepare_corpus.license_id(
            "https://creativecommons.org/licenses/by-sa/4.0/"
        ))
        self.assertIsNone(prepare_corpus.license_id(
            "https://creativecommons.org/licenses/by-nc/4.0/"
        ))
        self.assertIsNone(prepare_corpus.license_id("no known copyright restrictions"))

    def test_merge_and_release_manifest_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({
                "catalog": "pass-v3", "format": prepare_corpus.CANDIDATE_FORMAT,
                "source_id": "pass-v3:b",
            }, sort_keys=True) + "\n", encoding="utf-8")
            second.write_text(json.dumps({
                "catalog": "openimages-cvdf-v5-boxable",
                "format": prepare_corpus.CANDIDATE_FORMAT,
                "source_id": "openimages-cvdf-v5-boxable:a",
            }, sort_keys=True) + "\n", encoding="utf-8")
            merged = root / "merged.jsonl"
            self.assertEqual(prepare_corpus.main([
                "merge", str(merged), str(first), str(second),
            ]), 0)
            merged_values = [json.loads(line) for line in merged.read_text().splitlines()]
            self.assertEqual([value["source_id"] for value in merged_values], [
                "openimages-cvdf-v5-boxable:a", "pass-v3:b",
            ])

            artifacts = []
            for index, name in enumerate((
                    "corpus.jsonl", "corpus.tgpc", "corpus.tgpc.gz", "NOTICE.txt",
                    "statistics.json", "rights.json")):
                path = root / name
                path.write_bytes(f"artifact-{index}\n".encode())
                artifacts.append(path)
            release = root / "release.json"
            arguments = [
                "release-manifest", *(str(path) for path in artifacts), str(release),
                "--zenodo-doi", "10.5281/zenodo.fixture",
                "--github-release-url", "https://example.invalid/release",
            ]
            self.assertEqual(prepare_corpus.main(arguments), 0)
            first_bytes = release.read_bytes()
            release.unlink()
            self.assertEqual(prepare_corpus.main(arguments), 0)
            self.assertEqual(release.read_bytes(), first_bytes)
            value = json.loads(first_bytes)
            self.assertEqual(value["corpus_id"], "tgmr-corpus-v1")
            self.assertEqual(len(value["artifacts"]), 6)

    def test_openimages_normalize_fetch_and_assemble(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            data = b"catalog-authenticated-image"
            source.write_bytes(data)
            csv_path = root / "openimages.csv"
            md5 = hashlib.md5(data).digest()  # noqa: S324 - catalog fixture identity.
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "ImageID", "OriginalURL", "OriginalLandingURL", "License",
                    "AuthorProfileURL", "Author", "Title", "OriginalMD5",
                ])
                writer.writeheader()
                writer.writerow({
                    "ImageID": "abc1234567890def", "OriginalURL": source.as_uri(),
                    "OriginalLandingURL": "https://www.flickr.com/photos/example/123456789/",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                    "AuthorProfileURL": "https://www.flickr.com/people/example/",
                    "Author": "Example Author", "Title": "Example",
                    "OriginalMD5": base64.b64encode(md5).decode(),
                })
            candidates = root / "candidates.jsonl"
            self.assertEqual(prepare_corpus.main([
                "normalize", "openimages", str(csv_path), str(candidates),
                "--revision", "fixture-boxable",
            ]), 0)
            first = json.loads(candidates.read_text())
            self.assertEqual(first["catalog"], "openimages-cvdf-v5-boxable")
            self.assertEqual(first["license"], "CC-BY-2.0")
            self.assertEqual(first["author_id"], "flickr-user:example")
            self.assertEqual(first["upstream_flickr_id"], "123456789")

            cache = root / "cache"
            fetched = root / "fetched.jsonl"
            report = root / "fetch-report.json"
            self.assertEqual(prepare_corpus.main([
                "fetch", str(candidates), str(cache), str(fetched),
                "--retry", "0", "--report", str(report),
            ]), 0)
            fetched_record = json.loads(fetched.read_text())
            source_id = "openimages-cvdf-v5-boxable:abc1234567890def"
            classification = root / "classifications.jsonl"
            classification.write_text(json.dumps({
                "cache_filename": fetched_record["cache_filename"],
                "classification": expanded_classification(),
                "decoded_pixel_sha256": "1" * 64,
                "file_type": "jpeg", "height": 800,
                "icc_identity": "assumed-srgb", "orientation": 1,
                "source_id": source_id, "width": 1200,
            }, sort_keys=True) + "\n", encoding="utf-8")
            reviews = root / "reviews.jsonl"
            reviews.write_text(json.dumps({
                "content_tags": ["people", "skin-hair-clothing"],
                "format": prepare_corpus.REVIEW_FORMAT,
                "people_review_status": "approved-no-minors-or-sensitive-content",
                "rights_evidence_revision": "fixture-review-1",
                "rights_evidence_sha256": "2" * 64,
                "rights_evidence_url": "https://example.invalid/review/abc123",
                "rights_review_status": "approved", "source_id": source_id,
            }, sort_keys=True) + "\n", encoding="utf-8")
            assembled = root / "assembled.jsonl"
            self.assertEqual(prepare_corpus.main([
                "assemble", str(fetched), str(classification), str(cache), str(assembled),
                "--reviews", str(reviews),
            ]), 0)
            record = reconstruct_corpus.read_manifest(assembled)[0]
            self.assertEqual(record["format"], reconstruct_corpus.FORMAT_V2)
            self.assertEqual(record["rights"]["review_status"], "approved")  # type: ignore[index]
            self.assertEqual(record["split"], "unassigned")

            proxy_record = json.loads(classification.read_text())
            proxy_record["source_sha256"] = fetched_record["sha256"]
            proxy_record["cache_filename"] = (
                "train/a/b/c/abc1234567890def.jpg"
            )
            local_source = cache / proxy_record["cache_filename"]
            local_source.parent.mkdir(parents=True)
            local_source.write_bytes((cache / fetched_record["cache_filename"]).read_bytes())
            classification.write_text(
                json.dumps(proxy_record, sort_keys=True) + "\n", encoding="utf-8"
            )
            local_assembled = root / "local-assembled.jsonl"
            self.assertEqual(prepare_corpus.main([
                "assemble", str(candidates), str(classification), str(cache),
                str(local_assembled), "--reviews", str(reviews),
            ]), 0)
            local_record = reconstruct_corpus.read_manifest(local_assembled)[0]
            self.assertEqual(local_record["sha256"], fetched_record["sha256"])
            self.assertEqual(
                local_record["original_url"],
                "https://open-images-dataset.s3.amazonaws.com/train/abc1234567890def.jpg",
            )

            proxy_record["format"] = "rawtherapee-tgmr-image-proxy-classification-v1"
            classification.write_text(
                json.dumps(proxy_record, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaises(prepare_corpus.CorpusPreparationError):
                prepare_corpus.main([
                    "assemble", str(fetched), str(classification), str(cache),
                    str(root / "proxy-assembled.jsonl"), "--reviews", str(reviews),
                ])

    def test_pass_commons_and_smithsonian_normalizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "pass.csv"
            metadata.write_text(
                ",unickname,datetaken,licensename,hash,latitude,longitude\n"
                "3,example,2020-01-01,Attribution License,"
                "21657882a4c879e3d08bf7eb5997451,,\n",
                encoding="utf-8",
            )
            urls = root / "pass-urls.txt"
            urls.write_text("https://example.invalid/pass.jpg\n", encoding="utf-8")
            pass_output = root / "pass.jsonl"
            prepare_corpus.main([
                "normalize", "pass", str(metadata), str(pass_output),
                "--revision", "3.0", "--urls", str(urls),
            ])
            pass_value = json.loads(pass_output.read_text())
            self.assertEqual(pass_value["license"], "CC-BY-4.0")
            self.assertEqual(pass_value["author_id"], "flickr-user:example")
            self.assertEqual(
                pass_value["upstream_source_id"],
                "21657882a4c879e3d08bf7eb5997451",
            )
            self.assertIsNone(pass_value["advertised_checksum"])

            commons_snapshot = root / "commons.jsonl"
            commons_snapshot.write_text(json.dumps({
                "imageinfo": {
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Fixture.jpg",
                    "extmetadata": {
                        "Artist": {"value": "Example"},
                        "LicenseShortName": {"value": "CC BY 4.0"},
                        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
                    },
                    "sha1": "a" * 40, "url": "https://upload.wikimedia.org/fixture.jpg",
                    "user": "Example", "userid": 42,
                },
                "pageid": 7, "title": "File:Fixture.jpg",
            }, sort_keys=True) + "\n", encoding="utf-8")
            commons_output = root / "commons-out.jsonl"
            prepare_corpus.main([
                "normalize", "commons", str(commons_snapshot), str(commons_output),
                "--revision", "fixture-revision",
            ])
            self.assertEqual(json.loads(commons_output.read_text())["license"], "CC-BY-4.0")

            smithsonian_snapshot = root / "smithsonian.jsonl"
            smithsonian_snapshot.write_text(json.dumps({
                "author": "Smithsonian Institution", "id": "edan-1",
                "media": {"url": "https://ids.si.edu/fixture.jpg", "usage": "CC0"},
                "record_url": "https://www.si.edu/object/fixture", "title": "Fixture",
                "unit_code": "NMNH",
            }, sort_keys=True) + "\n", encoding="utf-8")
            smithsonian_output = root / "smithsonian-out.jsonl"
            prepare_corpus.main([
                "normalize", "smithsonian", str(smithsonian_snapshot), str(smithsonian_output),
                "--revision", "fixture-revision",
            ])
            self.assertEqual(json.loads(smithsonian_output.read_text())["license"], "CC0-1.0")

    def test_smithsonian_aws_collector_needs_no_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = root / "nmnhento"
            unit.mkdir()
            shard = unit / "00.txt"
            eligible = {
                "content": {
                    "descriptiveNonRepeating": {
                        "metadata_usage": {"access": "CC0"},
                        "online_media": {"media": [{
                            "content": "https://ids.si.edu/ids/deliveryService/id/fallback",
                            "resources": [
                                {
                                    "height": 480, "label": "Screen Image",
                                    "url": "https://ids.si.edu/screen.jpg", "width": 640,
                                },
                                {
                                    "height": 9108, "label": "High-resolution JPEG",
                                    "url": "https://ids.si.edu/high-resolution.jpg",
                                    "width": 11608,
                                },
                            ],
                            "type": "Images", "usage": {"access": "CC0"},
                        }]},
                        "record_link": "https://www.si.edu/object/fixture",
                        "unit_code": "NMNHENTO",
                    },
                    "freetext": {
                        "name": [{"content": "Fixture Collector", "label": "Collector"}],
                    },
                    "indexedStructured": {
                        "online_media_type": ["Images"],
                        "scientific_name": ["Fixture example"],
                        "tax_class": ["Insecta"],
                        "topic": ["Insects"],
                    },
                },
                "id": "edanmdm-fixture-1", "title": "Fixture insect",
            }
            rejected = json.loads(json.dumps(eligible))
            rejected["id"] = "edanmdm-fixture-2"
            rejected["content"]["descriptiveNonRepeating"]["metadata_usage"]["access"] = "Usage conditions apply"
            shard.write_text(
                json.dumps(eligible, separators=(",", ":")) + "\n"
                + json.dumps(rejected, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            unit_index = unit / "index.txt"
            unit_index.write_text(shard.as_uri() + "\n", encoding="utf-8")
            root_index = root / "index.txt"
            root_index.write_text(unit_index.as_uri() + "\n", encoding="utf-8")

            output = root / "smithsonian-aws.jsonl"
            report = root / "smithsonian-aws-report.json"
            snapshot_dir = root / "frozen-smithsonian-metadata"
            self.assertEqual(prepare_corpus.main([
                "collect-smithsonian", str(output), "--unit", "nmnhento",
                "--prefix", "00", "--index-url", root_index.as_uri(),
                "--snapshot-dir", str(snapshot_dir), "--report", str(report),
            ]), 0)
            records = list(prepare_corpus._jsonl(output))
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0]["media"]["url"],  # type: ignore[index]
                "https://ids.si.edu/high-resolution.jpg",
            )
            self.assertEqual(records[0]["media"]["resource_label"], "High-resolution JPEG")  # type: ignore[index]
            self.assertEqual(
                records[0]["catalog_categories"],
                ["Fixture example", "Images", "Insecta", "Insects"],
            )
            self.assertEqual(records[0]["author"], "Smithsonian Institution")
            self.assertEqual(records[0]["author_id"], "smithsonian-record:edanmdm-fixture-1")
            normalized = root / "smithsonian-candidates.jsonl"
            self.assertEqual(prepare_corpus.main([
                "normalize", "smithsonian", str(output), str(normalized),
                "--revision", "fixture-aws-snapshot",
            ]), 0)
            candidate = json.loads(normalized.read_text(encoding="utf-8"))
            self.assertEqual(
                candidate["author_id"], "smithsonian-record:edanmdm-fixture-1"
            )
            report_value = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_value["records"], 1)
            self.assertEqual(report_value["units"][0]["shards"][0]["records"], 2)
            self.assertEqual(report_value["units"][0]["shards"][0]["eligible_records"], 1)
            self.assertEqual((snapshot_dir / "index.txt").read_bytes(), root_index.read_bytes())
            self.assertEqual(
                (snapshot_dir / "nmnhento" / "index.txt").read_bytes(),
                unit_index.read_bytes(),
            )
            self.assertEqual(
                (snapshot_dir / "nmnhento" / "00.txt").read_bytes(), shard.read_bytes()
            )

            # Offline replay must use only the frozen bytes, even after the live
            # fixture paths have changed, and reproduce both artifacts exactly.
            root_index.write_text("https://example.invalid/changed/index.txt\n", encoding="utf-8")
            unit_index.write_text("https://example.invalid/changed/ff.txt\n", encoding="utf-8")
            shard.write_text("{}\n", encoding="utf-8")
            offline_output = root / "smithsonian-offline.jsonl"
            offline_report = root / "smithsonian-offline-report.json"
            self.assertEqual(prepare_corpus.main([
                "collect-smithsonian", str(offline_output), "--unit", "nmnhento",
                "--prefix", "00", "--index-url", root_index.as_uri(),
                "--snapshot-dir", str(snapshot_dir), "--offline-snapshot",
                "--report", str(offline_report),
            ]), 0)
            self.assertEqual(offline_output.read_bytes(), output.read_bytes())
            self.assertEqual(offline_report.read_bytes(), report.read_bytes())

    def test_v2_archive_member_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"authenticated PASS archive member\n"
            archive = root / "pass.tar"
            with tarfile.open(archive, "w") as container:
                info = tarfile.TarInfo("PASS/fixture.bin")
                info.size = len(data)
                info.mtime = 0
                container.addfile(info, io.BytesIO(data))
            archive_sha, _ = prepare_corpus.sha256_file(archive)
            record = {
                "advertised_checksum": None,
                "archive_fallbacks": [{
                    "member": "PASS/fixture.bin",
                    "member_sha256": hashlib.sha256(data).hexdigest(),
                    "sha256": archive_sha,
                    "url": archive.as_uri(),
                }],
                "author": "Fixture", "author_id": "fixture-author",
                "author_url": "https://example.invalid/author",
                "cache_filename": "fixture.bin",
                "catalog": {"name": "pass-v3", "revision": "3.0", "snapshot_sha256": "3" * 64},
                "classification": expanded_classification(), "content_tags": [],
                "decoded_pixel_sha256": "4" * 64, "fallback_urls": [],
                "file_type": "fixture", "format": reconstruct_corpus.FORMAT_V2,
                "height": 7, "icc_identity": "none",
                "landing_page": "https://example.invalid/fixture",
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "orientation": 1,
                "original_url": (root / "missing.bin").as_uri(),
                "patch_coordinates": [], "patch_sampling_seed": 1,
                "people_review_status": "not-applicable",
                "rights": {
                    "evidence_revision": "fixture", "evidence_sha256": "5" * 64,
                    "evidence_url": "https://example.invalid/rights", "review_status": "approved",
                },
                "selected": True, "selection_status": "accepted-test-fixture",
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_id": "pass-v3:fixture", "split": "train", "title": "Fixture",
                "upstream_flickr_id": None, "upstream_source_id": "fixture", "width": 7,
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            cache = root / "cache"
            self.assertEqual(reconstruct_corpus.main([
                str(manifest), str(cache), "--retry", "0",
            ]), 0)
            self.assertEqual((cache / "fixture.bin").read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
