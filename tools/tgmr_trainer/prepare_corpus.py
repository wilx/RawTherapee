#!/usr/bin/env python3
"""Prepare authenticated catalog snapshots for the TGMR production corpus.

The program deliberately uses only the Python standard library.  Network I/O
is confined to the explicit ``snapshot``, ``collect-*``, and ``fetch``
commands.  Normalization and assembly consume already frozen inputs, so the
canonical corpus selection never depends on a live API or a changing catalog.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Iterable, Iterator
import urllib.request
import urllib.error
import urllib.parse


CANDIDATE_FORMAT = "rawtherapee-tgmr-catalog-candidate-v1"
SOURCE_FORMAT = "rawtherapee-tgmr-corpus-source-manifest-v2"
SNAPSHOT_FORMAT = "rawtherapee-tgmr-catalog-snapshot-v1"
FETCHED_FORMAT = "rawtherapee-tgmr-fetched-candidate-v1"
REVIEW_FORMAT = "rawtherapee-tgmr-source-review-v1"
ACCEPTED_LICENSES = frozenset(
    ("CC0-1.0", "PDM-1.0", "CC-BY-2.0", "CC-BY-3.0", "CC-BY-4.0")
)
CONTENT_TAGS = frozenset(
    (
        "people", "skin-hair-clothing", "foliage", "fur-feathers",
        "architecture-brick", "textile-print", "metal-specular-jewelry",
        "food", "water-sky", "low-light", "macro-specimen",
    )
)
CHUNK = 1024 * 1024
SMITHSONIAN_AWS_INDEX = (
    "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/index.txt"
)


class CorpusPreparationError(ValueError):
    """A catalog snapshot or normalized record violates the frozen contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_pretty(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def atomic_bytes(path: Path, payload: bytes, force: bool) -> None:
    if path.exists() and not force:
        raise CorpusPreparationError(f"refusing to replace output: {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, records: Iterable[dict[str, object]], force: bool) -> None:
    payload = "".join(canonical_json(record) + "\n" for record in records).encode()
    atomic_bytes(path, payload, force)


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CorpusPreparationError(f"{field} must be lowercase SHA-256")
    return value


def require_url(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(("https://", "http://", "file://")):
        raise CorpusPreparationError(f"{field} must be an HTTP(S) or file URL")
    return value


def clean_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    cleaned = re.sub(r"\s+", " ", html.unescape(str(value))).strip()
    return cleaned or fallback


def portable_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", value).strip("-")
    if not cleaned:
        raise CorpusPreparationError("catalog record produced an empty portable ID")
    return cleaned


def flickr_photo_id(*urls: str) -> str | None:
    for url in urls:
        match = re.search(r"/photos/[^/]+/([0-9]{5,})(?:/|$)", url)
        if not match:
            match = re.search(
                r"/([0-9]{5,})_[A-Za-z0-9]+(?:_[a-z])?\.(?:jpe?g|png)(?:[?#]|$)",
                url,
                re.IGNORECASE,
            )
        if match:
            return match.group(1)
    return None


def normalized_author_id(author_url: str, author: str, prefix: str) -> str:
    match = re.search(r"flickr\.com/(?:people|photos)/([^/?#]+)", author_url, re.IGNORECASE)
    if match:
        return portable_id("flickr-user:" + urllib.parse.unquote(match.group(1)).casefold())
    return portable_id(prefix + ":" + author.casefold())


def candidate_cache_filename(record: dict[str, object]) -> str:
    identity = portable_id(
        f"{clean_text(record.get('catalog'), '')}:{clean_text(record.get('upstream_source_id'), '')}"
    )
    parsed = urllib.parse.urlparse(str(record.get("original_url", "")))
    suffix = Path(parsed.path).suffix.lower()
    if not suffix:
        identifiers = urllib.parse.parse_qs(parsed.query).get("id", [])
        if identifiers:
            suffix = Path(identifiers[0]).suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in (".jpg", ".png", ".tif", ".tiff"):
        suffix = ".img"
    return identity + suffix


def safe_cache_path(cache: Path, relative: str) -> Path:
    name = Path(relative)
    if (
        name.is_absolute()
        or not name.parts
        or any(component in ("", ".", "..") for component in name.parts)
    ):
        raise CorpusPreparationError("cache filename must be a safe relative path")
    root = cache.resolve()
    destination = (cache / name).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise CorpusPreparationError("cache filename escapes the cache") from error
    return destination


def license_id(value: object) -> str | None:
    text = str(value or "").strip().lower().rstrip("/")
    aliases = {
        "attribution license": "CC-BY-2.0",
        "cc0": "CC0-1.0",
        "public domain mark": "PDM-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0": "CC0-1.0",
        "https://creativecommons.org/publicdomain/mark/1.0": "PDM-1.0",
        "https://creativecommons.org/licenses/by/2.0": "CC-BY-2.0",
        "https://creativecommons.org/licenses/by/3.0": "CC-BY-3.0",
        "https://creativecommons.org/licenses/by/4.0": "CC-BY-4.0",
        "cc-by-2.0": "CC-BY-2.0",
        "cc-by-3.0": "CC-BY-3.0",
        "cc-by-4.0": "CC-BY-4.0",
        "cc by 2.0": "CC-BY-2.0",
        "cc by 3.0": "CC-BY-3.0",
        "cc by 4.0": "CC-BY-4.0",
    }
    return aliases.get(text)


def license_url(identifier: str) -> str:
    return {
        "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
        "PDM-1.0": "https://creativecommons.org/publicdomain/mark/1.0/",
        "CC-BY-2.0": "https://creativecommons.org/licenses/by/2.0/",
        "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
        "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    }[identifier]


def candidate(
    *, catalog: str, revision: str, snapshot_sha256: str,
    upstream_id: str, original_url: str, landing_page: str,
    author: str, author_id: str, author_url: str, title: str,
    license_name: str | None, advertised_checksum: str | None,
    flickr_id: str | None = None, rights_evidence_url: str | None = None,
    archive_fallbacks: list[dict[str, object]] | None = None,
    catalog_categories: list[str] | None = None,
) -> dict[str, object]:
    accepted = license_name in ACCEPTED_LICENSES
    return {
        "advertised_checksum": advertised_checksum,
        "author": clean_text(author, "unknown"),
        "author_id": portable_id(author_id),
        "author_url": author_url,
        "archive_fallbacks": archive_fallbacks or [],
        "catalog": catalog,
        "catalog_categories": catalog_categories or [],
        "catalog_revision": revision,
        "catalog_snapshot_sha256": snapshot_sha256,
        "format": CANDIDATE_FORMAT,
        "landing_page": landing_page,
        "license": license_name or "REJECTED-OR-UNKNOWN",
        "license_url": license_url(license_name) if accepted else rights_evidence_url or landing_page,
        "original_url": original_url,
        "rights_evidence_url": rights_evidence_url or landing_page,
        "rights_review_status": "pending" if accepted else "rejected-license",
        "title": clean_text(title, upstream_id),
        "upstream_flickr_id": flickr_id,
        "upstream_source_id": upstream_id,
    }


def normalize_openimages(path: Path, revision: str, digest: str) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ImageID", "OriginalURL", "OriginalLandingURL", "License", "Author"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise CorpusPreparationError("Open Images CSV is missing required columns")
        for row in reader:
            identifier = clean_text(row.get("ImageID"), "")
            if not identifier:
                continue
            original = require_url(row.get("OriginalURL"), "Open Images OriginalURL")
            landing = require_url(row.get("OriginalLandingURL"), "Open Images OriginalLandingURL")
            author_url = str(row.get("AuthorProfileURL") or landing)
            author = clean_text(row.get("Author"), "unknown")
            encoded_md5 = str(row.get("OriginalMD5") or "")
            advertised = f"md5-base64:{encoded_md5}" if encoded_md5 else None
            flickr = flickr_photo_id(landing, original)
            yield candidate(
                catalog="openimages-cvdf-v5-boxable", revision=revision,
                snapshot_sha256=digest,
                upstream_id=identifier, original_url=original, landing_page=landing,
                author=author,
                author_id=normalized_author_id(author_url, author, "openimages-author"),
                author_url=author_url,
                title=clean_text(row.get("Title"), identifier),
                license_name=license_id(row.get("License")), advertised_checksum=advertised,
                flickr_id=flickr, rights_evidence_url=str(row.get("License") or landing),
            )


def normalize_pass(
    path: Path, urls_path: Path, revision: str, digest: str,
    archive_index: Path | None = None,
) -> Iterator[dict[str, object]]:
    urls_digest, _ = sha256_file(urls_path)
    with urls_path.open("r", encoding="utf-8", newline="") as stream:
        urls = [line.strip() for line in stream if line.strip()]
    archives: dict[str, list[dict[str, object]]] = {}
    if archive_index is not None:
        for value in _jsonl(archive_index):
            identity = clean_text(value.get("hash"), "")
            entry = {
                "member": clean_text(value.get("member"), ""),
                "member_sha256": require_sha256(value.get("member_sha256"), "member_sha256"),
                "sha256": require_sha256(value.get("sha256"), "archive sha256"),
                "url": require_url(value.get("url"), "archive URL"),
            }
            if Path(str(entry["member"])).is_absolute() or ".." in Path(str(entry["member"])).parts:
                raise CorpusPreparationError("PASS archive index contains an unsafe member")
            archives.setdefault(identity, []).append(entry)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"unickname", "licensename", "hash"}.issubset(reader.fieldnames):
            raise CorpusPreparationError("PASS CSV is missing required columns")
        for index, row in enumerate(reader):
            if index >= len(urls):
                raise CorpusPreparationError("PASS URL list is shorter than metadata CSV")
            identity = clean_text(row.get("hash"), "")
            # PASS calls this column ``hash``, but it is the hexadecimal
            # source/filename identity used to join pass_metadata.csv to the
            # image URL list.  It is not a checksum of the downloaded image:
            # the published catalog contains identities from 23 to 32 digits,
            # and their values do not match the image MD5.  Authenticate the
            # actual image bytes with our SHA-256 after retrieval instead.
            if not re.fullmatch(r"[0-9a-f]{23,32}", identity):
                raise CorpusPreparationError(
                    "PASS hash is not a lowercase hexadecimal source identity"
                )
            author = clean_text(row.get("unickname"), "unknown")
            original_url = require_url(urls[index], "PASS URL")
            author_url = "https://www.flickr.com/people/" + author
            yield candidate(
                catalog="pass-v3", revision=revision,
                snapshot_sha256=hashlib.sha256((digest + urls_digest).encode()).hexdigest(),
                upstream_id=identity, original_url=original_url,
                landing_page="https://robots.ox.ac.uk/~vgg/data/pass/",
                author=author,
                author_id=normalized_author_id(author_url, author, "pass-author"),
                author_url=author_url,
                title=identity, license_name="CC-BY-4.0",
                advertised_checksum=None,
                flickr_id=flickr_photo_id(original_url),
                rights_evidence_url="https://robots.ox.ac.uk/~vgg/data/pass/",
                archive_fallbacks=archives.get(identity, []),
            )


def _jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n") or not line.strip():
                raise CorpusPreparationError(f"{path}:{number}: noncanonical JSONL line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CorpusPreparationError(f"{path}:{number}: expected object")
            yield value


def normalize_commons(path: Path, revision: str, digest: str) -> Iterator[dict[str, object]]:
    """Normalize frozen MediaWiki imageinfo results, one page per JSONL line."""
    for value in _jsonl(path):
        title = clean_text(value.get("title"), "")
        info = value.get("imageinfo")
        if not title or not isinstance(info, dict):
            raise CorpusPreparationError("Commons snapshot needs title and imageinfo object")
        metadata = info.get("extmetadata")
        if not isinstance(metadata, dict):
            raise CorpusPreparationError("Commons imageinfo lacks extmetadata")
        def meta(name: str) -> str:
            entry = metadata.get(name)
            return clean_text(entry.get("value") if isinstance(entry, dict) else entry, "")
        short = license_id(meta("LicenseShortName")) or license_id(meta("LicenseUrl"))
        page_id = str(value.get("pageid") or title)
        author = re.sub(r"<[^>]+>", "", meta("Artist")) or clean_text(info.get("user"), "unknown")
        yield candidate(
            catalog="wikimedia-commons", revision=revision, snapshot_sha256=digest,
            upstream_id=page_id, original_url=require_url(info.get("url"), "Commons original URL"),
            landing_page=require_url(info.get("descriptionurl"), "Commons description URL"),
            author=author, author_id=str(info.get("userid") or info.get("user") or author),
            author_url=str(info.get("userpage") or info.get("descriptionurl")),
            title=title, license_name=short,
            advertised_checksum=f"sha1:{str(info.get('sha1') or '')}" if info.get("sha1") else None,
            rights_evidence_url=str(info.get("descriptionurl")),
        )


def normalize_smithsonian(path: Path, revision: str, digest: str) -> Iterator[dict[str, object]]:
    """Normalize frozen Smithsonian Open Access AWS/API records."""
    for value in _jsonl(path):
        identifier = clean_text(value.get("id"), "")
        media = value.get("media")
        if not identifier or not isinstance(media, dict):
            raise CorpusPreparationError("Smithsonian snapshot needs id and selected media object")
        usage = clean_text(media.get("usage"), "")
        accepted = "cc0" in usage.lower()
        landing = require_url(value.get("record_url"), "Smithsonian record URL")
        categories = value.get("catalog_categories", [])
        if not isinstance(categories, list):
            raise CorpusPreparationError("Smithsonian catalog_categories must be a list")
        yield candidate(
            catalog="smithsonian-open-access", revision=revision, snapshot_sha256=digest,
            upstream_id=identifier, original_url=require_url(media.get("url"), "Smithsonian media URL"),
            landing_page=landing, author=clean_text(value.get("author"), "Smithsonian Institution"),
            author_id=clean_text(
                value.get("author_id"), "smithsonian-record:" + identifier
            ), author_url=landing,
            title=clean_text(value.get("title"), identifier),
            license_name="CC0-1.0" if accepted else None,
            advertised_checksum=str(media.get("checksum")) if media.get("checksum") else None,
            rights_evidence_url=landing,
            catalog_categories=[
                clean_text(category, "") for category in categories
                if clean_text(category, "")
            ],
        )


def snapshot(arguments: argparse.Namespace) -> int:
    if arguments.output.exists() and not arguments.force:
        raise CorpusPreparationError(f"refusing to replace output: {arguments.output}")
    temporary = arguments.output.with_name(arguments.output.name + ".part")
    request = urllib.request.Request(
        arguments.url, headers={"User-Agent": "RawTherapee-TGMR-catalog-snapshot/1"}
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while block := response.read(CHUNK):
                stream.write(block)
                digest.update(block)
                size += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        actual = digest.hexdigest()
        if arguments.sha256 and actual != arguments.sha256:
            raise CorpusPreparationError("catalog snapshot SHA-256 differs from --sha256")
        os.replace(temporary, arguments.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "bytes": size, "format": SNAPSHOT_FORMAT, "sha256": actual,
        "url": arguments.url,
    }
    sys.stdout.buffer.write(canonical_pretty(report))
    return 0


def _read_id_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        values = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    if len(values) != len(set(values)):
        raise CorpusPreparationError("identifier list contains duplicates")
    return values


def collect_commons(arguments: argparse.Namespace) -> int:
    titles = _read_id_list(arguments.ids)
    records: list[dict[str, object]] = []
    for offset in range(0, len(titles), 50):
        query = urllib.parse.urlencode({
            "action": "query", "format": "json", "formatversion": "2",
            "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl",
            "iiprop": "url|sha1|size|timestamp|user|userid|extmetadata",
            "prop": "imageinfo", "redirects": "1",
            "titles": "|".join(titles[offset:offset + 50]),
        })
        request = urllib.request.Request(
            "https://commons.wikimedia.org/w/api.php?" + query,
            headers={"User-Agent": "RawTherapee-TGMR-catalog-snapshot/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            raise CorpusPreparationError("Commons API response lacks pages")
        for page in pages:
            if not isinstance(page, dict) or page.get("missing"):
                continue
            imageinfo = page.get("imageinfo")
            if not isinstance(imageinfo, list) or len(imageinfo) != 1:
                raise CorpusPreparationError("Commons page lacks one imageinfo revision")
            records.append({
                "imageinfo": imageinfo[0], "pageid": page.get("pageid"),
                "title": page.get("title"),
            })
    records.sort(key=lambda value: str(value.get("title")))
    write_jsonl(arguments.output, records, arguments.force)
    sys.stdout.buffer.write(canonical_pretty({
        "format": "rawtherapee-tgmr-commons-snapshot-report-v1",
        "records": len(records), "requested": len(titles),
        "sha256": sha256_file(arguments.output)[0],
    }))
    return 0 if len(records) == len(titles) else 1


def _smithsonian_record(response: dict[str, object]) -> dict[str, object] | None:
    content = response.get("content")
    if not isinstance(content, dict):
        return None
    descriptive = content.get("descriptiveNonRepeating")
    if not isinstance(descriptive, dict):
        return None
    metadata_usage = descriptive.get("metadata_usage")
    metadata_access = (
        metadata_usage.get("access") if isinstance(metadata_usage, dict) else metadata_usage
    )
    if str(metadata_access).casefold() != "cc0":
        return None
    online = descriptive.get("online_media")
    media_values = online.get("media") if isinstance(online, dict) else None
    if not isinstance(media_values, list):
        return None
    chosen = None
    for media in media_values:
        if not isinstance(media, dict):
            continue
        usage = media.get("usage")
        access = usage.get("access") if isinstance(usage, dict) else usage
        media_type = str(media.get("type") or "").lower()
        resources = media.get("resources")
        resource_values = resources if isinstance(resources, list) else []
        preferred_labels = ("High-resolution JPEG", "Screen Image")
        selected_resource = None
        for label in preferred_labels:
            selected_resource = next(
                (
                    item for item in resource_values
                    if isinstance(item, dict)
                    and clean_text(item.get("label"), "") == label
                    and isinstance(item.get("url"), str)
                ),
                None,
            )
            if selected_resource is not None:
                break
        url = selected_resource.get("url") if selected_resource else media.get("content")
        if str(access).casefold() == "cc0" and "image" in media_type and isinstance(url, str):
            chosen = {
                "checksum": media.get("checksum"),
                "height": selected_resource.get("height") if selected_resource else None,
                "resource_label": (
                    selected_resource.get("label") if selected_resource else "Delivery Service"
                ),
                "url": url,
                "usage": "CC0",
                "width": selected_resource.get("width") if selected_resource else None,
            }
            break
    if chosen is None:
        return None
    freetext = descriptive.get("freetext")
    author = "Smithsonian Institution"
    author_id = "smithsonian-record:" + clean_text(
        descriptive.get("record_ID"), clean_text(response.get("id"), "unknown")
    )
    if isinstance(freetext, dict):
        names = freetext.get("name")
        creator_labels = (
            "architect", "artist", "author", "creator", "designer", "maker",
            "manufacturer", "photographer",
        )
        creators = []
        if isinstance(names, list):
            for name in names:
                if not isinstance(name, dict):
                    continue
                label = clean_text(name.get("label"), "").casefold()
                content_value = clean_text(name.get("content"), "")
                if content_value and any(token in label for token in creator_labels):
                    creators.append(content_value)
        if creators:
            creators = sorted(set(creators), key=str.casefold)
            author = "; ".join(creators)
            author_id = "smithsonian-creator:" + ";".join(
                creator.casefold() for creator in creators
            )
    categories = []
    indexed = content.get("indexedStructured")
    if isinstance(indexed, dict):
        for name in (
            "object_type", "online_media_type", "scientific_name", "tax_class",
            "tax_family", "tax_kingdom", "tax_order", "tax_phylum", "topic",
        ):
            values = indexed.get(name)
            if isinstance(values, list):
                categories.extend(clean_text(value, "") for value in values)
            elif values is not None:
                categories.append(clean_text(values, ""))
    return {
        "author": author,
        "author_id": author_id,
        "catalog_categories": sorted(set(filter(None, categories))),
        "id": response.get("id"),
        "media": chosen,
        "record_url": descriptive.get("record_link"),
        "title": response.get("title"),
        "unit_code": descriptive.get("unit_code"),
    }


def _read_catalog_bytes(
    url: str, *, snapshot_path: Path | None = None,
    snapshot_name: str | None = None, offline: bool = False, force: bool = False,
) -> tuple[bytes, dict[str, object]]:
    require_url(url, "remote catalog URL")
    if offline:
        if snapshot_path is None or not snapshot_path.is_file():
            raise CorpusPreparationError(f"missing offline catalog snapshot: {snapshot_path}")
        payload = snapshot_path.read_bytes()
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "RawTherapee-TGMR-catalog-snapshot/1"},
        )
        chunks = []
        with urllib.request.urlopen(request, timeout=120) as response:
            while block := response.read(CHUNK):
                chunks.append(block)
        payload = b"".join(chunks)
        if snapshot_path is not None:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(snapshot_path, payload, force)
    identity: dict[str, object] = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "url": url,
    }
    if snapshot_name is not None:
        identity["snapshot_file"] = snapshot_name
    return payload, identity


def _catalog_index(
    url: str, *, snapshot_path: Path | None = None,
    snapshot_name: str | None = None, offline: bool = False, force: bool = False,
) -> tuple[list[str], dict[str, object]]:
    payload, identity = _read_catalog_bytes(
        url, snapshot_path=snapshot_path, snapshot_name=snapshot_name,
        offline=offline, force=force,
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusPreparationError(f"Smithsonian index is not UTF-8: {url}") from error
    values = [line.strip() for line in text.splitlines() if line.strip()]
    if not values:
        raise CorpusPreparationError(f"Smithsonian index is empty: {url}")
    if len(values) != len(set(values)):
        raise CorpusPreparationError(f"Smithsonian index contains duplicates: {url}")
    return [require_url(value, "Smithsonian index entry") for value in values], identity


def _smithsonian_unit(index_url: str) -> str:
    path = Path(urllib.parse.urlparse(index_url).path)
    if path.name != "index.txt" or path.parent.name == "edan":
        raise CorpusPreparationError(f"invalid Smithsonian unit-index URL: {index_url}")
    return path.parent.name.casefold()


def _scan_smithsonian_shard(
    url: str, visitor: Callable[[dict[str, object]], None],
    *, snapshot_path: Path | None = None, snapshot_name: str | None = None,
    offline: bool = False, force: bool = False,
) -> dict[str, object]:
    require_url(url, "Smithsonian shard URL")
    if offline:
        if snapshot_path is None or not snapshot_path.is_file():
            raise CorpusPreparationError(f"missing offline Smithsonian shard: {snapshot_path}")
        source = snapshot_path.open("rb")
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "RawTherapee-TGMR-catalog-snapshot/1"},
        )
        source = urllib.request.urlopen(request, timeout=120)
    temporary = None
    snapshot_stream = None
    if snapshot_path is not None and not offline:
        if snapshot_path.exists() and not force:
            source.close()
            raise CorpusPreparationError(f"refusing to replace output: {snapshot_path}")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot_path.with_name(snapshot_path.name + ".tmp")
        snapshot_stream = temporary.open("wb")
    digest = hashlib.sha256()
    size = 0
    records = 0
    eligible = 0
    try:
        with source:
            for number, raw_line in enumerate(source, 1):
                if snapshot_stream is not None:
                    snapshot_stream.write(raw_line)
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    continue
                records += 1
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CorpusPreparationError(
                        f"malformed Smithsonian JSON record at {url}:{number}"
                    ) from error
                if not isinstance(value, dict):
                    raise CorpusPreparationError(
                        f"non-object Smithsonian JSON record at {url}:{number}"
                    )
                normalized = _smithsonian_record(value)
                if normalized is not None:
                    eligible += 1
                    visitor(normalized)
        if snapshot_stream is not None:
            snapshot_stream.flush()
            os.fsync(snapshot_stream.fileno())
            snapshot_stream.close()
            snapshot_stream = None
            os.replace(temporary, snapshot_path)
    finally:
        if snapshot_stream is not None:
            snapshot_stream.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    identity: dict[str, object] = {
        "bytes": size, "eligible_records": eligible, "records": records,
        "sha256": digest.hexdigest(), "url": url,
    }
    if snapshot_name is not None:
        identity["snapshot_file"] = snapshot_name
    return identity


def collect_smithsonian_aws(arguments: argparse.Namespace) -> int:
    for path in (arguments.output, arguments.report):
        if path is not None and path.exists() and not arguments.force:
            raise CorpusPreparationError(f"refusing to replace output: {path}")
    requested_units = [unit.casefold() for unit in arguments.unit]
    if len(requested_units) != len(set(requested_units)):
        raise CorpusPreparationError("--unit values must be unique")
    prefixes = sorted(set(prefix.casefold() for prefix in (arguments.prefix or [])))
    if any(not re.fullmatch(r"[0-9a-f]{2}", prefix) for prefix in prefixes):
        raise CorpusPreparationError("--prefix values must be two lowercase hexadecimal digits")

    snapshot_root = arguments.snapshot_dir
    if arguments.offline_snapshot and snapshot_root is None:
        raise CorpusPreparationError("--offline-snapshot requires --snapshot-dir")
    if arguments.offline_snapshot and not snapshot_root.is_dir():
        raise CorpusPreparationError(f"offline snapshot directory is missing: {snapshot_root}")
    root_snapshot = snapshot_root / "index.txt" if snapshot_root is not None else None
    unit_index_urls, root_identity = _catalog_index(
        arguments.index_url, snapshot_path=root_snapshot,
        snapshot_name="index.txt" if snapshot_root is not None else None,
        offline=arguments.offline_snapshot, force=arguments.force,
    )
    available = {_smithsonian_unit(url): url for url in unit_index_urls}
    if len(available) != len(unit_index_urls):
        raise CorpusPreparationError("Smithsonian root index contains duplicate unit names")
    missing = sorted(set(requested_units) - available.keys())
    if missing:
        raise CorpusPreparationError(
            "Smithsonian root index lacks requested units: " + ", ".join(missing)
        )

    output_records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    unit_reports = []
    for unit in sorted(requested_units):
        unit_snapshot = snapshot_root / unit / "index.txt" if snapshot_root is not None else None
        shard_urls, unit_identity = _catalog_index(
            available[unit], snapshot_path=unit_snapshot,
            snapshot_name=f"{unit}/index.txt" if snapshot_root is not None else None,
            offline=arguments.offline_snapshot, force=arguments.force,
        )
        selected_shards = []
        for url in shard_urls:
            name = Path(urllib.parse.urlparse(url).path).name
            if not re.fullmatch(r"[0-9a-f]{2}\.txt", name.casefold()):
                raise CorpusPreparationError(f"invalid Smithsonian shard URL: {url}")
            if arguments.all_shards or name[:2].casefold() in prefixes:
                selected_shards.append(url)
        selected_shards.sort()
        if not selected_shards:
            raise CorpusPreparationError(f"no selected Smithsonian shards for unit {unit}")

        emitted = 0
        def add_record(record: dict[str, object]) -> None:
            nonlocal emitted
            if arguments.per_unit_limit is not None and emitted >= arguments.per_unit_limit:
                return
            identifier = clean_text(record.get("id"), "")
            if not identifier:
                raise CorpusPreparationError("Smithsonian record lacks an ID")
            if identifier in seen_ids:
                raise CorpusPreparationError(f"duplicate Smithsonian record ID: {identifier}")
            seen_ids.add(identifier)
            output_records.append(record)
            emitted += 1

        shard_reports = []
        for url in selected_shards:
            name = Path(urllib.parse.urlparse(url).path).name.casefold()
            shard_snapshot = snapshot_root / unit / name if snapshot_root is not None else None
            shard_reports.append(_scan_smithsonian_shard(
                url, add_record, snapshot_path=shard_snapshot,
                snapshot_name=f"{unit}/{name}" if snapshot_root is not None else None,
                offline=arguments.offline_snapshot, force=arguments.force,
            ))
        unit_reports.append({
            "emitted_records": emitted,
            "index": unit_identity,
            "shards": shard_reports,
            "unit": unit,
        })

    output_records.sort(key=lambda value: str(value.get("id")))
    write_jsonl(arguments.output, output_records, arguments.force)
    report = {
        "all_shards": arguments.all_shards,
        "format": "rawtherapee-tgmr-smithsonian-aws-snapshot-report-v1",
        "output_sha256": sha256_file(arguments.output)[0],
        "per_unit_limit": arguments.per_unit_limit,
        "prefixes": prefixes,
        "records": len(output_records),
        "root_index": root_identity,
        "units": unit_reports,
    }
    if arguments.report is not None:
        atomic_bytes(arguments.report, canonical_pretty(report), arguments.force)
    sys.stdout.buffer.write(canonical_pretty(report))
    return 0


def collect_smithsonian_api(arguments: argparse.Namespace) -> int:
    identifiers = _read_id_list(arguments.ids)
    records = []
    for identifier in identifiers:
        url = "https://api.si.edu/openaccess/api/v1.0/content/" + urllib.parse.quote(
            identifier, safe=""
        ) + "?" + urllib.parse.urlencode({"api_key": arguments.api_key})
        request = urllib.request.Request(
            url, headers={"User-Agent": "RawTherapee-TGMR-catalog-snapshot/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        raw = payload.get("response")
        if not isinstance(raw, dict):
            raise CorpusPreparationError(f"Smithsonian API response lacks record {identifier}")
        normalized = _smithsonian_record(raw)
        if normalized is not None:
            records.append(normalized)
    records.sort(key=lambda value: str(value.get("id")))
    write_jsonl(arguments.output, records, arguments.force)
    sys.stdout.buffer.write(canonical_pretty({
        "format": "rawtherapee-tgmr-smithsonian-snapshot-report-v1",
        "records": len(records), "requested": len(identifiers),
        "sha256": sha256_file(arguments.output)[0],
    }))
    return 0 if len(records) == len(identifiers) else 1


def normalize(arguments: argparse.Namespace) -> int:
    digest, _ = sha256_file(arguments.input)
    if arguments.snapshot_sha256 and digest != arguments.snapshot_sha256:
        raise CorpusPreparationError("input does not match --snapshot-sha256")
    if arguments.catalog == "openimages":
        records = normalize_openimages(arguments.input, arguments.revision, digest)
    elif arguments.catalog == "pass":
        if arguments.urls is None:
            raise CorpusPreparationError("PASS normalization requires --urls")
        records = normalize_pass(
            arguments.input, arguments.urls, arguments.revision, digest,
            arguments.archive_index,
        )
    elif arguments.catalog == "commons":
        records = normalize_commons(arguments.input, arguments.revision, digest)
    else:
        records = normalize_smithsonian(arguments.input, arguments.revision, digest)
    materialized = []
    for record in records:
        if arguments.eligible_only and record["license"] not in ACCEPTED_LICENSES:
            continue
        materialized.append(record)
        if arguments.limit is not None and len(materialized) >= arguments.limit:
            break
    write_jsonl(arguments.output, materialized, arguments.force)
    summary = {
        "catalog": arguments.catalog, "format": "rawtherapee-tgmr-catalog-normalization-v1",
        "input_sha256": digest, "records": len(materialized),
    }
    sys.stdout.buffer.write(canonical_pretty(summary))
    return 0


def _advertised_matches(path: Path, advertised: object) -> bool:
    if advertised is None:
        return True
    value = str(advertised)
    algorithm, separator, expected = value.partition(":")
    if not separator:
        raise CorpusPreparationError("advertised checksum lacks an algorithm")
    if algorithm == "md5":
        digest = hashlib.md5()  # noqa: S324 - verifies a catalog identity, not security.
        expected_bytes = expected.lower()
    elif algorithm == "md5-base64":
        digest = hashlib.md5()  # noqa: S324 - verifies a catalog identity, not security.
        try:
            expected_bytes = base64.b64decode(expected, validate=True).hex()
        except (ValueError, base64.binascii.Error) as error:
            raise CorpusPreparationError("malformed base64 catalog MD5") from error
    elif algorithm == "sha1":
        digest = hashlib.sha1()  # noqa: S324 - verifies a catalog identity, not security.
        expected_bytes = expected.lower()
    else:
        raise CorpusPreparationError(f"unsupported advertised checksum: {algorithm}")
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest() == expected_bytes


def fetch(arguments: argparse.Namespace) -> int:
    records = list(_jsonl(arguments.candidates))
    chosen = records[arguments.start:]
    if arguments.limit is not None:
        chosen = chosen[:arguments.limit]
    arguments.cache.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    failures = 0
    for record in chosen:
        if record.get("format") != CANDIDATE_FORMAT:
            raise CorpusPreparationError("fetch input has the wrong candidate format")
        filename = candidate_cache_filename(record)
        destination = safe_cache_path(arguments.cache, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        status = "authenticated-cache"
        if not destination.exists() or not _advertised_matches(
            destination, record.get("advertised_checksum")
        ):
            status = "unavailable-or-changed"
            part = destination.with_name(destination.name + ".part")
            for attempt in range(arguments.retry + 1):
                try:
                    request = urllib.request.Request(
                        str(record["original_url"]),
                        headers={"User-Agent": "RawTherapee-TGMR-candidate-fetcher/1"},
                    )
                    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as stream:
                        while block := response.read(CHUNK):
                            stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if not _advertised_matches(part, record.get("advertised_checksum")):
                        part.unlink(missing_ok=True)
                        status = "checksum-mismatch"
                        break
                    os.replace(part, destination)
                    status = "downloaded"
                    break
                except (OSError, urllib.error.URLError, urllib.error.HTTPError):
                    part.unlink(missing_ok=True)
                    if attempt < arguments.retry:
                        time.sleep(min(2 ** attempt, 8))
            part.unlink(missing_ok=True)
        if status not in ("authenticated-cache", "downloaded"):
            failures += 1
            results.append({
                "source_id": portable_id(
                    f"{record.get('catalog', '')}:{record.get('upstream_source_id', '')}"
                ),
                "status": status,
                "url": record["original_url"],
            })
            continue
        sha256, size = sha256_file(destination)
        fetched = dict(record)
        fetched.update({
            "bytes": size, "cache_filename": filename, "format": FETCHED_FORMAT,
            "sha256": sha256,
        })
        output.append(fetched)
        results.append({
            "bytes": size, "sha256": sha256,
            "source_id": portable_id(
                f"{record.get('catalog', '')}:{record.get('upstream_source_id', '')}"
            ),
            "status": status,
        })
    write_jsonl(arguments.output, output, arguments.force)
    report = {
        "failures": failures, "format": "rawtherapee-tgmr-candidate-fetch-report-v1",
        "fetched": len(output), "requested": len(chosen), "results": results,
    }
    if arguments.report:
        atomic_bytes(arguments.report, canonical_pretty(report), arguments.force)
    sys.stdout.buffer.write(canonical_pretty(report))
    return 1 if failures else 0


def read_reviews(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    output: dict[str, dict[str, object]] = {}
    for value in _jsonl(path):
        if value.get("format") != REVIEW_FORMAT:
            raise CorpusPreparationError("wrong review record format")
        source_id = clean_text(value.get("source_id"), "")
        if source_id in output:
            raise CorpusPreparationError(f"duplicate review for {source_id}")
        tags = value.get("content_tags")
        if not isinstance(tags, list) or any(tag not in CONTENT_TAGS for tag in tags):
            raise CorpusPreparationError(f"invalid content tags for {source_id}")
        if value.get("rights_review_status") not in ("approved", "rejected", "pending"):
            raise CorpusPreparationError(f"invalid rights review for {source_id}")
        if value.get("people_review_status") not in (
            "not-applicable", "approved-no-minors-or-sensitive-content", "rejected", "pending"
        ):
            raise CorpusPreparationError(f"invalid people review for {source_id}")
        if value.get("normalized_author_id") is not None:
            portable_id(clean_text(value.get("normalized_author_id"), ""))
        output[source_id] = value
    return output


def assemble(arguments: argparse.Namespace) -> int:
    classifications: dict[str, dict[str, object]] = {}
    for value in _jsonl(arguments.classifications):
        source_id = clean_text(value.get("source_id"), "")
        if source_id in classifications:
            raise CorpusPreparationError(f"duplicate classification for {source_id}")
        classifications[source_id] = value
    reviews = read_reviews(arguments.reviews)
    output: list[dict[str, object]] = []
    for candidate_value in _jsonl(arguments.candidates):
        candidate_format = candidate_value.get("format")
        if candidate_format not in (FETCHED_FORMAT, CANDIDATE_FORMAT):
            raise CorpusPreparationError(
                "assemble requires fetched or local catalog candidate records"
            )
        catalog = clean_text(candidate_value.get("catalog"), "")
        upstream = clean_text(candidate_value.get("upstream_source_id"), "")
        source_id = portable_id(f"{catalog}:{upstream}")
        classification_record = classifications.get(source_id)
        if classification_record is None:
            continue
        if classification_record.get("format") == (
            "rawtherapee-tgmr-image-proxy-classification-v1"
        ):
            raise CorpusPreparationError(
                f"proxy classification cannot be assembled as final metadata: {source_id}"
            )
        cache_filename = clean_text(classification_record.get("cache_filename"), "")
        if not cache_filename:
            cache_filename = clean_text(candidate_value.get("cache_filename"), "")
        source_path = safe_cache_path(arguments.cache, cache_filename)
        if not source_path.is_file():
            raise CorpusPreparationError(f"classified cache file is missing: {source_path}")
        source_sha256, _ = sha256_file(source_path)
        classified_sha256 = clean_text(classification_record.get("source_sha256"), "")
        if classified_sha256 and require_sha256(
            classified_sha256, "classification source_sha256"
        ) != source_sha256:
            raise CorpusPreparationError(
                f"classified source bytes changed before assembly: {source_id}"
            )
        if candidate_format == FETCHED_FORMAT:
            if require_sha256(candidate_value.get("sha256"), "fetched sha256") != source_sha256:
                raise CorpusPreparationError(
                    f"fetched source bytes changed before assembly: {source_id}"
                )
        elif not classified_sha256:
            raise CorpusPreparationError(
                f"local catalog classification lacks source_sha256: {source_id}"
            )
        local_openimages = (
            candidate_format == CANDIDATE_FORMAT
            and catalog == "openimages-cvdf-v5-boxable"
        )
        if local_openimages:
            parts = Path(cache_filename).parts
            expected_name = upstream + ".jpg"
            if (
                len(parts) != 5
                or parts[0] not in ("train", "validation", "test")
                or parts[1:4] != (upstream[0], upstream[1], upstream[2])
                or parts[4] != expected_name
            ):
                raise CorpusPreparationError(
                    f"local Open Images cache layout is malformed: {source_id}"
                )
            reconstruction_url = (
                "https://open-images-dataset.s3.amazonaws.com/"
                f"{parts[0]}/{expected_name}"
            )
        else:
            reconstruction_url = candidate_value["original_url"]
        review = reviews.get(source_id, {})
        rights_status = str(review.get("rights_review_status") or candidate_value["rights_review_status"])
        tags = sorted(set(review.get("content_tags") or []))
        people_status = str(review.get("people_review_status") or (
            "pending" if "people" in tags else "not-applicable"
        ))
        classification = classification_record.get("classification")
        if not isinstance(classification, dict):
            raise CorpusPreparationError(f"classification object missing for {source_id}")
        output.append({
            # Open Images metadata describes the Flickr original, whereas the
            # CVDF archive is a <=1024-pixel mirror rendition.  Its exact
            # SHA-256 is the authoritative source identity.
            "advertised_checksum": (
                None if local_openimages else candidate_value.get("advertised_checksum")
            ),
            "archive_fallbacks": candidate_value.get("archive_fallbacks", []),
            "author": candidate_value["author"],
            "author_id": portable_id(clean_text(
                review.get("normalized_author_id"), str(candidate_value["author_id"])
            )),
            "author_url": candidate_value["author_url"],
            "cache_filename": cache_filename,
            "catalog": {
                "name": catalog,
                "revision": candidate_value["catalog_revision"],
                "snapshot_sha256": candidate_value["catalog_snapshot_sha256"],
            },
            "classification": classification,
            "content_tags": tags,
            "decoded_pixel_sha256": classification_record["decoded_pixel_sha256"],
            "fallback_urls": [],
            "file_type": classification_record["file_type"],
            "format": SOURCE_FORMAT,
            "height": classification_record["height"],
            "icc_identity": classification_record["icc_identity"],
            "landing_page": candidate_value["landing_page"],
            "license": candidate_value["license"],
            "license_url": candidate_value["license_url"],
            "orientation": classification_record["orientation"],
            "original_url": reconstruction_url,
            "patch_coordinates": [],
            "patch_sampling_seed": "0x" + hashlib.sha256(source_id.encode()).hexdigest()[:16],
            "people_review_status": people_status,
            "rights": {
                "evidence_revision": str(review.get("rights_evidence_revision") or candidate_value["catalog_revision"]),
                "evidence_sha256": str(review.get("rights_evidence_sha256") or candidate_value["catalog_snapshot_sha256"]),
                "evidence_url": str(review.get("rights_evidence_url") or candidate_value["rights_evidence_url"]),
                "review_status": rights_status,
            },
            "selected": False,
            "selection_status": "candidate-reviewed" if rights_status == "approved" else "candidate-pending-review",
            "sha256": source_sha256,
            "source_id": source_id,
            "split": "unassigned",
            "title": candidate_value["title"],
            "upstream_flickr_id": candidate_value.get("upstream_flickr_id"),
            "upstream_source_id": upstream,
            "width": classification_record["width"],
        })
    write_jsonl(arguments.output, output, arguments.force)
    sys.stdout.buffer.write(canonical_pretty({
        "format": "rawtherapee-tgmr-source-assembly-v1", "records": len(output),
    }))
    return 0


def release_manifest(arguments: argparse.Namespace) -> int:
    artifacts = []
    for role, path in (
        ("source-manifest", arguments.source_manifest),
        ("tgpc", arguments.tgpc),
        ("tgpc-gzip", arguments.tgpc_gzip),
        ("attribution-notice", arguments.attribution),
        ("corpus-statistics", arguments.statistics),
        ("rights-report", arguments.rights_report),
    ):
        digest, size = sha256_file(path)
        artifacts.append({
            "bytes": size, "filename": path.name, "role": role, "sha256": digest,
        })
    value = {
        "artifacts": artifacts,
        "corpus_id": "tgmr-corpus-v1",
        "format": "rawtherapee-tgmr-corpus-release-manifest-v1",
        "github_release_url": arguments.github_release_url,
        "zenodo_doi": arguments.zenodo_doi,
    }
    atomic_bytes(arguments.output, canonical_pretty(value), arguments.force)
    sys.stdout.buffer.write(canonical_pretty(value))
    return 0


def merge_jsonl(arguments: argparse.Namespace) -> int:
    records = []
    identities = set()
    expected_format = None
    for path in arguments.inputs:
        for value in _jsonl(path):
            record_format = value.get("format")
            if expected_format is None:
                expected_format = record_format
            if record_format != expected_format:
                raise CorpusPreparationError("cannot merge different JSONL record formats")
            identity = (
                value.get("source_id")
                or f"{value.get('catalog', '')}:{value.get('upstream_source_id', '')}"
            )
            if identity in identities:
                raise CorpusPreparationError(f"duplicate merged identity: {identity}")
            identities.add(identity)
            records.append(value)
    catalog_order = {
        "openimages-cvdf-v5-boxable": 0, "pass-v3": 1,
        "wikimedia-commons": 2, "smithsonian-open-access": 3,
    }
    def key(value: dict[str, object]) -> tuple[int, str]:
        catalog = value.get("catalog")
        if isinstance(catalog, dict):
            catalog = catalog.get("name")
        return catalog_order.get(str(catalog), 99), str(
            value.get("source_id") or value.get("upstream_source_id") or ""
        )
    records.sort(key=key)
    write_jsonl(arguments.output, records, arguments.force)
    sys.stdout.buffer.write(canonical_pretty({
        "format": "rawtherapee-tgmr-jsonl-merge-report-v1",
        "record_format": expected_format, "records": len(records),
        "sha256": sha256_file(arguments.output)[0],
    }))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    snapshot_parser = commands.add_parser("snapshot")
    snapshot_parser.add_argument("url")
    snapshot_parser.add_argument("output", type=Path)
    snapshot_parser.add_argument("--sha256")
    snapshot_parser.add_argument("--force", action="store_true")
    snapshot_parser.set_defaults(function=snapshot)

    commons_parser = commands.add_parser("collect-commons")
    commons_parser.add_argument("ids", type=Path)
    commons_parser.add_argument("output", type=Path)
    commons_parser.add_argument("--force", action="store_true")
    commons_parser.set_defaults(function=collect_commons)

    smithsonian_parser = commands.add_parser(
        "collect-smithsonian",
        help="collect Smithsonian records anonymously from the public AWS metadata",
    )
    smithsonian_parser.add_argument("output", type=Path)
    smithsonian_parser.add_argument("--unit", action="append", required=True)
    shard_selection = smithsonian_parser.add_mutually_exclusive_group(required=True)
    shard_selection.add_argument("--prefix", action="append")
    shard_selection.add_argument("--all-shards", action="store_true")
    smithsonian_parser.add_argument("--per-unit-limit", type=int)
    smithsonian_parser.add_argument("--index-url", default=SMITHSONIAN_AWS_INDEX)
    smithsonian_parser.add_argument("--snapshot-dir", type=Path)
    smithsonian_parser.add_argument("--offline-snapshot", action="store_true")
    smithsonian_parser.add_argument("--report", type=Path)
    smithsonian_parser.add_argument("--force", action="store_true")
    smithsonian_parser.set_defaults(function=collect_smithsonian_aws)

    smithsonian_api_parser = commands.add_parser(
        "collect-smithsonian-api",
        help="optional diagnostic collector for an explicit Smithsonian ID list",
    )
    smithsonian_api_parser.add_argument("ids", type=Path)
    smithsonian_api_parser.add_argument("output", type=Path)
    smithsonian_api_parser.add_argument("--api-key", required=True)
    smithsonian_api_parser.add_argument("--force", action="store_true")
    smithsonian_api_parser.set_defaults(function=collect_smithsonian_api)

    normalize_parser = commands.add_parser("normalize")
    normalize_parser.add_argument("catalog", choices=("openimages", "pass", "commons", "smithsonian"))
    normalize_parser.add_argument("input", type=Path)
    normalize_parser.add_argument("output", type=Path)
    normalize_parser.add_argument("--revision", required=True)
    normalize_parser.add_argument("--snapshot-sha256")
    normalize_parser.add_argument("--urls", type=Path)
    normalize_parser.add_argument("--archive-index", type=Path)
    normalize_parser.add_argument("--limit", type=int)
    normalize_parser.add_argument("--eligible-only", action="store_true")
    normalize_parser.add_argument("--force", action="store_true")
    normalize_parser.set_defaults(function=normalize)

    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("candidates", type=Path)
    fetch_parser.add_argument("cache", type=Path)
    fetch_parser.add_argument("output", type=Path)
    fetch_parser.add_argument("--start", type=int, default=0)
    fetch_parser.add_argument("--limit", type=int)
    fetch_parser.add_argument("--retry", type=int, default=2)
    fetch_parser.add_argument("--report", type=Path)
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.set_defaults(function=fetch)

    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument("candidates", type=Path)
    assemble_parser.add_argument("classifications", type=Path)
    assemble_parser.add_argument("cache", type=Path)
    assemble_parser.add_argument("output", type=Path)
    assemble_parser.add_argument("--reviews", type=Path)
    assemble_parser.add_argument("--force", action="store_true")
    assemble_parser.set_defaults(function=assemble)

    release_parser = commands.add_parser("release-manifest")
    release_parser.add_argument("source_manifest", type=Path)
    release_parser.add_argument("tgpc", type=Path)
    release_parser.add_argument("tgpc_gzip", type=Path)
    release_parser.add_argument("attribution", type=Path)
    release_parser.add_argument("statistics", type=Path)
    release_parser.add_argument("rights_report", type=Path)
    release_parser.add_argument("output", type=Path)
    release_parser.add_argument("--zenodo-doi", required=True)
    release_parser.add_argument("--github-release-url", required=True)
    release_parser.add_argument("--force", action="store_true")
    release_parser.set_defaults(function=release_manifest)

    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("output", type=Path)
    merge_parser.add_argument("inputs", nargs="+", type=Path)
    merge_parser.add_argument("--force", action="store_true")
    merge_parser.set_defaults(function=merge_jsonl)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if getattr(arguments, "limit", None) is not None and arguments.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if getattr(arguments, "start", 0) < 0 or getattr(arguments, "retry", 0) < 0:
        raise SystemExit("--start and --retry must be non-negative")
    if getattr(arguments, "per_unit_limit", None) is not None and arguments.per_unit_limit < 0:
        raise SystemExit("--per-unit-limit must be non-negative")
    if getattr(arguments, "sha256", None):
        require_sha256(arguments.sha256, "--sha256")
    if getattr(arguments, "snapshot_sha256", None):
        require_sha256(arguments.snapshot_sha256, "--snapshot-sha256")
    return int(arguments.function(arguments))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CorpusPreparationError as error:
        print(f"prepare_corpus.py: {error}", file=sys.stderr)
        raise SystemExit(2)
