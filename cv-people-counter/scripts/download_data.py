#!/usr/bin/env python3
"""Download and verify evaluation datasets (selective, disk-friendly).

    python scripts/download_data.py mot17      # 5 MOT17 sequences (FRCNN)
    python scripts/download_data.py mot20      # MOT20-01 (extreme crowd)
    python scripts/download_data.py visdrone   # 1 VisDrone-MOT val sequence
    python scripts/download_data.py --all      # everything above
    python scripts/download_data.py --verify   # integrity check only

MOT17/MOT20 are fetched with HTTP-range selective zip extraction (remotezip):
only the members of the sequences we evaluate are downloaded (~400 MB instead
of the full ~6 GB archives). VisDrone comes from the official distribution
(Google Drive); it is optional — the pipeline skips it when absent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
MOT20_URL = "https://motchallenge.net/data/MOT20.zip"

MOT17_SEQS = ["02", "04", "05", "09", "11"]          # FRCNN detector variant
MOT20_SEQS = ["01"]

DATA = REPO / "data"


def _extract_members(zip_url: str, patterns: list[str], dest: Path) -> int:
    """Extract zip members matching any prefix pattern — one HTTP range
    request per contiguous byte span instead of one request per file.

    Members of one sequence are stored consecutively in the archive, so a
    handful of large range requests (each tens of MB) is ~100x faster than
    remotezip's per-member fetches on high-latency links.
    """
    import struct
    import zlib

    import requests
    from remotezip import RemoteZip

    with RemoteZip(zip_url) as zf:
        infos = sorted(
            (i for i in zf.infolist()
             if any(i.filename.startswith(p) for p in patterns)
             and not i.filename.endswith("/")
             and "/det/" not in i.filename),   # raw detections not needed
            key=lambda i: i.header_offset,
        )
        print(f"[download] {zip_url}: {len(infos)} matching members")
        if not infos:
            return 0

        # merge members into byte spans (bridge small gaps such as the
        # skipped det/det.txt, but not huge unrelated chunks)
        GAP_TOLERANCE = 16 * 1024 * 1024
        spans: list[list[int]] = []
        for info in infos:
            start = info.header_offset
            end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size
            if spans and start - spans[-1][1] < GAP_TOLERANCE:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])

        total_bytes = sum(e - s for s, e in spans)
        print(f"[download] fetching {len(spans)} span(s), "
              f"~{total_bytes / 1e6:.0f} MB total", flush=True)

        n = 0
        for span_start, span_end in spans:
            resp = requests.get(
                zip_url,
                headers={"Range": f"bytes={span_start}-{span_end}"},
                stream=True, timeout=600,
            )
            resp.raise_for_status()
            buf = resp.content
            span_members = [i for i in infos
                            if span_start <= i.header_offset < span_end]
            for info in span_members:
                off = info.header_offset - span_start
                sig, = struct.unpack_from("<I", buf, off)
                if sig != 0x04034B50:
                    raise RuntimeError(
                        f"bad local header for {info.filename} "
                        f"(sig={sig:#x}) — span parsing out of sync")
                fn_len, extra_len = struct.unpack_from("<HH", buf, off + 26)
                data_start = off + 30 + fn_len + extra_len
                data = buf[data_start:data_start + info.compress_size]
                if len(data) < info.compress_size:
                    raise RuntimeError(f"truncated member {info.filename}")
                if info.compress_type == 8:
                    data = zlib.decompressobj(-15).decompress(data)
                out_path = dest / info.filename
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                n += 1
                if n % 250 == 0:
                    print(f"  ... {n}/{len(infos)} files", flush=True)
        print(f"[download] extracted {n} files")
        return n


def download_mot17(only: list[str] | None = None) -> None:
    seqs = only or MOT17_SEQS
    patterns = [f"MOT17/train/MOT17-{s}-FRCNN/" for s in seqs]
    _extract_members(MOT17_URL, patterns, DATA)


def download_mot20(only: list[str] | None = None) -> None:
    seqs = only or MOT20_SEQS
    patterns = [f"MOT20/train/MOT20-{s}/" for s in seqs]
    _extract_members(MOT20_URL, patterns, DATA)


def download_visdrone() -> bool:
    """Fetch one VisDrone2019-MOT val sequence from the official mirror.

    Returns True on success. The official Google-Drive share id is parsed at
    runtime from the VisDrone-Dataset README (no hard-coded ids). If Google
    Drive is unreachable, the caller proceeds without aerial data — the
    pipeline skips the VisDrone entry gracefully.
    """
    import re
    import urllib.request
    import zipfile

    readme_url = ("https://raw.githubusercontent.com/VisDrone/"
                  "VisDrone-Dataset/master/README.md")
    print("[download] parsing VisDrone README for the MOT-val link ...")
    try:
        text = urllib.request.urlopen(readme_url, timeout=30).read().decode()
    except Exception as exc:
        print(f"[download] cannot fetch README ({exc}) — skipping VisDrone")
        return False
    # look for the Drive link on the line mentioning 'MOT' + 'val'
    drive_id = None
    for line in text.splitlines():
        low = line.lower()
        if "drive.google" in low and "mot" in low and "val" in low \
                and "test" not in low:
            m = re.search(r"[-\w]{25,}", line.split("drive.google", 1)[1])
            if m:
                drive_id = m.group(0)
                break
    if not drive_id:
        print("[download] MOT-val Drive link not found — skipping VisDrone")
        return False

    import gdown
    archive = DATA / "VisDrone" / "VisDrone2019-MOT-val.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] VisDrone2019-MOT-val.zip via Google Drive id={drive_id}")
    try:
        gdown.download(id=drive_id, output=str(archive), quiet=False)
    except Exception as exc:
        print(f"[download] gdown failed ({exc}) — skipping VisDrone")
        return False
    if not archive.is_file() or archive.stat().st_size < 10 ** 7:
        print("[download] VisDrone archive missing/too small — skipping")
        return False
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist()
                   if m.startswith("VisDrone2019-MOT-val/sequences/0000001")
                   or m.startswith("VisDrone2019-MOT-val/annotations/0000001")]
        zf.extractall(DATA, members=members)
    archive.unlink()                                   # keep only the sequence
    return True


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _verify_mot(dataset: str, seqs: list[str]) -> list[str]:
    from people_counter.datasets.mot import parse_seqinfo

    problems = []
    for seq in seqs:
        if dataset == "MOT17":
            d = DATA / "MOT17" / "train" / f"MOT17-{seq}-FRCNN"
        else:
            d = DATA / "MOT20" / "train" / f"MOT20-{seq}"
        info_ini = d / "seqinfo.ini"
        gt = d / "gt" / "gt.txt"
        if not info_ini.is_file():
            problems.append(f"{d.name}: seqinfo.ini missing")
            continue
        if not gt.is_file():
            problems.append(f"{d.name}: gt/gt.txt missing")
        info = parse_seqinfo(info_ini)
        img_dir = d / info.get("imDir", "img")
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0
        expected = int(info.get("seqLength", "0"))
        if n_img != expected:
            problems.append(f"{d.name}: {n_img} images, expected {expected}")
        else:
            print(f"[verify] {d.name}: OK ({n_img} frames, "
                  f"{info.get('imWidth')}x{info.get('imHeight')}, "
                  f"{info.get('frameRate')} fps)")
    return problems


def verify() -> int:
    problems = []
    problems += _verify_mot("MOT17", MOT17_SEQS)
    problems += _verify_mot("MOT20", MOT20_SEQS)
    vd = DATA / "VisDrone" / "VisDrone2019-MOT-val" / "sequences"
    if vd.is_dir():
        seqs = sorted(p for p in vd.iterdir() if p.is_dir())
        print(f"[verify] VisDrone: {len(seqs)} sequence(s): "
              f"{[p.name for p in seqs]}")
        for s in seqs:
            n = len(list(s.glob('*.jpg')))
            ann = DATA / "VisDrone" / "VisDrone2019-MOT-val" / "annotations" \
                / f"{s.name}.txt"
            print(f"[verify] VisDrone/{s.name}: {n} frames, "
                  f"gt={'yes' if ann.is_file() else 'MISSING'}")
            if not ann.is_file():
                problems.append(f"VisDrone/{s.name}: annotation missing")
    else:
        print("[verify] VisDrone: not downloaded (optional)")
    if problems:
        print("\n".join(f"[verify] PROBLEM: {p}" for p in problems))
        return 1
    print("[verify] all good")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?",
                        choices=["mot17", "mot20", "visdrone", "all"],
                        help="what to download")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", action="append",
                        help="limit MOT download to these sequence numbers "
                             "(e.g. --only 02 --only 04)")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        raise SystemExit(verify())

    what = args.dataset or ("all" if args.all else None)
    if not what:
        parser.error("choose a dataset (mot17 | mot20 | visdrone | all) "
                     "or --verify")
    if what in ("mot17", "all"):
        download_mot17(args.only)
    if what in ("mot20", "all"):
        download_mot20(args.only)
    if what in ("visdrone", "all"):
        download_visdrone()
    raise SystemExit(verify())


if __name__ == "__main__":
    main()
