#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Import pinned OSS dictionary data, not Mozc code or Google's proprietary data."""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "18e76511af9371a9a4409b1d13c3e30884b53e9c"
BASE = f"https://raw.githubusercontent.com/google/mozc/{REVISION}/"
ROOT = Path(__file__).resolve().parent.parent


def fetch(relative, cache):
    path = cache / relative.replace("/", "_")
    if not path.exists():
        print(f"Downloading {relative}", flush=True)
        with urllib.request.urlopen(BASE + relative, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    return path.read_bytes()


def convert(line):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) not in (5, 6):
        raise ValueError("expected reading, left ID, right ID, cost, surface")
    reading, left, right, cost, surface = fields[:5]
    left, right, cost = int(left), int(right), int(cost)
    if not reading or not surface or len(reading) > 128 or len(surface) > 128:
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in reading + surface):
        return None
    # The original IDs and cost are retained in POS metadata. This is an
    # approximate cost-to-weight mapping, not Mozc's trained transition model.
    weight = max(1, min(5000, (16000 - cost) // 4))
    tag = (":" + fields[5]) if len(fields) == 6 and fields[5] else ""
    return f"{reading}\t{surface}\tmozc:{left}:{right}:{cost}{tag}\t{weight}\n"


def main():
    cache = ROOT / "dictionaries/.cache/mozc"
    cache.mkdir(parents=True, exist_ok=True)
    dictionaries = ROOT / "dictionaries"
    records = []
    id_path = "src/data/dictionary_oss/id.def"
    ids = fetch(id_path, cache)
    (dictionaries / "mozc-id.def").write_bytes(ids)
    records.append({"path": id_path, "sha256": hashlib.sha256(ids).hexdigest()})
    notice = fetch("LICENSE", cache).decode("utf-8") + "\n\n" + fetch("src/data/dictionary_oss/README.txt", cache).decode("utf-8")
    (dictionaries / "MOZC-NOTICES.txt").write_text(notice, encoding="utf-8")
    count = skipped = 0
    destination = dictionaries / "mozc-ja.tsv"
    temporary = destination.with_suffix(".tsv.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"# Derived from google/mozc {REVISION}; see MOZC-NOTICES.txt\n")
        for number in range(10):
            relative = f"src/data/dictionary_oss/dictionary{number:02}.txt"
            data = fetch(relative, cache)
            rows = 0
            for line in data.decode("utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                row = convert(line)
                if row is None:
                    skipped += 1
                else:
                    output.write(row)
                    count += 1
                    rows += 1
            records.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "entries": rows})
            print(f"Imported {relative}: {rows}", flush=True)
        for name in ["sample-ja.tsv", "extended-ja.tsv", "english.tsv", "phrases-ja.tsv"]:
            text = (dictionaries / name).read_text(encoding="utf-8")
            output.write(text + "\n")
    temporary.replace(destination)
    manifest = {"repository": "https://github.com/google/mozc", "revision": REVISION,
                "imported_rows": count, "excluded_out_of_bounds_or_control": skipped,
                "files": records, "tsv_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (dictionaries / "mozc-source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {count} Mozc rows; excluded {skipped}; appended NextIME vocabulary.", flush=True)


if __name__ == "__main__":
    main()
