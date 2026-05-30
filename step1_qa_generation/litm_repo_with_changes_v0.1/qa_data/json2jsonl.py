#!/usr/bin/env python3
"""
json2jsonl.py — Convert JSON to JSONL (NDJSON).

- Accepts a JSON file that's either:
  * a list of objects, or
  * a single object (will be written as one line), or
  * an object containing a list under --array-key.

- Reads from stdin / writes to stdout by default.
- Transparently handles .gz input/output by file extension.
- Optionally select a subset of fields.

Usage:
  json2jsonl.py -i data.json -o data.jsonl
  json2jsonl.py -i data.json.gz -o data.jsonl.gz
  cat data.json | json2jsonl.py > data.jsonl
  json2jsonl.py -i nested.json --array-key items -o out.jsonl
  json2jsonl.py -i data.json -o out.jsonl --fields id name score
"""

import argparse
import gzip
import io
import json
import os
import sys
from typing import Iterable, Any, Dict, List

def _open_read(path: str):
    if path == "-" or not path:
        return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def _open_write(path: str):
    if path == "-" or not path:
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8")
    return open(path, "w", encoding="utf-8", newline="\n")

def iter_records(obj: Any, array_key: str | None) -> Iterable[Dict]:
    """
    Yield dict-like records from the loaded JSON object.
    """
    if array_key:
        if not isinstance(obj, dict) or array_key not in obj:
            raise ValueError(f"--array-key '{array_key}' not found in top-level object")
        obj = obj[array_key]

    if isinstance(obj, list):
        for rec in obj:
            yield rec
    else:
        # Single object → single line
        yield obj

def project_fields(rec: Dict, fields: List[str] | None) -> Dict:
    if not fields:
        return rec
    return {k: rec.get(k) for k in fields}

def main():
    ap = argparse.ArgumentParser(description="Convert JSON to JSONL (one JSON object per line).")
    ap.add_argument("-i", "--input", default="-", help="Input JSON file (or - for stdin). Supports .gz")
    ap.add_argument("-o", "--output", default="-", help="Output JSONL file (or - for stdout). Supports .gz")
    ap.add_argument("--array-key", help="If the top-level object contains a list under this key, use it.")
    ap.add_argument("--fields", nargs="*", help="Optional subset of fields to keep in each record.")
    ap.add_argument("--ensure-ascii", action="store_true", help="Escape non-ASCII characters.")
    args = ap.parse_args()

    try:
        with _open_read(args.input) as fin:
            data = json.load(fin)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ JSON parse error: {e}")
    except FileNotFoundError:
        sys.exit(f"❌ Input not found: {args.input}")

    try:
        with _open_write(args.output) as fout:
            for rec in iter_records(data, args.array_key):
                # Accept any JSON-serializable record; force to dict if you prefer strictness
                if args.fields and isinstance(rec, dict):
                    rec = project_fields(rec, args.fields)
                line = json.dumps(rec, ensure_ascii=args.ensure_ascii)
                fout.write(line + "\n")
    except Exception as e:
        sys.exit(f"❌ Write error: {e}")

    if args.output != "-":
        print(f"✅ Wrote JSONL to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()