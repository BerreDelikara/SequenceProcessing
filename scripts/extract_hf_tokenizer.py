#!/usr/bin/env python3
"""Extracts vocab.json + merges.txt + special_tokens.txt from a HuggingFace
``tokenizer.json``. Used as an offline preprocessing step for TabiBERT and
Mursit, so the Java implementation only has to parse simple flat formats.

Usage:
    python3 scripts/extract_hf_tokenizer.py \\
        --hf-repo boun-tabilab/TabiBERT \\
        --output-dir src/main/resources/tabi-bert

The script will:
    1. Download tokenizer.json from the HF repo (cached via urllib).
    2. Write model.vocab to <output-dir>/vocab.json (flat {token: id}).
    3. Write model.merges to <output-dir>/merges.txt (one merge per line,
       rank = line number).
    4. Write added_tokens (special tokens) to <output-dir>/special_tokens.txt
       as "id<TAB>token" per line.

The original tokenizer.json is intentionally not kept — we only ship the
extracted, simpler forms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def fetch_tokenizer_json(hf_repo: str) -> dict:
    url = f"https://huggingface.co/{hf_repo}/raw/main/tokenizer.json"
    print(f"Fetching {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def write_vocab(model: dict, output_dir: str) -> None:
    vocab = model.get("vocab", {})
    path = os.path.join(output_dir, "vocab.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Wrote {len(vocab):,} vocab entries -> {path}", file=sys.stderr)


def write_merges(model: dict, output_dir: str) -> None:
    merges = model.get("merges", [])
    path = os.path.join(output_dir, "merges.txt")
    with open(path, "w", encoding="utf-8") as f:
        # HF tokenizer.json stores merges either as a list of strings like
        # "Ġ a" or as a list of two-element arrays ["Ġ", "a"]. Normalize.
        for merge in merges:
            if isinstance(merge, list):
                f.write(f"{merge[0]} {merge[1]}\n")
            else:
                f.write(f"{merge}\n")
    print(f"Wrote {len(merges):,} merge rules -> {path}", file=sys.stderr)


def write_special_tokens(tok: dict, output_dir: str) -> None:
    added = tok.get("added_tokens", [])
    path = os.path.join(output_dir, "special_tokens.txt")
    with open(path, "w", encoding="utf-8") as f:
        for entry in added:
            tid = entry.get("id")
            content = entry.get("content", "")
            if tid is None or not content:
                continue
            f.write(f"{tid}\t{content}\n")
    print(f"Wrote {len(added):,} special tokens -> {path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", required=True,
                        help='e.g. "boun-tabilab/TabiBERT"')
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write vocab.json/merges.txt/special_tokens.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = fetch_tokenizer_json(args.hf_repo)
    model = tok.get("model", {})
    if model.get("type") != "BPE":
        sys.exit(f"Expected BPE tokenizer, got type={model.get('type')!r}")
    write_vocab(model, args.output_dir)
    write_merges(model, args.output_dir)
    write_special_tokens(tok, args.output_dir)


if __name__ == "__main__":
    main()
