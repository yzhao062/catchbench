"""Scan distributed data files for direct identifier forms removed from PRE text."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "email_addresses": re.compile(
        r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
    ),
    "linkedin_profile_urls": re.compile(
        r"(?i)https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company|posts)/[^\s\"<>\\)\]]+"
    ),
    "other_profile_urls": re.compile(
        r"(?i)https?://(?:www\.)?(?:"
        r"(?:twitter\.com|x\.com|instagram\.com|facebook\.com|tiktok\.com)/@?[A-Za-z0-9_.-]+"
        r"|youtube\.com/@[A-Za-z0-9_.-]+"
        r"|bsky\.app/profile/[A-Za-z0-9_.-]+"
        r")"
    ),
    "home_paths_unix": re.compile(r"(?i)(?<![\w/])/home/[A-Za-z0-9._-]+"),
    "home_paths_windows": re.compile(r"(?i)(?<![\w\\])C:\\Users\\[A-Za-z0-9._-]+"),
    "handles": re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_.-]{1,38}"),
}
PHONE_PATTERNS = (
    re.compile(r"(?<!\w)\+\d(?:[ .()-]*\d){7,14}(?!\w)"),
    re.compile(r"(?<!\d)(?:\(\d{3}\)|\d{3})[-. ]\d{3}[-. ]\d{4}(?!\d)"),
)


def scan(root: Path) -> dict[str, dict[str, int]]:
    hits: dict[str, list[str]] = {name: [] for name in (*PATTERNS, "phone_numbers")}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            hits[name].extend(match.group(0) for match in pattern.finditer(text))
        seen_phone_spans: set[tuple[int, int]] = set()
        for pattern in PHONE_PATTERNS:
            for match in pattern.finditer(text):
                if match.span() not in seen_phone_spans:
                    hits["phone_numbers"].append(match.group(0))
                    seen_phone_spans.add(match.span())
    return {
        name: {"occurrences": len(values), "distinct": len(set(values))}
        for name, values in hits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(scan(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
