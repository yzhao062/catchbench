"""Print and verify the exact Hugging Face revisions recorded by the board."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from auditablebench.corpora import CORPUS_REVISIONS, verify_corpus_heads  # noqa: E402


def main() -> None:
    resolved = verify_corpus_heads()
    for corpus in CORPUS_REVISIONS:
        print(f"{corpus.name}\t{corpus.repo_id}\t{resolved[corpus.name]}")


if __name__ == "__main__":
    main()
