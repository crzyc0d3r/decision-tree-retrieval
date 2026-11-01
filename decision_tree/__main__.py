"""Retrieve evidence and print the complete navigation trace as JSON."""

import argparse
import json
from pathlib import Path
import sys

from .judges import JevJudge, JudgeError, LexicalJudge
from .models import load_corpus
from .navigate import retrieve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="How long can I restore a deleted object?")
    parser.add_argument("--corpus", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples" / "corpus.json")
    parser.add_argument("--judge", choices=("lexical", "jev"), default="lexical")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--documents", type=int, default=2)
    parser.add_argument("--passages", type=int, default=3)
    parser.add_argument("--rerank", type=int, default=0,
                        help="Rerank at most this many retained passages (0 disables)")
    parser.add_argument("--weight-by-document", action="store_true")
    args = parser.parse_args(argv)
    try:
        documents = load_corpus(args.corpus)
        judge = JevJudge(model=args.model) if args.judge == "jev" else LexicalJudge()
        result = retrieve(args.query, documents, judge, document_budget=args.documents,
                          passage_budget=args.passages, rerank=args.rerank,
                          weight_by_document=args.weight_by_document)
        result["judge_calls"] = judge.calls
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, JudgeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
