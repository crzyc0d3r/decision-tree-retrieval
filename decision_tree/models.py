"""Validated, explicit document hierarchy and judge candidates."""

from dataclasses import dataclass
import json
from pathlib import Path


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str


@dataclass(frozen=True)
class Passage:
    id: str
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    summary: str
    passages: tuple[Passage, ...]


def validate_corpus(documents):
    if not isinstance(documents, (tuple, list)) or not documents:
        raise ValueError("corpus must contain documents")
    seen = set()
    for document in documents:
        if not isinstance(document, Document):
            raise ValueError("expected Document")
        for field in ("id", "title", "summary"):
            text(getattr(document, field), f"document {field}")
        if document.id in seen:
            raise ValueError("all document and passage ids must be unique")
        seen.add(document.id)
        if not isinstance(document.passages, (tuple, list)) or not document.passages:
            raise ValueError("each document must have passages")
        for passage in document.passages:
            if not isinstance(passage, Passage):
                raise ValueError("expected Passage")
            text(passage.id, "passage id")
            text(passage.text, "passage text")
            if passage.id in seen:
                raise ValueError("all document and passage ids must be unique")
            seen.add(passage.id)
    return tuple(documents)


def load_corpus(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("corpus exceeds the 4 MiB example limit")
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("corpus JSON must be a list")
    try:
        documents = tuple(Document(
            id=d["id"], title=d["title"], summary=d["summary"],
            passages=tuple(Passage(id=p["id"], text=p["text"]) for p in d["passages"]),
        ) for d in value)
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid document or passage structure") from exc
    return validate_corpus(documents)
