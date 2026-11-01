"""Two-level navigation with irreversible pruning and inspectable stage traces."""

from .judges import validate_scores
from .models import Candidate, text, validate_corpus


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def ranked(candidates, scores):
    # Stable corpus order resolves ties, independent of dictionary response order.
    return sorted(candidates, key=lambda c: -scores[c.id])


def retrieve(query, documents, judge, *, document_budget=2, passage_budget=3,
             rerank=0, weight_by_document=False):
    text(query, "query")
    if not query.split():
        raise ValueError("query needs visible terms")
    documents = validate_corpus(documents)
    positive(document_budget, "document budget")
    positive(passage_budget, "passage budget")
    if isinstance(rerank, bool) or not isinstance(rerank, int) or not 0 <= rerank <= 255:
        raise ValueError("rerank must be an integer from 0 to 255")
    if not isinstance(weight_by_document, bool):
        raise ValueError("weight_by_document must be boolean")

    document_candidates = [Candidate(d.id, f"{d.title}\n{d.summary}") for d in documents]
    document_scores = validate_scores(judge.score(query, document_candidates, "documents"), document_candidates)
    selected = ranked(document_candidates, document_scores)[:document_budget]
    selected_ids = {c.id for c in selected}
    # Only selected branches contribute passage text to the second judge request.
    expanded = [(d, p) for d in documents if d.id in selected_ids for p in d.passages]
    passage_candidates = [Candidate(p.id, f"{d.title}\n{p.text}") for d, p in expanded]
    passage_scores = validate_scores(judge.score(query, passage_candidates, "passages"), passage_candidates)
    parent = {p.id: d for d, p in expanded}
    final_scores = {pid: score * (document_scores[parent[pid].id] if weight_by_document else 1)
                    for pid, score in passage_scores.items()}
    retained = ranked(passage_candidates, final_scores)[:passage_budget]
    choice_scores = None
    shortlist = retained[:rerank]
    if len(shortlist) >= 2:
        choice_scores = validate_scores(judge.choose(query, shortlist), shortlist, distribution=True)
        retained = ranked(shortlist, choice_scores) + retained[len(shortlist):]
    passage_by_id = {p.id: p for _, p in expanded}
    return {"query": query, "judge": judge.label,
            "quality_evidence": False,
            "results": [{"id": c.id, "document_id": parent[c.id].id,
                         "text": passage_by_id[c.id].text,
                         "binary_score": passage_scores[c.id],
                         "ranking_score": final_scores[c.id],
                         "choice_probability": choice_scores.get(c.id) if choice_scores else None}
                        for c in retained],
            "trace": {"document_scores": document_scores,
                      "retained_documents": [c.id for c in selected],
                      "pruned_documents": [d.id for d in documents if d.id not in selected_ids],
                      "scored_passages": [c.id for c in passage_candidates],
                      "passage_scores": passage_scores,
                      "choice_scores": choice_scores,
                      "weight_by_document": weight_by_document}}
