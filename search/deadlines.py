"""Conservative ranking signals for explicit submission dates, not final extraction."""
from __future__ import annotations

import re
from schemas import SearchChunk

MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
DATE_PATTERN = (r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
                r'\d{1,2}-[A-Za-z]{3}-\d{4}|'
                + MONTH + r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|'
                r'\d{1,2}(?:st|nd|rd|th)?\s+' + MONTH + r'\.?\s+\d{4})\b')
LABEL = re.compile(r'\b(?:due\s*date|closing\s*date|submission\s*deadline|'
                   r'proposal\s*deadline|bid\s*deadline|solicitation\s*due)\b', re.I)
DATE = re.compile(DATE_PATTERN, re.I)


def current_deadline_intent(query: str) -> bool:
    """Exclude historical and other milestone questions from this ranking policy."""
    return bool(re.search(r'\b(deadline|due date|closing date)\b', query, re.I)) and not bool(
        re.search(r'\b(original|initial|previous|before|historical|questions?|clarifications?|'
                  r'meeting|delivery|payment|registration)\b', query, re.I))


def dated_deadline_priority(chunk: SearchChunk) -> tuple[int, int]:
    """Prefer dated submission evidence, then explicit numbered deadline revisions.

    Merely being a newer addendum is insufficient: a date must follow a deadline
    label, and a revision cue must occur in the same local statement. The policy
    does not resolve conflicting time zones, conditional clauses, or contract terms.
    """
    text = re.sub(r'[*`#|]', ' ', chunk.text.replace('<br>', ' '))
    text = re.sub(r'\s+', ' ', text)
    best = (0, 0)
    for label in LABEL.finditer(text):
        before = text[max(0, label.start() - 45):label.start()]
        after = text[label.end():label.end() + 120]
        date = DATE.search(after)
        if not date or '?' in after[:date.end()]:
            continue
        if re.search(r'\b(questions?|clarifications?|not|no)\b', before[-30:], re.I):
            continue
        # A separate intervening milestone label must not donate its date.
        gap = after[:date.start()]
        if re.search(r'\b(prebid|pre-bid|meeting|publication|questions?|delivery)\b', gap, re.I):
            continue
        priority = (1, 0)
        local = before + label.group() + gap
        if (chunk.metadata.doc_type == 'addendum' and chunk.metadata.addendum_number
                and re.search(r'\b(new|revised|extended|extend|extension|changed)\b', local, re.I)):
            priority = (2, chunk.metadata.addendum_number)
        best = max(best, priority)
    return best


def prioritize_deadline_evidence(
    query: str, ranked: list[tuple[SearchChunk, float | None]],
) -> list[tuple[SearchChunk, float | None]]:
    """Apply documented evidence ordering only to current submission deadline intent."""
    if not current_deadline_intent(query):
        return ranked
    return sorted(ranked, key=lambda item: (dated_deadline_priority(item[0]),
                  item[1] if item[1] is not None else float('-inf')), reverse=True)
