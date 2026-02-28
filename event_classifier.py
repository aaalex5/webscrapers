"""event_classifier.py
--------------------
Scoring-based event category classifier for the venue scraper.

Architecture
============
Events are scored against named *internal sub-categories* (e.g. "live_music",
"dj_night") using a three-tier keyword system:

    strong  (+10 pts)  Unambiguous; rarely appears outside this event type.
    medium  (+5 pts)   Clear but occasionally seen in passing elsewhere.
    weak    (+2 pts)   Loosely correlated; easily overridden by stronger signals.

After raw scoring, two *cross-category adjustment rules* suppress signals that
are almost certainly incidental:

  Rule 1 – Strong music presence zeros out food's weak tier.
            "Happy Hour & Live Music (Doors at 7)" should not score as food.

  Rule 2 – Sport with only weak signals is suppressed entirely.
            "Game night" usually means board games, not a UFC watch party.

The winning category must clear a minimum score threshold (CONFIDENCE_THRESHOLD)
or the event falls back to "other".

Exact ties are resolved by a PRECEDENCE list that favours more-specific or more
common event types. Near-ties (gap ≤ SCORE_WEAK) rely purely on raw score; the
tie-break only applies when both categories have genuinely equal totals.

Internal sub-categories map to one of five final output categories:
    music   ← live_music, dj_night, open_mic, dance, karaoke
    trivia  ← trivia
    sport   ← sport
    food    ← food
    other   ← comedy, other

Usage
=====
    from event_classifier import classify_event

    result = classify_event("Doors at 7, Show at 9 w/ Horsegirl", description)
    event.event_type = result.internal_category   # e.g. "live_music"
    output["category"] = result.final_category    # e.g. "music"

    # For debugging
    print(result.confidence)   # "high" / "medium" / "low"
    print(result.scores)       # {category: score, ...}

Future improvements
===================
- Genre-specific keywords ("jazz", "blues", "reggae") can be added to
  live_music weak/medium once tested against real-world data.
- Per-venue override rules could be layered on top of the base scorer to handle
  venues with idiosyncratic language (e.g. a comedy club that always says "show").
- Confidence scores could be exposed in the output JSON for downstream filtering.
- A small labelled test set (see tests/test_classifier.py) would make threshold
  tuning much safer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

SCORE_STRONG = 10   # Unambiguous indicator; rarely false-positive
SCORE_MEDIUM = 5    # Clear signal but can appear in other contexts
SCORE_WEAK   = 2    # Loose association; easily overridden

# Minimum total score required to assign any category other than "other".
# A value of 5 means at least one medium match must fire — bare weak matches
# (e.g. a single mention of "dinner") are insufficient on their own.
CONFIDENCE_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Keyword rules by internal sub-category
#
# Design notes
# ------------
# * Keep *strong* keywords specific enough that a single match is convincing.
# * Keep *weak* keywords honest — if you can imagine an event of a completely
#   different type also containing the word, it belongs here.
# * Multi-word phrases are preferred over single words in strong/medium tiers
#   to reduce false-positive risk.
# * Phrase matching is case-insensitive and uses word boundaries, so "dj"
#   does not match "djembe" and "show" does not match "showcase" (the latter
#   contains an extra word character after the phrase).
# ---------------------------------------------------------------------------

RULES: dict[str, dict[str, list[str]]] = {

    "live_music": {
        # Ticket logistics and performer billing language is the most reliable
        # signal that an event is a ticketed musical performance.
        "strong": [
            "doors open", "doors at", "doors @", "doors:",
            "show at",                   # "Show at 9PM"
            "set time",
            "opening act", "support act",
            "w/",                        # "Band A w/ Band B" billing convention
            "tickets available", "ticket link", "on sale now",
            "general admission",
        ],
        # Unambiguous performance vocabulary that could theoretically appear in
        # other contexts but is dominant in venue event listings.
        "medium": [
            "live music", "live jazz", "live blues", "live folk",
            "live acoustic", "live band", "live entertainment",
            "concert", "gig", "performance",
            "headline", "headliner",
            "touring",
            "show",          # Generic but overwhelmingly used for performances
        ],
        # Present in non-music contexts; require corroboration from other tiers.
        "weak": [
            "band", "tour", "album release", "record release", "ep release",
        ],
    },

    "dj_night": {
        "strong": ["dj set", "dance party", "back to back", "b2b set"],
        "medium": ["dj", "club night", "dance floor"],
        "weak":   ["edm", "rave", "house music", "techno"],
    },

    "open_mic": {
        "strong": ["open mic", "open mike"],
        "medium": ["songwriter night", "writers night", "acoustic showcase"],
        "weak":   ["acoustic night", "unplugged"],
    },

    "karaoke": {
        # "karaoke" is self-defining; no medium or weak needed.
        "strong": ["karaoke"],
        "medium": [],
        "weak":   [],
    },

    "dance": {
        "strong": [
            "swing dance", "salsa night", "line dancing", "line dance",
            "ballroom night",
        ],
        "medium": [
            "bachata", "cumbia", "salsa lesson", "dance lesson", "dance class",
        ],
        "weak": ["two-step", "swing lesson"],
    },

    "food": {
        # Formal dining formats with a distinct event structure — these are
        # unambiguously food-primary events even when live music is present.
        "strong": [
            "prix fixe", "tasting menu", "chef's table", "wine pairing",
            "reservation required", "coursed dinner", "omakase",
            "degustation", "pop-up dinner", "supper club",
            "winemaker dinner", "beer dinner",
        ],
        # Dedicated food or beverage education events.
        "medium": [
            "wine tasting", "beer tasting", "cocktail class", "cooking class",
            "food festival", "farm to table",
        ],
        # Extremely common at all venue events; do not let these override a
        # stronger music signal. Rule 1 below zeros this tier out when strong
        # music keywords are present.
        "weak": [
            "brunch", "dinner", "happy hour", "tasting",
        ],
    },

    "trivia": {
        "strong": ["pub quiz", "quiz night", "trivia night"],
        "medium": ["trivia"],
        "weak":   ["brain game", "general knowledge"],
    },

    "sport": {
        "strong": [
            "watch party", "fight night", "ufc", "pay-per-view", "ppv",
            "boxing match", "mma event",
        ],
        "medium": [
            "nfl", "nba", "mlb", "nhl", "mls",
            "playoffs", "super bowl", "world cup", "championship",
            "monday night football", "game day",
        ],
        # "game night" deliberately lives in weak: it usually means board games,
        # not a sports-watching event. Rule 2 suppresses sport entirely when
        # only weak signals fire.
        "weak": ["game night", "football night"],
    },

    "comedy": {
        "strong": [
            "stand-up comedy", "stand up comedy", "improv show",
            "comedy show", "comedy night",
        ],
        "medium": [
            "stand-up", "stand up", "improv", "comedy", "comedian",
        ],
        "weak": ["laugh", "comic"],
    },
}


# ---------------------------------------------------------------------------
# Final output category map: internal sub-category → one of 5 output values
# ---------------------------------------------------------------------------

FINAL_CATEGORY_MAP: dict[str, str] = {
    "live_music": "music",
    "dj_night":   "music",
    "open_mic":   "music",
    "dance":      "music",
    "karaoke":    "music",
    "trivia":     "trivia",
    "food":       "food",
    "sport":      "sport",
    "comedy":     "other",
    "other":      "other",
}


# ---------------------------------------------------------------------------
# Tie-breaking precedence
#
# When two categories have exactly equal total scores the one with the higher
# index wins. This ordering reflects two priorities:
#   1. More specific categories beat generic ones (trivia > other).
#   2. Among equally specific categories, music sub-types beat food/sport
#      because venue event pages skew towards music events, and the weak food
#      keywords ("dinner", "happy hour") appear on almost every listing.
#
# Changing this list has NO effect when there is a clear score winner; it only
# matters for genuine ties.
# ---------------------------------------------------------------------------

PRECEDENCE: list[str] = [
    "other",
    "comedy",
    "food",
    "sport",
    "trivia",
    "dance",
    "open_mic",
    "karaoke",
    "dj_night",
    "live_music",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Returned by :func:`classify_event`."""

    internal_category: str          # Internal sub-category, e.g. "live_music"
    final_category: str             # Mapped output category, e.g. "music"
    score: float                    # Winning category's total score
    confidence: str                 # "high", "medium", or "low"
    scores: dict[str, float] = field(default_factory=dict)
    """Per-category totals — useful for debugging and threshold tuning."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace to a single space."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _phrase_match(phrase: str, text: str) -> bool:
    """Return True if *phrase* appears in *text* as a standalone token.

    Uses negative lookbehind/lookahead on ``\\w`` so that, for example:
      - ``"dj"``   matches ``" DJ set"``     but NOT ``"djembe"``
      - ``"show"`` matches ``"a great show"`` but NOT ``"showcase"``
      - ``"w/"``   matches ``"Band A w/ B"``  but NOT ``"w/o warning"``
    """
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return bool(re.search(pattern, text))


def _score_tiers(category: str, text: str) -> dict[str, float]:
    """Return ``{strong, medium, weak}`` point totals for *category*.

    Each phrase is tested for *presence* only — appearing three times counts
    the same as appearing once. This prevents inflated scores from repetitive
    descriptions while still rewarding phrase co-occurrence across tiers.
    """
    rules = RULES.get(category, {})
    tiers: dict[str, float] = {"strong": 0.0, "medium": 0.0, "weak": 0.0}

    for phrase in rules.get("strong", []):
        if _phrase_match(phrase, text):
            tiers["strong"] += SCORE_STRONG

    for phrase in rules.get("medium", []):
        if _phrase_match(phrase, text):
            tiers["medium"] += SCORE_MEDIUM

    for phrase in rules.get("weak", []):
        if _phrase_match(phrase, text):
            tiers["weak"] += SCORE_WEAK

    return tiers


# ---------------------------------------------------------------------------
# Cross-category adjustment rules
# ---------------------------------------------------------------------------

def _apply_adjustments(
    tier_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Apply inter-category context rules and return adjusted tier scores.

    Rule 1 — Strong music presence suppresses food's weak tier
        When any music sub-category lands at least one strong-tier keyword
        (e.g. "doors at 7", "opening act", "general admission"), the food
        weak-tier keywords ("dinner", "happy hour", "brunch") are almost
        certainly incidental. For example::

            "Live Music (Doors 7PM) + Happy Hour Specials"

        should classify as live_music, not food.  Zeroing the food weak tier
        removes that 2-point noise without touching food's medium/strong
        signals — a genuine food event like a wine pairing dinner still scores
        correctly even when a performer is mentioned.

    Rule 2 — Sport with only weak signals is suppressed
        "Game night" and "football night" on their own usually mean board-game
        nights or generic sports viewing at a bar, not a dedicated watch-party
        event with a clear sport focus. We zero the sport weak tier when no
        medium or strong sport keyword has fired, ensuring the threshold is not
        crossed by a single ambiguous phrase.
    """
    adjusted = {cat: dict(tiers) for cat, tiers in tier_scores.items()}

    # --- Rule 1 ---------------------------------------------------------------
    music_categories = {"live_music", "dj_night", "open_mic", "dance", "karaoke"}
    music_strong_total = sum(
        adjusted[cat]["strong"]
        for cat in music_categories
        if cat in adjusted
    )
    if music_strong_total >= SCORE_STRONG:
        adjusted["food"]["weak"] = 0.0

    # --- Rule 2 ---------------------------------------------------------------
    sport = adjusted.get("sport", {})
    if sport.get("strong", 0.0) == 0.0 and sport.get("medium", 0.0) == 0.0:
        adjusted["sport"]["weak"] = 0.0

    return adjusted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_event(name: str, description: str = "") -> ClassificationResult:
    """Classify an event using weighted keyword scoring.

    Parameters
    ----------
    name:
        Event name or title (e.g. ``"Horsegirl w/ Narrow Head"``).
    description:
        Optional additional text such as a full event description.  Both
        strings are concatenated before scoring so keywords can appear in
        either field.

    Returns
    -------
    ClassificationResult
        ``internal_category`` — one of the keys in :data:`RULES` or ``"other"``
        ``final_category``    — one of ``music / trivia / sport / food / other``
        ``score``             — numeric total for the winning category
        ``confidence``        — ``"high"`` / ``"medium"`` / ``"low"``
        ``scores``            — full per-category score dict for debugging

    Algorithm summary
    -----------------
    1. Normalise text (lowercase, collapse whitespace).
    2. Score every category across three tiers (strong/medium/weak).
    3. Apply cross-category adjustment rules (Rule 1, Rule 2).
    4. Collapse tier scores to a single total per category.
    5. If the best total is below ``CONFIDENCE_THRESHOLD`` → return ``"other"``.
    6. Resolve exact ties using ``PRECEDENCE`` ordering.
    7. Assign confidence level and return.
    """
    text = _normalize(f"{name} {description}")

    # Step 1 — Raw per-tier scoring
    tier_scores: dict[str, dict[str, float]] = {
        cat: _score_tiers(cat, text) for cat in RULES
    }

    # Step 2 — Cross-category adjustments
    adjusted = _apply_adjustments(tier_scores)

    # Step 3 — Collapse to a single score per category
    total_scores: dict[str, float] = {
        cat: sum(tiers.values()) for cat, tiers in adjusted.items()
    }

    # Step 4 — Find the leading category (primary sort: score; secondary: precedence)
    def _sort_key(cat: str) -> tuple[float, int]:
        return (total_scores[cat], PRECEDENCE.index(cat) if cat in PRECEDENCE else 0)

    best_cat = max(total_scores, key=_sort_key)
    best_score = total_scores[best_cat]

    # Step 5 — Confidence threshold
    if best_score < CONFIDENCE_THRESHOLD:
        return ClassificationResult(
            internal_category="other",
            final_category="other",
            score=best_score,
            confidence="low",
            scores=total_scores,
        )

    # Step 6 — Assign confidence band
    #   high   ≥ 20 pts  Two or more strong matches — very reliable.
    #   medium ≥ 10 pts  At least one strong match, or two mediums.
    #   low    ≥  5 pts  Only medium/weak signals; accept but note uncertainty.
    if best_score >= SCORE_STRONG * 2:
        confidence = "high"
    elif best_score >= SCORE_STRONG:
        confidence = "medium"
    else:
        confidence = "low"

    return ClassificationResult(
        internal_category=best_cat,
        final_category=FINAL_CATEGORY_MAP.get(best_cat, "other"),
        score=best_score,
        confidence=confidence,
        scores=total_scores,
    )
