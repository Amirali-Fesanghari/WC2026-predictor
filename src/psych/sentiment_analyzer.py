"""
src/psych/sentiment_analyzer.py
Sentence-level sentiment scoring for soccer psychological risk assessment.

Uses VADER (vaderSentiment) as the primary engine, augmented by a keyword
multiplier layer drawn from config.py. Scores are aggregated to a psych_score
in the range [-1, +1] and mapped to a qualitative risk level.

Dependencies (pip-installable):
    vaderSentiment, loguru
"""

import re
from pathlib import Path

from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))
from config import PSYCH_NEGATIVE_KEYWORDS, PSYCH_POSITIVE_KEYWORDS


# ── Constants ─────────────────────────────────────────────────────────────────

KEYWORD_NEGATIVE_WEIGHT = 1.50   # multiply sentence score if negative kw present
KEYWORD_POSITIVE_WEIGHT = 1.30   # multiply sentence score if positive kw present

RISK_THRESHOLDS = {
    # psych_score thresholds (mean of weighted sentence compound scores)
    "high":   -0.20,   # score <= -0.20 -> high risk
    "medium": -0.05,   # score <= -0.05 -> medium risk
    # above -0.05 -> low risk
}

MIN_SENTENCES_FOR_SIGNAL = 3     # return 0.0 if fewer than this many sentences scored

# Indirect-quote hedging patterns: sentences matching these get halved weight
INDIRECT_QUOTE_PATTERNS = re.compile(
    r"\baccording to\b|\bsources say\b|\breportedly\b|\bsaid to be\b|\bbelieved to be\b",
    re.IGNORECASE,
)

# Direct quote detector (text inside "..." or curly quotes)
DIRECT_QUOTE_PATTERN = re.compile(r'["“].{10,}["”]')

# Sentence splitter — split on . ! ? while keeping context
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_sentences(text):
    """Return a list of non-empty sentences from a block of text."""
    if not text:
        return []
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _keyword_multiplier(sentence):
    """
    Return a weight multiplier based on keyword presence.
    Negative keywords dominate; if both present, negative wins.
    """
    lower = sentence.lower()
    has_negative = any(kw.lower() in lower for kw in PSYCH_NEGATIVE_KEYWORDS)
    has_positive = any(kw.lower() in lower for kw in PSYCH_POSITIVE_KEYWORDS)

    if has_negative:
        return KEYWORD_NEGATIVE_WEIGHT
    if has_positive:
        return KEYWORD_POSITIVE_WEIGHT
    return 1.0


def _quote_weight(sentence):
    """
    Return 0.5 for indirect quotes, 1.0 otherwise.
    Direct player quotes (inside double-quotes) are intentionally kept at 1.0
    even if they also contain an indirect phrase.
    """
    if DIRECT_QUOTE_PATTERN.search(sentence):
        return 1.0
    if INDIRECT_QUOTE_PATTERNS.search(sentence):
        return 0.5
    return 1.0


def _entity_match(sentence, name):
    """
    Return True if the name (or any single word of it for short names) appears
    in the sentence, case-insensitive.
    """
    lower_sentence = sentence.lower()
    lower_name = name.lower()
    if lower_name in lower_sentence:
        return True
    # Also accept any word from the name with 5+ characters (reduces false negatives)
    for part in lower_name.split():
        if len(part) >= 5 and part in lower_sentence:
            return True
    return False


def _score_sentences(sentences, subject_name, vader):
    """
    Score each sentence that mentions subject_name.

    Returns
    -------
    scored : list of (sentence, weighted_compound) tuples
    signals : list of signal strings for sentences with |weighted_score| >= 0.25
    """
    scored = []
    signals = []

    for sentence in sentences:
        if not _entity_match(sentence, subject_name):
            continue

        raw = vader.polarity_scores(sentence)
        compound = raw["compound"]           # already in [-1, +1]

        kw_mult = _keyword_multiplier(sentence)
        q_weight = _quote_weight(sentence)
        weighted = compound * kw_mult * q_weight

        # Clip to [-1, +1] after multipliers
        weighted = max(-1.0, min(1.0, weighted))

        scored.append((sentence, weighted))

        # Build a signal string for notable sentences (|weighted| >= 0.25, max 10)
        if abs(weighted) >= 0.25:
            polarity_label = "NEGATIVE" if weighted < 0 else "POSITIVE"
            snippet = sentence[:100].rstrip() + ("..." if len(sentence) > 100 else "")
            signals.append(f"[{polarity_label} {weighted:+.2f}] {snippet}")

    return scored, signals


def _aggregate(scored_sentences):
    """
    Compute the final psych_score from a list of (sentence, score) tuples.

    Returns (psych_score, n_sentences).
    Returns (0.0, 0) if the list is empty.
    """
    if not scored_sentences:
        return 0.0, 0

    total = sum(score for _, score in scored_sentences)
    n = len(scored_sentences)
    mean_score = total / n
    # Clip to [-1, +1] as a safety measure (individual scores are already clipped)
    psych_score = max(-1.0, min(1.0, mean_score))
    return psych_score, n


def _risk_level(psych_score):
    if psych_score <= RISK_THRESHOLDS["high"]:
        return "high"
    if psych_score <= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _build_result(psych_score, n_sentences, signals):
    """Return the standard result dict."""
    return {
        "psych_score": round(psych_score, 4),
        "risk_level":  _risk_level(psych_score),
        "top_signals": signals[:10],           # cap at 10
        "n_sentences": n_sentences,
    }


def _null_result():
    """Returned when there are no relevant articles or insufficient signal."""
    return {
        "psych_score": 0.0,
        "risk_level":  "low",
        "top_signals": [],
        "n_sentences": 0,
    }


# ── Main class ────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Sentence-level VADER sentiment scorer with keyword multipliers and
    named-entity filtering, producing a psych_score per team or player.

    Usage
    -----
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_team("Brazil", articles)
    result = analyzer.analyze_player("Vinicius", articles)

    Each result is a dict with:
        psych_score  float  [-1.0, +1.0]  negative = psychological risk
        risk_level   str    'low' / 'medium' / 'high'
        top_signals  list   up to 10 descriptive strings from notable sentences
        n_sentences  int    number of sentences that contributed to the score
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        logger.debug("SentimentAnalyzer initialised with VADER")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_team(self, team_name, articles):
        """
        Score the psychological environment of a team from a list of articles.

        Parameters
        ----------
        team_name : str
            The team's name (must match how it appears in article text).
        articles : list of dict
            Each dict must contain at least a 'text' key (and optionally 'title').

        Returns
        -------
        dict  {psych_score, risk_level, top_signals, n_sentences}
        """
        if not articles:
            logger.info(f"analyze_team({team_name}): no articles — returning null result")
            return _null_result()

        logger.info(f"Analysing team sentiment: {team_name} ({len(articles)} articles)")

        all_scored = []
        all_signals = []

        for article in articles:
            text = (article.get("title", "") + " " + article.get("text", "")).strip()
            sentences = _split_sentences(text)
            scored, signals = _score_sentences(sentences, team_name, self._vader)
            all_scored.extend(scored)
            all_signals.extend(signals)

        psych_score, n = _aggregate(all_scored)

        if n < MIN_SENTENCES_FOR_SIGNAL:
            logger.info(
                f"analyze_team({team_name}): only {n} sentences scored "
                f"(min {MIN_SENTENCES_FOR_SIGNAL}) — returning null result"
            )
            return _null_result()

        result = _build_result(psych_score, n, all_signals)
        logger.info(
            f"Team {team_name}: psych_score={result['psych_score']:+.4f}, "
            f"risk={result['risk_level']}, sentences={n}"
        )
        return result

    def analyze_player(self, player_name, articles):
        """
        Score the psychological environment of a player from a list of articles.

        Parameters
        ----------
        player_name : str
            Player's full name (e.g. "Kylian Mbappe").
        articles : list of dict
            Each dict must contain at least a 'text' key (and optionally 'title').

        Returns
        -------
        dict  {psych_score, risk_level, top_signals, n_sentences}
        """
        if not articles:
            logger.info(f"analyze_player({player_name}): no articles — returning null result")
            return _null_result()

        logger.info(f"Analysing player sentiment: {player_name} ({len(articles)} articles)")

        all_scored = []
        all_signals = []

        for article in articles:
            text = (article.get("title", "") + " " + article.get("text", "")).strip()
            sentences = _split_sentences(text)
            scored, signals = _score_sentences(sentences, player_name, self._vader)
            all_scored.extend(scored)
            all_signals.extend(signals)

        psych_score, n = _aggregate(all_scored)

        if n < MIN_SENTENCES_FOR_SIGNAL:
            logger.info(
                f"analyze_player({player_name}): only {n} sentences scored "
                f"(min {MIN_SENTENCES_FOR_SIGNAL}) — returning null result"
            )
            return _null_result()

        result = _build_result(psych_score, n, all_signals)
        logger.info(
            f"Player {player_name}: psych_score={result['psych_score']:+.4f}, "
            f"risk={result['risk_level']}, sentences={n}"
        )
        return result


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MOCK_ARTICLES = [
        {
            "title": "France captain doubt for opener after training injury",
            "text": (
                "France face a major concern as their captain picked up an injury in training. "
                "The player is reportedly doubtful for the opening group-stage match. "
                "According to sources say close to the camp, the coach is considering alternatives. "
                "France have struggled with fitness issues throughout the preparation phase. "
                "The squad remains confident despite the setback, with the manager insisting "
                "the team is united and motivated to perform at the World Cup. "
                "France were dominant in qualifying and showed great form across all competitions."
            ),
            "url": "https://example.com/france-injury",
            "date": "2026-05-28T10:00:00+00:00",
            "source": "mock",
        },
        {
            "title": "Mbappe fit and ready — France star targets golden boot",
            "text": (
                "Kylian Mbappe declared himself fully fit and confident ahead of France's "
                "World Cup campaign. Mbappe, the team leader, has been in outstanding form "
                "and is motivated to win the golden boot. France coaching staff confirmed "
                "Mbappe recovered from his minor knock and will be available for selection. "
                "The captain's record-breaking performances have France fans excited for the "
                "tournament. Mbappe was awarded player of the month before the squad gathered."
            ),
            "url": "https://example.com/mbappe-fit",
            "date": "2026-05-29T08:00:00+00:00",
            "source": "mock",
        },
        {
            "title": "Scandal surrounds France ahead of World Cup",
            "text": (
                "A controversy has erupted in the France camp days before the tournament. "
                "There are reports of a fallout between senior players and the coaching staff. "
                "According to sources say with knowledge of the situation, a rift has developed "
                "over playing time and tactical decisions. France management have denied any "
                "argument or conflict within the squad. The scandal is threatening to undermine "
                "France's preparation and could affect the team's performance in their opening match."
            ),
            "url": "https://example.com/france-scandal",
            "date": "2026-05-30T06:00:00+00:00",
            "source": "mock",
        },
    ]

    analyzer = SentimentAnalyzer()

    print("\n=== Team analysis: France ===")
    team_result = analyzer.analyze_team("France", MOCK_ARTICLES)
    print(f"  psych_score : {team_result['psych_score']:+.4f}")
    print(f"  risk_level  : {team_result['risk_level']}")
    print(f"  n_sentences : {team_result['n_sentences']}")
    print(f"  top_signals :")
    for sig in team_result["top_signals"]:
        print(f"    {sig}")

    print("\n=== Player analysis: Mbappe ===")
    player_result = analyzer.analyze_player("Mbappe", MOCK_ARTICLES)
    print(f"  psych_score : {player_result['psych_score']:+.4f}")
    print(f"  risk_level  : {player_result['risk_level']}")
    print(f"  n_sentences : {player_result['n_sentences']}")
    print(f"  top_signals :")
    for sig in player_result["top_signals"]:
        print(f"    {sig}")

    print("\n=== Edge case: empty articles ===")
    empty_result = analyzer.analyze_team("Germany", [])
    print(f"  psych_score : {empty_result['psych_score']}")
    print(f"  risk_level  : {empty_result['risk_level']}")
    print(f"  n_sentences : {empty_result['n_sentences']}")
