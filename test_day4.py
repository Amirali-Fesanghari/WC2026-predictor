"""Quick smoke test for all Day 4 modules."""
import sys
sys.path.insert(0, ".")

print("=" * 55)
print("  Day 4 Module Smoke Tests")
print("=" * 55)

# ── 1. Poisson model ────────────────────────────────────────
try:
    import pandas as pd
    from src.models.poisson_model import DixonColesModel

    data = {
        "home_team": ["France","Brazil","Germany","Argentina","Spain","England"],
        "away_team":  ["Germany","Argentina","Spain","France","England","Brazil"],
        "home_goals": [2, 1, 1, 2, 3, 1],
        "away_goals": [0, 0, 2, 1, 1, 2],
    }
    df = pd.DataFrame(data)
    m = DixonColesModel()
    m.fit(df)
    pred = m.predict_match("France", "Brazil")
    hw = round(pred["home_win_prob"] * 100, 1)
    dr = round(pred["draw_prob"] * 100, 1)
    aw = round(pred["away_win_prob"] * 100, 1)
    ms = pred["most_likely_score"]
    sh = pred["score_matrix"].shape
    print(f"[OK] poisson_model     France vs Brazil: {hw}% / {dr}% / {aw}%  likely={ms}  matrix={sh}")
except Exception as e:
    print(f"[FAIL] poisson_model: {e}")

# ── 2. Tournament simulator ─────────────────────────────────
try:
    from src.simulation.tournament_simulator import TournamentSimulator
    sim = TournamentSimulator(n_simulations=3000)
    sim.run()
    res = sim.get_results()
    top = sorted(res.items(), key=lambda x: x[1]["champion_prob"], reverse=True)[:3]
    names = ", ".join(f"{t} {round(s['champion_prob']*100,1)}%" for t, s in top)
    print(f"[OK] tournament_sim    top3: {names}")
except Exception as e:
    print(f"[FAIL] tournament_sim: {e}")

# ── 3. Tactical classifier ──────────────────────────────────
try:
    from src.tactics.tactical_classifier import TacticalClassifier
    tc = TacticalClassifier()
    enc = tc.encode_formation("4-3-3")
    mu = tc.analyze_matchup("4-3-3", "4-4-2")
    recs = tc.recommend_formation("4-4-2")
    print(f"[OK] tactical_classif  4-3-3 enc={enc}  matchup_keys={list(mu.keys())[:4]}  recs={len(recs)}")
except Exception as e:
    print(f"[FAIL] tactical_classif: {e}")

# ── 4. Sentiment analyzer ───────────────────────────────────
try:
    from src.psych.sentiment_analyzer import SentimentAnalyzer
    sa = SentimentAnalyzer()
    articles = [
        {"title": "France crisis", "text": "France is in crisis. The manager was fired after a terrible controversy.  Players are injured and suspended amid political conflict in the team.", "url": "", "date": "2026-05-01"},
        {"title": "Mbappe fit", "text": "Mbappe is confident and fully fit. He is motivated and ready to lead France to victory at the World Cup.", "url": "", "date": "2026-05-02"},
        {"title": "France training", "text": "France trained well today. The captain looked sharp and the team seems united ahead of the tournament.", "url": "", "date": "2026-05-03"},
    ]
    result = sa.analyze_team("France", articles)
    score = round(result["psych_score"], 3)
    risk = result["risk_level"]
    nsent = result["n_sentences"]
    print(f"[OK] sentiment_analyz  France psych_score={score}  risk={risk}  n_sent={nsent}")
except Exception as e:
    print(f"[FAIL] sentiment_analyz: {e}")

# ── 5. Ensemble predictor ───────────────────────────────────
try:
    from src.models.ensemble import EnsemblePredictor
    ep = EnsemblePredictor()
    pred2 = ep.predict_tournament_match("Argentina", "France", "final")
    hw2 = round(pred2["home_win_prob"] * 100, 1)
    dr2 = round(pred2["draw_prob"] * 100, 1)
    aw2 = round(pred2["away_win_prob"] * 100, 1)
    conf = pred2["confidence"]
    agree = round(pred2["model_agreement"], 3)
    print(f"[OK] ensemble           Argentina vs France: {hw2}% / {dr2}% / {aw2}%  conf={conf}  agree={agree}")
except Exception as e:
    print(f"[FAIL] ensemble: {e}")

# ── 6. Injury tracker ───────────────────────────────────────
try:
    from src.pipeline.injury_tracker import InjuryTracker
    it = InjuryTracker()
    feat = it.get_availability_feature("France")
    print(f"[OK] injury_tracker    France availability_feature={round(feat, 3)}")
except Exception as e:
    print(f"[FAIL] injury_tracker: {e}")

print("=" * 55)
print("  Done. Fix any [FAIL] lines above before proceeding.")
print("=" * 55)
