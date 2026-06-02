import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st

st.set_page_config(
    page_title="WC2026 AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark CSS theme with green accent ─────────────────────────
st.markdown(
    """
    <style>
      :root {
        --accent:  #00ff87;
        --bg:      #0d0d0d;
        --surface: #1a1a1a;
        --border:  #2a2a2a;
        --text:    #e0e0e0;
        --muted:   #888888;
      }
      html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text);
      }
      .stApp { background: var(--bg); }
      [data-testid="stSidebar"] { background: #111 !important; }
      .stTabs [data-baseweb="tab-list"] {
        background: #111;
        border-bottom: 1px solid var(--border);
        gap: 4px;
      }
      .stTabs [data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
      .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent);
      }
      .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-bottom: 8px;
      }
      .metric-card h2 { color: var(--accent); font-size: 2.1rem; margin: 0; }
      .metric-card p  { color: var(--muted); margin: 4px 0 0; font-size: 0.85rem; }
      .risk-high   { color: #ff4444; font-weight: 700; }
      .risk-medium { color: #ffa500; font-weight: 700; }
      .risk-low    { color: #00ff87; font-weight: 700; }
      div[data-testid="stMetricValue"] { color: var(--accent) !important; }
      .stButton > button {
        background: var(--accent);
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
      }
      .stButton > button:hover { background: #00cc6a; }
      h1, h2, h3 { color: #ffffff; }
      .stSelectbox label, .stSlider label { color: #cccccc; }
      .stDataFrame { background: var(--surface); }
      .banner {
        background: linear-gradient(90deg, #003300 0%, #005522 50%, #003300 100%);
        border: 1px solid var(--accent);
        border-radius: 8px;
        padding: 10px 20px;
        text-align: center;
        margin-bottom: 16px;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0.04em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Standard library + third-party imports ────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import math
from datetime import datetime

# ── Config imports ─────────────────────────────────────────────
try:
    from config import WC_2026_TEAMS, CACHE_DIR, DB_PATH
    _CONFIG_OK = True
except Exception as _e:
    st.warning(f"config.py not found or has errors: {_e}")
    _CONFIG_OK = False
    WC_2026_TEAMS = []
    CACHE_DIR = Path(__file__).parents[1] / "data" / "cache"
    DB_PATH = Path(__file__).parents[1] / "data" / "wc2026.db"

# ── Optional pipeline imports ──────────────────────────────────
try:
    from src.pipeline.elo import EloEngine, _expected_score
    _ELO_OK = True
except ImportError:
    _ELO_OK = False

try:
    from src.pipeline.football_data_loader import load_open_international_results
    _LOADER_OK = True
except ImportError:
    _LOADER_OK = False

# ── Model class import (real class name from xgboost_model.py) ─
try:
    from src.models.xgboost_model import WCOutcomeModel
    _MODEL_CLASS_OK = True
except ImportError:
    _MODEL_CLASS_OK = False

# ── FeatureEngineer import (for build_prediction_vector) ───────
try:
    from src.pipeline.feature_engineer import FeatureEngineer
    _FE_OK = True
except ImportError:
    _FE_OK = False

# ── TournamentSimulator import ─────────────────────────────────
try:
    from src.simulation.tournament_simulator import TournamentSimulator, DEFAULT_ELO
    _SIM_OK = True
except ImportError:
    _SIM_OK = False

# ── Plotly dark layout defaults ───────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1a1a",
    plot_bgcolor="#1a1a1a",
    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)
ACCENT = "#00ff87"
TEAMS_SORTED = sorted(WC_2026_TEAMS) if WC_2026_TEAMS else ["(no teams)"]


# ═══════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    """
    Load the most recently saved WCOutcomeModel.
    Returns a WCOutcomeModel instance, or None if no saved model exists.

    Uses @st.cache_resource so the model object is shared across sessions
    and is not re-loaded on every Streamlit rerun.
    """
    if not _MODEL_CLASS_OK:
        return None
    try:
        # WCOutcomeModel.save() writes to src/models/saved/xgboost_outcome_<ts>.joblib
        model_dir = Path(__file__).parents[1] / "src" / "models" / "saved"
        candidates = sorted(model_dir.glob("xgboost_outcome_*.joblib"), reverse=True)
        if not candidates:
            return None
        return WCOutcomeModel.load(candidates[0])
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading FeatureEngineer…")
def load_feature_engineer():
    """
    Load a FeatureEngineer with all data sources pre-loaded.
    Returns None if the module is unavailable or data loading fails.

    Uses @st.cache_resource so the heavy data load only happens once.
    """
    if not _FE_OK:
        return None
    try:
        fe = FeatureEngineer()
        fe.load_data()
        return fe
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading ELO ratings…")
def get_elo_ratings():
    """Build current ELO ratings from historical data; returns {} on failure."""
    if not _ELO_OK or not _LOADER_OK:
        return {}
    try:
        results = load_open_international_results()
        engine = EloEngine()
        df90 = results[results["match_date"] >= "1990-01-01"].copy()
        engine.process_dataframe(df90)
        return dict(engine.ratings)
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner="Loading ELO history…")
def get_elo_history():
    """Load ELO history parquet if available."""
    path = CACHE_DIR / "elo_history.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_psych_signals():
    """Load psychological risk signals from DB or parquet cache."""
    parquet = CACHE_DIR / "psych_signals.parquet"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if DB_PATH.exists():
        try:
            import sqlite3
            con = sqlite3.connect(str(DB_PATH))
            df = pd.read_sql("SELECT * FROM psych_signals WHERE reviewed = 1", con)
            con.close()
            return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_recent_form(team, n=5):
    """Return list of last-N match dicts for a team; empty list on failure."""
    if not _LOADER_OK:
        return []
    try:
        results = load_open_international_results()
        home = results[results["home_team"] == team].copy()
        home["opponent"] = home["away_team"]
        home["gf"] = home["home_goals"]
        home["ga"] = home["away_goals"]
        away = results[results["away_team"] == team].copy()
        away["opponent"] = away["home_team"]
        away["gf"] = away["away_goals"]
        away["ga"] = away["home_goals"]
        combined = pd.concat([home, away]).sort_values("match_date", ascending=False).head(n)
        form = []
        for _, row in combined.iterrows():
            gf, ga = int(row["gf"]), int(row["ga"])
            result = "W" if gf > ga else ("D" if gf == ga else "L")
            form.append({"date": str(row["match_date"])[:10],
                         "opponent": row["opponent"],
                         "score": f"{gf}-{ga}",
                         "result": result})
        return form
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
#  ELO PROBABILITY HELPER
# ═══════════════════════════════════════════════════════════════

def elo_wdl(elo_home, elo_away):
    """Return (p_win, p_draw, p_loss) using simple ELO + draw heuristic."""
    p_win_raw = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    gap = abs(p_win_raw - 0.5)
    p_draw = max(0.10, 0.28 - 0.25 * gap)
    scale = 1 - p_draw
    p_win = p_win_raw * scale
    p_loss = (1 - p_win_raw) * scale
    total = p_win + p_draw + p_loss
    return p_win / total, p_draw / total, p_loss / total


# ═══════════════════════════════════════════════════════════════
#  SHAP FALLBACK FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def build_shap_fallback(home_team, away_team, elo_ratings):
    """Return a synthetic feature-importance DataFrame when model is absent."""
    elo_h = elo_ratings.get(home_team, 1500)
    elo_a = elo_ratings.get(away_team, 1500)
    features = [
        ("home_elo",            elo_h,           (elo_h - 1500) * 0.0004),
        ("away_elo",            elo_a,           -(elo_a - 1500) * 0.0004),
        ("elo_diff",            elo_h - elo_a,   (elo_h - elo_a) * 0.0003),
        ("home_advantage",      1.0,             0.0250),
        ("home_form_pts",       9.0,             0.0180),
        ("away_form_pts",       6.0,            -0.0120),
        ("home_goals_scored",   1.8,             0.0150),
        ("away_goals_conceded", 1.1,             0.0090),
        ("head2head_home_wins", 3.0,             0.0070),
        ("stage_weight",        1.3,             0.0060),
    ]
    df = pd.DataFrame(features, columns=["Feature", "Value", "SHAP Impact"])
    df = df.reindex(df["SHAP Impact"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════

# ── Countdown banner ──────────────────────────────────────────
st.markdown(
    "<div class='banner'>⚽ WC 2026 is 11 days away!</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center; color:#00ff87; margin-bottom:0;'>⚽ WC 2026 AI Predictor</h1>"
    "<p style='text-align:center; color:#888; margin-top:4px;'>Machine Learning · ELO · Monte Carlo Simulation</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Match Predictor", "Tournament Simulator", "Team Intel"])


# ───────────────────────────────────────────────────────────────
#  TAB 1 — MATCH PREDICTOR
# ───────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## Match Predictor")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox(
            "Home Team",
            TEAMS_SORTED,
            index=TEAMS_SORTED.index("France") if "France" in TEAMS_SORTED else 0,
            key="t1_home",
        )
    with col2:
        away_team = st.selectbox(
            "Away Team",
            TEAMS_SORTED,
            index=TEAMS_SORTED.index("Germany") if "Germany" in TEAMS_SORTED else 1,
            key="t1_away",
        )

    stage = st.selectbox(
        "Stage",
        ["Group Stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"],
        key="t1_stage",
    )

    predict_clicked = st.button("Predict", key="btn_predict")

    if predict_clicked:
        with st.spinner("Running prediction…"):
            elo_ratings = get_elo_ratings()
            # Load model via WCOutcomeModel.load_latest() wrapper
            # Returns WCOutcomeModel instance or None
            model_obj = load_model()

            elo_h = elo_ratings.get(home_team, 1500.0)
            elo_a = elo_ratings.get(away_team, 1500.0)

            used_model = False

            if model_obj is None:
                # ELO-only fallback — works even before training
                p_win, p_draw, p_loss = elo_wdl(elo_h, elo_a)
                source = "ELO fallback"
                st.warning(
                    "Trained model not found — using ELO-based probability. "
                    "Run `python -m src.models.xgboost_model` to train.",
                    icon="⚠️",
                )
            else:
                try:
                    # Try to build a full feature vector via FeatureEngineer
                    fe = load_feature_engineer()
                    if fe is not None:
                        # build_prediction_vector is the real method in feature_engineer.py
                        feat = fe.build_prediction_vector(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=datetime(2026, 6, 11),
                            stage=stage.lower().replace(" ", "_").replace("-", "_"),
                        )
                    else:
                        # Minimal fallback vector — model.predict() will fill
                        # missing columns with 0 via reindex
                        feat = pd.DataFrame([{
                            "home_elo":   elo_h,
                            "away_elo":   elo_a,
                            "elo_diff":   elo_h - elo_a,
                            "stage_weight": 1.0,
                        }])

                    # WCOutcomeModel.predict() returns a dict with p_home_win / p_draw / p_away_win
                    result = model_obj.predict(
                        feature_vector=feat,
                        home_team=home_team,
                        away_team=away_team,
                    )
                    p_win  = result["p_home_win"]
                    p_draw = result["p_draw"]
                    p_loss = result["p_away_win"]
                    source = "XGBoost model"
                    used_model = True
                except Exception as exc:
                    st.warning(f"Model prediction failed ({exc}); using ELO fallback.", icon="⚠️")
                    p_win, p_draw, p_loss = elo_wdl(elo_h, elo_a)
                    source = "ELO fallback"

            st.session_state["pred"] = {
                "home": home_team, "away": away_team, "stage": stage,
                "p_win": p_win, "p_draw": p_draw, "p_loss": p_loss,
                "elo_h": elo_h, "elo_a": elo_a, "source": source,
                "elo_ratings": elo_ratings,
                "used_model": used_model,
            }

    pred = st.session_state.get("pred")

    if pred:
        p_win  = pred["p_win"]
        p_draw = pred["p_draw"]
        p_loss = pred["p_loss"]
        h      = pred["home"]
        a      = pred["away"]

        # Horizontal bar chart
        fig = go.Figure(go.Bar(
            x=[p_win * 100, p_draw * 100, p_loss * 100],
            y=[f"{h} Win", "Draw", f"{a} Win"],
            orientation="h",
            marker_color=[ACCENT, "#888888", "#ff6b6b"],
            text=[f"{v:.1f}%" for v in [p_win * 100, p_draw * 100, p_loss * 100]],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=14),
        ))
        fig.update_layout(
            **_LAYOUT,
            title=dict(text=f"{h}  vs  {a} — {pred['stage']}", font=dict(size=17, color="#fff")),
            xaxis=dict(range=[0, 105], ticksuffix="%", showgrid=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=13)),
            height=230,
            bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3 metric columns
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric(f"{h} Win %", f"{p_win:.1%}")
        with mc2:
            st.metric("Draw %", f"{p_draw:.1%}")
        with mc3:
            st.metric(f"{a} Win %", f"{p_loss:.1%}")

        # Expected score text
        exp_home_g = p_win * 2.2 + p_draw * 1.1
        exp_away_g = p_loss * 2.2 + p_draw * 1.1
        st.info(
            f"**Expected Score:** {h} {exp_home_g:.1f} — {exp_away_g:.1f} {a}"
            f"  *(ELO-weighted xG estimate · source: {pred['source']})*"
        )

        # SHAP feature importance
        st.markdown("### Top 10 Feature Importances (SHAP)")
        shap_df = build_shap_fallback(h, a, pred.get("elo_ratings", {}))

        if pred.get("used_model"):
            # Re-load the model object (cached, no cost) to run SHAP
            model_obj = load_model()
            if model_obj is not None:
                try:
                    import shap
                    fe = load_feature_engineer()
                    if fe is not None:
                        feat_vec = fe.build_prediction_vector(
                            home_team=h, away_team=a,
                            match_date=datetime(2026, 6, 11),
                            stage=pred["stage"].lower().replace(" ", "_").replace("-", "_"),
                        )
                    else:
                        feat_vec = pd.DataFrame([{
                            "home_elo": pred["elo_h"], "away_elo": pred["elo_a"],
                            "elo_diff": pred["elo_h"] - pred["elo_a"], "stage_weight": 1.0,
                        }])

                    explanation = model_obj.explain_prediction(
                        feature_vector=feat_vec,
                        home_team=h,
                        away_team=a,
                        outcome_idx=2,
                    )
                    shap_df = pd.DataFrame({
                        "Feature":     explanation["feature"].tolist(),
                        "Value":       explanation["value"].tolist(),
                        "SHAP Impact": explanation["shap"].tolist(),
                    }).head(10).reset_index(drop=True)
                except Exception:
                    pass  # keep fallback

        st.dataframe(
            shap_df.head(10),
            column_config={
                "Feature":     st.column_config.TextColumn("Feature", width="medium"),
                "Value":       st.column_config.NumberColumn("Value", format="%.3f"),
                "SHAP Impact": st.column_config.ProgressColumn(
                    "SHAP Impact",
                    help="Positive = favours Home Win",
                    min_value=float(shap_df["SHAP Impact"].min()),
                    max_value=float(shap_df["SHAP Impact"].max()),
                    format="%.4f",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.markdown(
            "<div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;"
            "padding:50px;text-align:center;color:#555;'>"
            "<p style='font-size:1.1rem;'>Select two teams and click <strong>Predict</strong></p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────
#  TAB 2 — TOURNAMENT SIMULATOR
#
#  Uses TournamentSimulator from src.simulation.tournament_simulator
#  which implements the correct WC 2026 format:
#    12 groups of 4, top-2 + best 8 third-placed = 32 advance,
#    then R32 -> R16 -> QF -> SF -> Final.
#  Falls back to the inline ELO-only simulator if the module is missing.
# ───────────────────────────────────────────────────────────────

def _run_simulator(elo_ratings, n_sims=10_000):
    """
    Run the tournament simulation and return a DataFrame with columns:
      Team, Champion, Final, Semi, Quarter, R16, R32
    (values are fractions 0-1, sorted by Champion desc).
    """
    if _SIM_OK:
        # Merge live ELO ratings with DEFAULT_ELO fallback
        merged_elo = {**DEFAULT_ELO, **elo_ratings}
        sim = TournamentSimulator(elo_ratings=merged_elo, seed=None)
        sim.run(n_simulations=n_sims)
        results = sim.get_results()
        rows = []
        for team, probs in results.items():
            rows.append({
                "Team":     team,
                "Champion": probs["champion_prob"],
                "Final":    probs["final_prob"],
                "Semi":     probs["sf_prob"],
                "Quarter":  probs["qf_prob"],
                "R16":      probs["r16_prob"],
                "R32":      probs["r32_prob"],
            })
        return pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)
    else:
        # Inline ELO-only fallback (48 teams, 16 groups of 3)
        teams = list(WC_2026_TEAMS)
        if not teams:
            return pd.DataFrame()

        default_elo = 1500.0
        ratings = {t: elo_ratings.get(t, default_elo) for t in teams}

        counts = {t: {"champion": 0, "final": 0, "semi": 0, "quarter": 0,
                      "r16": 0, "r32": 0} for t in teams}

        def sim_knockout(t1, t2):
            pa, _, pb = elo_wdl(ratings[t1], ratings[t2])
            return t1 if random.random() < pa / (pa + pb) else t2

        for _ in range(n_sims):
            pool = teams[:]
            random.shuffle(pool)
            survivors = []
            for g in range(16):
                grp = pool[g * 3: g * 3 + 3]
                pts = {t: 0 for t in grp}
                for i in range(len(grp)):
                    for j in range(i + 1, len(grp)):
                        pa, pd_, pb = elo_wdl(ratings[grp[i]], ratings[grp[j]])
                        r = random.random()
                        if r < pa:
                            pts[grp[i]] += 3
                        elif r < pa + pd_:
                            pts[grp[i]] += 1
                            pts[grp[j]] += 1
                        else:
                            pts[grp[j]] += 3
                top2 = sorted(grp, key=lambda t: pts[t], reverse=True)[:2]
                survivors.extend(top2)

            def run_round(competitors, label):
                winners = []
                for k in range(0, len(competitors), 2):
                    if k + 1 >= len(competitors):
                        winners.append(competitors[k])
                        counts[competitors[k]][label] += 1
                        continue
                    t1, t2 = competitors[k], competitors[k + 1]
                    counts[t1][label] += 1
                    counts[t2][label] += 1
                    winners.append(sim_knockout(t1, t2))
                return winners

            r32   = run_round(survivors, "r32")
            r16   = run_round(r32,       "r16")
            qf    = run_round(r16,       "quarter")
            sf    = run_round(qf,        "semi")
            final = run_round(sf,        "final")
            if final:
                counts[final[0]]["champion"] += 1

        rows = []
        for team in teams:
            c = counts[team]
            rows.append({
                "Team":     team,
                "Champion": c["champion"] / n_sims,
                "Final":    c["final"]    / n_sims,
                "Semi":     c["semi"]     / n_sims,
                "Quarter":  c["quarter"]  / n_sims,
                "R16":      c["r16"]      / n_sims,
                "R32":      c["r32"]      / n_sims,
            })
        return pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)


with tab2:
    st.markdown("## Tournament Simulator")
    if _SIM_OK:
        st.markdown(
            "<p style='color:#888;'>10,000 independent Monte Carlo tournaments using the real "
            "WC 2026 format (12 groups of 4) with ELO win probabilities.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:#888;'>10,000 independent Monte Carlo tournaments using ELO win "
            "probabilities (simplified 48-team format — install project modules for full bracket).</p>",
            unsafe_allow_html=True,
        )

    sim_btn = st.button("Run 10,000 Simulations", key="btn_sim")

    if sim_btn:
        with st.spinner("Simulating 10,000 tournaments…"):
            elo_ratings = get_elo_ratings()
            sim_df = _run_simulator(elo_ratings, n_sims=10_000)
            st.session_state["sim_df"] = sim_df

    sim_df = st.session_state.get("sim_df")

    if sim_df is not None and not sim_df.empty:
        # Plotly bar chart — top 16 by champion probability
        top16 = sim_df.head(16)
        fig = px.bar(
            top16,
            x="Team",
            y=top16["Champion"] * 100,
            labels={"y": "Champion Probability (%)"},
            title="Top 16 Teams by Champion Probability",
            color=top16["Champion"] * 100,
            color_continuous_scale=[[0, "#1a1a1a"], [0.3, "#005533"], [1, ACCENT]],
        )
        fig.update_layout(
            **_LAYOUT,
            xaxis_tickangle=-35,
            coloraxis_showscale=False,
            yaxis=dict(ticksuffix="%"),
            height=430,
        )
        fig.update_traces(
            text=[f"{v:.1f}%" for v in top16["Champion"] * 100],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        n_teams = len(sim_df)
        st.markdown(f"### All {n_teams} Teams — Stage Probabilities")
        display = sim_df.copy()
        for col in ["Champion", "Final", "Semi", "Quarter", "R16", "R32"]:
            display[col] = (display[col] * 100).round(1)
        display.index = range(1, len(display) + 1)

        st.dataframe(
            display,
            column_config={
                "Team":     st.column_config.TextColumn("Team", width="medium"),
                "Champion": st.column_config.ProgressColumn(
                    "Champion %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Final":    st.column_config.ProgressColumn(
                    "Final %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Semi":     st.column_config.ProgressColumn(
                    "Semi %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Quarter":  st.column_config.ProgressColumn(
                    "QF %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "R16":      st.column_config.NumberColumn("R16 %", format="%.1f%%"),
                "R32":      st.column_config.NumberColumn("R32 %", format="%.1f%%"),
            },
            use_container_width=True,
            height=600,
        )

    else:
        st.markdown(
            "<div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;"
            "padding:50px;text-align:center;color:#555;'>"
            "<p style='font-size:1.1rem;'>Click <strong>Run 10,000 Simulations</strong> to start</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────
#  TAB 3 — TEAM INTEL
#
#  ELO rating: live from get_elo_ratings() (processes historical DB).
#  Squad quality: reads FBref parquet cache via load_cached_squad().
#  Psych signals: get_psych_signals() — tries parquet first, then
#    queries SQLite wc2026.db WHERE reviewed = 1.
#  Recent form: get_recent_form() — loads international results CSV/parquet.
# ───────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## Team Intel")

    intel_team = st.selectbox(
        "Select Team",
        TEAMS_SORTED,
        index=TEAMS_SORTED.index("Brazil") if "Brazil" in TEAMS_SORTED else 0,
        key="t3_team",
    )

    elo_ratings = get_elo_ratings()
    elo_val = elo_ratings.get(intel_team, None)

    col_a, col_b = st.columns([1, 2])

    # ── Left column: ELO + squad quality ──────────────────────
    with col_a:
        st.markdown("### ELO Rating")
        if elo_val is not None:
            wc_elos = [(t, elo_ratings.get(t, 1500)) for t in WC_2026_TEAMS]
            rank = sorted(wc_elos, key=lambda x: x[1], reverse=True)
            rank_pos = next((i + 1 for i, (t, _) in enumerate(rank) if t == intel_team), "?")
            st.markdown(
                f"<div class='metric-card'>"
                f"<h2>{elo_val:.0f}</h2>"
                f"<p>ELO Rating &nbsp;|&nbsp; WC Rank #{rank_pos} / {len(WC_2026_TEAMS)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No ELO data available — run the data pipeline first.")

        st.markdown("### Squad Quality Score")
        try:
            from src.pipeline.fbref_scraper import load_cached_squad, compute_squad_quality_score
            squad_df = load_cached_squad(intel_team)
            if not squad_df.empty:
                quality = compute_squad_quality_score(squad_df)
                for label, key in [
                    ("Top 11 Avg", "top11_avg_rating"),
                    ("Attack",     "att_rating"),
                    ("Midfield",   "mid_rating"),
                    ("Defence",    "def_rating"),
                    ("GK",         "gk_rating"),
                    ("Depth",      "depth_score"),
                ]:
                    val = quality.get(key)
                    if val is not None:
                        st.metric(label, f"{val:.1f}")
            else:
                st.info("FBref squad data not cached for this team.")
        except ImportError:
            st.info("Squad scoring module not available.")
        except Exception as exc:
            st.warning(f"Squad data error: {exc}", icon="⚠️")

    # ── Right column: psych risk + recent form ─────────────────
    with col_b:
        st.markdown("### Psych Risk Level")

        # get_psych_signals() queries SQLite wc2026.db (reviewed=1) or parquet cache
        psych_df = get_psych_signals()
        psych_team = pd.DataFrame()

        if not psych_df.empty and "team_name" in psych_df.columns:
            psych_team = psych_df[psych_df["team_name"] == intel_team].copy()

        if not psych_team.empty:
            max_sev  = psych_team["severity"].max() if "severity" in psych_team.columns else 0
            avg_sent = psych_team["sentiment_score"].mean() if "sentiment_score" in psych_team.columns else 0

            if max_sev >= 4 or avg_sent < -0.3:
                risk_label, risk_class, risk_delta = "HIGH", "risk-high", "↑ elevated"
            elif max_sev >= 2 or avg_sent < -0.1:
                risk_label, risk_class, risk_delta = "MEDIUM", "risk-medium", "→ moderate"
            else:
                risk_label, risk_class, risk_delta = "LOW", "risk-low", "↓ stable"

            st.metric(
                "Overall Psych Risk",
                f"{risk_label}",
                delta=risk_delta,
                delta_color="inverse",
            )
            st.markdown(
                f"<p style='color:#888;'>{len(psych_team)} signal(s) found · "
                f"avg sentiment <strong>{avg_sent:+.2f}</strong> · "
                f"max severity <strong>{max_sev:.0f}/5</strong></p>",
                unsafe_allow_html=True,
            )
        else:
            st.metric("Overall Psych Risk", "LOW", delta="↓ no signals", delta_color="off")
            st.info("No psych signals found. Run `python -m src.psych.collector` to collect data.")

        st.markdown("### Recent Form — Last 5 Matches")
        form = get_recent_form(intel_team, n=5)

        if form:
            result_color = {"W": "#00ff87", "D": "#ffa500", "L": "#ff4444"}
            result_bg    = {"W": "rgba(0,255,135,0.08)", "D": "rgba(255,165,0,0.08)", "L": "rgba(255,68,68,0.08)"}

            for match in form:
                r = match["result"]
                bg    = result_bg.get(r, "#1a1a1a")
                col   = result_color.get(r, "#555")
                tcol  = result_color.get(r, "#aaa")
                score = match['score']
                opp   = match['opponent']
                dt    = match['date']
                st.markdown(
                    f"<div style='background:{bg};border-left:3px solid "
                    f"{col};border-radius:4px;padding:8px 14px;margin:4px 0;'>"
                    f"<span style='color:{tcol};font-weight:700;'>{r}</span>"
                    f"&nbsp;&nbsp;<span style='color:#e0e0e0;'>{intel_team} {score} {opp}</span>"
                    f"<span style='color:#555;font-size:0.8rem;float:right;'>{dt}</span></div>",
                    unsafe_allow_html=True,
                )

            wins   = sum(1 for m in form if m["result"] == "W")
            draws  = sum(1 for m in form if m["result"] == "D")
            losses = sum(1 for m in form if m["result"] == "L")
            st.markdown(
                f"<p style='color:#888;margin-top:8px;'>"
                f"Last 5: <strong style='color:#00ff87;'>{wins}W</strong>"
                f" <strong style='color:#ffa500;'>{draws}D</strong>"
                f" <strong style='color:#ff4444;'>{losses}L</strong>"
                f" — {wins*3+draws}/15 pts</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No recent match data. Run the data loader pipeline first.")

# Run with: streamlit run dashboard/app.py
