"""
cli.py — WC2026 Predictor command-line interface.

Usage examples:
    python cli.py predict --home "France" --away "Brazil" --stage "quarter-final"
    python cli.py simulate --n 10000 [--seed 42]
    python cli.py elo --team "Argentina"
    python cli.py update [--skip-scrape]
    python cli.py psych --team "France" --days 7
"""
import sys
import argparse
import random
import math
from pathlib import Path
from datetime import datetime, timedelta

# ── Project root on path ─────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Rich: optional dependency ─────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.columns import Columns
    from rich.rule import Rule
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None


# ─────────────────────────────────────────────────────────────
# Output helpers: degrade gracefully when rich is absent
# ─────────────────────────────────────────────────────────────

def _print(msg: str = ""):
    if RICH:
        console.print(msg)
    else:
        print(msg)


def _rule(title: str = ""):
    if RICH:
        console.print(Rule(title))
    else:
        width = 60
        if title:
            pad = (width - len(title) - 2) // 2
            print("─" * pad + f" {title} " + "─" * pad)
        else:
            print("─" * width)


def _error(msg: str):
    if RICH:
        console.print(f"[bold red]ERROR:[/bold red] {msg}")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def _warn(msg: str):
    if RICH:
        console.print(f"[yellow]WARNING:[/yellow] {msg}")
    else:
        print(f"WARNING: {msg}")


def _success(msg: str):
    if RICH:
        console.print(f"[bold green]{msg}[/bold green]")
    else:
        print(msg)


def _pct(val: float) -> str:
    """Format a 0-1 probability as a percentage string."""
    return f"{val * 100:.1f}%"


def _bar(val: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Simple ASCII progress bar."""
    filled = round(val * width)
    return fill * filled + empty * (width - filled)


def _check_db():
    """
    Return a live SQLAlchemy session, or exit with a helpful message
    if the database does not exist yet.
    Detects missing DB and suggests running build_pipeline or `cli.py update`.
    """
    from config import DB_PATH
    if not DB_PATH.exists():
        _error(f"Database not found at: {DB_PATH}")
        _print()
        _print("Run the build pipeline first:" if not RICH else
               "Run the build pipeline first:")
        _print("  [bold cyan]python -m src.pipeline.build_pipeline[/bold cyan]" if RICH
               else "  python -m src.pipeline.build_pipeline")
        _print("or use the CLI shortcut:" if not RICH else
               "or use the CLI shortcut:")
        _print("  [bold cyan]python cli.py update[/bold cyan]" if RICH
               else "  python cli.py update")
        sys.exit(1)
    from src.pipeline.database import get_session
    return get_session()


# ─────────────────────────────────────────────────────────────
# Sub-command: predict
# ─────────────────────────────────────────────────────────────

def _stage_to_key(stage: str) -> str:
    """Normalise user-supplied stage string to ELO engine key."""
    s = stage.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "group":           "group",
        "group_stage":     "group",
        "round_of_32":     "round_of_32",
        "round_of_16":     "round_of_16",
        "r16":             "round_of_16",
        "last_16":         "round_of_16",
        "quarter_final":   "quarter_final",
        "quarter-final":   "quarter_final",
        "quarterfinal":    "quarter_final",
        "qf":              "quarter_final",
        "semi_final":      "semi_final",
        "semifinal":       "semi_final",
        "sf":              "semi_final",
        "third_place":     "third_place",
        "final":           "final",
    }
    return mapping.get(s, "group")


def _estimate_score(p_home: float, p_away: float, elo_diff: float) -> tuple:
    """
    Rough expected goals estimate derived from win probabilities.
    Uses a Poisson-inspired heuristic — not a trained model, just a proxy
    for the prediction card when the ML model is not available.

    Returns (home_xg, away_xg).
    """
    # Base rate: average WC match has ~1.35 goals per team
    base = 1.35
    # Stronger team scores slightly more, weaker slightly less
    # Normalise elo_diff so 200 pts ~ 0.25 goal swing
    swing = min(0.4, abs(elo_diff) / 800)
    if elo_diff >= 0:          # home team stronger
        home_xg = base + swing
        away_xg = base - swing
    else:                      # away team stronger
        home_xg = base - swing
        away_xg = base + swing
    return round(home_xg, 2), round(away_xg, 2)


def _confidence_label(p_max: float) -> str:
    """Return a human-readable confidence label from the leading probability."""
    if p_max >= 0.65:
        return "HIGH"
    if p_max >= 0.50:
        return "MEDIUM"
    if p_max >= 0.38:
        return "LOW"
    return "VERY LOW"


def cmd_predict(args):
    """Print a formatted prediction card for a single match."""
    from src.pipeline.elo import EloEngine
    from src.utils.team_name_map import normalize
    from config import CACHE_DIR

    home = normalize(args.home)
    away = normalize(args.away)
    stage_key = _stage_to_key(args.stage)

    # Load cached ELO history if available, otherwise fall back to bare engine
    cache = CACHE_DIR / "elo_history.parquet"
    engine = EloEngine()
    if cache.exists():
        import pandas as pd
        hist = pd.read_parquet(cache)
        # Reconstruct current ratings from the last row per team
        for team_col, elo_col in [("home_team", "elo_home_after"), ("away_team", "elo_away_after")]:
            if team_col in hist.columns and elo_col in hist.columns:
                last = hist.sort_values("date").groupby(team_col)[elo_col].last()
                for team, elo in last.items():
                    engine.ratings[team] = float(elo)
    else:
        _warn("ELO cache not found — using default ratings (1500). Run `python cli.py update` first.")

    try:
        pred = engine.predict_match(home, away, neutral=True)
    except Exception as exc:
        _error(f"Prediction failed: {exc}")
        sys.exit(1)

    elo_diff = pred["home_elo"] - pred["away_elo"]
    home_xg, away_xg = _estimate_score(pred["p_home_win"], pred["p_away_win"], elo_diff)
    p_max = max(pred["p_home_win"], pred["p_away_win"], pred["p_draw"])
    confidence = _confidence_label(p_max)

    stage_display = args.stage.replace("-", " ").title()

    if RICH:
        # ── Rich card ──────────────────────────────────────────
        title_text = Text(f"  {home}  vs  {away}  ", style="bold white")
        subtitle = f"Stage: {stage_display}   |   ELO: {pred['home_elo']:.0f} vs {pred['away_elo']:.0f}"

        # Probability table
        prob_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        prob_table.add_column("Outcome", style="bold", min_width=14)
        prob_table.add_column("Probability", justify="right", min_width=10)
        prob_table.add_column("Bar", min_width=22)

        outcomes = [
            (f"{home} win",  pred["p_home_win"], "green"),
            ("Draw",          pred["p_draw"],     "yellow"),
            (f"{away} win",  pred["p_away_win"], "red"),
        ]
        for label, prob, colour in outcomes:
            bar = f"[{colour}]{_bar(prob)}[/{colour}]"
            prob_table.add_row(label, f"[bold]{_pct(prob)}[/bold]", bar)

        # Score / confidence row
        score_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        score_table.add_column("Key", style="dim")
        score_table.add_column("Value", style="bold")
        score_table.add_row("Expected score", f"{home_xg:.2f} – {away_xg:.2f}")
        score_table.add_row("ELO difference", f"{elo_diff:+.0f} (home advantage)")
        conf_colour = {"HIGH": "green", "MEDIUM": "yellow",
                       "LOW": "orange1", "VERY LOW": "red"}.get(confidence, "white")
        score_table.add_row("Confidence", f"[{conf_colour}]{confidence}[/{conf_colour}]")

        console.print()
        console.print(Panel(
            f"{prob_table}\n{score_table}",
            title=f"[bold yellow]WC2026 PREDICTION CARD[/bold yellow]",
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style="bright_blue",
            padding=(1, 2),
        ))
        console.print()
    else:
        # ── Plain-text card ────────────────────────────────────
        print()
        print("=" * 60)
        print(f"  WC2026 PREDICTION: {home} vs {away}")
        print(f"  Stage: {stage_display}")
        print(f"  ELO:   {pred['home_elo']:.0f} vs {pred['away_elo']:.0f}")
        print("=" * 60)
        print(f"  {home} win:  {_pct(pred['p_home_win']):>6}  {_bar(pred['p_home_win'])}")
        print(f"  Draw:       {_pct(pred['p_draw']):>6}  {_bar(pred['p_draw'])}")
        print(f"  {away} win: {_pct(pred['p_away_win']):>6}  {_bar(pred['p_away_win'])}")
        print()
        print(f"  Expected score : {home_xg:.2f} – {away_xg:.2f}")
        print(f"  ELO difference : {elo_diff:+.0f}")
        print(f"  Confidence     : {confidence}")
        print("=" * 60)
        print()


# ─────────────────────────────────────────────────────────────
# Sub-command: simulate
# ─────────────────────────────────────────────────────────────

def _simulate_one_tournament(engine, teams: list, rng: random.Random) -> str:
    """
    Run a single Monte Carlo WC tournament simulation.
    Groups → R32 → R16 → QF → SF → Final.
    Returns the name of the champion.

    Simplified bracket: 48 teams, 12 groups of 4, top 2 + 8 best 3rd = 32 advance.
    For speed we skip full group-stage simulation and instead seed a 32-team
    knockout bracket weighted by current ELO.
    """
    def win_prob(t1: str, t2: str) -> float:
        e1 = engine.ratings.get(t1, 1500.0)
        e2 = engine.ratings.get(t2, 1500.0)
        return 1.0 / (1.0 + 10 ** (-(e1 - e2) / 400))

    def play_match(t1: str, t2: str, knockout: bool = True) -> str:
        """Return winner. In knockouts there is no draw."""
        p = win_prob(t1, t2)
        return t1 if rng.random() < p else t2

    # ── Group stage: probabilistic qualification ──────────────
    # Sort teams by ELO, split into 12 groups of 4, top 2 advance
    # plus 8 best 3rd-place finishers (simplified: take top 32 by "group points")
    sorted_teams = sorted(teams, key=lambda t: engine.ratings.get(t, 1500.0), reverse=True)

    # Assign teams to groups (pot-based draw simulation)
    groups = [[] for _ in range(12)]
    shuffled = sorted_teams[:]
    rng.shuffle(shuffled)
    for i, team in enumerate(shuffled):
        groups[i % 12].append(team)

    qualifiers = []
    third_place_teams = []

    for group in groups:
        # Play round-robin within group (6 matches)
        points = {t: 0 for t in group}
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                t1, t2 = group[i], group[j]
                p = win_prob(t1, t2)
                r = rng.random()
                # ~25% draw chance weighted by closeness of teams
                elo_gap = abs(engine.ratings.get(t1, 1500) - engine.ratings.get(t2, 1500))
                draw_prob = max(0.10, 0.28 - elo_gap / 4000)
                if r < draw_prob:
                    points[t1] += 1
                    points[t2] += 1
                elif r < draw_prob + (1 - draw_prob) * p:
                    points[t1] += 3
                else:
                    points[t2] += 3

        ranked = sorted(group, key=lambda t: (points[t], engine.ratings.get(t, 1500)), reverse=True)
        qualifiers.append(ranked[0])  # 1st
        qualifiers.append(ranked[1])  # 2nd
        third_place_teams.append((ranked[2], points[ranked[2]]))

    # Best 8 third-place finishers
    third_place_teams.sort(key=lambda x: (x[1], engine.ratings.get(x[0], 1500)), reverse=True)
    qualifiers.extend(t for t, _ in third_place_teams[:8])

    # Shuffle bracket (simple random seeding)
    rng.shuffle(qualifiers)

    # ── Knockout rounds ───────────────────────────────────────
    bracket = qualifiers  # 32 teams
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            winner = play_match(bracket[i], bracket[i + 1], knockout=True)
            next_round.append(winner)
        bracket = next_round

    return bracket[0]


def cmd_simulate(args):
    """Run N tournament simulations and print champion probability table."""
    from src.pipeline.elo import EloEngine
    from config import CACHE_DIR, WC_2026_TEAMS

    n = args.n
    seed = getattr(args, "seed", None)
    rng = random.Random(seed)

    # Load ELO ratings
    engine = EloEngine()
    cache = CACHE_DIR / "elo_history.parquet"
    if cache.exists():
        import pandas as pd
        hist = pd.read_parquet(cache)
        for team_col, elo_col in [("home_team", "elo_home_after"), ("away_team", "elo_away_after")]:
            if team_col in hist.columns and elo_col in hist.columns:
                last = hist.sort_values("date").groupby(team_col)[elo_col].last()
                for team, elo in last.items():
                    engine.ratings[team] = float(elo)
    else:
        _warn("ELO cache not found — using default ratings. Run `python cli.py update` first.")

    teams = [t for t in WC_2026_TEAMS if t in engine.ratings or True]
    # Ensure all WC teams have a rating
    for t in teams:
        if t not in engine.ratings:
            engine.ratings[t] = 1500.0

    champion_counts: dict[str, int] = {}

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]Simulating {n:,} tournaments...", total=n)
            for _ in range(n):
                champ = _simulate_one_tournament(engine, teams, rng)
                champion_counts[champ] = champion_counts.get(champ, 0) + 1
                progress.advance(task)
    else:
        print(f"Simulating {n:,} tournaments...")
        step = max(1, n // 20)
        for i in range(n):
            champ = _simulate_one_tournament(engine, teams, rng)
            champion_counts[champ] = champion_counts.get(champ, 0) + 1
            if (i + 1) % step == 0:
                pct_done = (i + 1) / n * 100
                bar = _bar((i + 1) / n, width=30)
                print(f"\r  {bar} {pct_done:.0f}%", end="", flush=True)
        print()

    # Sort by frequency
    sorted_results = sorted(champion_counts.items(), key=lambda x: x[1], reverse=True)

    if RICH:
        table = Table(
            title=f"[bold yellow]WC2026 Champion Probabilities[/bold yellow]  "
                  f"[dim]({n:,} simulations)[/dim]",
            box=box.ROUNDED,
            border_style="bright_blue",
            show_lines=False,
        )
        table.add_column("Rank", justify="right", style="dim", width=5)
        table.add_column("Team", style="bold white", min_width=22)
        table.add_column("Wins", justify="right", style="cyan", width=7)
        table.add_column("Probability", justify="right", style="bold", width=12)
        table.add_column("Bar", min_width=24)

        top_prob = sorted_results[0][1] / n if sorted_results else 1.0
        for rank, (team, wins) in enumerate(sorted_results, 1):
            prob = wins / n
            rel = prob / top_prob  # relative bar scaled to leader
            colour = (
                "gold1"       if rank == 1 else
                "silver"      if rank == 2 else
                "dark_orange" if rank == 3 else
                "white"
            )
            table.add_row(
                str(rank),
                f"[{colour}]{team}[/{colour}]",
                str(wins),
                f"[bold]{_pct(prob)}[/bold]",
                f"[bright_blue]{_bar(rel, width=22)}[/bright_blue]",
            )

        console.print()
        console.print(table)
        console.print()
    else:
        print()
        print("=" * 58)
        print(f"  WC2026 CHAMPION PROBABILITIES  ({n:,} simulations)")
        print("=" * 58)
        print(f"  {'Rank':>4}  {'Team':<26}  {'Wins':>5}  {'Prob':>6}  Bar")
        print("  " + "-" * 54)
        top_prob = sorted_results[0][1] / n if sorted_results else 1.0
        for rank, (team, wins) in enumerate(sorted_results, 1):
            prob = wins / n
            rel = prob / top_prob
            print(f"  {rank:>4}  {team:<26}  {wins:>5}  {_pct(prob):>6}  {_bar(rel, width=18)}")
        print("=" * 58)
        print()


# ─────────────────────────────────────────────────────────────
# Sub-command: elo
# ─────────────────────────────────────────────────────────────

def cmd_elo(args):
    """Print a team's current ELO and their last 10 matches with ELO changes."""
    from src.utils.team_name_map import normalize
    from config import CACHE_DIR

    team = normalize(args.team)

    # Load ELO history parquet
    cache = CACHE_DIR / "elo_history.parquet"
    if not cache.exists():
        _warn("ELO history cache not found. Run `python cli.py update` first.")
        # Try DB fallback
        sess = _check_db()
        from src.pipeline.database import Team as TeamModel, EloHistory as EloHistoryModel
        db_team = sess.query(TeamModel).filter_by(name=team).first()
        sess.close()
        if not db_team:
            _error(f"Team not found: {team}")
            sys.exit(1)
        current_elo = db_team.elo_current
        recent_matches = []
    else:
        import pandas as pd
        hist = pd.read_parquet(cache)

        # Filter rows where this team appears as home or away
        home_mask = hist["home_team"] == team
        away_mask = hist["away_team"] == team

        home_rows = hist[home_mask].copy()
        home_rows["team_elo_before"] = home_rows["elo_home_before"]
        home_rows["team_elo_after"]  = home_rows["elo_home_after"]
        home_rows["delta"]           = home_rows["delta_home"]
        home_rows["opponent"]        = home_rows["away_team"]
        home_rows["team_goals"]      = home_rows["home_goals"]
        home_rows["opp_goals"]       = home_rows["away_goals"]
        home_rows["venue"]           = "H"

        away_rows = hist[away_mask].copy()
        away_rows["team_elo_before"] = away_rows["elo_away_before"]
        away_rows["team_elo_after"]  = away_rows["elo_away_after"]
        away_rows["delta"]           = away_rows["delta_away"]
        away_rows["opponent"]        = away_rows["home_team"]
        away_rows["team_goals"]      = away_rows["away_goals"]
        away_rows["opp_goals"]       = away_rows["home_goals"]
        away_rows["venue"]           = "A"

        cols = ["date", "opponent", "team_goals", "opp_goals", "venue",
                "team_elo_before", "team_elo_after", "delta", "stage"]
        combined = pd.concat([
            home_rows[cols],
            away_rows[cols],
        ]).sort_values("date", ascending=False)

        if combined.empty:
            _error(f"No match history found for team: {team}")
            sys.exit(1)

        current_elo = float(combined.iloc[0]["team_elo_after"])
        recent_matches = combined.head(10).to_dict("records")

    if RICH:
        # Header panel
        conf_style = "bold cyan"
        from src.utils.team_name_map import get_confederation
        confederation = get_confederation(team)
        header = (
            f"[bold white]{team}[/bold white]\n"
            f"[{conf_style}]{confederation}[/{conf_style}]   "
            f"Current ELO: [bold yellow]{current_elo:.0f}[/bold yellow]"
        )
        console.print()
        console.print(Panel(header, title="[bold yellow]ELO PROFILE[/bold yellow]",
                            border_style="bright_blue", padding=(0, 2)))

        if recent_matches:
            match_table = Table(
                title="[bold]Last 10 Matches[/bold]",
                box=box.SIMPLE_HEAD,
                border_style="dim",
                show_lines=False,
            )
            match_table.add_column("Date",      style="dim",   min_width=12)
            match_table.add_column("Opponent",  style="white", min_width=22)
            match_table.add_column("Score",     justify="center", min_width=7)
            match_table.add_column("H/A",       justify="center", width=4)
            match_table.add_column("Stage",     style="dim",   min_width=10)
            match_table.add_column("ELO Before",justify="right", min_width=10)
            match_table.add_column("ELO After", justify="right", min_width=10)
            match_table.add_column("Change",    justify="right", min_width=8)

            for m in recent_matches:
                tg = int(m.get("team_goals", 0) or 0)
                og = int(m.get("opp_goals", 0) or 0)
                if tg > og:
                    result_style = "green"
                    result_char = "W"
                elif tg == og:
                    result_style = "yellow"
                    result_char = "D"
                else:
                    result_style = "red"
                    result_char = "L"

                delta = float(m.get("delta", 0) or 0)
                delta_str = f"[{'green' if delta >= 0 else 'red'}]{delta:+.1f}[/{'green' if delta >= 0 else 'red'}]"

                date_str = str(m.get("date", ""))[:10]
                score_str = f"[{result_style}]{result_char} {tg}–{og}[/{result_style}]"
                elo_b = float(m.get("team_elo_before", 0) or 0)
                elo_a = float(m.get("team_elo_after", 0) or 0)
                stage = str(m.get("stage", "")).replace("_", " ").capitalize()

                match_table.add_row(
                    date_str,
                    str(m.get("opponent", "")),
                    score_str,
                    str(m.get("venue", "")),
                    stage,
                    f"{elo_b:.0f}",
                    f"{elo_a:.0f}",
                    delta_str,
                )
            console.print(match_table)
        else:
            console.print("  [dim]No match history available in cache.[/dim]")
        console.print()
    else:
        from src.utils.team_name_map import get_confederation
        confederation = get_confederation(team)
        print()
        print("=" * 62)
        print(f"  ELO PROFILE: {team}  [{confederation}]")
        print(f"  Current ELO: {current_elo:.0f}")
        print("=" * 62)
        if recent_matches:
            print(f"  {'Date':<12} {'Opponent':<24} {'Score':<7} {'H/A':<4} {'ΔELO':>7}  {'After':>7}")
            print("  " + "-" * 58)
            for m in recent_matches:
                tg = int(m.get("team_goals", 0) or 0)
                og = int(m.get("opp_goals", 0) or 0)
                result = "W" if tg > og else ("D" if tg == og else "L")
                delta = float(m.get("delta", 0) or 0)
                date_str = str(m.get("date", ""))[:10]
                after = float(m.get("team_elo_after", 0) or 0)
                print(
                    f"  {date_str:<12} {str(m.get('opponent','')):<24} "
                    f"{result} {tg}–{og:<3} {str(m.get('venue','')):<4} "
                    f"{delta:>+7.1f}  {after:>7.0f}"
                )
        else:
            print("  No match history available in cache.")
        print("=" * 62)
        print()


# ─────────────────────────────────────────────────────────────
# Sub-command: update
# ─────────────────────────────────────────────────────────────

def cmd_update(args):
    """Run the full data refresh pipeline (FBref + news scraper)."""
    skip_scrape = getattr(args, "skip_scrape", False)
    _rule("WC2026 DATA REFRESH")
    _print()

    if skip_scrape:
        _warn("--skip-scrape: web scraping steps will be skipped.")
        _print()

    steps = [
        ("Initialising database",          _update_init_db),
        ("Loading historical results",     _update_load_results),
        ("Computing ELO ratings",          _update_elo),
        ("Seeding teams",                  _update_seed_teams),
    ]
    if not skip_scrape:
        steps += [
            ("Scraping FBref squad data",      _update_fbref),
            ("Running news / psych scraper",   _update_news),
        ]

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running pipeline...", total=len(steps))
            _state = {}
            for label, fn in steps:
                progress.update(task, description=f"[cyan]{label}...")
                try:
                    fn(_state)
                    progress.advance(task)
                    console.print(f"  [green]✓[/green] {label}")
                except Exception as exc:
                    console.print(f"  [red]✗[/red] {label}: {exc}")
                    progress.advance(task)
    else:
        _state = {}
        for i, (label, fn) in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {label}...")
            try:
                fn(_state)
                print(f"  OK")
            except Exception as exc:
                print(f"  FAILED: {exc}")

    _print()
    _success("Data refresh complete.")
    _print()


def _update_init_db(state: dict):
    from src.pipeline.build_pipeline import step1_init_db
    state["Session"] = step1_init_db()


def _update_load_results(state: dict):
    from src.pipeline.build_pipeline import step2_load_historical_data
    state["df"] = step2_load_historical_data()


def _update_elo(state: dict):
    from src.pipeline.build_pipeline import step3_compute_elo
    if "df" not in state:
        raise RuntimeError("No historical data loaded — run update from scratch")
    state["engine"] = step3_compute_elo(state["df"])


def _update_seed_teams(state: dict):
    from src.pipeline.build_pipeline import step4_seed_teams
    if "Session" not in state or "engine" not in state:
        raise RuntimeError("DB or ELO engine not ready")
    step4_seed_teams(state["Session"], state["engine"])


def _update_fbref(state: dict):
    from src.pipeline.fbref_scraper import scrape_all_wc_teams
    scrape_all_wc_teams(delay_between_teams=4.0)


def _update_news(state: dict):
    # The psych/news scraper module may not be built yet — graceful fallback
    try:
        from src.psych.news_scraper import run_all  # type: ignore
        run_all()
    except ImportError:
        _warn("News scraper module (src/psych/news_scraper.py) not found — skipping.")


# ─────────────────────────────────────────────────────────────
# Sub-command: psych
# ─────────────────────────────────────────────────────────────

def _sentiment_label(score: float) -> str:
    if score >= 0.4:  return "POSITIVE"
    if score >= 0.1:  return "MILDLY POSITIVE"
    if score >= -0.1: return "NEUTRAL"
    if score >= -0.4: return "MILDLY NEGATIVE"
    return "NEGATIVE"


def _sentiment_colour(score: float) -> str:
    if score >= 0.3:  return "green"
    if score >= 0.0:  return "yellow"
    if score >= -0.3: return "orange1"
    return "red"


def cmd_psych(args):
    """Run news scraper + sentiment analysis and print a psych risk report."""
    from src.utils.team_name_map import normalize
    from config import PSYCH_NEGATIVE_KEYWORDS, PSYCH_POSITIVE_KEYWORDS, CACHE_DIR

    team = normalize(args.team)
    days = args.days

    _rule(f"Psych Risk Report — {team}")
    _print()

    # ── Step 1: try to fetch fresh news ───────────────────────
    articles = []
    try:
        from src.psych.news_scraper import fetch_team_news  # type: ignore
        _print(f"Fetching news for [bold]{team}[/bold] (last {days} days)..." if RICH
               else f"Fetching news for {team} (last {days} days)...")
        articles = fetch_team_news(team, days=days)
    except ImportError:
        _warn("News scraper not available. Loading signals from database instead.")
        # Fall back to DB signals
        try:
            sess = _check_db()
            from src.pipeline.database import PsychSignal, Team as TeamModel
            db_team = sess.query(TeamModel).filter_by(name=team).first()
            if db_team:
                cutoff = datetime.utcnow() - timedelta(days=days)
                signals = (
                    sess.query(PsychSignal)
                    .filter(PsychSignal.team_id == db_team.id)
                    .filter(PsychSignal.recorded_at >= cutoff)
                    .order_by(PsychSignal.recorded_at.desc())
                    .all()
                )
                articles = [
                    {
                        "headline": s.headline or "(no headline)",
                        "sentiment_score": s.sentiment_score or 0.0,
                        "risk_category": s.risk_category or "unknown",
                        "severity": s.severity or 1,
                        "source": s.source_type or "db",
                        "text": s.raw_text or "",
                        "date": s.recorded_at,
                    }
                    for s in signals
                ]
            sess.close()
        except Exception as exc:
            _warn(f"Database fallback also failed: {exc}")

    except Exception as exc:
        _warn(f"News fetch failed: {exc}")

    # ── Step 2: keyword-based sentiment if no ML scores ───────
    if not articles:
        _print("[dim]No recent signals found. Try running `python cli.py update` first.[/dim]"
               if RICH else "No recent signals found. Try running `python cli.py update` first.")
        _print()
        return

    # Ensure sentiment_score field exists via simple keyword fallback
    for art in articles:
        if "sentiment_score" not in art or art["sentiment_score"] is None:
            text_lower = (art.get("text", "") + " " + art.get("headline", "")).lower()
            neg = sum(1 for kw in PSYCH_NEGATIVE_KEYWORDS if kw in text_lower)
            pos = sum(1 for kw in PSYCH_POSITIVE_KEYWORDS if kw in text_lower)
            total = neg + pos
            art["sentiment_score"] = (pos - neg) / max(total, 1)

    # ── Step 3: aggregate metrics ─────────────────────────────
    scores = [a["sentiment_score"] for a in articles if a.get("sentiment_score") is not None]
    avg_sentiment = sum(scores) / len(scores) if scores else 0.0

    risk_flags = [a for a in articles
                  if a.get("sentiment_score", 0) < -0.2 or
                     (isinstance(a.get("severity"), int) and a["severity"] >= 3)]
    risk_count = len(risk_flags)

    categories: dict[str, int] = {}
    for a in articles:
        cat = a.get("risk_category", "general") or "general"
        categories[cat] = categories.get(cat, 0) + 1

    overall_risk = (
        "CRITICAL" if avg_sentiment < -0.5 or risk_count >= 5 else
        "HIGH"     if avg_sentiment < -0.3 or risk_count >= 3 else
        "MEDIUM"   if avg_sentiment < -0.1 or risk_count >= 1 else
        "LOW"
    )

    if RICH:
        risk_colours = {"CRITICAL": "red", "HIGH": "orange1", "MEDIUM": "yellow", "LOW": "green"}
        rc = risk_colours.get(overall_risk, "white")
        sc = _sentiment_colour(avg_sentiment)

        # Summary panel
        summary = (
            f"Team: [bold white]{team}[/bold white]   "
            f"Period: last [bold]{days}[/bold] days   "
            f"Articles: [bold]{len(articles)}[/bold]\n"
            f"Avg sentiment: [{sc}]{avg_sentiment:+.3f} ({_sentiment_label(avg_sentiment)})[/{sc}]   "
            f"Risk flags: [bold]{risk_count}[/bold]   "
            f"Overall risk: [{rc}]{overall_risk}[/{rc}]"
        )
        console.print(Panel(summary, title="[bold yellow]PSYCH RISK SUMMARY[/bold yellow]",
                            border_style="bright_blue", padding=(0, 2)))

        # Category breakdown
        if categories:
            cat_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan",
                              title="Signal Categories")
            cat_table.add_column("Category", style="white", min_width=18)
            cat_table.add_column("Count",    justify="right", width=7)
            for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
                cat_table.add_row(cat.replace("_", " ").title(), str(cnt))
            console.print(cat_table)

        # Article list
        art_table = Table(
            title=f"[bold]Recent Articles / Signals[/bold]",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            border_style="dim",
        )
        art_table.add_column("Date",      style="dim",   min_width=12)
        art_table.add_column("Headline",  style="white", min_width=38)
        art_table.add_column("Sentiment", justify="center", min_width=10)
        art_table.add_column("Category",  style="dim",   min_width=14)

        for a in sorted(articles, key=lambda x: x.get("date", datetime.min), reverse=True)[:20]:
            score = a.get("sentiment_score", 0.0) or 0.0
            sc2 = _sentiment_colour(score)
            date_str = str(a.get("date", ""))[:10] if a.get("date") else "—"
            headline = str(a.get("headline", ""))[:60]
            cat = str(a.get("risk_category", "general") or "general").replace("_", " ").title()
            art_table.add_row(
                date_str,
                headline,
                f"[{sc2}]{score:+.2f}[/{sc2}]",
                cat,
            )
        console.print(art_table)
        console.print()
    else:
        print()
        print("=" * 64)
        print(f"  PSYCH RISK REPORT: {team}  (last {days} days)")
        print("=" * 64)
        print(f"  Articles found   : {len(articles)}")
        print(f"  Avg sentiment    : {avg_sentiment:+.3f} ({_sentiment_label(avg_sentiment)})")
        print(f"  Risk flags       : {risk_count}")
        print(f"  Overall risk     : {overall_risk}")
        if categories:
            print()
            print("  Signal categories:")
            for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"    {cat:<20} {cnt}")
        print()
        print(f"  {'Date':<12} {'Sentiment':>10}  {'Category':<16}  Headline")
        print("  " + "-" * 60)
        for a in sorted(articles, key=lambda x: x.get("date", datetime.min), reverse=True)[:20]:
            score = a.get("sentiment_score", 0.0) or 0.0
            date_str = str(a.get("date", ""))[:10] if a.get("date") else "—"
            headline = str(a.get("headline", ""))[:45]
            cat = str(a.get("risk_category", "general") or "general")[:14]
            print(f"  {date_str:<12} {score:>+10.3f}  {cat:<16}  {headline}")
        print("=" * 64)
        print()


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="WC2026 Predictor — command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py predict --home "France" --away "Brazil" --stage "quarter-final"
  python cli.py simulate --n 10000
  python cli.py elo --team "Argentina"
  python cli.py update
  python cli.py psych --team "France" --days 7
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ── predict ───────────────────────────────────────────────
    p_predict = sub.add_parser(
        "predict",
        help="Predict a single match outcome",
        description="Print a formatted prediction card with probabilities, expected score, and confidence.",
    )
    p_predict.add_argument("--home",  required=True, metavar="TEAM",
                           help="Home / first team name")
    p_predict.add_argument("--away",  required=True, metavar="TEAM",
                           help="Away / second team name")
    p_predict.add_argument("--stage", default="group", metavar="STAGE",
                           help="Match stage: group, round-of-16, quarter-final, semi-final, final "
                                "(default: group)")

    # ── simulate ──────────────────────────────────────────────
    p_sim = sub.add_parser(
        "simulate",
        help="Run N Monte Carlo tournament simulations",
        description="Simulate the full WC2026 tournament N times and report champion probabilities.",
    )
    p_sim.add_argument("--n", type=int, default=10000, metavar="N",
                       help="Number of simulations (default: 10000)")
    p_sim.add_argument("--seed", type=int, default=None, metavar="SEED",
                       help="Random seed for reproducibility")

    # ── elo ───────────────────────────────────────────────────
    p_elo = sub.add_parser(
        "elo",
        help="Show a team's ELO rating and recent match history",
        description="Print the team's current ELO and last 10 matches with ELO changes.",
    )
    p_elo.add_argument("--team", required=True, metavar="TEAM",
                       help="Team name (any recognised variant, e.g. 'USA', 'Korea Republic')")

    # ── update ────────────────────────────────────────────────
    p_update = sub.add_parser(
        "update",
        help="Run the full data refresh pipeline (FBref + news scraper)",
        description="Downloads latest results, recomputes ELO, scrapes FBref squad data, "
                    "and runs the news/psych scraper.",
    )
    p_update.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip web scraping step (use only cached/local data)",
    )

    # ── psych ─────────────────────────────────────────────────
    p_psych = sub.add_parser(
        "psych",
        help="Run news scraper and print a psych risk report for a team",
        description="Fetches recent news, analyses sentiment, and prints a psychological risk "
                    "report for the specified team.",
    )
    p_psych.add_argument("--team", required=True, metavar="TEAM",
                         help="Team name")
    p_psych.add_argument("--days", type=int, default=7, metavar="DAYS",
                         help="How many days back to search for news (default: 7)")

    return parser


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

COMMANDS = {
    "predict":  cmd_predict,
    "simulate": cmd_simulate,
    "elo":      cmd_elo,
    "update":   cmd_update,
    "psych":    cmd_psych,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    fn = COMMANDS.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    try:
        fn(args)
    except KeyboardInterrupt:
        _print()
        _warn("Interrupted by user.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        _error(str(exc))
        if "--debug" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
