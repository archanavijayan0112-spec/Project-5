#!/usr/bin/env python3
"""
MoodLens CLI — Analyze emotions from the terminal.

Usage:
  python cli.py "I feel amazing today!"
  python cli.py --file journal.txt
  python cli.py --interactive
  python cli.py --compare "I am happy" "I am sad"
  python cli.py "Your text" --json
  python cli.py "Your text" --no-transformer
"""

import argparse
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional rich output — fallback to plain text if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class _FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip Rich markup for plain print
            text = " ".join(str(a) for a in args)
            import re
            print(re.sub(r'\[.*?\]', '', text))
        def rule(self, title=""):
            print(f"\n{'─'*60}  {title}  {'─'*60}\n")
    console = _FallbackConsole()

from moodlens import MoodLens

EMOTION_EMOJI = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢",
    "trust": "🤝", "anticipation": "🔮", "neutral": "😐",
}

SENTIMENT_COLOR = {
    "Positive": "green", "Negative": "red", "Neutral": "yellow"
}

BAR_FULL  = "█"
BAR_EMPTY = "░"


def emotion_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def print_profile(profile, title: str = "MoodLens Analysis"):
    """Render the EmotionProfile to the terminal."""

    if RICH:
        # ── Header ──────────────────────────────────────────────────────
        console.rule(f"[bold cyan]🔬 {title}[/bold cyan]")

        # ── Overview panel ───────────────────────────────────────────────
        sent_color = SENTIMENT_COLOR.get(profile.sentiment_label, "white")
        dom_emoji  = EMOTION_EMOJI.get(profile.dominant_emotion, "🧠")
        overview = (
            f"[bold]Dominant Emotion:[/bold]  {dom_emoji}  [magenta]{profile.dominant_emotion.title()}[/magenta]\n"
            f"[bold]Sentiment:[/bold]         [{sent_color}]{profile.sentiment_label}[/{sent_color}] "
            f"(score: [bold]{profile.sentiment_score:+.3f}[/bold])\n"
            f"[bold]Subjectivity:[/bold]      {profile.subjectivity:.0%}  "
            f"({'very subjective' if profile.subjectivity > 0.7 else 'balanced'})\n"
            f"[bold]Emotional Intensity:[/bold] {profile.emotional_intensity:.0%}\n"
            f"[bold]Emotional Complexity:[/bold] {profile.emotional_complexity} active emotion(s)\n"
            f"[bold]Word Count:[/bold]       {profile.word_count}"
        )
        console.print(Panel(overview, title="📊 Overview", border_style="cyan", expand=False))

        # ── Emotion wheel ─────────────────────────────────────────────────
        table = Table(title="🎭 Emotion Breakdown", box=box.ROUNDED, border_style="dim")
        table.add_column("Emotion", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Distribution", style="cyan")
        for emotion, score in profile.emotions.items():
            emoji = EMOTION_EMOJI.get(emotion, "•")
            bar   = emotion_bar(score)
            table.add_row(f"{emoji} {emotion.title()}", f"{score:.3f}", bar)
        console.print(table)

        # ── Mood summary ──────────────────────────────────────────────────
        console.print(Panel(f"[italic]{profile.mood_summary}[/italic]",
                            title="💬 Mood Summary", border_style="blue"))

        # ── Key phrases ───────────────────────────────────────────────────
        if profile.key_phrases:
            kp = "  •  ".join(f"[bold]{p}[/bold]" for p in profile.key_phrases)
            console.print(Panel(kp, title="🔑 Key Phrases", border_style="dim"))

        # ── Sentence timeline ─────────────────────────────────────────────
        if len(profile.sentence_sentiments) > 1:
            st = Table(title="📈 Sentence Sentiment Timeline", box=box.SIMPLE, border_style="dim")
            st.add_column("#", style="dim", width=3)
            st.add_column("Snippet")
            st.add_column("Sentiment", justify="right")
            for s in profile.sentence_sentiments[:10]:
                color = SENTIMENT_COLOR.get(s["label"], "white")
                st.add_row(str(s["index"]), s["sentence"][:80],
                           f"[{color}]{s['sentiment']:+.2f}[/{color}]")
            console.print(st)

        # ── Psychological insights ────────────────────────────────────────
        if profile.psychological_insights:
            insights_text = "\n".join(f"  → {i}" for i in profile.psychological_insights)
            console.print(Panel(insights_text, title="🧠 Psychological Insights", border_style="magenta"))

        # ── Recommendations ───────────────────────────────────────────────
        if profile.recommendations:
            recs_text = "\n".join(f"  ✦ {r}" for r in profile.recommendations)
            console.print(Panel(recs_text, title="💡 Recommendations", border_style="green"))

        console.rule("[dim]End of analysis[/dim]")

    else:
        # Plain-text fallback
        print(f"\n=== {title} ===")
        print(f"Dominant Emotion : {profile.dominant_emotion}")
        print(f"Sentiment        : {profile.sentiment_label} ({profile.sentiment_score:+.3f})")
        print(f"Subjectivity     : {profile.subjectivity:.0%}")
        print(f"Intensity        : {profile.emotional_intensity:.0%}")
        print(f"Word Count       : {profile.word_count}")
        print("\nEmotions:")
        for e, s in profile.emotions.items():
            print(f"  {e:<14} {s:.3f}  {emotion_bar(s, 15)}")
        print(f"\nSummary: {profile.mood_summary}")
        if profile.psychological_insights:
            print("\nInsights:")
            for i in profile.psychological_insights:
                print(f"  → {i}")
        if profile.recommendations:
            print("\nRecommendations:")
            for r in profile.recommendations:
                print(f"  • {r}")


def interactive_mode(lens: MoodLens):
    console.print(Panel(
        "[bold cyan]MoodLens Interactive Mode[/bold cyan]\n"
        "Type or paste any text and press Enter. Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
        border_style="cyan"
    ))
    session = 1
    while True:
        try:
            if RICH:
                from rich.prompt import Prompt
                text = Prompt.ask(f"\n[bold green]Text #{session}[/bold green]")
            else:
                text = input(f"\nText #{session}: ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if text.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if not text.strip():
            continue

        try:
            if RICH:
                with Progress(SpinnerColumn(), TextColumn("[cyan]Analyzing…"), transient=True) as p:
                    p.add_task("", total=None)
                    profile = lens.analyze(text)
            else:
                profile = lens.analyze(text)
            print_profile(profile, title=f"Analysis #{session}")
            session += 1
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        prog="moodlens",
        description="MoodLens — AI-powered emotion & sentiment analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file",        "-f", help="Path to a .txt file to analyze")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive session")
    parser.add_argument("--compare",     "-c", nargs=2, metavar=("TEXT_A", "TEXT_B"),
                        help="Compare two texts")
    parser.add_argument("--json",        "-j", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-transformer", action="store_true",
                        help="Skip transformer model (faster, less accurate)")
    parser.add_argument("--version",     "-v", action="version", version="MoodLens 1.0.0")

    args = parser.parse_args()

    # Greeting banner
    if not args.json:
        if RICH:
            console.print(Panel.fit(
                "[bold cyan]🔬 MoodLens[/bold cyan]  [dim]v1.0.0 — AI Emotion & Sentiment Analyzer[/dim]",
                border_style="cyan"
            ))
        else:
            print("MoodLens v1.0.0 — AI Emotion & Sentiment Analyzer\n")

    use_tf = not args.no_transformer
    lens   = MoodLens(use_transformer=use_tf)

    # ── Compare mode ──────────────────────────────────────────────────────
    if args.compare:
        comparison = lens.compare(args.compare[0], args.compare[1])
        if args.json:
            print(json.dumps(comparison, indent=2))
        else:
            print_profile(MoodLens.__new__(MoodLens), "Text A")   # dummy; use dict display
            console.print_json(json.dumps(comparison, indent=2)) if RICH else print(json.dumps(comparison, indent=2))
        return

    # ── Interactive mode ──────────────────────────────────────────────────
    if args.interactive:
        interactive_mode(lens)
        return

    # ── File mode ─────────────────────────────────────────────────────────
    if args.file:
        path = Path(args.file)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        text = path.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(0)

    try:
        if RICH and not args.json:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Analyzing…"), transient=True) as p:
                p.add_task("", total=None)
                profile = lens.analyze(text)
        else:
            profile = lens.analyze(text)

        if args.json:
            print(profile.to_json())
        else:
            print_profile(profile)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
