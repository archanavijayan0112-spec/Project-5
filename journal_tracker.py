"""
MoodLens — Journal Tracker Example

Track your emotional journey over multiple journal entries.
Saves a mood history to JSON and prints a timeline summary.

Run: python examples/journal_tracker.py
"""

import json
import datetime
from pathlib import Path
from moodlens import MoodLens

JOURNAL_FILE = Path("examples/journal_history.json")

SAMPLE_ENTRIES = [
    ("2024-01-01", "New year, new beginnings. I feel hopeful and excited about everything ahead. There's so much possibility."),
    ("2024-01-08", "Work has been stressful. My boss criticized my project in front of everyone. I feel embarrassed and angry."),
    ("2024-01-15", "Had a long talk with my best friend. Feeling much better now — grateful for the support and connection."),
    ("2024-01-22", "Received some bad news about a family member's health. I'm worried and scared. Trying to stay calm."),
    ("2024-01-29", "Finished a big personal project. Proud of myself. It wasn't perfect but I did it and that matters."),
]


def load_history():
    if JOURNAL_FILE.exists():
        return json.loads(JOURNAL_FILE.read_text())
    return []


def save_history(history):
    JOURNAL_FILE.parent.mkdir(exist_ok=True)
    JOURNAL_FILE.write_text(json.dumps(history, indent=2))


def analyze_journal(entries: list[tuple[str, str]], lens: MoodLens) -> list[dict]:
    results = []
    for date_str, text in entries:
        profile = lens.analyze(text)
        results.append({
            "date": date_str,
            "text_snippet": text[:80] + ("…" if len(text) > 80 else ""),
            "dominant_emotion": profile.dominant_emotion,
            "sentiment_label": profile.sentiment_label,
            "sentiment_score": profile.sentiment_score,
            "emotional_intensity": profile.emotional_intensity,
            "top_emotions": dict(list(profile.emotions.items())[:3]),
            "mood_summary": profile.mood_summary,
        })
    return results


def print_timeline(history: list[dict]):
    SENTIMENT_SYMBOLS = {"Positive": "▲", "Neutral": "●", "Negative": "▼"}
    EMOTION_COLORS_CLI = {
        "joy": "😊", "sadness": "😢", "anger": "😠",
        "fear": "😨", "surprise": "😲", "disgust": "🤢",
        "trust": "🤝", "anticipation": "🔮",
    }
    print("\n" + "═" * 65)
    print("  📔  Mood Journal Timeline")
    print("═" * 65)

    scores = [e["sentiment_score"] for e in history]
    avg_score = sum(scores) / len(scores) if scores else 0
    trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"

    for entry in history:
        sym = SENTIMENT_SYMBOLS.get(entry["sentiment_label"], "●")
        emoji = EMOTION_COLORS_CLI.get(entry["dominant_emotion"], "🧠")
        bar_val = (entry["sentiment_score"] + 1) / 2  # map -1..1 → 0..1
        bar = "█" * round(bar_val * 15) + "░" * (15 - round(bar_val * 15))
        print(f"\n  {entry['date']}  {sym} {emoji}  {entry['dominant_emotion'].title():<14}  {bar}  {entry['sentiment_score']:+.2f}")
        print(f"  ├─ {entry['text_snippet']}")
        print(f"  └─ {entry['mood_summary'][:90]}…" if len(entry['mood_summary']) > 90 else f"  └─ {entry['mood_summary']}")

    print("\n" + "─" * 65)
    print(f"  Entries analyzed   : {len(history)}")
    print(f"  Average sentiment  : {avg_score:+.3f}")
    print(f"  Mood trend         : {trend.upper()}")
    dominant_emotions = [e["dominant_emotion"] for e in history]
    most_common = max(set(dominant_emotions), key=dominant_emotions.count)
    print(f"  Most frequent mood : {most_common.title()}")
    print("═" * 65 + "\n")


def main():
    print("MoodLens — Journal Mood Tracker\n")
    lens = MoodLens(use_transformer=False)

    # Analyze sample entries
    print(f"Analyzing {len(SAMPLE_ENTRIES)} journal entries…")
    history = analyze_journal(SAMPLE_ENTRIES, lens)
    save_history(history)
    print(f"✓ Saved to {JOURNAL_FILE}")

    print_timeline(history)

    # Generate HTML report for the most recent entry
    from moodlens.visualizer import MoodVisualizer
    last_text = SAMPLE_ENTRIES[-1][1]
    last_profile = lens.analyze(last_text)
    viz = MoodVisualizer(last_profile)
    viz.save_html_report("examples/journal_report.html")
    print("✓ HTML report → examples/journal_report.html")


if __name__ == "__main__":
    main()
