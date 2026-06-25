"""
MoodLens Examples — Quick-start demos
Run: python examples/demo.py
"""

from moodlens import MoodLens
from moodlens.visualizer import MoodVisualizer

print("=" * 60)
print("  MoodLens — Quick Start Demo")
print("=" * 60)

lens = MoodLens(use_transformer=False)   # set True for deep-learning mode

# ── 1. Single text ────────────────────────────────────────────────
print("\n[1] Single text analysis")
text = (
    "I finally got the job offer I've been waiting for! "
    "I'm so thrilled and grateful. But I'm also a little nervous "
    "about the big change ahead."
)
profile = lens.analyze(text)
print(f"    Text             : {text[:70]}…")
print(f"    Dominant Emotion : {profile.dominant_emotion}")
print(f"    Sentiment        : {profile.sentiment_label} ({profile.sentiment_score:+.3f})")
print(f"    Complexity       : {profile.emotional_complexity} emotions active")
print(f"    Summary          : {profile.mood_summary}")

# ── 2. Emotion breakdown ───────────────────────────────────────────
print("\n[2] Emotion breakdown")
for emotion, score in profile.emotions.items():
    bar = "█" * round(score * 20) + "░" * (20 - round(score * 20))
    print(f"    {emotion:<14} {bar}  {score:.3f}")

# ── 3. Sentence timeline ───────────────────────────────────────────
print("\n[3] Sentence-level timeline")
for s in profile.sentence_sentiments:
    label_symbol = "+" if s["label"] == "Positive" else ("-" if s["label"] == "Negative" else "~")
    print(f"    [{label_symbol}] {s['sentiment']:+.2f}  {s['sentence'][:70]}")

# ── 4. Compare two texts ───────────────────────────────────────────
print("\n[4] Comparing two texts")
text_a = "I am devastated. Everything I worked for has fallen apart. I feel hopeless."
text_b = "Despite the setback, I feel determined. I know I can rebuild and come back stronger."
comparison = lens.compare(text_a, text_b)
print(f"    Text A dominant  : {comparison['profile_a']['dominant_emotion']}")
print(f"    Text B dominant  : {comparison['profile_b']['dominant_emotion']}")
print(f"    Sentiment shift  : {comparison['sentiment_shift']:+.3f}")
print(f"    Summary          : {comparison['summary']}")

# ── 5. Batch analysis ──────────────────────────────────────────────
print("\n[5] Batch analysis (3 texts)")
batch_texts = [
    "I love spending time with my family.",
    "The traffic was absolutely infuriating today.",
    "I'm not sure how I feel about all of this.",
]
profiles = lens.analyze_batch(batch_texts)
for i, (t, p) in enumerate(zip(batch_texts, profiles), 1):
    print(f"    [{i}] {p.dominant_emotion:<14}  {t[:55]}…" if len(t) > 55 else f"    [{i}] {p.dominant_emotion:<14}  {t}")

# ── 6. Generate HTML report ────────────────────────────────────────
print("\n[6] Generating HTML report → examples/report.html")
try:
    viz = MoodVisualizer(profile)
    viz.save_html_report("examples/report.html")
    print("    ✓ Open examples/report.html in your browser.")
except Exception as e:
    print(f"    ⚠ Could not generate report: {e}")

# ── 7. Psychological insights ──────────────────────────────────────
print("\n[7] Psychological insights")
for insight in profile.psychological_insights:
    print(f"    → {insight}")

print("\n[8] Recommendations")
for rec in profile.recommendations:
    print(f"    • {rec}")

print("\n" + "=" * 60)
print("  Done! Try the CLI:  python cli.py --interactive")
print("  Or the API:         uvicorn api:app --reload")
print("=" * 60)
