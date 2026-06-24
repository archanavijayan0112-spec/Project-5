"""
MoodLens Visualizer — Generate charts and HTML reports from EmotionProfiles.

Usage:
    from moodlens import MoodLens
    from moodlens.visualizer import MoodVisualizer

    lens    = MoodLens()
    profile = lens.analyze("I feel hopeful yet anxious about the future.")
    viz     = MoodVisualizer(profile)
    viz.save_radar_chart("radar.png")
    viz.save_html_report("report.html")
"""

import json
from pathlib import Path
from moodlens.analyzer import EmotionProfile

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

EMOTION_COLORS = {
    "joy":          "#FFD700",
    "sadness":      "#4A90D9",
    "anger":        "#E74C3C",
    "fear":         "#8E44AD",
    "surprise":     "#F39C12",
    "disgust":      "#27AE60",
    "trust":        "#1ABC9C",
    "anticipation": "#E67E22",
}


class MoodVisualizer:
    def __init__(self, profile: EmotionProfile):
        self.profile = profile

    # ── Matplotlib charts ──────────────────────────────────────────────────

    def save_radar_chart(self, path: str = "radar.png"):
        """Save a Plutchik radar/spider chart as PNG."""
        if not MATPLOTLIB:
            print("matplotlib not installed. Run: pip install matplotlib")
            return
        emotions = list(self.profile.emotions.items())
        if not emotions:
            return
        labels = [e[0].title() for e in emotions]
        values = [e[1] for e in emotions]
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")

        ax.plot(angles, values, "o-", linewidth=2, color="#58A6FF")
        ax.fill(angles, values, alpha=0.25, color="#58A6FF")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="white", size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="#888", size=8)
        ax.tick_params(colors="white")
        ax.spines["polar"].set_color("#30363D")
        ax.grid(color="#30363D", linestyle="--", alpha=0.5)
        ax.set_title(f"Emotion Radar — {self.profile.dominant_emotion.title()}",
                     color="white", size=13, pad=20)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"✓ Radar chart saved → {path}")

    def save_bar_chart(self, path: str = "emotions_bar.png"):
        """Save a horizontal bar chart of all emotions."""
        if not MATPLOTLIB:
            print("matplotlib not installed.")
            return
        emotions = list(self.profile.emotions.items())
        if not emotions:
            return
        labels = [e[0].title() for e in emotions]
        values = [e[1] for e in emotions]
        colors = [EMOTION_COLORS.get(e[0], "#888") for e in emotions]

        fig, ax = plt.subplots(figsize=(8, max(3, len(emotions) * 0.5 + 1)))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")

        bars = ax.barh(labels, values, color=colors, edgecolor="#30363D", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", color="white", fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Score", color="#888")
        ax.set_title("Emotion Distribution", color="white", size=13)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"✓ Bar chart saved → {path}")

    # ── HTML report ────────────────────────────────────────────────────────

    def save_html_report(self, path: str = "report.html"):
        """Generate a self-contained HTML report with inline charts."""
        p = self.profile
        emotions_json = json.dumps(p.emotions)
        sent_json     = json.dumps([s["sentiment"] for s in p.sentence_sentiments])
        sent_labels   = json.dumps([s["sentence"][:40] + "…" for s in p.sentence_sentiments])

        sent_color_map = {"Positive": "#2EA043", "Negative": "#DA3633", "Neutral": "#D29922"}
        sent_color = sent_color_map.get(p.sentiment_label, "#888")

        insights_html = "".join(f"<li>{i}</li>" for i in p.psychological_insights)
        recs_html     = "".join(f"<li>{r}</li>" for r in p.recommendations)
        phrases_html  = "".join(f'<span class="chip">{ph}</span>' for ph in p.key_phrases)
        emotion_rows  = "".join(
            f'<tr><td class="emo-label">{e.title()}</td>'
            f'<td><div class="bar-wrap"><div class="bar-fill" style="width:{v*100:.1f}%;'
            f'background:{EMOTION_COLORS.get(e,"#888")}"></div></div></td>'
            f'<td class="score">{v:.3f}</td></tr>'
            for e, v in p.emotions.items()
        )
        sentence_rows = "".join(
            f'<tr class="sent-{s["label"].lower()}">'
            f'<td class="snum">{s["index"]}</td>'
            f'<td>{s["sentence"]}</td>'
            f'<td class="spol">{s["sentiment"]:+.2f}</td></tr>'
            for s in p.sentence_sentiments
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoodLens Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0D1117;color:#E6EDF3;line-height:1.6}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 20px}}
h1{{font-size:1.8rem;color:#58A6FF;margin-bottom:4px}}
.tagline{{color:#8B949E;margin-bottom:32px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:#161B22;border:1px solid #30363D;border-radius:10px;padding:20px}}
.card h2{{font-size:0.85rem;color:#8B949E;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}}
.big-stat{{font-size:2rem;font-weight:700;color:#58A6FF}}
.sentiment{{font-size:1.4rem;font-weight:700;color:{sent_color}}}
table{{width:100%;border-collapse:collapse;font-size:0.9rem}}
th{{text-align:left;color:#8B949E;font-weight:500;padding:4px 8px;border-bottom:1px solid #30363D}}
td{{padding:6px 8px;border-bottom:1px solid #21262D}}
.emo-label{{font-weight:500;width:110px}}
.bar-wrap{{background:#21262D;border-radius:4px;height:10px;width:100%}}
.bar-fill{{height:10px;border-radius:4px;transition:width .3s}}
.score{{color:#8B949E;width:50px;text-align:right;font-variant-numeric:tabular-nums}}
ul{{padding-left:20px;color:#C9D1D9}}
li{{margin:4px 0}}
.chip{{display:inline-block;background:#21262D;border:1px solid #30363D;border-radius:20px;
       padding:3px 12px;margin:3px;font-size:0.85rem;color:#8B949E}}
canvas{{max-height:280px}}
.sent-positive td{{color:#2EA043}}
.sent-negative td{{color:#DA3633}}
.spol{{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}}
.snum{{color:#8B949E;width:30px}}
footer{{margin-top:40px;text-align:center;color:#30363D;font-size:0.8rem}}
</style>
</head>
<body>
<div class="wrap">
  <h1>🔬 MoodLens Analysis Report</h1>
  <p class="tagline">AI-powered emotion &amp; sentiment analysis</p>

  <div class="grid">
    <div class="card">
      <h2>Dominant Emotion</h2>
      <div class="big-stat">{p.dominant_emotion.title()}</div>
    </div>
    <div class="card">
      <h2>Overall Sentiment</h2>
      <div class="sentiment">{p.sentiment_label}</div>
      <div style="color:#8B949E;font-size:0.85rem">Score: {p.sentiment_score:+.3f}</div>
    </div>
    <div class="card">
      <h2>Subjectivity</h2>
      <div class="big-stat">{p.subjectivity:.0%}</div>
    </div>
    <div class="card">
      <h2>Emotional Intensity</h2>
      <div class="big-stat">{p.emotional_intensity:.0%}</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:24px">
    <h2>💬 Mood Summary</h2>
    <p style="margin-top:8px;color:#C9D1D9;font-style:italic">{p.mood_summary}</p>
  </div>

  <div class="grid">
    <div class="card">
      <h2>🎭 Emotion Breakdown</h2>
      <table style="margin-top:8px">
        <tr><th>Emotion</th><th>Distribution</th><th>Score</th></tr>
        {emotion_rows}
      </table>
    </div>
    <div class="card">
      <h2>🕸️ Radar Chart</h2>
      <canvas id="radar"></canvas>
    </div>
  </div>

  <div class="card" style="margin-bottom:24px">
    <h2>📈 Sentiment Timeline</h2>
    <canvas id="timeline" style="max-height:180px;margin:12px 0"></canvas>
    <table style="margin-top:8px;font-size:0.83rem">
      <tr><th>#</th><th>Sentence</th><th>Pol.</th></tr>
      {sentence_rows}
    </table>
  </div>

  <div class="card" style="margin-bottom:24px">
    <h2>🔑 Key Phrases</h2>
    <div style="margin-top:8px">{phrases_html}</div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>🧠 Psychological Insights</h2>
      <ul style="margin-top:8px">{insights_html}</ul>
    </div>
    <div class="card">
      <h2>💡 Recommendations</h2>
      <ul style="margin-top:8px">{recs_html}</ul>
    </div>
  </div>

  <footer>Generated by MoodLens v1.0.0</footer>
</div>

<script>
const COLORS = {json.dumps(EMOTION_COLORS)};
const emotions = {emotions_json};
const labels = Object.keys(emotions);
const values = Object.values(emotions);
const bgColors = labels.map(l => COLORS[l] || '#888');

// Radar
new Chart(document.getElementById('radar'), {{
  type: 'radar',
  data: {{
    labels: labels.map(l=>l[0].toUpperCase()+l.slice(1)),
    datasets: [{{data:values, backgroundColor:'rgba(88,166,255,0.2)',
                  borderColor:'#58A6FF', pointBackgroundColor:bgColors, borderWidth:2}}]
  }},
  options: {{
    scales: {{r: {{min:0, max:1,
      angleLines:{{color:'#30363D'}},
      grid:{{color:'#30363D'}},
      ticks:{{color:'#8B949E',font:{{size:10}}}},
      pointLabels:{{color:'#E6EDF3',font:{{size:11}}}}
    }}}},
    plugins:{{legend:{{display:false}}}},
    animation:{{duration:800}}
  }}
}});

// Timeline
const sentVals = {sent_json};
const sentLabels = {sent_labels};
new Chart(document.getElementById('timeline'), {{
  type:'line',
  data:{{
    labels:sentLabels,
    datasets:[{{data:sentVals, fill:true,
      borderColor:'#58A6FF', backgroundColor:'rgba(88,166,255,0.1)',
      tension:0.3, pointRadius:4, pointBackgroundColor:'#58A6FF'}}]
  }},
  options:{{
    scales:{{
      x:{{display:false}},
      y:{{min:-1,max:1, grid:{{color:'#30363D'}},
         ticks:{{color:'#8B949E'}}
      }}
    }},
    plugins:{{legend:{{display:false}}}},
    animation:{{duration:600}}
  }}
}});
</script>
</body>
</html>"""
        Path(path).write_text(html, encoding="utf-8")
        print(f"✓ HTML report saved → {path}")
