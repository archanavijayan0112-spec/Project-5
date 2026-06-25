# 🔬 MoodLens

> **AI-powered Emotion & Sentiment Analysis Toolkit for Python**

[![CI](https://github.com/yourusername/moodlens/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/moodlens/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MoodLens analyzes any text — journal entries, tweets, product reviews, chat messages — and returns a **rich psychological profile**: emotion scores across Plutchik's 8-emotion wheel, sentence-level sentiment timelines, subjectivity, key phrases, psychological insights, and actionable recommendations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 **Plutchik Emotion Wheel** | Scores across joy, sadness, anger, fear, surprise, disgust, trust, anticipation |
| 🤖 **Transformer Model** | Powered by `distilroberta-base` for deep-learning accuracy |
| 📈 **Sentence Timeline** | Per-sentence sentiment to track emotional arc |
| 🧠 **Psychological Insights** | Evidence-based commentary on detected patterns |
| 💡 **Recommendations** | Practical suggestions based on dominant emotions |
| 🔑 **Key Phrases** | Automatic extraction of the most meaningful phrases |
| 📊 **HTML Reports** | Self-contained visual reports with interactive charts |
| 🌐 **REST API** | FastAPI server with `/analyze`, `/compare`, `/batch` endpoints |
| 🖥️ **CLI** | Beautiful terminal interface with Rich formatting |
| ⚖️ **Compare Mode** | Side-by-side emotional delta between two texts |
| 📦 **Batch Mode** | Analyze up to 20 texts in one call |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/moodlens.git
cd moodlens

# Install dependencies (CPU-only torch recommended for most users)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Python API

```python
from moodlens import MoodLens

lens = MoodLens()
profile = lens.analyze("I finally got the promotion! I'm thrilled, but also a little nervous.")

print(profile.dominant_emotion)      # → "joy"
print(profile.sentiment_label)       # → "Positive"
print(profile.sentiment_score)       # → 0.62
print(profile.emotional_complexity)  # → 3
print(profile.mood_summary)
# "The text conveys an intense, subjective tone dominated by joy. ..."

for emotion, score in profile.emotions.items():
    print(f"{emotion:15} {score:.3f}")
```

### 3. CLI

```bash
# Analyze a single text
python cli.py "I'm so nervous about tomorrow's exam, but also strangely excited."

# Analyze a file
python cli.py --file my_journal.txt

# Interactive session
python cli.py --interactive

# Compare two texts
python cli.py --compare "I am devastated." "I feel hopeful now."

# Output raw JSON
python cli.py "Some text here" --json

# Skip transformer (faster, lexicon-only)
python cli.py "Some text" --no-transformer
```

### 4. REST API

```bash
uvicorn api:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

```bash
# Analyze
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love building things that make people smile."}'

# Compare
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text_a": "I am broken.", "text_b": "I am healing."}'

# Batch
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I am happy.", "I am angry.", "I feel empty."]}'
```

---

## 📊 Example Output

```
╭──────────────────────────────────────────╮
│  📊 Overview                             │
│  Dominant Emotion:   😊  Joy             │
│  Sentiment:          Positive (+0.621)   │
│  Subjectivity:       74%  very subjective│
│  Emotional Intensity: 68%               │
│  Emotional Complexity: 3 active emotions │
│  Word Count:         22                  │
╰──────────────────────────────────────────╯

 Emotion        Score   Distribution
 ─────────────────────────────────────
 Joy            0.482   ████████████░░░░░░░░
 Anticipation   0.291   ██████░░░░░░░░░░░░░░
 Fear           0.144   ███░░░░░░░░░░░░░░░░░
 Trust          0.083   ██░░░░░░░░░░░░░░░░░░

 Mood Summary:
 The text conveys an intense, subjective tone dominated by joy.
 Overall sentiment is positive and the writing is emotionally layered.

 Psychological Insights:
  → Expression of positive affect — associated with reward and motivation.
  → Coexistence of fear and trust may indicate vulnerability paired with resilience.

 Recommendations:
  ✦ Savor this positive state — gratitude journaling can prolong positive affect.
```

---

## 🗂️ Project Structure

```
moodlens/
├── moodlens/
│   ├── __init__.py          # Package exports
│   ├── analyzer.py          # Core analysis engine
│   └── visualizer.py        # Charts & HTML reports
├── tests/
│   └── test_moodlens.py     # Full test suite (pytest)
├── examples/
│   ├── demo.py              # Quick-start demo
│   └── journal_tracker.py   # Multi-entry mood timeline
├── .github/
│   └── workflows/ci.yml     # GitHub Actions CI
├── cli.py                   # Terminal interface
├── api.py                   # FastAPI REST server
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧠 How It Works

MoodLens uses a **3-layer analysis pipeline**:

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 1 — Lexicon Scoring          │
│  Plutchik keyword seeds × intensity │
│  adverbs → raw emotion scores       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Layer 2 — Transformer Model        │
│  distilroberta-base fine-tuned on   │
│  emotion classification (6 classes) │
└────────────────┬────────────────────┘
                 │  Weighted blend (40/60)
                 ▼
┌─────────────────────────────────────┐
│  Layer 3 — TextBlob                 │
│  Polarity + subjectivity scores     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Post-processing                    │
│  • Sentence timeline                │
│  • Key phrase extraction            │
│  • Flesch-Kincaid readability       │
│  • Psychological insight engine     │
│  • Recommendation engine            │
└─────────────────────────────────────┘
                 │
                 ▼
         EmotionProfile
```

---

## 🔌 API Reference

### `MoodLens(use_transformer=True)`
Initialize the analyzer. Set `use_transformer=False` for faster, lighter lexicon-only mode.

### `.analyze(text: str) → EmotionProfile`
Full single-text analysis.

### `.analyze_batch(texts: list[str]) → list[EmotionProfile]`
Analyze multiple texts.

### `.compare(text_a: str, text_b: str) → dict`
Returns emotion delta and sentiment shift between two texts.

### `EmotionProfile` fields

| Field | Type | Description |
|---|---|---|
| `emotions` | `dict[str, float]` | Normalized scores per emotion |
| `dominant_emotion` | `str` | Highest-scoring emotion |
| `sentiment_score` | `float` | Polarity: -1 (negative) to +1 (positive) |
| `sentiment_label` | `str` | "Positive" / "Neutral" / "Negative" |
| `subjectivity` | `float` | 0 (objective) to 1 (subjective) |
| `emotional_intensity` | `float` | Overall emotional strength (0–1) |
| `emotional_complexity` | `int` | Number of significant emotions (>8%) |
| `sentence_sentiments` | `list[dict]` | Per-sentence polarity timeline |
| `key_phrases` | `list[str]` | Top noun phrases |
| `mood_summary` | `str` | Auto-generated narrative |
| `psychological_insights` | `list[str]` | Evidence-based observations |
| `recommendations` | `list[str]` | Practical suggestions |
| `word_count` | `int` | Token count |
| `readability_grade` | `float` | Flesch-Kincaid grade level |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=moodlens --cov-report=html   # with coverage report
```

---

## 🗺️ Roadmap

- [ ] Multilingual support (Spanish, French, Hindi, Arabic)
- [ ] Streamlit web dashboard
- [ ] CSV / JSONL bulk file ingestion
- [ ] Emotion trend graphs (matplotlib / plotly)
- [ ] Discord & Slack bot integration
- [ ] Fine-tuned model on therapy/journal datasets
- [ ] Real-time WebSocket streaming API

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests for new functionality
4. Run `pytest` and confirm all pass
5. Open a pull request

---

## 📜 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- [TextBlob](https://textblob.readthedocs.io/)
- [NLTK](https://www.nltk.org/)
- [Plutchik's Wheel of Emotions](https://en.wikipedia.org/wiki/Robert_Plutchik)
