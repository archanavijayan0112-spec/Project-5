"""
MoodLens - Core Emotion & Sentiment Analysis Engine
Analyzes text using multiple NLP models and returns rich psychological insights.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
import numpy as np

# Download required NLTK data
for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Plutchik's 8 basic emotions → keyword seeds
EMOTION_SEEDS = {
    "joy":        ["happy", "joyful", "excited", "delighted", "cheerful", "thrilled", "wonderful", "love", "elated", "glad", "enjoy", "amazing", "great", "fantastic", "awesome"],
    "sadness":    ["sad", "unhappy", "depressed", "miserable", "sorrowful", "grief", "cry", "tears", "lonely", "heartbroken", "hopeless", "lost", "empty", "melancholy", "despair"],
    "anger":      ["angry", "furious", "rage", "mad", "irritated", "frustrated", "hate", "annoyed", "hostile", "outraged", "livid", "bitter", "resentful", "disgusted"],
    "fear":       ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "dread", "panic", "frightened", "uneasy", "threatened", "horror", "phobia", "alarmed"],
    "surprise":   ["surprised", "shocked", "astonished", "amazed", "unexpected", "sudden", "wow", "unbelievable", "startled", "stunned", "incredible", "whoa"],
    "disgust":    ["disgusting", "revolting", "gross", "awful", "repulsive", "horrible", "nasty", "vile", "sick", "nausea", "loathe", "detest", "yuck", "awful"],
    "trust":      ["trust", "believe", "reliable", "honest", "faithful", "sincere", "loyal", "confident", "certain", "assured", "dependable", "safe", "secure"],
    "anticipation":["excited", "looking forward", "eager", "hopeful", "expect", "anticipate", "await", "soon", "planning", "future", "goal", "dream", "wish", "hope"],
}

# Intensity adverbs
INTENSIFIERS   = {"very", "extremely", "absolutely", "incredibly", "deeply", "truly", "utterly", "overwhelmingly", "totally"}
DIMINISHERS    = {"slightly", "somewhat", "barely", "a little", "kind of", "sort of", "mildly", "vaguely"}

@dataclass
class EmotionProfile:
    emotions: dict[str, float]          = field(default_factory=dict)
    dominant_emotion: str               = ""
    sentiment_score: float              = 0.0        # -1 to +1
    sentiment_label: str                = ""         # Positive / Neutral / Negative
    subjectivity: float                 = 0.0        # 0 = objective, 1 = subjective
    emotional_intensity: float          = 0.0        # 0 to 1
    emotional_complexity: int           = 0          # number of significant emotions
    mood_summary: str                   = ""
    key_phrases: list[str]              = field(default_factory=list)
    sentence_sentiments: list[dict]     = field(default_factory=list)
    word_count: int                     = 0
    readability_grade: float            = 0.0
    psychological_insights: list[str]   = field(default_factory=list)
    recommendations: list[str]          = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


class MoodLens:
    """
    MoodLens: Multi-layer emotion & sentiment analyzer.

    Layers:
      1. Lexicon-based Plutchik emotion scoring
      2. TextBlob polarity + subjectivity
      3. Transformer-based sentiment (optional, GPU-friendly)
      4. Sentence-level timeline
      5. Psychological insight generation
    """

    def __init__(self, use_transformer: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words  = set(stopwords.words("english"))
        self._transformer = None

        if use_transformer:
            try:
                self._transformer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=-1,          # CPU; change to 0 for GPU
                )
                print("✓ Transformer emotion model loaded.")
            except Exception as e:
                print(f"⚠ Transformer not available ({e}). Using lexicon mode only.")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> EmotionProfile:
        """Full analysis pipeline — returns an EmotionProfile."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        text = text.strip()
        profile = EmotionProfile()

        tokens        = self._tokenize(text)
        profile.word_count = len(word_tokenize(text))

        # Layer 1 – Lexicon emotions
        lex_emotions  = self._lexicon_emotions(tokens)

        # Layer 2 – Transformer emotions (if available)
        if self._transformer:
            tf_emotions = self._transformer_emotions(text)
            # Weighted blend: 40 % lexicon, 60 % transformer
            all_keys    = set(lex_emotions) | set(tf_emotions)
            emotions    = {k: round(0.4 * lex_emotions.get(k, 0) + 0.6 * tf_emotions.get(k, 0), 4)
                           for k in all_keys}
        else:
            emotions    = lex_emotions

        # Normalize
        total = sum(emotions.values()) or 1
        profile.emotions = {k: round(v / total, 4) for k, v in sorted(emotions.items(), key=lambda x: -x[1])}
        profile.dominant_emotion   = max(profile.emotions, key=profile.emotions.get) if profile.emotions else "neutral"
        profile.emotional_intensity = round(min(1.0, sum(emotions.values()) / (len(tokens) + 1) * 10), 4)
        profile.emotional_complexity = sum(1 for v in profile.emotions.values() if v > 0.08)

        # Layer 3 – TextBlob polarity/subjectivity
        blob = TextBlob(text)
        profile.sentiment_score = round(blob.sentiment.polarity, 4)
        profile.subjectivity    = round(blob.sentiment.subjectivity, 4)
        profile.sentiment_label = self._polarity_label(profile.sentiment_score)

        # Layer 4 – Sentence-level timeline
        profile.sentence_sentiments = self._sentence_timeline(text)

        # Layer 5 – Key phrases
        profile.key_phrases = self._extract_key_phrases(text, tokens)

        # Layer 6 – Readability
        profile.readability_grade = self._flesch_kincaid(text)

        # Layer 7 – Narrative summary + insights
        profile.mood_summary             = self._mood_summary(profile)
        profile.psychological_insights   = self._insights(profile)
        profile.recommendations          = self._recommendations(profile)

        return profile

    def analyze_batch(self, texts: list[str]) -> list[EmotionProfile]:
        """Analyze multiple texts and return a list of profiles."""
        return [self.analyze(t) for t in texts]

    def compare(self, text_a: str, text_b: str) -> dict:
        """Compare emotional profiles of two texts."""
        pa, pb = self.analyze(text_a), self.analyze(text_b)
        emotions_a, emotions_b = pa.emotions, pb.emotions
        all_keys = set(emotions_a) | set(emotions_b)
        delta = {k: round(emotions_b.get(k, 0) - emotions_a.get(k, 0), 4) for k in all_keys}
        return {
            "profile_a": pa.to_dict(),
            "profile_b": pb.to_dict(),
            "sentiment_shift": round(pb.sentiment_score - pa.sentiment_score, 4),
            "emotion_delta": delta,
            "summary": f"Mood shifted from {pa.dominant_emotion} → {pb.dominant_emotion}."
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> list[str]:
        tokens = word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(t) for t in tokens
                if t.isalpha() and t not in self.stop_words]

    def _lexicon_emotions(self, tokens: list[str]) -> dict[str, float]:
        scores = {e: 0.0 for e in EMOTION_SEEDS}
        token_set = set(tokens)
        for emotion, seeds in EMOTION_SEEDS.items():
            for seed in seeds:
                if seed in token_set:
                    scores[emotion] += 1.0
        return scores

    def _transformer_emotions(self, text: str) -> dict[str, float]:
        # Truncate to 512 tokens for transformer
        truncated = " ".join(text.split()[:400])
        try:
            results = self._transformer(truncated)[0]
            # Map model labels → Plutchik wheel
            label_map = {
                "joy":     "joy",     "sadness": "sadness",
                "anger":   "anger",   "fear":    "fear",
                "surprise":"surprise","disgust": "disgust",
                "neutral": "trust",
            }
            return {label_map.get(r["label"].lower(), r["label"].lower()): r["score"]
                    for r in results}
        except Exception:
            return {}

    def _polarity_label(self, score: float) -> str:
        if score > 0.15:  return "Positive"
        if score < -0.15: return "Negative"
        return "Neutral"

    def _sentence_timeline(self, text: str) -> list[dict]:
        sentences = sent_tokenize(text)
        timeline  = []
        for i, sent in enumerate(sentences):
            blob  = TextBlob(sent)
            pol   = round(blob.sentiment.polarity, 3)
            timeline.append({
                "index":     i + 1,
                "sentence":  sent[:120] + ("…" if len(sent) > 120 else ""),
                "sentiment": pol,
                "label":     self._polarity_label(pol),
            })
        return timeline

    def _extract_key_phrases(self, text: str, tokens: list[str]) -> list[str]:
        blob    = TextBlob(text)
        phrases = list(blob.noun_phrases)
        # Fallback: top frequent content words
        if len(phrases) < 3:
            freq    = Counter(tokens)
            phrases = [w for w, _ in freq.most_common(8)]
        return phrases[:8]

    def _flesch_kincaid(self, text: str) -> float:
        """Approximate Flesch-Kincaid grade level."""
        sentences = sent_tokenize(text)
        words     = word_tokenize(text)
        if not sentences or not words:
            return 0.0
        syllables = sum(self._count_syllables(w) for w in words if w.isalpha())
        asl       = len(words) / len(sentences)          # avg sentence length
        asw       = syllables / max(len(words), 1)       # avg syllables per word
        grade     = round(0.39 * asl + 11.8 * asw - 15.59, 1)
        return max(0.0, grade)

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def _mood_summary(self, p: EmotionProfile) -> str:
        dom = p.dominant_emotion
        sent = p.sentiment_label.lower()
        intensity = "intense" if p.emotional_intensity > 0.5 else "mild"
        complexity = "emotionally layered" if p.emotional_complexity >= 3 else "emotionally focused"
        subj = "subjective" if p.subjectivity > 0.5 else "objective"
        return (f"The text conveys a {intensity}, {subj} tone dominated by {dom}. "
                f"Overall sentiment is {sent} and the writing is {complexity}, "
                f"with {p.emotional_complexity} notable emotional thread(s).")

    def _insights(self, p: EmotionProfile) -> list[str]:
        insights = []
        dom = p.dominant_emotion

        emotion_insights = {
            "joy":          "Expression of positive affect — associated with reward, connection, and motivation.",
            "sadness":      "Sadness signals loss or unmet needs. It often drives reflection and empathy.",
            "anger":        "Anger frequently signals a perceived injustice or boundary violation.",
            "fear":         "Fear activates protective responses; may indicate perceived threat or uncertainty.",
            "surprise":     "Surprise indicates an unexpected cognitive update — can be positive or negative.",
            "disgust":      "Disgust is a strong rejection signal, often tied to moral or physical aversion.",
            "trust":        "High trust language correlates with secure attachment and social safety.",
            "anticipation": "Future-oriented thinking — linked to hope, planning, and motivation.",
        }
        if dom in emotion_insights:
            insights.append(emotion_insights[dom])

        if p.emotional_complexity >= 4:
            insights.append("High emotional complexity — the writer may be processing conflicting feelings simultaneously.")
        if p.subjectivity > 0.7:
            insights.append("Highly subjective writing — rich in personal opinion and emotional coloring.")
        if p.sentiment_score < -0.4:
            insights.append("Strong negative valence detected. The writer may be under stress or experiencing difficulty.")
        if p.sentiment_score > 0.6:
            insights.append("Strong positive valence — the writer is expressing enthusiasm or contentment.")
        if "fear" in p.emotions and p.emotions["fear"] > 0.2 and "trust" in p.emotions and p.emotions["trust"] > 0.2:
            insights.append("Coexistence of fear and trust may indicate vulnerability paired with resilience.")

        return insights

    def _recommendations(self, p: EmotionProfile) -> list[str]:
        recs = []
        dom  = p.dominant_emotion
        if dom in ("sadness", "fear"):
            recs += ["Consider journaling your feelings to gain clarity.",
                     "Talking to a trusted person can help process difficult emotions."]
        elif dom == "anger":
            recs += ["Physical activity (e.g. a brisk walk) can help discharge anger safely.",
                     "Try naming the specific boundary that was crossed to clarify the source."]
        elif dom == "joy":
            recs += ["Savor this positive state — gratitude journaling can prolong positive affect.",
                     "Share your mood; positive emotions are socially contagious in healthy ways."]
        elif dom == "anticipation":
            recs += ["Break your goal into concrete next steps to channel this forward momentum.",
                     "Visualization exercises can strengthen anticipatory motivation."]
        if p.emotional_complexity >= 4:
            recs.append("With many emotions active, mindful breathing can help you identify the core feeling.")
        return recs
