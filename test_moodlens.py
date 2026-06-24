"""
MoodLens Test Suite
Run: pytest tests/ -v
"""

import pytest
from moodlens import MoodLens, EmotionProfile


@pytest.fixture(scope="module")
def lens():
    return MoodLens(use_transformer=False)   # fast lexicon-only mode for CI


class TestBasicAnalysis:
    def test_returns_emotion_profile(self, lens):
        p = lens.analyze("I am very happy today!")
        assert isinstance(p, EmotionProfile)

    def test_word_count(self, lens):
        p = lens.analyze("Hello world this is a test sentence.")
        assert p.word_count > 0

    def test_empty_text_raises(self, lens):
        with pytest.raises(ValueError):
            lens.analyze("")

    def test_whitespace_raises(self, lens):
        with pytest.raises(ValueError):
            lens.analyze("   ")

    def test_dominant_emotion_present(self, lens):
        p = lens.analyze("I am thrilled and overjoyed!")
        assert p.dominant_emotion in p.emotions or p.dominant_emotion == "neutral"

    def test_emotions_sum_to_one(self, lens):
        p = lens.analyze("I feel fear and anger and disgust all at once.")
        if p.emotions:
            total = sum(p.emotions.values())
            assert abs(total - 1.0) < 0.01


class TestSentiment:
    def test_positive_text(self, lens):
        p = lens.analyze("Today is absolutely wonderful. Everything is amazing and beautiful!")
        assert p.sentiment_score > 0

    def test_negative_text(self, lens):
        p = lens.analyze("This is terrible. I hate everything. Nothing works and I feel awful.")
        assert p.sentiment_score < 0

    def test_sentiment_label_positive(self, lens):
        p = lens.analyze("Life is beautiful and full of joy!")
        assert p.sentiment_label == "Positive"

    def test_sentiment_label_negative(self, lens):
        p = lens.analyze("I am devastated and heartbroken. Everything is ruined.")
        assert p.sentiment_label == "Negative"

    def test_subjectivity_range(self, lens):
        p = lens.analyze("The sky is blue. Water is wet. The year is 2024.")
        assert 0.0 <= p.subjectivity <= 1.0


class TestStructure:
    def test_sentence_sentiments_list(self, lens):
        p = lens.analyze("I love dogs. Cats are also nice. I hate traffic.")
        assert isinstance(p.sentence_sentiments, list)
        assert len(p.sentence_sentiments) >= 1

    def test_sentence_sentiment_fields(self, lens):
        p = lens.analyze("I am happy. The weather is bad.")
        for s in p.sentence_sentiments:
            assert "index" in s
            assert "sentence" in s
            assert "sentiment" in s
            assert "label" in s

    def test_key_phrases(self, lens):
        p = lens.analyze("The beautiful summer day filled my heart with joy and wonder.")
        assert isinstance(p.key_phrases, list)

    def test_to_dict(self, lens):
        p = lens.analyze("Testing serialization.")
        d = p.to_dict()
        assert isinstance(d, dict)
        assert "emotions" in d
        assert "sentiment_score" in d

    def test_to_json(self, lens):
        import json
        p = lens.analyze("Testing JSON output.")
        data = json.loads(p.to_json())
        assert "dominant_emotion" in data


class TestCompare:
    def test_compare_returns_dict(self, lens):
        result = lens.compare("I am happy.", "I am devastated.")
        assert isinstance(result, dict)
        assert "sentiment_shift" in result
        assert "emotion_delta" in result
        assert "summary" in result

    def test_compare_sentiment_shift_direction(self, lens):
        result = lens.compare("I am very sad.", "I feel great!")
        assert result["sentiment_shift"] > 0   # moved positive


class TestBatch:
    def test_batch_length(self, lens):
        texts = ["I am happy.", "I am sad.", "I am angry."]
        profiles = lens.analyze_batch(texts)
        assert len(profiles) == 3

    def test_batch_types(self, lens):
        texts = ["Joy is wonderful.", "Fear is real."]
        profiles = lens.analyze_batch(texts)
        for p in profiles:
            assert isinstance(p, EmotionProfile)


class TestInsights:
    def test_insights_list(self, lens):
        p = lens.analyze("I feel so scared and worried about the future.")
        assert isinstance(p.psychological_insights, list)

    def test_recommendations_list(self, lens):
        p = lens.analyze("I am furious and outraged. This is unacceptable!")
        assert isinstance(p.recommendations, list)

    def test_mood_summary_string(self, lens):
        p = lens.analyze("The world is strange and uncertain today.")
        assert isinstance(p.mood_summary, str) and len(p.mood_summary) > 10
