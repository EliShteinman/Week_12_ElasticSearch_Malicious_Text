import logging
import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, path_download: str = None):
        nltk_dir = (
            path_download if path_download else os.path.join(os.getcwd(), "nltk_data")
        )
        try:
            nltk.download("vader_lexicon", download_dir=nltk_dir, quiet=True)
            logger.info("VADER lexicon installed successfully")
        except Exception as e:
            logger.error(f"Error installing VADER lexicon: {e}")
        self.sid = SentimentIntensityAnalyzer()

    def get_sentiment_score(self, text: str) -> float:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text for sentiment analysis")
            return 0.0
        try:
            score = self.sid.polarity_scores(text)
            return score["compound"]
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0

    def get_sentiment_label(
        self,
        text: str,
        positive_threshold: float = 0.5,
        negative_threshold: float = -0.5,
    ) -> str:
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid text for sentiment analysis")
            return "neutral"
        score = self.get_sentiment_score(text)
        return self.convert_to_sentiment_label(
            score, positive_threshold, negative_threshold
        )

    def convert_to_sentiment_label(
        self, score: float, positive_threshold: float, negative_threshold: float
    ) -> str:
        if score >= positive_threshold:
            return "positive"
        elif score <= negative_threshold:
            return "negative"
        else:
            return "neutral"