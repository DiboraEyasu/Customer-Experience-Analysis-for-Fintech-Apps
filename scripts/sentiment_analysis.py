import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

class SentimentAnalyzer:

    def __init__(self):
        self.df = None
        self.sentiment_model = None

    
    def load_data(self):
        try:
            df = pd.read_csv('../data/raw/reviews_raw.csv')  # Your actual file
            print(f"✅ Successfully loaded {len(df)} reviews")
            return df
        except FileNotFoundError:
            print("❌ Error: 'reviews_raw.csv' file not found")
            print("   Make sure the file is in the same directory")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Clean text for analysis"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def initialize_sentiment_model(self):
        """Initialize the sentiment model - using TextBlob instead of DistilBERT"""
        print("⚠️ Using TextBlob for sentiment analysis (faster setup)")
        self.sentiment_model = "textblob"  # Just mark that we're using TextBlob

    def get_sentiment(self, text):
        """Use TextBlob for sentiment analysis - works immediately"""
        if not isinstance(text, str) or text.strip() == "":
            return "NEUTRAL", 0.5
        
        try:
            # Use TextBlob instead of transformers
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            # Convert polarity to label and score
            if polarity > 0.1:
                return "POSITIVE", polarity
            elif polarity < -0.1:
                return "NEGATIVE", polarity
            else:
                return "NEUTRAL", abs(polarity)  # Use absolute value for score
                
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return "NEUTRAL", 0.5

    def polarity_to_label(self, polarity):
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    def analyze_real_data(self):
        """Analyze your real scraped data"""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
            
        print("Applying sentiment analysis to real data...")
        
        # Apply sentiment analysis
        sentiment_results = self.df['review_text'].apply(self.get_sentiment)
        self.df[['sentiment_label', 'sentiment_score']] = sentiment_results.apply(
            lambda x: pd.Series([x[0], x[1]])
        )
        
        # ✅ ADD: Rating-based sentiment labels (REQUIRED for Task 2)
        def rating_to_sentiment_label(rating):
            if rating >= 4:
                return "positive"
            elif rating == 3:
                return "neutral" 
            else:
                return "negative"
        
        self.df['rating_sentiment_label'] = self.df['rating'].apply(rating_to_sentiment_label)
        
        # Add TextBlob analysis
        self.df['textblob_polarity'] = self.df['review_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        self.df['textblob_sentiment'] = self.df['textblob_polarity'].apply(self.polarity_to_label)
        
        # Preprocess text for frequency analysis
        self.df['clean_text'] = self.df['review_text'].apply(self.preprocess_text)
        
        print("✅ Sentiment analysis complete!")
        return self.df

    def aggregate_sentiment_by_bank_rating(self):
        """Aggregate sentiment by bank and rating - REQUIRED for Task 2"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
            
        print("=== TASK 2: SENTIMENT AGGREGATION BY BANK & RATING ===")
        
        aggregation = self.df.groupby(['bank_name', 'rating']).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'textblob_polarity': 'mean',
            'sentiment_label': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NEUTRAL'
        }).round(3)
        
        # Flatten column names
        aggregation.columns = ['sentiment_mean', 'sentiment_std', 'review_count', 
                              'textblob_polarity_mean', 'most_common_sentiment']
        
        aggregation = aggregation.reset_index()
        print(aggregation)
        
        return aggregation

    # Keep your other methods (frequency_test, tf_idf, topic_modeling) as they are
    def frequency_test(self):
        if 'clean_text' not in self.df.columns:
            print("❌ Run analyze_real_data() first to create clean_text")
            return pd.DataFrame()
            
        count_vec = CountVectorizer(stop_words="english")
        X_counts = count_vec.fit_transform(self.df["clean_text"])

        word_counts = np.asarray(X_counts.sum(axis=0)).flatten()
        vocab = np.array(count_vec.get_feature_names_out())

        freq_df = pd.DataFrame({"word": vocab, "count": word_counts})
        freq_df = freq_df.sort_values("count", ascending=False)
        return freq_df
    
    def tf_idf(self):
        if 'clean_text' not in self.df.columns:
            print("❌ Run analyze_real_data() first to create clean_text")
            return pd.DataFrame()
            
        tfidf_vec = TfidfVectorizer(stop_words="english")
        X_tfidf = tfidf_vec.fit_transform(self.df["clean_text"])

        tfidf_means = np.asarray(X_tfidf.mean(axis=0)).flatten()
        vocab_tfidf = np.array(tfidf_vec.get_feature_names_out())

        tfidf_df = pd.DataFrame({"word": vocab_tfidf, "tfidf": tfidf_means})
        tfidf_df = tfidf_df.sort_values("tfidf", ascending=False)
        return tfidf_df

    def topic_modeling(self):
        # ... keep your existing topic_modeling method
        pass