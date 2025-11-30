"""
Configuration file for Bank Reviews
"""
import os
from dotenv import load_dotenv

load_dotenv()

#Google Play Store APP_IDS
APP_IDS = {
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    'Abyssinia': os.getenv('ABBYSINIA_APP_ID', 'com.boa.boaMobileBanking'),
    'Dashen': os.getenv('DASHEN_APP_ID ', 'com.dashen.dashensuperapp')
}

#Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial bank of Ethiopia',
    'Abyssinia': 'Bank of Abyssinia',
    'Dashen': 'Dashen Bank'
}

#Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 400)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',
    'country': 'et'  # Ethiopia 
}

#File Pathsr
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'raw_reviews': 'data/raw/reviews_raw.csv',
    'processed_reviews': 'data/processed/reviews_processed.csv',
    'sentiment_results': 'data/processed/reviews~_with_sentiment.csv',
    'final_results': 'data/processed/reviews_final.csv'
}
