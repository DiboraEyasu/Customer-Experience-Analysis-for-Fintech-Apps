# Customer-Experience-Analysis-for-Fintech-Apps
Analyzing cstomers reaction to most common bank apps by scraping google playstore.
## 📋 Project Overview
This project analyzes customer satisfaction for three Ethiopian banks' mobile applications through Google Play Store reviews. As part of the 10 Academy Artificial Intelligence Mastery program, we scrape, process, and analyze user reviews to provide actionable insights for improving mobile banking experiences.

## 🎯 Business Context
Omega Consultancy is supporting Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank to enhance their mobile banking applications by identifying key pain points and satisfaction drivers from user feedback.

## 📊 Core Objectives
Scrape and preprocess user reviews from Google Play Store

Analyze sentiment and identify key customer feedback themes

Store cleaned data in PostgreSQL database

Derive actionable insights through visualizations and recommendations

##📂 Project Structure Implementation
Fintech-Apps-Customer-Experience/
├── .vscode/              # IDE configuration
│   └── settings.json
├── .github/              # CI/CD workflows
│   └── workflows/
│       └── ci.yml
├── data/                 # Data management
│   ├── raw/             # Raw datasets
│   └── processed/       # Processed data
├── notebooks/           # Exploratory analysis
├── src/                 # Source code
├── tests/               # Test suites
├── scripts/             # Utility scripts
├── .gitignore          # Version control exclusions
├── requirements.txt    # Dependencies
└── README.md          # Project document

## 🚀 Installation & Setup
**Prerequisites**
* ##### Python 3.8+

* ##### Git

* ##### PostgreSQL (for Task 3)

### Installation Steps
**Clone the repository**

bash
git clone https://github.com/DiboraEyasu/Fintech-Apps-Customer-Experience.git
cd Fintech-Apps-Customer-Experience

### Create virtual environment
bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Download NLP models

bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
📊 Data Collection
Sources
Google Play Store reviews for:

Commercial Bank of Ethiopia (CBE)

Bank of Abyssinia (BOA)

Dashen Bank

## Dataset Content
Data Fields Collected
* review_text: User feedback content

* rating: 1-5 star rating
* review_date: Date of review
* user_name: Reviewer name (if available)
* bank_name: Bank identifier
* thumbs_up: Helpful votes count
* source: Always "Google Play"
 **Target Volume**
Minimum: 400 reviews per bank (1,200+ total)

Quality: <5% missing data or errors