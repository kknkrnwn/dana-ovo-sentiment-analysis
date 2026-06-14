# 💳 DANA vs OVO: Sentiment Analysis & Play Store Reviews Analysis using Support Vector Machine

This repository focuses on data mining, comprehensive text preprocessing, and advanced sentiment classification of user reviews from two of the biggest fintech/digital wallet applications in Indonesia: **DANA** (`id.dana`) and **OVO** (`ovo.id`). The goal is to extract live user feedback from the Google Play Store, clean and normalize unstructured Indonesian text, and perform sentiment classification using Natural Language Processing (NLP) and Support Vector Machine (SVM) to compare user satisfaction and system stability.

## 📊 Project Overview
The comprehensive pipeline of this project consists of:
1. **App Review Scraping:** Extracting live user reviews directly from the Google Play Store utilizing the `google-play-scraper` library based on specific App IDs.
2. **Text Preprocessing & NLP:** Cleaning unstructured Indonesian text through multiple stages including HTML tag removal, regex filtering, tokenizing, stop-word removal, and stemming using Sastrawi.
3. **Machine Learning Classification:** Training and evaluating a Support Vector Machine (SVM) model to accurately categorize reviews into Positive, Negative, and Neutral classes.
4. **Sentiment & Model Analytics:** Visualizing classification counts, cross-validating with a Confusion Machine, and benchmarking evaluation metrics side-by-side.

### Tech Stack:
* **Language:** Python
* **Data Scraping:** `google-play-scraper`
* **Data Frame Management:** `pandas`, `numpy`
* **NLP & Text Mining:** `nltk`, `Sastrawi` (Indonesian Stemmer), `regex`, `BeautifulSoup`
* **Machine Learning Model:** `scikit-learn` (`SVC`, `TfidfVectorizer`)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 🛠️ Step-by-Step Notebook Workflow (`ANALISIS_DANA_&_OVO_2.ipynb`)

The notebook is systematically structured into the following core development phases:

### 1. Library Installation & Environment Setup
Installing the necessary packages via `pip` (including `google-play-scraper` and `Sastrawi`) and importing fundamental libraries like `pandas`, `numpy`, and `re`].

### 2. Live Data Scraping (Google Play Store API)
Targeting the specific application packages for DANA (`id.dana`) and OVO (`ovo.id`). The scraper fetches user reviews filtered by language (`lang='id'`) and sorted by the most relevant parameters.

### 3. Data Transformation & Structuring
Converting the raw scraped data payload into a structured `pandas.DataFrame`. This captures crucial metrics for analysis, including `userName`, `content` (review text), `score` (ratings), and `app_name`.

### 4. Sentiment Labeling (Initial Baseline)
Auto-tagging sentiments based on the star rating (`score`) given by users. Ratings $\ge$ 3.0 are initially classified as **positif**, while ratings < 3.0 are classified as **negatif**.

### 5. Text Cleaning & Normalization
Implementing a rigorous cleaning function (`clean_text`) to sanitize raw text. This includes:
* Removing HTML tags using `BeautifulSoup`.
* Removing usernames, URLs, non-alphabet characters, and emojis.
* Converting all text to lowercase.
* Handling character duplications (e.g., converting elongated words like "yukkk" to normal forms)].

### 6. Tokenizing
Breaking down the cleaned sentences into individual words/tokens utilizing `nltk.tokenize.word_tokenize`.

### 7. Stopword Removal
Filtering out non-essential Indonesian words (such as "yang", "untuk", "pada", "di") using the `Sastrawi` StopWordRemover factory to focus strictly on meaningful terms.

### 8. Stemming
Reducing words to their base or root form (e.g., converting "perbaiki" or "mengisi" to their root words) using the `Sastrawi` Stemmer factory to standardize the text data.

### 9. SVM Modeling & Feature Extraction (Planned/Ongoing)
* Transforming the preprocessed text column into numerical features using **TF-IDF Vectorization**.
* Splitting the dataset into Training and Testing sets.
* Training a **Support Vector Machine (SVM)** classifier to predict multi-class sentiments (Positive, Negative, and Neutral).

---

## 📈 Sentiment Visualization & Previews

The processed text datasets and model prediction results are mapped into various visual charts to evaluate model performance and compare metrics between DANA and OVO:

##### 📊 A. Model Performance (Confusion Matrix)
This matrix evaluates the accuracy and prediction behavior of the SVM model, mapping true labels against predicted labels to highlight accurate predictions and misclassifications:

<img width="650" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/130db84c-ef2a-40e9-8a78-189b082e8360" />

##### 📊 B. SVM Sentiment Distribution (Bar Chart)
A structured bar chart displaying the final volume of reviews categorized into Positive, Negative, and Neutral classes based on the SVM classification results:

<img width="650" alt="Bar Chart" src="https://github.com/user-attachments/assets/becf5b0d-bc9c-4842-beae-aaf940a573d2" />


##### 📊 C. Model Evaluation Comparison 
A side-by-side comparison stacking DANA and OVO across major evaluation metrics (Accuracy, Precision, Recall, and F1-Score) to determine model stability:

<img width="650" alt="Metrics Benchmarking" src="https://github.com/user-attachments/assets/2a70cc14-8269-4a48-ad67-42ef435322ed" />


---

## 🚀 How to Run the Project

1. **Clone this repository** to your local machine:
```bash
   git clone [https://github.com/your-username/dana-ovo-sentiment-analysis.git](https://github.com/your-username/dana-ovo-sentiment-analysis.git)
