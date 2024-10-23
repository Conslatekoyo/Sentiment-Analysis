
# IMDb Movie Reviews Sentiment Analysis

A machine learning project to classify movie reviews as either positive or negative using various text processing techniques and machine learning models. The project explores basic Natural Language Processing (NLP) concepts, including data cleaning, tokenization, TF-IDF vectorization, and model evaluation.


## Introduction

This project focuses on building a sentiment analysis model that classifies movie reviews from the IMDb dataset as positive or negative. The project covers essential aspects of text processing and machine learning, including data exploration, text cleaning, feature extraction, and model evaluation. 

The following models were explored:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

## Dataset

The project uses the [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle, which contains 50,000 reviews labeled as either "positive" or "negative." The dataset is split into 25,000 training and 25,000 testing examples.

## Project Structure

```
.
├── data/
│   ├── imdb_reviews.csv        # Dataset (to be downloaded from Kaggle)
├── notebooks/
│   ├── Sentiment_Analysis.ipynb  # Jupyter notebook with step-by-step implementation
├── src/
│   ├── preprocess.py           # Script for text cleaning and preprocessing
│   ├── models.py               # Machine learning models and training scripts
├── README.md                   # Project README file
└── requirements.txt            # Python dependencies
```

## Requirements

To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk`
- `bs4` (BeautifulSoup)
- `re` (Regular Expressions)

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

Data preprocessing is crucial for the performance of the machine learning models. The following steps were applied to the dataset:
1. **Cleaning**: Removal of HTML tags, punctuation, special characters, and numbers.
2. **Tokenization**: Splitting reviews into individual words.
3. **Removing Stop Words**: Eliminating common words (e.g., "the," "is") that do not contribute to sentiment.
4. **Stemming/Lemmatization**: Reducing words to their base or root form.
5. **Handling Negations**: Replacing phrases like "not good" with "not_good" to preserve sentiment.

## Machine Learning Models

The following machine learning models were implemented and tested:
- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

### TF-IDF Representation

Text data was converted to numerical format using TF-IDF (Term Frequency-Inverse Document Frequency), which highlights important words in each document.

### Latent Semantic Analysis (Optional)

For dimensionality reduction and topic extraction, LSA was also implemented using `TruncatedSVD`.

## Evaluation

Models were evaluated using the following metrics:
- **Accuracy**: Percentage of correctly predicted reviews.
- **Confusion Matrix**: Visual representation of prediction results.
- **Precision, Recall, F1-Score**: For deeper insights into model performance.

## Results

- **Best Performing Model**: Logistic Regression achieved an accuracy of 80% on the test dataset.
- **Insights**:
  - Handling negations and stop words significantly improved performance.
  - TF-IDF worked better than simple Bag-of-Words (BoW) due to the high dimensionality of text data.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/username/sentiment-analysis-imdb.git
    cd sentiment-analysis-imdb
    ```
2. Download the IMDb dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the `data` folder.

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook to train and test models:
    ```bash
    jupyter notebook notebooks/Sentiment_Analysis.ipynb
    ```

## Contributing

Contributions are welcome! If you'd like to add improvements, please fork the repository, make changes, and submit a pull request.

This README template outlines the purpose, workflow, and methods used in the sentiment analysis project, providing clear instructions on how to set up and run the project.
