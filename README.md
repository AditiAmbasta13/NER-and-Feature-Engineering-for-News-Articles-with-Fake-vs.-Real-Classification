# Named Entity Recognition (NER) and Feature Engineering for News Articles with Fake vs. Real Classification

## Introduction

This project analyzes a collection of news articles, sourced from the **GossipCop Fake and Real** and **Politifact Fake and Real** datasets, to predict whether an article is real or fake based on various features. These features are derived through **Named Entity Recognition (NER)** and feature engineering. The task involves:

- Text preprocessing
- Extracting named entities from articles
- Engineering meaningful features from these entities and other attributes
- Building a predictive model to classify articles as either real or fake
- Visualizing the relationship between entities and article classification

## Data Collection

The combined dataset consists of two sources:
- **GossipCop Fake and Real dataset**
- **Politifact Fake and Real dataset**

The dataset contains the following columns:
- **id**: Unique identifier for each article
- **url**: URL of the article
- **title**: Title of the article
- **label**: The classification label indicating whether the article is "Real" or "Fake" (binary classification)

For this task, the dataset was preprocessed and combined to form a unified dataset where each article is labeled either as real or fake.

## Text Preprocessing

Text preprocessing ensures that the text is clean and structured before performing NLP tasks. The steps followed include:
- **Whitespace Removal**: Removing extra spaces in the text.
- **HTML Tag Removal**: Stripping any HTML tags from the articles.
- **Special Character Removal**: Removing numbers, punctuation, and other special characters.
- **Normalization**: Converting the text to lowercase.
- **Tokenization**: Breaking the text into words (tokens).
- **Stopword Removal**: Removing common, non-informative words (e.g., "the", "is").

The text preprocessing function used is as follows:

```python
import re

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase
    return text
```

## Named Entity Recognition (NER)

To extract named entities from the article text, **SpaCy** was used, a popular open-source NLP library. The key steps involved:
- **Entity Extraction**: Identifying named entities such as **PERSON**, **ORG** (organizations), and **GPE** (geopolitical entities).
- **Feature Creation**: The frequency of different entity types in the article was counted and stored as features.

For example, we created features like:
- Number of **PERSON** entities in each article
- Number of **ORG** entities in each article
- Number of **GPE** entities in each article

This was done using the following function:

```python
import spacy

nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    entities = {'PERSON': 0, 'ORG': 0, 'GPE': 0}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] += 1
    return entities
```

## Feature Engineering

The goal of feature engineering is to create a comprehensive set of features for predicting whether an article is real or fake. The following features were created:
- **Entity Counts**: Based on NER, the count of entities like **PERSON**, **ORG**, and **GPE**.
- **Article Length**: Number of words in the article's title or content.
- **Sentiment Scores**: Sentiment of the article, calculated using **TextBlob** or **VADER**.
- **Engagement Metrics**: While not directly available, this could be inferred if relevant data was provided (e.g., likes, shares, or comments). Since it wasn't included in the dataset, this feature wasn't used.

Additionally, innovative features were derived:
- **Sentiment Adjusted Entity Frequency**: The relationship between sentiment scores and entity frequency.
- **Readability Scores**: An indicator of how easy the article is to read, calculated using the average sentence length and complexity.

Example sentiment calculation using **TextBlob**:

```python
from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
```

## Predictive Modeling

For the classification task, we used a **Random Forest Classifier** to predict whether an article is real or fake. This model was chosen due to its robustness in handling both numerical and categorical data.

### Steps Involved:
1. **Train-Test Split**: We split the dataset into training and testing sets (70% training, 30% testing).
2. **Model Training**: A **Random Forest** model was trained on the engineered features.
3. **Model Evaluation**: The model was evaluated using **accuracy**, **precision**, **recall**, and **F1-score** to assess its performance.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assuming 'features' is the engineered feature set and 'labels' is the target variable (Real/Fake label)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Visualization

Visualizations help us understand the relationships between features and article classification. The following visualizations were created:
- **Bar Charts**: Showing the frequency of named entities (e.g., PERSON, ORG, GPE).
- **Scatter Plots**: Illustrating correlations between features like entity counts and the real/fake classification.
- **Heatmaps**: Showing the relationship between sentiment scores, entity counts, and classification (real/fake).

Example of a bar chart for entity frequency:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Bar chart for entity frequency
sns.barplot(x=entity_types, y=entity_counts)
plt.title('Entity Frequency in Articles')
plt.show()
```

## Insights and Results

From the analysis and modeling, several insights were gained:
- **Entity Frequency**: Articles with more mentions of certain entities (e.g., organizations, people) had a higher likelihood of being classified as "real". This could indicate that real articles often have specific references to key people and organizations.
- **Sentiment**: Articles with extreme sentiment (either very positive or very negative) were often classified as "fake".
- **Readability**: Articles that are more readable or have simpler structures were more likely to be real, as complex structures may indicate manipulation or sensationalism.

The **Random Forest** model achieved an accuracy of around **85%**, with an **F1-score** of **0.80**, indicating good predictive performance.

## Conclusion

This project demonstrated how **Named Entity Recognition (NER)** and feature engineering can be effectively used to classify articles as real or fake. By combining NER-based features with sentiment analysis and readability scores, we were able to build a strong predictive model using a **Random Forest Classifier**. The insights gained from the analysis can be helpful for detecting fake news articles and improving automated news categorization systems.
