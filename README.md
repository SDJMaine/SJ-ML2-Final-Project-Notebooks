# Emotionality of Tweets – ML2 Final Project

This repo contains my Machine Learning II capstone project:  
classifying the emotion of tweets into six categories using:

1. TF–IDF + Multinomial Logistic Regression - classic baseline  
2. BiLSTM - neural, non-transformer baseline 
3. DistilBERT - pretrained transformer fine-tuned on the dataset

Target emotions: **sadness, joy, love, anger, fear, surprise**.

---

## 1. Dataset

Primary dataset (Kaggle):

- **Emotion Dataset for NLP**  
  https://www.kaggle.com/datasets/parulpandey/emotion-dataset  

The original data is already split into:

- `training.csv`
- `validation.csv`
- `test.csv`

## 2. Repository Structure
```text
.
├── data/
│   └── primary_emotions/
│       ├── training.csv
│       ├── validation.csv
│       └── test.csv
├── notebooks/
│   ├── Tfidf_logreg.ipynb
│   ├── Lstm_baseline.ipynb
│   └── Distillbert_finetune.ipynb
├── src/
│   └── data_utils.py        
├── models/  
