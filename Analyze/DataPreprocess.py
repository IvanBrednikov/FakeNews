import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

file_name = '..\\fake_news.csv'

#проверка верности меток
def validate_count(df):
    count = len(df) # count = 6335
    fake_count = len(df[df['label'] == 0]) # FAKE == 3164
    real_count = len(df[df['label'] == 1]) # REAL == 3171
    print('count: %d; fake_count: %d; real_count: %d; summ: %d'
          %(count, fake_count, real_count, fake_count+real_count))

def get_df():
    df = pd.read_csv(file_name)
    # преобразуем метки в числовые 0 - fake, 1 - real
    df['label'] = df['label'].replace('FAKE', 0)
    df['label'] = df['label'].replace('REAL', 1)
    df.reset_index()  # сброс индексов
    df['idx'] = df.index  # создаём поле idx для получения индекса

    return df

def get_tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    # Применение TF-IDF к текстовым данным
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

    # Получение списка ключевых слов и их значения TF-IDF для первого документа
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


#validate_count(get_df())