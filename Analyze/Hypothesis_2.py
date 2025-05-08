import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import DataPreprocess as Data

df = Data.get_df()
tfidf_matrix, feature_names = Data.get_tfidf(df)

#проверяем гипотезу: есть такое множество ключевых слов
#где отдельно взятое слово имеет большой вес в правдивых новостях и низкий вес в ложных,
#и наоборот, есть такие слова которые имеют большой вес в ложных новостях и низкий в правдивых

#очистка данных от бессмысленных слов
df_tfidf = pd.DataFrame(tfidf_matrix.toarray())

print(df_tfidf.shape)
del_list = []
for word_i in range(len(feature_names)):
    for ch in str(feature_names[word_i]):
        if ch in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            print(feature_names[word_i])
            del_list.append(word_i)
            break

df_tfidf = df_tfidf.drop(columns=del_list)

print(df_tfidf.shape)