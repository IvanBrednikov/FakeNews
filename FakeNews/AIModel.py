import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import DataPreprocess as Data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

df = Data.get_df()
tfidf_matrix, feature_names = Data.get_tfidf(df)

#очистка данных от бессмысленных слов
df_tfidf = pd.DataFrame(tfidf_matrix.toarray())

del_list = []
for word_i in range(len(feature_names)):
    for ch in str(feature_names[word_i]):
        if ch in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            del_list.append(word_i)
            break

df_tfidf = df_tfidf.drop(columns=del_list)

#разделяем данные на обучающие и тестовые
train_data, test_data, train_label, test_label = train_test_split(df_tfidf, df['label'], test_size=0.2, random_state=0)

#обучение модели
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(train_data, train_label)

# Предсказание категорий для тестовых данных
test_pred = classifier.predict(test_data)

# Оценка точности модели
accuracy = accuracy_score(test_label, test_pred)
print(f"Точность модели: {accuracy*100:.2f}")

#создание матрицы ошибок
conf_matrix = confusion_matrix(test_label, test_pred, labels=[0, 1])
confusion_normalized = conf_matrix / conf_matrix.sum(axis=1)

labels_ = ['Fake', 'Real']
fig = plt.figure(figsize=(4, 4))
ax1 = sns.heatmap(confusion_normalized, xticklabels=labels_, yticklabels=labels_, annot=True)
plt.title("Матрица ошибок")
plt.ylabel("Истинные метки")
plt.xlabel("Предсказанные метки")

plt.show()