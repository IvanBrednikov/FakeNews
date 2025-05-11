import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as Data

df = Data.get_df()
tfidf_matrix, feature_names = Data.get_tfidf(df)

#проверяем гипотезу: есть такие ключевые слова которые имеют большой вес в фейковых новостях и низкий в реальных

#очистка данных от бессмысленных слов
df_tfidf = pd.DataFrame(tfidf_matrix.toarray())

del_list = []
for word_i in range(len(feature_names)):
    for ch in str(feature_names[word_i]):
        if ch in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            del_list.append(word_i)
            feature_names[word_i] = -1
            break

df_tfidf = df_tfidf.drop(columns=del_list)
feature_names = feature_names[feature_names != -1]

#выведем матрицу как изображение с оттенками серого
tfidf_matrix = df_tfidf.values #преборазование в матрицу
#укажем диапазон расчётов и вывода одинаковыми для опитмизации
rows = 1000
columns = 500

#создадим матрицу для визуализации данных
color_matrix = np.zeros((rows, columns, 3))
for i in range(rows):
    for j in range(columns):
        if tfidf_matrix[i, j] != 0:
            label = df[df['idx'] == i].values[0][3] #получаем все значения строки и возвращаем метку
            if label:
                color_matrix[i, j] = [0, 0, 255] #реальная новость - синий
            else:
                color_matrix[i, j] = [255, 0, 0] #фейковая новость - красный
        else:
            color_matrix[i, j] = [255, 255, 255] #слово не используется - белый

#получение средних значений весов для каждого слова
def get_means(matrix):
    nrows = matrix.shape[0]
    ncolumns = matrix.shape[1]
    counts = np.zeros(ncolumns) #массив значений - количества ненулевых весов в матрице
    summ = np.sum(matrix, axis=0) #сумма каждого столбца

    #считаем ненулевые веса в столбце
    for j_ in range(ncolumns):
        for i_ in range(nrows):
            if matrix[i_, j_] != 0:
                counts[j_] += 1
    #вычисляем средние значения
    means = summ/counts
    return means

#отделим набор данных о фейковых новостях от реальных
fake_matrix = tfidf_matrix[df['label'] == 0]
real_matrix = tfidf_matrix[df['label'] == 1]
#вычислим средние значения весов для каждого слова
fake_means = get_means(fake_matrix[0:rows,0:columns])
real_means = get_means(real_matrix[0:rows,0:columns])

#выведем результаты
plt.subplot(1, 2, 1)
plt.imshow(color_matrix, vmin=0, vmax=255, interpolation=None)
plt.title('Частота использования слов в документах')
plt.xlabel('Документ')
plt.ylabel('Слово')

plt.subplot(1, 2, 2)
plt.bar(feature_names[0:fake_means.shape[0]], fake_means, color='r')
plt.bar(feature_names[0:fake_means.shape[0]], real_means, color='b')
plt.title('Соотношение значимости слов в фейковых новостях и в реальных')
plt.xlabel('Слова')
plt.ylabel('Средний вес')
plt.xticks(rotation=90)

plt.show()

#гипотеза подтвердилась
#на графике присутствуют столбцы показывающие значительный перевес в сторону фейковых или реальных новостей для значимости отдельно взятого слова
#на основе этой закономерности можно создать модель обучения