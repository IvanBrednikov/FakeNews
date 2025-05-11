import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('parkinsons.data')
#извлечём признаки и метки
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

#масштабируем данные от -1 до 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#разбиение датасета на обучающие и тестовые наборы данных
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

#создание и обучение модели
model=XGBClassifier()
model.fit(x_train,y_train)

#вывод результата
y_pred=model.predict(x_test)
print('Точность на тестовой выборке: %.2f' % (accuracy_score(y_test, y_pred)*100))

#создание матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

#выведем диаграмму соотношения данных, матрицу ошибок
plt.pie([labels[labels==1].shape[0], labels[labels==0].shape[0]],
        labels=['Здоров', 'Болен'], colors=['green', 'red'])
plt.title('Распределение данных в наборе')

plt.matshow(conf_matrix, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.title("Матрица ошибок")
plt.ylabel("Истинные метки")
plt.xlabel("Предсказанные метки")

plt.show()