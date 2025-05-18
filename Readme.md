<h1>Практическая работа №1</h1>
<h2>Задача 1</h2>
<p>
 Создать модель ИИ, с 90% точностью определяющая фейковые новости.
</p>
<p>
 Инструменты: Python 3.12, sklearn, TfidfVectorizer, PassiveAggressiveClassifier<br>
 Для обучения модели применён инструмент TfidVectorizer так
как необходимо преобразование текстовых данных в численные признаки для 
обучения модели. Была выявлена зависимость: есть такие ключевые слова, имеющие 
большой вес в реальных новостях и низкий вес в фейковых, и наоборот. <br>

|||
|-------|--------|
|<img src="FakeNews/Bar1.png" width=300 height=300> | <img src="FakeNews/Bar2.png" width=300 height=300>|

Данные в наборе распределены равномерно<br>
<img src="FakeNews/Bar3.png" width=300 height=300><br>
Полученная модель имеет точность - 93.05% <br>
<img src="FakeNews/Conf_matrix.png" width=300 height=300><br>
</p>

<h2>Задача 2</h2>
<p>
 Создать модель ИИ, с 95% точностью определяющая заболевание Паркинсона.
</p>

<p>
Инструменты: Python 3.12, sklearn, XGBClassifier<br>

<img src="ParkinsonDisease/data_raspred.png" width=300 height=300><br>
Полученная модель имеет точность - 94.87% <br>
<img src="ParkinsonDisease/conf_matrix.png" width=300 height=300><br>

</p>

<h2>Задача 3</h2>
<p>
 Создать модель ИИ, классифицирующая отходы по фотографии.
</p>

<p>
Инструменты: Python 3.12, sklearn, keras, tensorflow, ImageGenerator, Xception<br>

| Распределение данных                                                         | Примеры элементов набора данных |
|------------------------------------------------------------------------------|----------------------------|
 | <img src="GarbageClassification/data_distribution.png" width=300 height=300> | <img src="GarbageClassification/garbagre images.png" width=300 height=300>|
 | Ошибка перекрёстной энтропии от эпохи                                        | Полученная матрица ошибок|
 | <img src="GarbageClassification/effecincy.png" width=300 height=300>         |<img src="GarbageClassification/conf_matrix.png" width=300 height=300>|

Полученная модель имеет точность ~95% <br>
<img src="GarbageClassification/accuracy.png" width=300 height=50><br>
</p>
