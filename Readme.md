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
большой вес в реальных новостях и низкий вес в фейковых, и наоборот.
<img src="FakeNews/Bar1.png" width=300 height=300>
<img src="FakeNews/Bar2.png" width=300 height=300><br>
Данные в наборе распределены равномерно<br>
<img src="FakeNews/Bar3.png" width=300 height=300><br>
Полученная модель имеет точность - 93.05% <br>
<img src="FakeNews/conf_matrix.png" width=300 height=300><br>
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