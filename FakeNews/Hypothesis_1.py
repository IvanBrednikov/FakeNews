import matplotlib.pyplot as plt
import DataPreprocess as Data

#проверяем гипотезу: слова из заголовков часто совпадают со значимыми словами из текста статьи в правдивых новостях

df = Data.get_df()
tfidf_matrix, feature_names = Data.get_tfidf(df)

#подсчёт процента совпадения слов из заголовка со значимыми из текста
def preprocess(row):
    title = str(row['title']).lower()
    title = title.split(' ')
    tfidf_scores_ = tfidf_matrix.toarray()[int(row['idx'])]
    sorted_keywords_ = [word for _, word in sorted(zip(tfidf_scores_, feature_names), reverse=True)]
    cross_words = list(set(title) & set(sorted_keywords_[0:10]))
    return len(cross_words)/len(title)

rows_proceed = 100
df['cross_percent'] = df[df['idx'] < rows_proceed].apply(preprocess, axis=1)

data_fake = df.loc[(df['idx']<rows_proceed) & (df['label'] == 0), ['idx', 'cross_percent']]
data_real = df.loc[(df['idx']<rows_proceed) & (df['label'] == 1), ['idx', 'cross_percent']]

plt.ylabel('Процент совпадения')

plt.plot(data_fake['idx'], data_fake['cross_percent'], 'o', color='red')
plt.plot(data_real['idx'], data_real['cross_percent'], 'o', color='blue')

plt.show() #на графике очевидно что гипотеза не подтвердилась и зависимость не прослеживается