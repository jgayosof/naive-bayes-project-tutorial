# imports:
import pandas as pd
import numpy as np
import pickle
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Example importing the CSV here
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')


df_interim = df_raw.copy()
df_raw.to_csv('../data/raw/playstore_reviws_raw.csv')

df_interim = df_interim.drop(columns=['package_name'])

# Lowercase & drop spaces (begin and end):
df_interim['review'] = df_interim['review'].str.lower()
df_interim['review'] = df_interim['review'].str.strip()

df = df_interim.copy()
df.to_csv('../data/processed/playstore_reviws_raw.csv')


# X, y split:
X = df['review']
y = df['polarity']

# "Vectorize" the input
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()

X = X.toarray()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)

### GaussianNB ###

# construct model & fit
clf = GaussianNB()
clf.fit(X_train, y_train)

# predict on X_train & X_test
y_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Report
target_names = ['bad', 'good']
print(f'GAUSSIAN-NB RESULTS:')
print(f'\n Classification report on X_train:')
print(classification_report(y_train, y_pred, target_names=target_names))
print(f'\n Classification report on X_test:')
print(classification_report(y_test, y_test_pred, target_names=target_names))

print(f'Los resultados del modelo con GaussianNB no fueron buenos, probablemente hubo overfit')
print(f'Lo voy a intentar de nuevo con MultinomialNB. \n')


### MultinomialNB ###

# construct model & fit
clf_multinomial = MultinomialNB()
clf_multinomial.fit(X_train, y_train)

# predict on X_train & X_test
y_pred = clf_multinomial.predict(X_train)
y_test_pred = clf_multinomial.predict(X_test)

print(f'MULTINOMIAL-NB RESULTS:')
print(f'\n Classification report on X_train:')
print(classification_report(y_train, y_pred, target_names=target_names))
print(f'\n Classification report on X_test:')
print(classification_report(y_test, y_test_pred, target_names=target_names))

print(f'Los resultados del modelo con MultinomialNB fueron notoriamente mejores')


### Prueba del modelo ###
msj = 'I Like this app'
msj2 = 'I hate this app'
Z = vectorizer.transform([msj])
Z2 = vectorizer.transform([msj2])

# pediction w GaussianNB:
print(f'Prediction with GaussianNB for: {msj}')
print(clf.predict(Z.toarray()))
print(f'Prediction with GaussianNB for: {msj2}')
print(clf.predict(Z2.toarray()))


# pediction w MultinomialNB:
print(f'Prediction with MultinomialNB for: {msj}')
print(clf_multinomial.predict(Z.toarray()))
print(f'Prediction with MultinomialNB for: {msj2}')
print(clf_multinomial.predict(Z2.toarray()))


# Save models
pickle.dump(clf, open('../models/GaussianNB.pkl', 'wb'))
pickle.dump(clf_multinomial, open('../models/MultinomialNB.pkl', 'wb'))



# Tests
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
sentiment_pipeline(data)