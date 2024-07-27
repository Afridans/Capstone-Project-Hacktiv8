# Original file is located at
#    https://colab.research.google.com/drive/1fAn9ipz1XysEzIi0nmpWJQntE32COV0I

#Latar Belakang Project
# Internet Movie Database (IMDB) adalah platform besar yang menyediakan informasi tentang film dan memungkinkan pengguna memberikan rating dan ulasan, yang menjadi data berharga untuk analisis sentimen. Dalam kasus ini NLP digunakan untuk mengukur reaksi penonton, memahami tren opini publik, dan membantu pembuatan konten dalam mengambil keputusan.

# Import Libraries

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
import tensorflow as tf

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from oauth2client.service_account import ServiceAccountCredentials
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, Dropout, TextVectorization
from tensorflow.keras import layers, models

from google.colab import drive
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV

# Load Data

sheet_url = 'https://docs.google.com/spreadsheets/d/11y4bcPa2y_bc6avLEM1A8wUPqM2f73W6G50ODGuNG3o/edit?usp=sharing'
sheet_url_trf = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
df = pd.read_csv(sheet_url_trf)
df.head()


df.info()

# Cleaning Data

#cek missing values
df.isna().sum()


#Mengidentifikasi data duplikat
duplicates = df.duplicated()
print(df[duplicates])


#Menghapus data duplikat
df.drop_duplicates(inplace=True)

# Mendownload stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


# Membersihkan teks ulasan
def clean_text(text):
    text = re.sub(r'<br />', ' ', text)  # Menghapus tag HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Menghapus karakter khusus dan angka
    text = text.lower()  # Mengonversi teks ke huruf kecil
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]  # Menghapus stop words dan melakukan stemming
    return ' '.join(words)

# Membersihkan kolom 'review'
df['cleaned_review'] = df['review'].apply(clean_text)

"""Hasil Pembersihan akan dibuatkan kolom baru yaitu cleaned_review"""

# Menampilkan beberapa baris pertama dari dataset yang sudah dibersihkan
df.head()

# EDA

# Melakukan analisis distribusi data, dan didapatkan data yang balance 50:50

# Distribusi sentimen
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title('Distribusi Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.show()

# Panjang ulasan
df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(6, 4))
sns.histplot(df['review_length'], bins=30, kde=True)
plt.title('Distribusi Panjang Ulasan')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.show()

# WordCloud untuk ulasan positif
positive_reviews = df[df['sentiment'] == 'positive']['cleaned_review'].tolist()
positive_text = ' '.join(positive_reviews)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud untuk Ulasan Positif')
plt.axis('off')
plt.show()


# WordCloud untuk ulasan negatif
negative_reviews = df[df['sentiment'] == 'negative']['cleaned_review'].tolist()
negative_text = ' '.join(negative_reviews)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud untuk Ulasan Negatif')
plt.axis('off')
plt.show()

# Preprocessing

# Mengubah Value di kolom sentiment dari Positive menjadi 1 dan Negative menjadi 0

label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

df.head()

df.info()

# Menginisiasi variable banyaknya karakter unik (max_words) dan panjang rata2 ulasan (max_len)

max_words = 5000
max_len = 150

# Melakukan Vektorisasi Text

vectorizer = TextVectorization(
    max_tokens=max_words,
    output_mode='int',
    output_sequence_length=max_len
)

vectorizer.adapt(df['cleaned_review'])

# Menyimpan Vocab untuk Pengunaan API

vocab = vectorizer.get_vocabulary()
with open('vocab.txt', 'w') as f:
    for word in vocab:
        f.write(f"{word}\n")

vector_review = vectorizer(df['cleaned_review'].values)
vector_review = np.array(vector_review)

# Melakukan Split Data untuk Train dan Test

X_train, X_test, y_train, y_test = train_test_split(vector_review, df['sentiment'], test_size=0.2, random_state=42)

# Model

#Menginisiasi Panjang Nilai Embedding

embedding_dim = 5000

# Perancangan Model Arsitektur LSTM

lstm_model = models.Sequential()
lstm_model.add(tf.keras.Input(shape=(max_len,), dtype=tf.int32))
lstm_model.add(layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
lstm_model.add(layers.LSTM(128, return_sequences=True))
lstm_model.add(layers.Dropout(0.5))
lstm_model.add(layers.LSTM(64))
lstm_model.add(layers.Dropout(0.5))
lstm_model.add(layers.Dense(2, activation='softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lstm_model.summary()

tf.keras.utils.plot_model(lstm_model)

# Melakukan training model dengan menggunakan lstm, dengan epoch 20 dan bacth_size 32"""

history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

loss, accuracy = lstm_model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Definisikan EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Latih model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Melakukan Ploting Hasil Training Model

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training accuracy')  # 'b-' for blue solid line
    plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')  # 'r-' for red solid line
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')  # 'b-' for blue solid line
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')  # 'r-' for red solid line
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(history)

# Membuat Evaluasi Model Menggunakan Confuse Matrix

predictions = lstm_model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, predicted_labels, target_names=['Negative', 'Positive']))

text = [['good good good'], ['bad bad bad']]
vector_text = vectorizer(text)
print(vector_text)
lstm_model.predict(vector_text)

# Save Model Dengan Format h5

lstm_model.save('lstm_model.h5')


