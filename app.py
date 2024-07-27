import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Fungsi untuk membersihkan teks ulasan
def clean_text(text):
    text = re.sub(r'<br />', ' ', text)  # Menghapus tag HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Menghapus karakter khusus dan angka
    text = text.lower()  # Mengonversi teks ke huruf kecil
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]  # Menghapus stop words dan melakukan stemming
    return ' '.join(words)

# UI
def user_interface ():
    st.title('Sentiment Analysis')
    st.text('Capstone Project')
    reviews = st.text_input('Write your reviews')
    return reviews

#Fungsi Predictions
def predict_reviews(reviews):
    # Clean and preprocess user input
    cleaned_reviews = clean_text(reviews)
    print("Preprocessed text:", cleaned_reviews)  # Print the preprocessed text

    # Load vocabulary from vocab.txt
    vocab = []
    with open('vocab.txt', 'r') as f:
        vocab = f.read().splitlines()

    # Define TextVectorization layer
    vectorizer = TextVectorization(
        max_tokens=5000,
        output_mode='int',
        output_sequence_length=150,
        vocabulary=vocab
    )

    # Vectorize the tweet
    vectorized_reviews = vectorizer([cleaned_reviews])
    # Load the pre-trained model
    model = tf.keras.models.load_model('lstm_model.h5')

    # Predict using the model
    prediction = model.predict(vectorized_reviews)

    # Get the positive probability
    positive_prob = prediction[0][0]

    # Calculate the negative probability
    negative_prob = 1 - positive_prob

    return [negative_prob, positive_prob], {0: 'negative', 1: 'positive'}

def main():
        reviews = user_interface()
        if st.button('Detect'):
            if reviews:  # Ensure the review is not empty
                prediction, label_mapping = predict_reviews(reviews)
                print("Prediction:", prediction)  # Print the prediction
            
             # Convert the list to a NumPy array
            prediction_array = np.array(prediction)

            # Display the predicted category with the highest probability
            max_index = prediction_array.argmax()
            predicted_label = label_mapping[max_index]
            st.success(f'Predicted Category: {predicted_label}')

        else:
            st.error('Please enter a review to analyze.')

if __name__ == "__main__":
    main()