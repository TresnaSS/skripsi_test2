import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import nltk
from googletrans import Translator
import time

st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* White background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }
    .st-cx {
        background-color: white; /* White background for chat messages */
    }
    .sidebar .block-container {
        background-color: #f0f0f0; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download('wordnet')
nltk.download('stopwords')

with open("./bookgenremodel.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

def cleantext(text):
    text = re.sub("'\''","",text)
    text = re.sub("[^a-zA-Z]"," ",text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

def lematizing(sentence):
    lemma = WordNetLemmatizer()
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    
    stemSentence = stemSentence.strip()
    return stemSentence

def stemming(sentence):
    stemmer = PorterStemmer()
    stemmed_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += " "

    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence

def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)

    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)

    newmapper = {0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller'}

    return newmapper[predicted[0]]

def predict_genre(book_summary):
    if not book_summary:
        st.warning("Mohon Masukkan Ringkasan Buku.")
    else:

        progress_placeholder = st.empty()
        if progress_placeholder is not None:
            progress_placeholder.info("Sedang melakukan prediksi...")

        time.sleep(2)

        cleaned_summary = cleantext(book_summary)

        with open("./tfidfvector.pkl", 'rb') as file:
            vectorizer = pickle.load(file)

        vectorized_summary = vectorizer.transform([cleaned_summary])

        with open("./bookgenremodel.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        prediction = loaded_model.predict(vectorized_summary)

        newmapper = {0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller'}
        predicted_genre = newmapper[prediction[0]]

        progress_placeholder.empty()

        st.write("Hasil Prediksi Genre Buku")
        st.title(predicted_genre)
        st.success("Prediksi selesai!")

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='margin: 0;'>Prediksi Genre Buku</h1>
    </div>
""", unsafe_allow_html=True)

book_summary = st.text_area("Masukkan Ringkasan Buku:")

translator = Translator()

def translate_to_english(text, max_retries=3):
    translation = None
    retries = 0
    while retries < max_retries:
        try:
            translation = translator.translate(text, dest='en')
            break
        except Exception as e:
            st.error(f"Error: {e}. Retrying...")
            retries += 1
            time.sleep(1)  # Add a short delay before retrying
    # Codingan Tresna geming
    # if translation:

        # return translation.text
    # else:
        # st.error("Translation failed after multiple retries.")
        # return ""

    # Codingan editan Mikhael Ganteng
    if translation and hasattr(translation, 'text'):
        return translation.text
    else:
        st.error("Translation failed after multiple retries.")
        return ""


if st.button("Terjemahkan ke Bahasa Inggris"):
    translated_summary = translate_to_english(book_summary)
    st.write("Ringkasan Terjemahan:")
    st.write(translated_summary)
    book_summary = translated_summary

if st.button("Prediksi Genre"):
    predict_genre(book_summary)
