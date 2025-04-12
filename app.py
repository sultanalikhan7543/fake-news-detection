import streamlit as st
import pickle
import nltk
import string
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----- Set Up Writable NLTK Path -----
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# ----- Download NLTK Resources -----
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    st.error(f"⚠️ NLTK Setup Error: {str(e)}")
    st.stop()

# ----- Load Model Components -----
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except Exception as e:
    st.error(f"🔧 Model Loading Error: {str(e)}")
    st.stop()

# ----- NLP Setup -----
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----- Preprocessing Function -----
def preprocess(text):
    try:
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                 if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"📝 Processing Error: {str(e)}")
        st.stop()

# ----- Streamlit UI -----
st.title("📰 Fake News Detection App")
user_input = st.text_area("Enter a News Article or Headline:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("🚩 Please enter some text!")
    else:
        try:
            clean_text = preprocess(user_input)
            vec = vectorizer.transform([clean_text])
            prediction = model.predict(vec)[0]
            result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.subheader(result)
        except Exception as e:
            st.error(f"❌ Prediction Failed: {str(e)}")