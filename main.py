import streamlit as st
import pickle
import pandas as pd
from preprocessing import preprocess_text

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

st.set_page_config(page_title="AI or NOT?", page_icon=":robot:", layout="wide")

title = st.markdown("<h1 style='text-align: center; color: white;'>AI or NOT?</h1>", unsafe_allow_html=True)

text = st.text_area(label="Enter the text",height=300,placeholder="Enter the text you want to check...")

col1, col2,col3 = st.columns(3)
st.empty()
st.empty()
button = col2.button("Check",use_container_width=True,type="primary")

if button:
    if text == "":
        st.write("Please enter some text")
    else:
        text = pd.DataFrame({"Article_content": [text]})
        text = preprocess_text(text)
        vector = vectorizer.transform(text['final_text'])
        vector = vector.todense()
        vector = pd.DataFrame(vector)
        prediction = loaded_model.predict(vector)
        st.empty()
        st.empty()
        col2.markdown(f"<h3>The content is: <b>{prediction[0]}</b></h3>",unsafe_allow_html=True)



