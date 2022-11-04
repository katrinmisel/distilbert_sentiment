import pandas as pd
import tensorflow as tf
from keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pickle
import streamlit as st

def construct_encodings(x, tkzr, trucation=True, padding=True):
    return tkzr(x, max_length=100, truncation=trucation, padding=padding)

def construct_tfdataset(encodings):
    return tf.data.Dataset.from_tensor_slices(dict(encodings))
    
def create_predictor(model):
  tkzr = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  def predict_proba(text):
      x = [text]

      encodings = construct_encodings(x, tkzr)
      tfdataset = construct_tfdataset(encodings)
      tfdataset = tfdataset.batch(1)

      preds = model.predict(tfdataset).logits
      preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
      return preds[0][0]
      
  return predict_proba
  
new_model = TFDistilBertForSequenceClassification.from_pretrained('katrinmisel/distilbert-tweets')

clf = create_predictor(new_model)

def predict_sentiment(text):
  p = clf(text)
  if p > 0.5 :
    sentiment = "Positive"
    face = ":slightly_smiling_face:"
  else: 
    sentiment = "Negative"
    face = ":slightly_frowning_face:"
  
  return sentiment, face
  

###### set up streamlit

st.title("Sentiment prediction :robot_face:")

text = st.text_input(label="Please enter your text here and click \"Predict\"")

st.write("")

if st.button('Predict'):
    st.subheader(f"Text: \"{text}\"")
    sentiment, face = predict_sentiment(text)
    st.subheader(f"Sentiment prediction: {sentiment} {face}")
