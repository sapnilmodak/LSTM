import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

if 'my_list' not in st.session_state:
    st.session_state.my_list = []

# Add an item to the list
if st.button('Add Item'):
    st.session_state.my_list.append("New Item")

# Try to pop an item
if 'my_list' not in st.session_state:
    st.session_state.my_list = []

# Button to add an item to the list with a unique key
if st.button('Add Item', key='add_item_button'):
    st.session_state.my_list.append("New Item")
    st.success("Item added!")

# Button to pop an item from the list with a unique key
if st.button('Pop Item', key='pop_item_button'):
    if st.session_state.my_list:  # Ensure the list is not empty
        popped_item = st.session_state.my_list.pop()
        st.success(f"Popped: {popped_item}")
    else:
        st.warning("No items to pop from the list!")

# Display current list
st.write("Current List:", st.session_state.my_list)