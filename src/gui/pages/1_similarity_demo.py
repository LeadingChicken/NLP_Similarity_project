import streamlit as st
import requests
st.title('Demo độ tương đồng ngữ nghĩa giữa các model')
options = ['mBERT','XML-Roberta','LaBSE']
selected_option = st.selectbox("Chọn một model:", options)

vietnam_sentence = st.text_input('Nhập vào một câu trong tiếng việt')
english_sentence = st.text_input('Nhập vào một câu trong tiếng anh')
if st.button('Kiểm tra độ tương đồng giữa 2 câu'):
	data = {
		"sentence1":vietnam_sentence,
		"sentence2":english_sentence,
		"model_name":selected_option
	}
	url = "http://127.0.0.1:8000/similarity"
	response = requests.post(url,json=data)
	st.write('Độ tương đồng ngữ nghĩa giữa 2 câu là:',response.json()['similarity_score'])