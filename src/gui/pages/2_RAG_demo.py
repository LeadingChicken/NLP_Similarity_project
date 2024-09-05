import streamlit as st
import requests
# Set up the app title
st.title("Demo trả lời paper bằng RAG và OpenAI Embedding")

# Dropdown for language selection
language = st.selectbox(
    "Chọn ngôn ngữ đầu ra",
    ("Tiếng Việt", "Tiếng Anh")
)

# Text input for the question
question = st.text_input("Câu hỏi:")

pdf_path = st.file_uploader("Chọn paper")

# Placeholder for model response
response_placeholder = st.empty()

# Dummy response generation (replace this with your model's response)
if st.button("Bấm vào đây để trả lời"):
    try:
        params = {
            "pdf_path" : pdf_path.name,
            "language" : language,
            "question" : question
        }
        print(params)
        # Fetching data from the API
        response = requests.get("http://127.0.0.1:8000/message/",params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            output = response.json()['model_output']
            
        else:
            output = "Có lỗi xảy ra. Hãy thử lại"
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    # Display the model's response
    response_placeholder.text_area("Câu trả lời:", value=output, height=150)