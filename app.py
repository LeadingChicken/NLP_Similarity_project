import streamlit as st
import requests
# Set up the app title
st.title("Demo paper answering application using RAG and multilingual similarity search")

# Dropdown for language selection
language = st.selectbox(
    "Select output language",
    ("Vietnamese", "English")
)

# Text input for the question
question = st.text_input("Question:")

pdf_path = st.file_uploader("Choose a paper")

# Placeholder for model response
response_placeholder = st.empty()

# Dummy response generation (replace this with your model's response)
if st.button("Click here to get response"):
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
            output = "Failed to fetch a your question. Please try again."
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    # Display the model's response
    response_placeholder.text_area("Model Response:", value=output, height=150)


