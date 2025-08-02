import streamlit as st
import fitz  # PyMuPDF
import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="StudyMate 📘", layout="wide")
st.title("📚 StudyMate - Ask Questions from Your PDF")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        # Extract text
        pdf_text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()

        st.success("PDF loaded successfully!")

        question = st.text_input("❓ Ask a question about the PDF")

        if question:
            with st.spinner("Thinking..."):
                # Prepare context
                context = Document(page_content=pdf_text[:4000])  # trim if needed

                # Call OpenAI model via OpenRouter
                chain = load_qa_chain(OpenAI(
                    temperature=0.2,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name="openrouter/openai/gpt-3.5-turbo",
                    openai_api_key=openai.api_key,
                ), chain_type="stuff")

                result = chain.run(input_documents=[context], question=question)
                st.markdown("### 💬 Answer")
                st.write(result)
