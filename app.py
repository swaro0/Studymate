import streamlit as st
from backend.pdf_processor import extract_text_from_pdf, chunk_text
from backend.embeddings import create_faiss_index
from backend.qa_engine import search_chunks, generate_answer

st.title("📚 StudyMate: AI-Powered PDF Q&A System")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    st.success(f"Processed {len(uploaded_files)} PDF(s) successfully!")
    index, embeddings = create_faiss_index(all_chunks)

question = st.text_input("Ask a question about your PDFs:")
if st.button("Get Answer") and question:
    relevant_chunks = search_chunks(question, all_chunks)
    context = " ".join(relevant_chunks)
    answer = generate_answer(question, context)
    st.markdown(f"### Answer:\n{answer}")
