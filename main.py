import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import AstraDB

import os
import re
import uuid

# Load environment variables
load_dotenv()
token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')

# Function to extract text from PDFs
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.replace('\x00', '')  # Remove null characters

# Function to extract metadata (e.g., years of experience, skills) from resume text
def extract_metadata(text):
    # Extract years of experience
    exp_match = re.search(r'(\d+)\+?\s+years? of experience', text, re.IGNORECASE)
    years_of_experience = exp_match.group(1) if exp_match else "Not mentioned"

    # Extract key skills
    skills_match = re.findall(r'\b(Java|Python|Machine Learning|SQL|React|AWS|DevOps|Data Science)\b', text, re.IGNORECASE)
    key_skills = ', '.join(set(skills_match)) if skills_match else "Not mentioned"

    # Extract a brief description (first 100 words)
    description = ' '.join(text.split()[:100]) if text else "Not available"

    return {
        "years_of_experience": years_of_experience,
        "key_skills": key_skills,
        "description": description,
    }

# Function to create documents with metadata
def create_docs(user_pdf_list, unique_id):
    docs = []
    for pdf_file in user_pdf_list:
        text = get_pdf_text(pdf_file)
        metadata = extract_metadata(text)
        docs.append(Document(
            page_content=text,
            metadata={
                "name": pdf_file.name,
                "size": pdf_file.size,
                "unique_id": unique_id,
                "years_of_experience": metadata["years_of_experience"],
                "key_skills": metadata["key_skills"],
                "description": metadata["description"],
            },
        ))
    return docs

# Function to initialize the vector store
def get_vectorstore():
    return AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="hiring_assistant",
        api_endpoint=api_endpoint,
        token=token,
    )

# Function to fetch similar documents
def similar_docs(vectorstore, search_query):
    results = vectorstore.similarity_search_with_score(search_query, k=5)
    filtered_results = [doc for doc in results if doc[1] > 0.90]
    return filtered_results

# Function to summarize a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0.5, max_tokens=2000)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary

# Main Streamlit application
def main():
    st.set_page_config(page_title="Gen AI Hiring Assistant", page_icon=":robot_face:")

    # Sidebar for resume upload
    with st.sidebar:
        st.header("Upload Resumes")
        pdf = st.file_uploader("Upload resumes here (PDF only)", type=["pdf"], accept_multiple_files=True)
        upload_button = st.button("Upload")

    st.markdown("<h3 style='text-align: center; color: violet;'>Gen AI powered Hiring Assistant</h3>", unsafe_allow_html=True)

    st.markdown("#### 📄 Job Description")
    job_description = st.text_area("Paste the job description here", key="job_description")

    st.markdown("#### 🏅 Key Skills")
    key_skills = st.text_input("Enter key skills (comma-separated)", key="key_skills")

    st.markdown("#### 🔍 Search for Best Matches")
    search_button = st.button("Find Best Matches")

    # Initialize Vector Store
    vectorstore = get_vectorstore()

    # Handle resume upload
    if upload_button and pdf:
        with st.spinner("Processing uploaded resumes..."):
            unique_id = uuid.uuid4().hex
            docs = create_docs(pdf, unique_id)
            vectorstore.add_documents(docs)
            st.success(f"Uploaded {len(docs)} resumes successfully!")

    # Handle search
    if search_button:
        if not job_description and not key_skills:
            st.warning("Please provide a job description or key skills for the search.")
        else:
            with st.spinner("Searching for best matches..."):
                search_query = f"{job_description} {key_skills}"
                relevant_docs = similar_docs(vectorstore, search_query)

                if not relevant_docs:
                    st.info("No matches found with a similarity score greater than 90%.")
                else:
                    st.markdown("### Top Matching Resumes:")
                    for idx, (doc, score) in enumerate(relevant_docs):
                        st.subheader(f"Match {idx + 1}: {doc.metadata['name']}")
                        st.write(f"**Match Score:** {score:.2f}")
                        st.write(f"**Years of Experience:** {doc.metadata.get('years_of_experience', 'Not mentioned')}")
                        st.write(f"**Key Skills:** {doc.metadata.get('key_skills', 'Not mentioned')}")
                        st.write(f"**Description:** {doc.metadata.get('description', 'Not available')}")
                        with st.expander("See Summary"):
                            summary = get_summary(doc)
                            st.write(summary)

if __name__ == '__main__':
    main()
