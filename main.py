import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import AstraDB

import os
import uuid

token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')

# Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = '1234'

# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    text = text.replace('\x00', '')  # Remove null characters
    return text

def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,
                      "size": filename.size,
                      "unique_id": unique_id},
        ))
    return docs

def get_vectorstore(text_chunks):
    vstore = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="hiring_assistant",
        api_endpoint=api_endpoint,
        token=token,
    )
    inserted_ids = vstore.add_documents(text_chunks)
    print(f"\nInserted {len(inserted_ids)} documents.")
    return vstore

def similar_docs(vectorstore, search_query):
    similar_docs = vectorstore.similarity_search_with_score(search_query, k=3)
    return similar_docs

def get_summary(current_doc):
    llm = OpenAI(temperature=0.5, max_tokens=2000)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary

def main():
    st.set_page_config(page_title="Gen AI powered Hiring Assistant", page_icon=":robot_face:")

    # Centering and coloring the subtitle using HTML in st.markdown
    st.markdown("<h3 style='text-align: center; color: violet;'>Gen AI powered Hiring Assistant. Made for polycab by Vishleshan using Astra Vector Store and OpenAI</h3>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Enter the job description, key skills, and upload resumes to get started</h3>", unsafe_allow_html=True)

    st.markdown("#### 📄 Job Description")
    job_description = st.text_area("Please paste the job description here", key="job_description")

    st.markdown("#### 🏅 Key Skills")
    key_skills = st.text_input("Enter key skills separated by commas (e.g., Python, Data Analysis, Machine Learning)", key="key_skills")

    st.markdown("#### 📥 Upload Resumes")
    document_count = st.text_input("Enter the number of resumes you want to screen", key="document_count")
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Who is the best fit?")

    if submit:
        with st.spinner('Processing...'):
            st.session_state['unique_id'] = uuid.uuid4().hex

            # Create a documents list out of all the user uploaded PDF files
            final_docs_list = create_docs(pdf, st.session_state['unique_id'])
            st.write(f"*Resumes uploaded:* {len(final_docs_list)}")

            # Push data to AstraDB
            vectorstore = get_vectorstore(final_docs_list)

            # Combine job description and key skills for the search query
            search_query = f"{job_description} {key_skills}"

            # Fetch relevant documents from AstraDB
            relevant_docs = similar_docs(vectorstore, search_query)
            
            st.write(":heavy_minus_sign:" * 30)
            
            # Displaying relevant documents
            for item in range(len(relevant_docs)):
                st.subheader(f"👉 {item + 1}")
                st.write(f"**File**: {relevant_docs[item][0].metadata['name']}")

                with st.expander('Show details 👀'):
                    st.info(f"**Match Score**: {relevant_docs[item][1]}")
                    summary = get_summary(relevant_docs[item][0])
                    st.write(f"**Summary**: {summary}")           
        
        st.success("Hope I was able to save your time ❤️")
        st.markdown(
            "<h3 style='text-align: center; font-size: 30px; color: violet;'> To know more about DataStax Astra Vector Store, visit <a href='https://docs.datastax.com/en/astra/astra-db-vector/' target='_blank'>here</a> </h3>",
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    load_dotenv()
    main()