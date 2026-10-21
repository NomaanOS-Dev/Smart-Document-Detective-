import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Smart Document Detective", page_icon="🔍", layout="wide")

st.title("🔍 Smart Document Detective - RAG Analyzer")
st.caption("Investigative AI Engine powered by LangChain, ChromaDB & LLMs | Built by NomaanOS-Dev")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Legal/Technical Document (PDF)", type="pdf")

if uploaded_file and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
        
    st.sidebar.success("Document uploaded successfully!")
    
    # Process PDF
    loader = PyPDFLoader("temp_doc.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Vector DB
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Detective Prompt System
    system_prompt = (
        "You are an expert technical and legal document investigator.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, say that you don't know.\n"
        "Keep the answer precise, rigorous, and cite facts directly from context.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    user_query = st.text_input("Enter your query or investigation point:")
    if user_query:
        with st.spinner("Analyzing document evidence..."):
            # Direct retrieval and execution via LCEL
            retrieved_docs = retriever.invoke(user_query)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context_text, "input": user_query})
            
            st.write("### Investigation Findings:")
            st.write(answer)
            
            with st.expander("Retrieved Document Evidence (Context)"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Evidence Chunk {idx+1}:**")
                    st.write(doc.page_content)
else:
    st.info("👈 Please enter your OpenAI API Key and upload a PDF document in the sidebar to begin investigation.")
