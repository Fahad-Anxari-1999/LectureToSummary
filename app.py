import logging
logging.basicConfig(level=logging.DEBUG)
import streamlit as st
import os
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import assemblyai as aai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain.schema import Document
def save_to_pdf(text, filename, type):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        margin = 50
        text_start_y = height - margin
        line_height = 15
        c.setFont("Times-Roman", 12)
        for line in textwrap.wrap(text, width=85):
            c.drawString(margin, text_start_y, line)
            text_start_y -= line_height
            if text_start_y < margin:
                c.showPage()
                c.setFont("Times-Roman", 12)
                text_start_y = height - margin
        c.save()
        if(type=="Transcribed"):
            st.download_button(f"Download Transcribed  as PDF", open(filename, "rb"), file_name=filename)
        if(type=="summary"):
            st.download_button(f"Download Summary as PDF", open(filename, "rb"), file_name=filename)
# ---- Streamlit App ----
st.title("Audio Transcription and Summarization App")

# 1. Upload JSON Credential File or Paste Credential
st.header("Google API Credentials")
uploaded_file = st.file_uploader("Upload Google Credential JSON File", type=["json"])
if uploaded_file:
    # Save credentials file temporarily
    temp_json_path = "temp_google_credentials.json"
    with open(temp_json_path, "wb") as f:
        f.write(uploaded_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path
    st.success("Credentials loaded successfully!")
else:
    credentials_json = st.text_area("Paste your Google Credential JSON here:")
    if credentials_json:
        with open("temp_google_credentials.json", "w") as f:
            f.write(credentials_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_google_credentials.json"
        st.success("Credentials set successfully!")

# 2. Upload and Transcribe Audio File
st.header("Audio File Upload and Transcription")
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a","mp4"])

if audio_file:
    # AssemblyAI Transcription
    filename = os.path.splitext(audio_file.name)[0]
    try:
        aai.settings.api_key = st.text_input("Enter your AssemblyAI API Key:", type="password")
        st.info("Transcribing your audio. Please wait...")
        transcriber = aai.Transcriber()
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        st.stop()
        # Save audio file temporarily
    audio_temp_path = "temp_audio_file"
    with open(audio_temp_path, "wb") as f:
        f.write(audio_file.read())
    
    try:
        transcript = transcriber.transcribe(audio_temp_path).text
        save_to_pdf(transcript, filename=f"{filename}_Translated.pdf",type="Transcribed")
        st.success("Transcription complete!")
        st.text_area("Transcribed Text", value=transcript, height=300)
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        st.stop()
    
    # 3. Summarization using LangChain
       
    # Initialize the embedding model (Google GenerativeAI example)

    st.header("Generate Summary")
    #text = "You are an assistant from question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise. \n\n {context}"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(transcript)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #documents = [Document(page_content=chunk) for chunk in texts]
    vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory="./chroma_db")
    vectorstore.persist()
    
    query = "Summarize this text"
    retrieved_docs = vectorstore.similarity_search(query, k=5)  
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the given content in approximately 20% of the original sentences."),
        ("human", "{context}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    try:
        summary = question_answer_chain.invoke({"context": retrieved_docs})
        st.text_area("Summary", value=summary, height=200)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        st.stop()
    
    # 4. Save to PDF Function
    
    
    # 5. PDF Download
    st.header("Save as PDF")
    save_to_pdf(summary, filename=f"{filename}",type="summary")
    

# End of Streamlit App
