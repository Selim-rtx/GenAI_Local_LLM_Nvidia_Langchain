import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Union
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain import hub
from langchain_community.llms import LlamaCpp
import gradio as gr

# Function to load HTML content from a URL
def html_document_loader(url: Union[str, bytes]) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for HTTP errors
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception: {e}")
        return ""

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        text = re.sub("\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"Exception {e} while parsing document from {url}")
        return ""

# Function to create embeddings
def create_embeddings(embedding_path: str = "./embed"):
    urls = [
        "https://github.com/LasseRegin/medical-question-answer-data/blob/master/ehealthforumQAs.json",
        "https://github.com/LasseRegin/medical-question-answer-data/blob/master/icliniqQAs.json",
    ]

    documents = [html_document_loader(url) for url in urls]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
    texts = text_splitter.create_documents(documents)
    index_docs(texts, embedding_path)
    print("Generated embeddings successfully")

# Function to index documents
def index_docs(documents: List[str], dest_embed_dir: str) -> None:
    embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")  # Updated model
    texts = [doc.page_content for doc in documents]

    if os.path.exists(dest_embed_dir):
        faiss_index = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
        faiss_index.add_texts(texts)
        faiss_index.save_local(folder_path=dest_embed_dir)
    else:
        faiss_index = FAISS.from_texts(texts, embedding=embeddings)
        faiss_index.save_local(folder_path=dest_embed_dir)

# Get NVIDIA API key
os.environ["NVIDIA_API_KEY"] = "YOUR_API_KEY"

# Create embeddings if they don't exist
embedding_path = "./embed"
if not os.path.exists(embedding_path):
    create_embeddings(embedding_path)

# Load embeddings and models
embedding_model = NVIDIAEmbeddings(model="NV-Embed-QA")  # Updated model
docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Set CUDA_VISIBLE_DEVICES to use the first GPU (you can adjust this based on your setup)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path to the downloaded GGUF model file
model_path = r"YOUR_PATH\mistral-7b-openorca.Q4_0.gguf" #Quantized Mistral-7b has been chosen for example. If you want to run the model on-device, such as a laptop, take a quantized model.
n_gpu_layers=22,  
n_batch=8,       
n_ctx=2048

# Configure model parameters for GPU usage
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,      # Number of GPU layers to use
    n_batch=n_batch,                # Batch size
    n_ctx=n_ctx,                    # Context size
    f16_kv=True,                    # Use 16-bit floating point for key-value pairs
    gpu=True,                       # Enable GPU
    verbose=True,                   # Enable verbose output
)

# Format documents for the chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Load prompt from hub
rag_prompt = hub.pull("rlm/rag-prompt")

# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Define department mapping function
def recommend_department(symptoms: str) -> str:
    departments = {
        "cardio": ["chest pain", "heart attack", "hypertension"],
        "neuro": ["headache", "stroke", "seizure"],
        "ortho": ["back pain", "joint pain", "fracture"],
        "ent": ["sore throat", "ear pain", "sinusitis"],
        "derm": ["rash", "itching", "skin infection"],
        "endocrinology": ["weight loss", "increased appetite", "palpitations", "hyperthyroidism"],
        # These are examples, and are not exhaustive. Add more mappings as needed
    }

    for department, keywords in departments.items():
        if any(keyword in symptoms.lower() for keyword in keywords):
            return department

    return "general"

# Define the chatbot function
def chatbot(input_text):
    docs = docsearch.similarity_search(input_text)
    response = chain.invoke({"context": docs, "question": input_text})
    department = recommend_department(input_text)
    return f"{response}\n\nRecommended Department: {department.capitalize()}"
    return response

# Create a Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=7, label="Enter your question"),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Dr Chatbot",
    description="Ask questions about patient's symptoms or observations"
)

# Launch the interface
iface.launch(share=True)
