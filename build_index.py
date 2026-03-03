import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"

def load_pdfs():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            reader = PdfReader(path)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            docs.append({"text": text, "source": file})
    return docs

def split_docs(docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in docs:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append({"text": chunk, "source": doc["source"]})
    return chunks

def build_index():
    docs = load_pdfs()
    chunks = split_docs(docs)

    texts = [c["text"] for c in chunks]
    metas = [{"source": c["source"]} for c in chunks]

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings, metadatas=metas)
    db.save_local("faiss_index")

if __name__ == "__main__":
    build_index()
