from PyPDF2 import PdfReader
# from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema import LLMResult
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os



# PDF processing block

pdf_folder_path = "./data/"
vectore_store_dir = "./vectorStore"
embedding = OllamaEmbeddings(model="mistral")
langchain_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(pdf_folder_path):
    raise FileNotFoundError
else:
    print("PDF folder exists")


def load_and_chunk_pdf(pdf_file_path):

    loaders = [UnstructuredPDFLoader(os.path.join(pdf_file_path, fn))
              for fn in os.listdir(pdf_file_path) if fn.endswith('.pdf') ]
    all_text= []
    for loader in loaders:
        data = loader.load()
        print("Text spliting initiated for file : " +loader.file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=0
        )
        texts = text_splitter.split_documents(data)
        all_text.extend(texts)
    
    return all_text

def vectorDBLoader(embeddingType):
    if not os.path.exists(vectore_store_dir) or not os.listdir(vectore_store_dir):
        all_text = load_and_chunk_pdf(pdf_folder_path)
        print("Strating data load into the Vectore Store")
        vectorStore = Chroma.from_documents(documents=all_text, 
                                            embedding=embeddingType, persist_directory=vectore_store_dir)
        print("Vector DB load completed")
        vectorStore.persist()
        print("Vector Db save to persist directory")
    else:
        print("Loading Vector DB from persistent store")
        vectorStore = Chroma(persist_directory=vectore_store_dir, embedding_function=embeddingType)
        print("Data loading from persistent store completed")
    return vectorStore


def main():
    vectorDB = vectorDBLoader(embeddingType=langchain_embedding)
    retriver = vectorDB.as_retriever()
    doc = retriver.get_relevant_documents("What is this document about ?")
    print(doc)

if __name__ == '__main__':
    main()
