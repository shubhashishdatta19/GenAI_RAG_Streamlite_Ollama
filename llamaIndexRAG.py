from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os.path


Persistent_dir = './Strorage'
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
ollm = Ollama(model='mistral', request_timeout=600)
Settings.llm = ollm
Settings.embed_model = embedding_model

if not os.path.exists(Persistent_dir):
    document = SimpleDirectoryReader('./Books', recursive=True).load_data()
    index = VectorStoreIndex.from_documents(document)
    index.storage_context.persist(persist_dir=Persistent_dir)
else:
    strorage_context = StorageContext.from_defaults(persist_dir=Persistent_dir)
    index = load_index_from_storage(strorage_context)
    
query_engine = index.as_query_engine(llm=ollm)
chatbot = index.as_chat_engine(llm=ollm)

print(query_engine.query("What is System testing, give answer in 500 words. Do talk about different technique used in System testing and write that in bullet points."))



