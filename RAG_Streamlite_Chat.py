from PyPDF2 import PdfReader
# from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain.chains import Retr
import os
import streamlit as st


# PDF processing block

pdf_folder_path = "./data/"
vectore_store_dir = "./vectorStore"
embedding = OllamaEmbeddings(model="mistral")
langchain_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm_ollama = Ollama(model="codellama",
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler]))
llm_openAI = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        model="gpt-3.5-turbo",
                        temperature=0.5)


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

def process_llm_response(llm_respose):
    print(llm_respose["result"])
    print('\n\nSources:')
    for source in llm_respose["source_documents"]:
        print(source.metadata['source'])


def generate_response(question):
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    - If you don't know the answer, just say I do not know.
    - build your answer based on provided documents only.
    - do NOT answer question out of context, but do try to answer it if you know it. 
    - respond to general greetings as "Selin". 
    {context}
    Question: {question}                                                                        
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    vectorDB = vectorDBLoader(embeddingType=langchain_embedding)
    retriever = vectorDB.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(llm=llm_ollama,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True, 
                                  chain_type_kwargs={"prompt":PROMPT})
    response = qa_chain({"query":question})
    return response

def main():

    # res = generate_response("talk about different type of Webdriver elemnt locators")
    # process_llm_response(res)
    # Streamlit interaction
    st.title("Chat with your PDFs")
    st.session_state.setdefault('messages', [{"role": "assistant", "content": "Ask me a question!"}])

    # Handle the chat input and response logic
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_response(prompt)
        assistant_response = response.get('result', "I'm sorry, I couldn't find an result.")
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
        if 'source_documents' in response and response['source_documents']:
        # Initialize an empty list to hold the source metadata
            sources_metadata = []

            # Loop through each document in the source documents
            for doc in response['source_documents']:
                # Assuming doc is an instance of Document and has a method or property to access metadata
                # And assuming metadata is an object or dictionary that includes a 'source' key or attribute
                try:
                    # Attempt to access the source information from the metadata
                    source_info = doc.metadata['source']  # Use this if metadata is a dictionary
                    # If metadata is accessed through methods or properties, adjust the above line accordingly
                    sources_metadata.append(source_info)
                except AttributeError:
                    # Handle cases where doc does not have a metadata attribute or metadata does not have a 'source' key
                    print(f"Document {doc} does not have the expected metadata structure.")

            # Concatenate all source metadata into one string, separated by new lines
            sources_concatenated = '\n'.join(sources_metadata)
        
        # Append the concatenated source metadata to the assistant's message
        st.session_state.messages.append({"role": "assistant", "content": f"Sources:\n{sources_concatenated}"})


    # Display messages
    for message in st.session_state.messages:
        with st.container():
            role = message["role"]
            #     st.success(message["content"])
            if role == "assistant":
                st.info(message["content"])
                # st.info(message["source"])
            else:
                st.success(message["content"])


    



if __name__ == '__main__':
    main()
