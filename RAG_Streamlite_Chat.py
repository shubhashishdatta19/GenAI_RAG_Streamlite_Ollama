from PyPDF2 import PdfReader
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
from langchain_community.llms.llamacpp import LlamaCpp
import os
import streamlit as st


# PDF processing block

pdf_folder_path = "./data/"
vectore_store_dir = "./vectorStore"
embedding = OllamaEmbeddings(model="phi")
langchain_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

llm_ollama = Ollama(model="codegemma",
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler]),
             temperature=0.2,
             verbose=True,
             num_gpu=4)

llm_openAI = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        model="gpt-3.5-turbo",
                        temperature=0.5)

llm_cpp = LlamaCpp(
    model_path="C:\\AI\\model\\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.75, n_ctx=2048,
    top_p=1,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler]),
    verbose=True, 
     n_gpu_layers=4 # Verbose is required to pass to the callback manager
)


if not os.path.exists(pdf_folder_path):
    raise FileNotFoundError
else:
    print("PDF folder exists")


def load_and_chunk_pdf(pdf_file_path):
    """
    Loads and chunks a PDF file into smaller pieces of text.

    Args:
        pdf_file_path (str): The path to the directory containing the PDF files.

    Returns:
        list: A list of extracted text chunks from all PDF files.
    """
    # Create a list of PDF loaders for all PDF files in the specified directory
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_file_path, fn))
              for fn in os.listdir(pdf_file_path) if fn.endswith('.pdf') ]
    
    # Initialize an empty list to store all the extracted text
    all_text= []
    
    # Iterate through each PDF loader
    for loader in loaders:
        # Load the PDF file
        data = loader.load()
        
        # Print a message indicating the initiation of text splitting for the loaded file
        print("Text splitting initiated for file : " +loader.file_path)
        
        # Define a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=0
        )
        
        # Split the document into chunks of specified size and append them to the all_text list
        texts = text_splitter.split_documents(data)
        all_text.extend(texts)
    
    # Return the combined list of extracted text from all PDF files
    return all_text

def vectorDBLoader(embeddingType):
    """
    Loads or creates a vector database using the Chroma library.

    Args:
        embeddingType (str): The type of embedding to use for vectorization.

    Returns:
        vectorStore (Chroma): The vector database object.

    Raises:
        FileNotFoundError: If the vector store directory doesn't exist or is empty.
    """

    # Check if vector store directory is empty or doesn't exist
    if not os.path.exists(vectore_store_dir) or not os.listdir(vectore_store_dir):
        # Load and chunk PDF documents from a specified folder path
        all_text = load_and_chunk_pdf(pdf_folder_path)
        print("Starting data load into the Vector Store")

        # Create a new vector store using Chroma library
        vectorStore = Chroma.from_documents(
            documents=all_text,
            embedding=embeddingType,
            persist_directory=vectore_store_dir
        )
        print("Vector DB load completed")

        # Persist the vector store to disk
        vectorStore.persist()
        print("Vector DB saved to persist directory")
    else:
        print("Loading Vector DB from persistent store")

        # Load existing vector store from disk
        vectorStore = Chroma(
            persist_directory=vectore_store_dir,
            embedding_function=embeddingType
        )
        print("Data loading from persistent store completed")

    return vectorStore

def process_llm_response(llm_response):
    """
    Prints the result value and sources from the given LLM response.

    Args:
        llm_response (dict): The LLM response dictionary.

    Returns:
        None
    """
    # Print the value associated with the key "result"
    print(llm_response["result"])

    # Print the sources from the "source_documents" list
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        # Access the metadata property of each source and print it
        print(source.metadata['source'])

def generate_response(question):
    # Define a prompt template with placeholders for context and question
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    - If you don't know the answer, just say I do not know.
    - build your answer based on provided documents only.
    - do NOT answer question out of context.
    - respond to general greetings as "Selin". 
    {context}
    Question: {question}                                                                        
    """
    
    # Create a PromptTemplate object with the defined template and input variables
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    # Load vectorDB with specified embeddingType
    vectorDB = vectorDBLoader(embeddingType=langchain_embedding)
    # Create a retriever from the vectorDB
    retriever = vectorDB.as_retriever()

    # Create a RetrievalQA object with specified parameters
    qa_chain = RetrievalQA.from_chain_type(llm=llm_cpp,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True, 
                                  chain_type_kwargs={"prompt":PROMPT})

    # Use the qa_chain to answer the provided question
    # The question is passed as a dictionary with "query" key
    response = qa_chain({"query":question})

    # Return the generated response
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
