# GenAI_RAG_Streamlite_Ollama
GenAI_RAG_Streamlite_Ollama

This code is an implementation of a chatbot that allows users to ask questions about the content of PDF documents. Here's a breakdown of the code:

It imports various libraries and modules required for the chatbot functionality.
It defines the file paths, folders, and models required for PDF processing.
It checks if the PDF folder path exists and raises an error if it doesn't.
It defines a function load_and_chunk_pdf that loads and chunks PDF files into smaller pieces of text.
It defines a function vectorDBLoader that loads or creates a vector database using the Chroma library for vectorization.
It defines a function process_llm_response that prints the result value and sources from the given language model response.
It defines a function generate_response that takes a user's question as input and generates a response using a language model and a retrieval question answering chain.
It defines a function main that handles the logic for interacting with the chatbot using Streamlit, including taking user input, generating a response, and displaying the conversation.
The main function is executed if the script is run directly.
Overall, this code sets up the necessary components for a chatbot that can answer questions based on the content of PDF documents.

Reference -
https://medium.com/@Stan_DS/creating-a-retrieval-augmented-generation-rag-model-on-your-laptop-for-free-0ca119d03c48
https://medium.com/@rubentak/talk-to-your-files-in-a-local-rag-application-using-mistral-7b-langchain-and-chroma-db-no-2b4ba77358e0

