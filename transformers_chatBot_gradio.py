from transformers import pipeline
from transformers import Conversation
import gradio as gr
from langchain.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm_ollama = Ollama(model="codegemma",
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler]),
             temperature=0.2,
             verbose=True,
             num_gpu=4)

chatbot = pipeline(task="conversational",model="facebook/blenderbot-400M-distill")
chatbot_ollama = pipeline(task="conversational",model=llm_ollama)
# msg = 'Hi I am Shaw, how are you?'
# conversation = Conversation(messages=msg)
# conversation = chatbot(conversation)
# print(conversation)
# conversation.add_user_input("Where do you work?")
# print(conversation)


message_list = []
response_list = []

def vanilla_chatbot(message, history):
    conversation = Conversation(text=message, past_user_inputs=message_list, generated_responses=response_list)
    conversation = chatbot(conversation)

    return conversation.generated_responses[-1]

demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting.")

demo_chatbot.launch()