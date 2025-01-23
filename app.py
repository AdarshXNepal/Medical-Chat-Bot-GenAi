# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import vertexai
# from vertexai.preview.generative_models import (
#     GenerationConfig,
#     GenerativeModel
# )
# from flask import Flask, render_template, request
# from src.prompt import *
# import os

# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# project_id = os.getenv("project_id")
# project_region = os.getenv("region")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# vertexai.init(project=project_id, location=project_region)

# from langchain_google_vertexai import ChatVertexAI

# embeddings = download_hugging_face_embeddings()


# index_name = "medicalbot"

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# llm = ChatVertexAI(
#     model_name="gemini-1.5-flash-001",  # Correct parameter
#     temperature=0.3,
#     max_tokens=300,
#     stop=None
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# @app.route("/")
# def index():
#     return render_template('index.html')


# @app.route("/<string:msg>", methods=["GET", "POST"])
# def chat():
#     user_message = ""
#     bot_response = ""

#     if request.method == "POST":
#         user_message = request.form["msg"]
#         # Process the user message and generate a bot response
#         bot_response = "This is a response to: " + user_message

#     return render_template("index.html", user_message=user_message, bot_response=bot_response)





# app = Flask(__name__)
# '''
# @app.route("/", methods=["GET", "POST"])
# def index():
#     user_message = ""
#     bot_response = ""

#     if request.method == "POST":
#         user_message = request.form["msg"]
#         # Process the user message and generate a bot response
#         bot_response = "This is a response to: " + user_message

#     return render_template("index.html", user_message=user_message, bot_response=bot_response)

# if __name__ == "__main__":
#     app.run(debug=True)'''

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8300, debug= True)
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel
from langchain_google_vertexai import ChatVertexAI
import os
from src.prompt import *

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# Retrieve necessary API keys and project information
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
project_id = os.getenv("project_id")
project_region = os.getenv("region")


os.environ["TOKENIZERS_PARALLELISM"] = "False" 

# Set Pinecone API key and initialize Vertex AI
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
vertexai.init(project=project_id, location=project_region)

# Download Hugging Face embeddings (this is a function defined in your helper module)
embeddings = download_hugging_face_embeddings()

# Set up Pinecone vector store and retriever
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize the language model with Vertex AI
llm = ChatVertexAI(
    model_name="gemini-1.5-flash-001",  # Correct parameter for Vertex AI model
    temperature=0.5,
    max_tokens=400,
    stop=None
)

# Define the prompt template for the system
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # Assuming system_prompt is defined elsewhere
        ("human", "{input}"),
    ]
)

# Create a question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Route for the home page (index)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8200, debug=True)