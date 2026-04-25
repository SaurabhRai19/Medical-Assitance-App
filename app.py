from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

'''
The app flow:
First we create trials.ipynb, which contains loaders, chunking creation, using those 
chunks we create the embeddings with HuggingFace, than upload it pine cone index "medical_
chatbot", and can use those embeddings with below app.py code where we are searching some-
thing from the existing pinecone index.

We have saved those loaders, create the chunks using huggingface in Helper.py file.
We have kept a system prompt saved for our model in prompt.py file.
Run store_index.py file to create the index in pinecone(in other words to create the vector
store.) - command: python store_index.py
Finally Run the app.py with python app.py to see the working version at localhost:8080 

'''
app = Flask(__name__)


load_dotenv()
'''
Using Pinecone as Vector Database:
[Pinecone](https://www.pinecone.io/) is a fully managed, cloud-native vector database specifically designed for high-performance similarity search over high-dimensional vector embeddings. Unlike traditional relational databases that optimize for exact keyword matches, Pinecone is purpose-built to handle the "physics" of AI data, enabling machines to find conceptually similar content (e.g., matching "fast car" with "speedy vehicle") in milliseconds. [1, 2, 3, 4] 
## Key Differences from Other Databases

* Infrastructure Abstraction: Unlike self-hosted options like [Milvus](https://milvus.io/) or [Weaviate](https://weaviate.io/), Pinecone is a proprietary SaaS. It requires zero server provisioning, tuning, or maintenance—you simply interact with it via an API.
* Specialization vs. Integration: While extensions like [pgvector](https://github.com/pgvector/pgvector) "bolt on" vector search to existing SQL databases, Pinecone is built from the ground up for vectors. This results in superior performance and cost-efficiency when scaling to hundreds of millions or billions of vectors.
* Storage-Compute Separation: Pinecone decouples storage from compute, using BLOB storage as its source of truth. This allows the database to scale query throughput independently of data size. [2, 4, 5, 6, 7, 8] 

## Core Definitions and Features

* Index: The primary organizational unit in Pinecone. It is an isolated environment that holds a collection of vectors with a fixed dimensionality (e.g., 1536 for OpenAI models) and a specific distance metric.
* Vector Embeddings: Numerical representations of data (text, images, audio) that capture semantic meaning. Pinecone stores these as arrays of floating-point numbers.
* Metadata Filtering: The ability to attach JSON-like tags (e.g., category: "billing", date: "2024-04-23") to vectors. You can then perform searches that combine vector similarity with hard constraints.
* Distance Metrics: Mathematical functions used to determine similarity. Pinecone supports three primary types:
* Cosine Similarity: Measures the angle between vectors; best for normalized text embeddings.
   * Dot Product: Measures direction and magnitude.
   * Euclidean Distance: Measures straight-line distance.
* Namespaces: Isolated partitions within a single index. They are ideal for multi-tenancy, allowing you to separate data for different customers without creating new indexes.
* Serverless Architecture: A deployment model that automatically scales resources based on usage. You pay only for storage and read/write operations performed, eliminating the need for capacity planning.
* Hybrid Search: A feature that blends semantic (dense) vector search with keyword-based (sparse) matching to provide more robust retrieval. [2, 3, 4, 8, 9, 10, 11, 12, 13] 

'''
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
#OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

#chatModel = ChatOpenAI(model="gpt-4o")
chatModel = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
