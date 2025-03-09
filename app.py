# Import necessary libraries
from flask import Flask, render_template, request
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize Flask application
app = Flask(__name__)

# Using OllamaEmbeddings with the 'llama3.2' model for generating embeddings
embeddings = OllamaEmbeddings(model = 'llama3.2')

# Initialize Chroma database with a specific directory and collection name
chroma_db = Chroma(persist_directory = './book_data/anna-sewell_black-beauty', 
                   embedding_function = embeddings, 
                   collection_name = 'books_data')

# Using ConversationBufferMemory to store and retrieve conversation history
memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

# Function to generate a response from the LLM based on the user's question
def llm_response(question):
    template = """
    You are an assistant for summarizing books and answering follow up questions.
    - If the user asks for a summary of a book, generate a detailed summary consisting of 10-15 sentences. 
    - If the user asks a follow-up question about the book (such as details about the author, characters, themes, or plot points), answer the question directly without generating a full summary.

    Context : {context}

    Question : {question}

    Here is the conversation history: {chat_history}

    Summary:
    """

    prompt = ChatPromptTemplate.from_template(template = template)
    model = ChatOllama(model = 'llama3.2')
    chain = prompt | model | StrOutputParser()

    # Create a retriever from the Chroma database
    retriever = chroma_db.as_retriever(search_type = "similarity", search_kwargs = {"k" : 1})
    
    # Create a conversational retrieval chain using the model, retriever, and memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = model, 
        retriever = retriever, 
        memory = memory, 
        combine_docs_chain_kwargs = {"prompt": prompt}
    )

    context = qa_chain.invoke(question)
    
    return context['answer']

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling chatbot responses
@app.route('/get', methods=["GET", "POST"])
def chatbot_response():
    userText = request.args.get('msg') if request.method == "GET" else request.form.get('msg')
    response = llm_response(userText)
    return str(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=False)
