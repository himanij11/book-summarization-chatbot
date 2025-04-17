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
    
    Your task is to answer questions about the book accurately and concisely.

    Follow these rules STRICTLY:
    1. ONLY provide summaries when explicitly asked with words like "summarize" or "overview"
    2. For all other questions, answer directly and specifically
    3. Never make up information - say "I don't know" if unsure
    4. Keep answers under 5 sentences unless it's a summary
    5. Never repeat full summaries when answering follow-ups

    Here are some examples of question-answer pairs:
    
    Question: Who is the main character in Black Beauty?
    Answer: The main character is Black Beauty, a handsome black horse who narrates his life story. Born on an English farm, he experiences both kind and cruel handlers throughout his life, serving as a carriage horse, cab horse, and work horse while observing human behavior from a unique equine perspective.
    
    Question: What are the main themes of the book?
    Answer: The main themes of Black Beauty include animal welfare, kindness versus cruelty, social justice, and the importance of empathy. The novel serves as a strong critique of animal mistreatment and advocates for compassion toward all living beings. It also explores how one's circumstances can change drastically and the importance of maintaining dignity through hardship.
    
    Question: Who wrote Black Beauty and when?
    Answer: Black Beauty was written by Anna Sewell and published in 1877. It was the only novel she wrote, and she died just five months after its publication. The book became one of the best-selling books of all time and was influential in raising awareness about animal welfare.

    User: What happens in Chapter 5?
    AI: In Chapter 5, Black Beauty is sold to Squire Gordon and meets his new stablemates including Merrylegs and Ginger.

    Context : {context}

    Here is the conversation history: {chat_history}

    Question : {question}

    Strict, concise answer: 
    """

    prompt = ChatPromptTemplate.from_template(template = template)
    model = ChatOllama(model = 'llama3.2')
    chain = prompt | model | StrOutputParser()

    # Create a retriever from the Chroma database
    retriever = chroma_db.as_retriever(search_type = "similarity", search_kwargs = {"k" : 2})
    
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