# Import necessary libraries
from flask import Flask, render_template, request
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
import re

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

model = ChatOllama(model = 'llama3.2')

# Define custom translation tool using @tool decorator
@tool
def translate_text(input: str) -> str:
    """
    Translates text into the specified target language.

    Format: "Translate to <language>: <text>"

    Example: "Translate to French: the book was written by Anna"
    """
    match = re.search(r'translate(?: to)?\s+([a-zA-Z]+)\s*:\s*(.+)', input, re.IGNORECASE)
    if not match:
        return "Invalid format. Use: 'Translate to <language>: <text>'"

    target_language = match.group(1).capitalize()
    input_text = match.group(2).strip()

    supported_languages = ["French", "Hindi", "Spanish", "German", "Chinese", 
                           "Japanese", "Italian", "Korean", "Russian"]

    if target_language not in supported_languages:
        return f"Unsupported language '{target_language}'. Supported: {', '.join(supported_languages)}"

    translation_template = """
    You are a professional translator. Translate the following text into {target_language}.
    Maintain the same tone, style, and meaning as the original text.

    Text to translate:
    {input_text}

    Translation:
    """

    prompt = ChatPromptTemplate.from_template(template=translation_template)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"input_text": input_text, "target_language": target_language})

# Function to generate a response from the LLM based on the user's question
def llm_response(question : str) -> str:
    
    chat_template = """
    You are an assistant for summarizing books and answering follow up questions.
    
    Your task is to answer questions about the book accurately and concisely.

    Follow these rules STRICTLY:
    1. ONLY provide summaries when explicitly asked with words like "summarize" or "overview" or "summary"
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

    chat_prompt = ChatPromptTemplate.from_template(template = chat_template)
    chain = chat_prompt | model | StrOutputParser()

    # Create a retriever from the Chroma database
    retriever = chroma_db.as_retriever(search_type = "similarity", search_kwargs = {"k" : 2})
    
    # Create a conversational retrieval chain using the model, retriever, and memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = model, 
        retriever = retriever, 
        memory = memory, 
        combine_docs_chain_kwargs = {"prompt": chat_prompt}
    )

    result = qa_chain.invoke(question)
    
    return result['answer']

# Define agent prompt template
agent_template = """
You are a helpful AI assistant for answering questions about books and translating content.

TOOLS:
{tools}

You can use tools like this:
Question: the user input
Thought: Do I need a tool? Yes/No
Action: one of [{tool_names}]
Action Input: the input string
Observation: the result
...
Thought: I now know the final answer
Final Answer: <your answer>

Question: {input}
{agent_scratchpad}
"""

# Setup tools
tools = [translate_text]
tool_names = [tool.name for tool in tools]

# Final prompt
agent_prompt = ChatPromptTemplate.from_template(template=agent_template).partial(
    tool_names=", ".join(tool_names),
    tools="\n".join(f"{tool.name}: {tool.description}" for tool in tools)
)

# Create agent and executor
agent = create_react_agent(llm=model, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Define the route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=["GET", "POST"])
def chatbot_response():
    userText = request.args.get('msg') if request.method == "GET" else request.form.get('msg')

    if any(word in userText.lower() for word in ["summarize", "overview", "summary"]):
        response = llm_response(userText)
    elif "translate" in userText.lower():
        try:
            result = agent_executor.invoke({"input": userText})
            response = result.get('output', result)  # Get 'output' key if returned as a dict
        except Exception as e:
            response = f"Translation failed: {str(e)}"
    else:
        response = llm_response(userText)

    return str(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=False)