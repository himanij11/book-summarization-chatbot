# Book Summarization and Q&A Chatbot

A Flask-based chatbot that provides book summaries and answers questions using LangChain, Ollama, and ChromaDB for retrieval-augmented generation.

## Overview

This application creates a conversational chatbot that can:
- Generate detailed summaries of books
- Answer follow-up questions about authors, characters, themes, and plot points
- Maintain conversation history for contextual responses

The current implementation is configured with data from Anna Sewell's "Black Beauty".

## Features

- **Retrieval-Augmented Generation**: Uses a vector database (ChromaDB) to retrieve relevant passages from the book
- **Conversational Memory**: Maintains context throughout the conversation
- **Web Interface**: Simple Flask-based web interface for interacting with the chatbot

## Technologies

- **Flask**: Web framework for the application
- **LangChain**: Framework for developing applications with large language models
- **Ollama**: Local LLM runtime using the llama3.2 model
- **ChromaDB**: Vector database for storing and retrieving text embeddings
- **ConversationalRetrievalChain**: LangChain chain for conversational question answering

## Project Structure
```
.
├── app.py                  # Main Flask application
├── templates/              # HTML templates
│   └── index.html          # Main page template
├── book_data/              # Directory containing book content for retrieval
│   └── anna-sewell_black-beauty/ # Example book data
├── requirements.txt        # Python dependencies
└── README.md               # This file

```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/book-summarization-chatbot.git
   cd book-summarization-chatbot
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure Ollama is running with the llama3.2 model:
   ```
   ollama pull llama3.2
   ollama run llama3.2
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`

3. Interact with the chatbot through the web interface. You can ask questions such as:
   - "Generate a summary for the book 'Black Beauty'"
   - "Who is the author of the book?"
   - "Tell me more about the author"
   - "Which are the main characters in the book?"