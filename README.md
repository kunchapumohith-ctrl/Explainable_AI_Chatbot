This project is an Explainable AI Chatbot that leverages Large Language Models (LLMs) and Vector Similarity Search to provide accurate, context-aware, and transparent responses.
Unlike traditional chatbots, this system not only answers queries but also explains the reasoning behind its responses by retrieving relevant information from a knowledge base
Features:
 Natural language conversation using LLMs
 Semantic search using vector embeddings
 Context-aware answers from knowledge base
 Explainable responses with source references
 Fast retrieval using similarity search
 Modular and scalable architecture
 Tech Stack:
Programming Language: Python
LLM Integration: OpenAI / Hugging Face
Vector Database: FAISS / Pinecone / ChromaDB
Embeddings: Sentence Transformers / OpenAI Embeddings
Frameworks: LangChain (optional)
Frontend (optional): Streamlit / React
How It Works:
User query is converted into embeddings
Vector database performs similarity search
Relevant documents are retrieved
LLM generates response using retrieved context
System provides explanation along with answer
Example:
User: What is machine learning?
Bot: Machine learning is a subset of AI...
Explanation: Retrieved from document X with similarity score 0.87


Live Demo:
Try the app here: 🎵 [Open App](https://explainable-ai-chatbot-mohith-dxtbgrtvkgjocy7uvi5z.streamlit.app/)
