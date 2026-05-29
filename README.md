# RAG Document Chatbot

Chat with your documents using LangChain + LangGraph + GPT-4. Upload PDFs, get answers with source citations. Built this for a project where we needed to let non-technical users query internal docs without writing SQL or learning any tools.

## Stack

- **Python 3.11**, FastAPI
- - **LangChain** - document loading, text splitting, chain composition
  - - **LangGraph** - multi-step agent with memory and tool use
    - - **ChromaDB** - local vector store (can swap for Pinecone or FAISS)
      - - **OpenAI** - GPT-4 for generation, text-embedding-3-small for embeddings
        - - **HuggingFace** - alternative local embedding model (sentence-transformers)
         
          - ## How it works
         
          - ```
            User uploads PDF
              -> PyPDFLoader extracts text
              -> RecursiveCharacterTextSplitter chunks it
              -> OpenAI embeddings computed for each chunk
              -> Stored in ChromaDB with metadata

            User asks a question
              -> Question embedded
              -> Top-k chunks retrieved from ChromaDB
              -> Retrieved context + question sent to GPT-4
              -> Answer returned with source page references
            ```

            ## Project layout

            ```
            rag-document-chatbot/
              app/
                main.py              # FastAPI app
                rag/
                  pipeline.py        # core RAG chain using LangChain
                  graph.py           # LangGraph agent for multi-turn
                  vectorstore.py     # ChromaDB wrapper
                  document_loader.py # PDF/text loading
                api/
                  routes.py          # REST endpoints
                models/
                  schemas.py         # Pydantic models
              requirements.txt
              docker-compose.yml
              .env.example
            ```

            ## Quick start

            ```bash
            git clone https://github.com/Venkatganga997/rag-document-chatbot.git
            cd rag-document-chatbot

            python -m venv venv
            source venv/bin/activate  # Windows: venv\Scripts\activate

            pip install -r requirements.txt

            cp .env.example .env
            # add OPENAI_API_KEY to .env

            uvicorn app.main:app --reload
            ```

            API docs at http://localhost:8000/docs

            ## API

            ```
            POST /upload          - upload a PDF
            POST /chat            - ask a question (returns answer + sources)
            GET  /documents       - list uploaded documents
            DELETE /documents/{id} - remove a document
            ```

            ## Example

            ```python
            # upload
            curl -X POST http://localhost:8000/upload -F "file=@report.pdf"

            # ask
            curl -X POST http://localhost:8000/chat \
              -H "Content-Type: application/json" \
              -d '{"question": "What was the Q3 revenue?", "doc_id": "abc123"}'

            # response
            {
              "answer": "Q3 revenue was $2.4B, up 12% YoY.",
              "sources": [{"page": 14, "text": "...Q3 revenue reached 2.4 billion..."}]
            }
            ```

            ## Notes

            - ChromaDB runs locally by default, Pinecone config available in config.py
            - - Multi-document mode lets you query across multiple uploaded files
              - - Conversation memory handled by LangGraph's state machine - supports follow-up questions
                - - Works well with dense technical PDFs, legal docs, financial reports
