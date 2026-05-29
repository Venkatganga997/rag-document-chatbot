from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.
If the answer is not in the context, say "I don't have enough information in the document to answer that."
Always cite the source page numbers in your answer.

Context:
{summaries}

Question: {question}

Answer:"""


class RAGPipeline:
      def __init__(self, persist_dir: str = "./chroma_db"):
                self.persist_dir = persist_dir
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                self.vectorstores: dict[str, Chroma] = {}

      def ingest_pdf(self, file_path: str, doc_id: str) -> int:
                """Load a PDF, chunk it, embed and store in ChromaDB. Returns chunk count."""
                logger.info(f"Ingesting document {doc_id} from {file_path}")

          loader = PyPDFLoader(file_path)
        pages = loader.load()

        # add doc_id to metadata so we can filter later
        for page in pages:
                      page.metadata["doc_id"] = doc_id

        chunks = self.text_splitter.split_documents(pages)
        logger.info(f"Split into {len(chunks)} chunks")

        collection_name = f"doc_{doc_id}"
        vectorstore = Chroma.from_documents(
                      documents=chunks,
                      embedding=self.embeddings,
                      collection_name=collection_name,
                      persist_directory=self.persist_dir
        )
        self.vectorstores[doc_id] = vectorstore
        logger.info(f"Stored {len(chunks)} chunks for doc {doc_id}")
        return len(chunks)

    def get_vectorstore(self, doc_id: str) -> Optional[Chroma]:
              if doc_id not in self.vectorstores:
                            # try loading from disk
                            collection_name = f"doc_{doc_id}"
                            try:
                                              vs = Chroma(
                                                                    collection_name=collection_name,
                                                                    embedding_function=self.embeddings,
                                                                    persist_directory=self.persist_dir
                                              )
                                              self.vectorstores[doc_id] = vs
except Exception:
                return None
        return self.vectorstores.get(doc_id)

    def chat(self, question: str, doc_id: str) -> dict:
              vectorstore = self.get_vectorstore(doc_id)
              if not vectorstore:
                            raise ValueError(f"Document {doc_id} not found. Upload it first.")

              retriever = vectorstore.as_retriever(
                  search_type="similarity",
                  search_kwargs={"k": 5}
              )

        prompt = PromptTemplate(
                      template=PROMPT_TEMPLATE,
                      input_variables=["summaries", "question"]
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
                      llm=self.llm,
                      retriever=retriever,
                      chain_type_kwargs={"prompt": prompt},
                      return_source_documents=True
        )

        result = chain.invoke({"question": question})

        # extract source pages
        sources = []
        for doc in result.get("source_documents", []):
                      page = doc.metadata.get("page", "unknown")
                      sources.append({
                          "page": page,
                          "text": doc.page_content[:200]
                      })

        return {
                      "answer": result["answer"],
                      "sources": sources
        }

    def delete_document(self, doc_id: str) -> bool:
              vectorstore = self.get_vectorstore(doc_id)
              if not vectorstore:
                            return False
                        vectorstore.delete_collection()
        self.vectorstores.pop(doc_id, None)
        return True
