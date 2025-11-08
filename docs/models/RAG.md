# Retrieval-Augmented Generation (RAG) Models

## Overview

**Retrieval-Augmented Generation (RAG)** is a hybrid AI architecture that combines large language models with external knowledge retrieval systems. This approach grounds AI responses in factual, up-to-date information while maintaining the fluency and reasoning capabilities of language models.

## Core Concept

Traditional LLMs are limited by their training data cutoff and can hallucinate facts. RAG solves this by:
1. **Retrieving** relevant documents from external knowledge bases
2. **Augmenting** the prompt with retrieved context
3. **Generating** responses grounded in retrieved information

```
User Query
    ↓
Retrieval System → Knowledge Base (documents, databases, APIs)
    ↓
Retrieved Context (top-K relevant documents)
    ↓
LLM + Retrieved Context
    ↓
Grounded, Factual Response
```

## Architecture Components

### 1. Knowledge Base
- **Document Store**: Vector databases (ChromaDB, Pinecone, Weaviate)
- **Embeddings**: Dense vector representations
- **Indexing**: Efficient similarity search (FAISS, HNSW)

### 2. Retrieval System
- **Query Encoder**: Converts queries to embeddings
- **Similarity Search**: Finds relevant documents
- **Re-ranking**: Orders results by relevance

### 3. Generation System
- **LLM**: GPT-4, Claude, Llama-2, Mistral
- **Prompt Engineering**: Context integration
- **Response Synthesis**: Combines sources

## Implementation

### Basic RAG System

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

class BasicRAG:
    def __init__(self, documents_path: str):
        # Load documents
        loader = TextLoader(documents_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
        )
    
    def query(self, question: str) -> str:
        """Ask a question and get grounded response."""
        return self.qa_chain.run(question)

# Usage
rag = BasicRAG("company_docs.txt")
answer = rag.query("What is our refund policy?")
print(answer)
```

### Advanced RAG with Re-ranking

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict
import numpy as np

class AdvancedRAG:
    def __init__(
        self,
        vector_db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        # Embedding model for retrieval
        self.embedder = SentenceTransformer(embedding_model)
        
        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder(reranker_model)
        
        # Vector store
        self.vectorstore = self._load_vectorstore(vector_db_path)
        
        # LLM
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def _load_vectorstore(self, path: str):
        """Load or create vector store."""
        embeddings = OpenAIEmbeddings()
        return Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )
    
    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 3
    ) -> List[Dict]:
        """Retrieve documents and re-rank for better relevance."""
        
        # Initial retrieval (cast wide net)
        retrieved_docs = self.vectorstore.similarity_search(
            query,
            k=top_k
        )
        
        # Prepare pairs for re-ranking
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        
        # Re-rank with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        doc_score_pairs = list(zip(retrieved_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K after re-ranking
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in doc_score_pairs[:rerank_top_k]
        ]
    
    def query(self, question: str) -> Dict:
        """Query with retrieval and generation."""
        
        # Retrieve and re-rank
        relevant_docs = self.retrieve_and_rerank(question)
        
        # Build context from top documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Generate response with context
        prompt = f"""Answer the question based on the provided context. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.predict(prompt)
        
        return {
            "answer": response,
            "sources": relevant_docs
        }

# Usage
rag = AdvancedRAG("./vector_db")
result = rag.query("How do I reset my password?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")
```

### Production RAG with Monitoring

```python
import chromadb
from chromadb.config import Settings
import anthropic
from dataclasses import dataclass
from typing import List, Optional
import logging
import time

@dataclass
class RAGMetrics:
    """Track RAG system performance."""
    query_time: float
    retrieval_time: float
    generation_time: float
    num_retrieved: int
    relevance_scores: List[float]

class ProductionRAG:
    def __init__(
        self,
        collection_name: str,
        anthropic_api_key: str
    ):
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Claude
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to knowledge base."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"Added {len(documents)} documents to collection")
    
    def query_with_metrics(
        self,
        query: str,
        top_k: int = 5
    ) -> tuple[Dict, RAGMetrics]:
        """Query with detailed metrics."""
        start_time = time.time()
        
        # Retrieval phase
        retrieval_start = time.time()
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        retrieval_time = time.time() - retrieval_start
        
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarity scores
        relevance_scores = [1 - d for d in distances]
        
        # Generation phase
        generation_start = time.time()
        context = "\n\n".join([
            f"[Document {i+1}] {doc}"
            for i, doc in enumerate(documents)
        ])
        
        message = self.claude.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Based on the following context, answer the question. If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}"""
            }]
        )
        
        answer = message.content[0].text
        generation_time = time.time() - generation_start
        
        # Calculate metrics
        total_time = time.time() - start_time
        metrics = RAGMetrics(
            query_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_retrieved=len(documents),
            relevance_scores=relevance_scores
        )
        
        self.logger.info(f"Query completed in {total_time:.2f}s")
        
        result = {
            "answer": answer,
            "sources": [
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance": score
                }
                for doc, meta, score in zip(documents, metadatas, relevance_scores)
            ]
        }
        
        return result, metrics

# Usage
rag = ProductionRAG(
    collection_name="company_knowledge",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Add documents
rag.add_documents(
    documents=["Company policy doc 1...", "Product info doc 2..."],
    metadatas=[{"type": "policy"}, {"type": "product"}],
    ids=["policy_1", "product_1"]
)

# Query with metrics
result, metrics = rag.query_with_metrics("What is the vacation policy?")
print(f"Answer: {result['answer']}")
print(f"Retrieval: {metrics.retrieval_time:.2f}s, Generation: {metrics.generation_time:.2f}s")
```

## Advanced Patterns

### 1. Hybrid Search (Dense + Sparse)

```python
from rank_bm25 import BM25Okapi

class HybridRAG:
    """Combine semantic search with keyword search."""
    def __init__(self):
        # Dense retrieval (semantic)
        self.vectorstore = Chroma(...)
        
        # Sparse retrieval (keyword)
        self.bm25 = None
        self.documents = []
    
    def index_documents(self, documents: List[str]):
        """Index for both dense and sparse retrieval."""
        # Vector store
        self.vectorstore.add_documents(documents)
        
        # BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        """Combine dense and sparse retrieval."""
        # Dense retrieval scores
        dense_results = self.vectorstore.similarity_search_with_score(
            query, k=top_k*2
        )
        dense_scores = {doc.page_content: 1-score for doc, score in dense_results}
        
        # Sparse retrieval scores
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores (weighted sum)
        combined_scores = {}
        for doc, dense_score in dense_scores.items():
            idx = self.documents.index(doc)
            sparse_score = sparse_scores[idx]
            combined_scores[doc] = 0.7 * dense_score + 0.3 * sparse_score
        
        # Return top K
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]
```

### 2. Multi-hop RAG

```python
class MultiHopRAG:
    """Perform multiple retrieval steps for complex queries."""
    def __init__(self):
        self.rag = AdvancedRAG("./vector_db")
        self.llm = ChatOpenAI(model="gpt-4")
    
    def multi_hop_query(self, query: str, max_hops: int = 3) -> Dict:
        """Answer complex questions requiring multiple retrieval steps."""
        context = []
        current_query = query
        
        for hop in range(max_hops):
            # Retrieve for current query
            docs = self.rag.retrieve_and_rerank(current_query, top_k=3)
            context.extend(docs)
            
            # Check if we have enough information
            check_prompt = f"""Given this context, can you answer: "{query}"?
            
Context:
{self._format_docs(context)}

Respond with YES or NO, then explain what's missing if NO."""

            check_response = self.llm.predict(check_prompt)
            
            if "YES" in check_response:
                break
            
            # Generate follow-up query
            followup_prompt = f"""Based on the context and the original question, what should we search for next?

Original question: {query}
Current context: {self._format_docs(context)}

Generate a focused search query:"""
            
            current_query = self.llm.predict(followup_prompt)
        
        # Final answer generation
        answer_prompt = f"""Answer the question using all provided context.

Question: {query}

Context from multiple sources:
{self._format_docs(context)}

Answer:"""
        
        answer = self.llm.predict(answer_prompt)
        
        return {
            "answer": answer,
            "sources": context,
            "hops": hop + 1
        }
```

### 3. Self-Querying RAG

```python
class SelfQueryingRAG:
    """Generate optimal queries automatically."""
    def __init__(self):
        self.rag = AdvancedRAG("./vector_db")
        self.llm = ChatOpenAI(model="gpt-4")
    
    def generate_queries(self, user_question: str) -> List[str]:
        """Generate multiple search queries from user question."""
        prompt = f"""Generate 3 different search queries to find information needed to answer this question:

Question: {user_question}

Provide 3 focused, specific search queries:
1."""
        
        response = self.llm.predict(prompt)
        queries = [line.split(". ", 1)[1] for line in response.split("\n") if line.strip()]
        return queries[:3]
    
    def query(self, user_question: str) -> Dict:
        """Answer using multiple generated queries."""
        # Generate optimal queries
        search_queries = self.generate_queries(user_question)
        
        # Retrieve for each query
        all_docs = []
        for query in search_queries:
            docs = self.rag.retrieve_and_rerank(query, top_k=2)
            all_docs.extend(docs)
        
        # Deduplicate and rank
        unique_docs = self._deduplicate(all_docs)
        
        # Generate final answer
        context = "\n\n".join([doc['content'] for doc in unique_docs])
        answer = self.rag.llm.predict(f"""Answer: {user_question}

Context:
{context}

Answer:""")
        
        return {
            "answer": answer,
            "queries_used": search_queries,
            "sources": unique_docs
        }
```

## Best Practices

### 1. Document Chunking Strategy

```python
def smart_chunk_documents(
    documents: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """Intelligently chunk documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc)
        chunks.extend(doc_chunks)
    
    return chunks
```

### 2. Embedding Model Selection

| Use Case | Model | Dimension | Speed |
|----------|-------|-----------|-------|
| General | all-MiniLM-L6-v2 | 384 | Fast |
| High Quality | all-mpnet-base-v2 | 768 | Medium |
| Multilingual | paraphrase-multilingual | 768 | Medium |
| Code | code-search-net | 768 | Fast |
| OpenAI | text-embedding-3-small | 1536 | API |

### 3. Evaluation Metrics

```python
class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth_docs: List[List[str]],
        retrieved_docs: List[List[str]]
    ) -> Dict:
        """Evaluate retrieval quality."""
        precisions = []
        recalls = []
        
        for truth, retrieved in zip(ground_truth_docs, retrieved_docs):
            truth_set = set(truth)
            retrieved_set = set(retrieved)
            
            if len(retrieved_set) > 0:
                precision = len(truth_set & retrieved_set) / len(retrieved_set)
                precisions.append(precision)
            
            if len(truth_set) > 0:
                recall = len(truth_set & retrieved_set) / len(truth_set)
                recalls.append(recall)
        
        return {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls))
        }
```

## Performance Optimization

### 1. Caching

```python
from functools import lru_cache
import hashlib

class CachedRAG:
    def __init__(self):
        self.rag = AdvancedRAG("./vector_db")
        self.cache = {}
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def query(self, question: str) -> Dict:
        # Check cache
        cache_key = self._hash_query(question)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Query RAG
        result = self.rag.query(question)
        
        # Cache result
        self.cache[cache_key] = result
        return result
```

### 2. Batch Processing

```python
def batch_query_rag(
    rag: BasicRAG,
    queries: List[str],
    batch_size: int = 10
) -> List[str]:
    """Process multiple queries efficiently."""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        # Parallel retrieval
        batch_results = [rag.query(q) for q in batch]
        results.extend(batch_results)
    
    return results
```

## Limitations and Solutions

| Limitation | Impact | Solution |
|------------|--------|----------|
| Context Length | Can't use all docs | Re-ranking, summarization |
| Retrieval Errors | Wrong docs retrieved | Hybrid search, multi-query |
| Outdated Info | Stale knowledge | Regular updates, timestamps |
| Hallucination | Still possible | Citation, confidence scores |
| Latency | Slower than pure LLM | Caching, optimization |

## Conclusion

RAG provides:
- **Factual Grounding**: Reduces hallucinations
- **Up-to-date Info**: Access to current knowledge
- **Transparency**: Traceable sources
- **Cost Efficiency**: Smaller models with retrieval
- **Domain Adaptation**: Easy knowledge updates

Essential for production AI systems requiring accuracy, traceability, and current information.

## Additional Resources

- **LangChain RAG Tutorial**: Complete implementation guide
- **LlamaIndex**: Advanced RAG framework
- **Vector Databases**: ChromaDB, Pinecone, Weaviate comparison
- **Evaluation**: RAGAS framework for RAG assessment

For implementation examples, see `examples/use-cases/` (customer service, business intelligence systems use RAG patterns).
