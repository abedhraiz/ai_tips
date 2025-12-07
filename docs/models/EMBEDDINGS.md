# Embedding Models

## Overview

**Embedding Models** convert data (text, images, audio) into dense vector representations (embeddings) that capture semantic meaning. These vectors enable similarity search, clustering, and serve as input features for other ML models. Embeddings are fundamental to search engines, recommendation systems, and RAG.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Input** | Text, images, audio |
| **Output** | Dense vectors (128-4096 dims) |
| **Key Property** | Semantic similarity |
| **Training** | Contrastive learning |
| **Use Cases** | Search, RAG, clustering |

## How It Works

```
Embedding Models: Semantic Vector Space

Text: "The cat sat on the mat"
              │
              ▼
    ┌───────────────────┐
    │  Embedding Model  │
    │   (Transformer)   │
    └─────────┬─────────┘
              │
              ▼
    [0.23, -0.45, 0.12, ..., 0.67]
           768 dimensions

Similarity in Vector Space:

    "happy"  ●────────── Close ────────●  "joyful"
                         │
                         │ Far
                         │
                         ●  "sad"

    cosine_similarity("happy", "joyful") ≈ 0.92
    cosine_similarity("happy", "sad") ≈ 0.15
```

## Types of Embeddings

### Text Embeddings

| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| **text-embedding-3-small** | 1536 | 8191 | General, cost-effective |
| **text-embedding-3-large** | 3072 | 8191 | High accuracy |
| **voyage-large-2** | 1024 | 16K | Long documents |
| **E5-large-v2** | 1024 | 512 | Open-source |
| **BGE-large** | 1024 | 512 | Multilingual |
| **GTE-large** | 1024 | 8192 | Long context |
| **nomic-embed-text** | 768 | 8192 | Open-source |
| **mxbai-embed-large** | 1024 | 512 | High quality |

### Image Embeddings

| Model | Dimensions | Best For |
|-------|------------|----------|
| **CLIP ViT-L/14** | 768 | Image-text matching |
| **DINOv2** | 768-1536 | Visual similarity |
| **ImageBind** | 1024 | Multimodal |
| **SigLIP** | 1152 | Better CLIP |

### Multimodal Embeddings

| Model | Modalities | Dimensions |
|-------|------------|------------|
| **ImageBind** | 6 modalities | 1024 |
| **CLIP** | Image + Text | 768 |
| **CLAP** | Audio + Text | 512 |

## Popular Models

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Get embeddings
embedding = get_embedding("Hello world")
print(f"Dimensions: {len(embedding)}")  # 1536 for small, 3072 for large

# Batch embeddings
texts = ["First text", "Second text", "Third text"]
response = client.embeddings.create(input=texts, model="text-embedding-3-small")
embeddings = [item.embedding for item in response.data]
```

### Sentence Transformers (Open Source)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Single embedding
embedding = model.encode("Hello world")
print(f"Shape: {embedding.shape}")  # (384,)

# Batch embeddings
sentences = [
    "This is a sentence",
    "This is another sentence",
    "Completely different topic"
]
embeddings = model.encode(sentences)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print(similarities)
# [[1.0, 0.85, 0.23],
#  [0.85, 1.0, 0.21],
#  [0.23, 0.21, 1.0]]
```

### Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load model
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(texts):
    # For BGE models, add instruction for retrieval
    if isinstance(texts, str):
        texts = [texts]
    
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        output = model(**encoded)
    
    embeddings = mean_pooling(output, encoded['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()

embeddings = get_embedding(["This is a test sentence"])
```

### Voyage AI

```python
import voyageai

client = voyageai.Client()

# Get embeddings
result = client.embed(
    texts=["Sample text to embed"],
    model="voyage-large-2",
    input_type="document"  # or "query"
)

embedding = result.embeddings[0]
print(f"Dimensions: {len(embedding)}")  # 1024
```

### Cohere

```python
import cohere

co = cohere.Client('api-key')

# Get embeddings
response = co.embed(
    texts=["Hello world", "Another text"],
    model="embed-english-v3.0",
    input_type="search_document"  # or "search_query", "classification", "clustering"
)

embeddings = response.embeddings
```

## Use Cases

### 1. Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Index documents
documents = [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Neural networks process information like brains",
    "Data science combines statistics and programming"
]
doc_embeddings = model.encode(documents)

# Search
def search(query, top_k=3):
    query_embedding = model.encode(query)
    
    # Cosine similarity
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Top results
    top_indices = similarities.argsort()[::-1][:top_k]
    
    return [(documents[i], similarities[i]) for i in top_indices]

results = search("How do AI systems learn?")
for doc, score in results:
    print(f"{score:.3f}: {doc}")
# 0.782: Machine learning is a subset of AI
# 0.651: Neural networks process information like brains
# 0.423: Data science combines statistics and programming
```

### 2. RAG with Vector Database

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents
documents = [
    {"id": "1", "text": "Python was created by Guido van Rossum"},
    {"id": "2", "text": "JavaScript runs in web browsers"},
    {"id": "3", "text": "Rust focuses on memory safety"},
]

for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    collection.add(
        embeddings=[embedding],
        documents=[doc["text"]],
        ids=[doc["id"]]
    )

# Query
query = "Who created Python?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)
print(results['documents'])
# [['Python was created by Guido van Rossum', ...]]
```

### 3. Clustering

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "I love pizza", "Pasta is delicious",
    "Python is great", "JavaScript is popular",
    "Dogs are loyal", "Cats are independent"
]

embeddings = model.encode(texts)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

for text, label in zip(texts, labels):
    print(f"Cluster {label}: {text}")
# Cluster 0: I love pizza
# Cluster 0: Pasta is delicious
# Cluster 1: Python is great
# Cluster 1: JavaScript is popular
# Cluster 2: Dogs are loyal
# Cluster 2: Cats are independent
```

### 4. Semantic Deduplication

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "The weather is nice today",
    "It's a beautiful day outside",
    "Python programming is fun",
    "Coding in Python is enjoyable",
    "I like coffee"
]

embeddings = model.encode(texts)

def find_duplicates(embeddings, threshold=0.8):
    similarity_matrix = cosine_similarity(embeddings)
    duplicates = []
    
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                duplicates.append((i, j, similarity_matrix[i][j]))
    
    return duplicates

duplicates = find_duplicates(embeddings)
for i, j, score in duplicates:
    print(f"Similar ({score:.2f}): '{texts[i]}' ~ '{texts[j]}'")
# Similar (0.89): 'The weather is nice today' ~ 'It's a beautiful day outside'
# Similar (0.85): 'Python programming is fun' ~ 'Coding in Python is enjoyable'
```

### 5. Classification with Embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model = SentenceTransformer('all-MiniLM-L6-v2')

# Training data
texts = [
    "Great product!", "Terrible experience", "Loved it",
    "Waste of money", "Highly recommend", "Disappointing",
    "Amazing quality", "Poor customer service"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Get embeddings
embeddings = model.encode(texts)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.25, random_state=42
)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict new text
new_text = "This product exceeded my expectations"
new_embedding = model.encode([new_text])
prediction = classifier.predict(new_embedding)
print(f"Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

## Benchmarks (MTEB)

### Text Retrieval

| Model | NDCG@10 | Speed | Cost |
|-------|---------|-------|------|
| **voyage-large-2** | 0.543 | Medium | Paid |
| **text-embedding-3-large** | 0.540 | Fast | Paid |
| **E5-Mistral-7B** | 0.536 | Slow | Free |
| **BGE-large-en-v1.5** | 0.519 | Fast | Free |
| **text-embedding-3-small** | 0.510 | Fast | Paid |
| **all-MiniLM-L6-v2** | 0.419 | Very Fast | Free |

### Embedding Dimensions vs Quality

| Dimensions | Quality | Storage | Speed |
|------------|---------|---------|-------|
| 384 | Good | Low | Fast |
| 768 | Better | Medium | Medium |
| 1024 | High | Higher | Slower |
| 1536 | Very High | High | Slow |
| 3072 | Highest | Very High | Slowest |

## Choosing Embeddings

```
What's your use case?
│
├── High accuracy, budget available
│   └── text-embedding-3-large or voyage-large-2
│
├── Cost-effective production
│   └── text-embedding-3-small or voyage-2
│
├── Open-source required
│   ├── Best quality → E5-large-v2, BGE-large
│   ├── Fast inference → all-MiniLM-L6-v2
│   └── Long context → GTE-large, nomic-embed-text
│
├── Multimodal
│   └── CLIP, ImageBind
│
└── Multilingual
    └── multilingual-e5-large, BGE-M3
```

## Dimensionality Reduction

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# OpenAI supports native dimensionality reduction
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Sample text",
    model="text-embedding-3-small",
    dimensions=256  # Reduce from 1536 to 256
)

# For other models, use PCA or Matryoshka
from sklearn.decomposition import PCA

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)  # (n, 384)

# Reduce dimensions
pca = PCA(n_components=128)
reduced = pca.fit_transform(embeddings)  # (n, 128)
```

## Best Practices

### 1. Query vs Document Embeddings

```python
# Some models benefit from different prefixes

# For E5 models
query = "query: How to learn Python?"
document = "passage: Python can be learned through online courses..."

# For BGE models
query = "Represent this sentence for searching relevant passages: How to learn Python?"
document = "Python can be learned through online courses..."
```

### 2. Batching for Efficiency

```python
# Process in batches
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def batch_encode(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)
```

### 3. Normalize Embeddings

```python
import numpy as np

def normalize(embeddings):
    """Normalize for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Most similarity searches expect normalized vectors
embeddings = normalize(model.encode(texts))
```

## Vector Databases

| Database | Type | Scaling | Best For |
|----------|------|---------|----------|
| **Pinecone** | Managed | Auto | Production |
| **Weaviate** | Self/Managed | Good | Hybrid search |
| **Milvus** | Self-hosted | Excellent | Large scale |
| **Qdrant** | Self/Managed | Good | Filtering |
| **ChromaDB** | Self-hosted | Limited | Prototyping |
| **FAISS** | Library | Excellent | Local/Research |
| **pgvector** | PostgreSQL | Good | Existing PG users |

## Related Models

- **[RAG](./RAG.md)** - Uses embeddings for retrieval
- **[CLIP](./CLIP.md)** - Image-text embeddings
- **[LLM](./LLM.md)** - Generate text from retrieved context
- **[MLM](./MLM.md)** - Source of text embeddings

## Resources

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Voyage AI](https://www.voyageai.com/)
- [BGE Models](https://huggingface.co/BAAI/bge-large-en-v1.5)
