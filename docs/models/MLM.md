# MLM - Masked Language Model

## Overview

**Masked Language Models** are bidirectional transformers trained to predict masked (hidden) tokens in a sequence. Unlike autoregressive LLMs that only see previous context, MLMs can attend to both left and right context, making them excellent for understanding tasks.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Size** | 100M to 1B+ parameters |
| **Training** | Masked token prediction (15% typically masked) |
| **Architecture** | Transformer encoder |
| **Context** | 512 to 8K tokens |
| **Modality** | Text |
| **Direction** | Bidirectional |

## How It Works

```
Original:     "The cat [MASK] on the mat"
                       ↓
              Transformer Encoder
              (attends to all tokens)
                       ↓
Prediction:   "The cat sat on the mat"
```

**Training Process:**
1. Randomly mask 15% of input tokens
2. 80% replaced with [MASK], 10% random token, 10% unchanged
3. Model predicts original tokens using full context
4. Loss computed only on masked positions

## Architecture Comparison

```
LLM (Decoder-only):    [Token1] → [Token2] → [Token3] → ...
                       Can only see: ←←←←←←←

MLM (Encoder-only):    [Token1] ↔ [Token2] ↔ [Token3] ↔ ...
                       Can see:    ←←←←←→→→→→→
```

## Popular Models

| Model | Parameters | Context | Best For |
|-------|-----------|---------|----------|
| **BERT-base** | 110M | 512 | General NLU baseline |
| **BERT-large** | 340M | 512 | Higher accuracy tasks |
| **RoBERTa** | 125M-355M | 512 | Improved BERT training |
| **DeBERTa** | 100M-1.5B | 512-24K | SOTA on many benchmarks |
| **ALBERT** | 12M-235M | 512 | Parameter-efficient |
| **DistilBERT** | 66M | 512 | Fast inference, 60% smaller |
| **ELECTRA** | 14M-335M | 512 | Efficient pre-training |
| **XLM-RoBERTa** | 270M-3.5B | 512 | 100+ languages |
| **ModernBERT** | 140M-395M | 8192 | 2024 state-of-the-art |

## Examples with Input/Output

### Example 1: Fill-Mask Task

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

result = fill_mask("The capital of France is [MASK].")
# Output: [{'token_str': 'paris', 'score': 0.95}, ...]
```

### Example 2: Named Entity Recognition (NER)

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

text = "Apple CEO Tim Cook announced the new iPhone in California."
entities = ner(text)

# Output:
# [{'entity_group': 'ORG', 'word': 'Apple', 'score': 0.99},
#  {'entity_group': 'PER', 'word': 'Tim Cook', 'score': 0.99},
#  {'entity_group': 'MISC', 'word': 'iPhone', 'score': 0.92},
#  {'entity_group': 'LOC', 'word': 'California', 'score': 0.99}]
```

### Example 3: Text Classification

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("This movie was absolutely fantastic!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Example 4: Semantic Similarity

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "The stock market crashed today"
]

embeddings = model.encode(sentences)
similarity = cosine_similarity(embeddings)

# Sentences 0 and 1: 0.82 (similar)
# Sentences 0 and 2: 0.12 (dissimilar)
```

### Example 5: Question Answering (Extractive)

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.
The tower is 330 metres tall and was the tallest man-made structure for 41 years.
"""

question = "How tall is the Eiffel Tower?"
result = qa(question=question, context=context)

# Output: {'answer': '330 metres', 'score': 0.95, 'start': 185, 'end': 196}
```

## Use Cases

| Task | Best Models | Accuracy |
|------|-------------|----------|
| **Text Classification** | DeBERTa, RoBERTa | 95-98% |
| **Named Entity Recognition** | BERT-NER, SpaCy-transformers | 92-97% |
| **Sentiment Analysis** | DistilBERT, RoBERTa | 94-96% |
| **Question Answering** | DeBERTa, ALBERT | 88-93% F1 |
| **Semantic Similarity** | Sentence-BERT, MiniLM | 85-92% |
| **Token Classification** | BERT-base, XLM-R | 90-96% |

## MLM vs LLM: When to Use Which

| Aspect | MLM (BERT-style) | LLM (GPT-style) |
|--------|------------------|-----------------|
| **Architecture** | Encoder-only | Decoder-only |
| **Context** | Bidirectional | Left-to-right only |
| **Best for** | Understanding tasks | Generation tasks |
| **Speed** | Faster inference | Slower (autoregressive) |
| **Size** | 100M-1B typical | 1B-405B typical |
| **Classification** | ✅ Excellent | Good |
| **NER** | ✅ Excellent | Good |
| **Text Generation** | ❌ Poor | ✅ Excellent |
| **Summarization** | ❌ Poor | ✅ Excellent |
| **Fine-tuning cost** | Low | High |

## Fine-Tuning Example

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset("imdb")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()
```

## Performance Benchmarks

### GLUE Benchmark Scores

| Model | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
|-------|------|-----|------|-------|------|-------|------|-----|-----|
| BERT-base | 84.6 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.1 |
| RoBERTa-large | 90.2 | 72.1 | 94.7 | 96.4 | 68.0 | 92.4 | 90.9 | 86.6 | 86.4 |
| DeBERTa-v3-large | 91.8 | 72.8 | 96.0 | 97.0 | 75.3 | 93.0 | 92.0 | 92.7 | 88.8 |

## Memory Requirements

| Model | Parameters | RAM (FP32) | RAM (FP16) | RAM (INT8) |
|-------|-----------|-----------|-----------|-----------|
| DistilBERT | 66M | 264 MB | 132 MB | 66 MB |
| BERT-base | 110M | 440 MB | 220 MB | 110 MB |
| BERT-large | 340M | 1.4 GB | 680 MB | 340 MB |
| DeBERTa-v3-large | 435M | 1.7 GB | 870 MB | 435 MB |
| XLM-R-XL | 3.5B | 14 GB | 7 GB | 3.5 GB |

## Hugging Face Quick Reference

```python
# Classification
from transformers import pipeline

# Sentiment
sentiment = pipeline("sentiment-analysis")

# NER
ner = pipeline("ner", aggregation_strategy="simple")

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
classifier("I love playing tennis", candidate_labels=["sports", "cooking", "music"])

# Feature extraction (embeddings)
extractor = pipeline("feature-extraction", model="bert-base-uncased")
```

## Choosing the Right MLM

```
Need multilingual support?
├── Yes → XLM-RoBERTa
└── No
    ├── Need long context (>512)?
    │   ├── Yes → Longformer, BigBird, ModernBERT
    │   └── No
    │       ├── Need fastest inference?
    │       │   ├── Yes → DistilBERT, MiniLM
    │       │   └── No
    │       │       ├── Need best accuracy?
    │       │       │   ├── Yes → DeBERTa-v3
    │       │       │   └── No → RoBERTa-base
    │       │       └── Need parameter efficiency?
    │       │           └── Yes → ALBERT
    │       └── Need domain-specific?
    │           ├── Scientific → SciBERT
    │           ├── Biomedical → BioBERT, PubMedBERT
    │           ├── Legal → LegalBERT
    │           ├── Financial → FinBERT
    │           └── Code → CodeBERT
```

## Common Mistakes to Avoid

1. **Using MLM for generation** - MLMs are not designed for text generation
2. **Ignoring context limits** - BERT truncates at 512 tokens
3. **Wrong tokenizer** - Always use the matching tokenizer
4. **Freezing wrong layers** - For fine-tuning, freeze early layers first
5. **Batch size too large** - Can cause memory issues with larger models

## Related Models

- **[LLM](./LLM.md)** - For text generation tasks
- **[Encoder-Decoder](./ENCODER_DECODER.md)** - For seq2seq tasks (translation, summarization)
- **[Embedding Models](./EMBEDDING.md)** - Specialized for semantic similarity

## Resources

- [Hugging Face BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [DeBERTa Paper](https://arxiv.org/abs/2006.03654)
