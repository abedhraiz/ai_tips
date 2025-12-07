# Encoder-Decoder Models (Seq2Seq)

## Overview

**Encoder-Decoder Models** combine both transformer encoder and decoder components to handle sequence-to-sequence tasks. The encoder processes the input, and the decoder generates the output, making them ideal for translation, summarization, and other transformation tasks.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Size** | 60M to 13B+ parameters |
| **Training** | Denoising, span corruption, translation |
| **Architecture** | Full Transformer (encoder + decoder) |
| **Context** | 512 to 16K tokens |
| **Modality** | Text (primarily) |
| **Direction** | Encoder: bidirectional, Decoder: autoregressive |

## How It Works

```
Input: "Translate to French: Hello, how are you?"
              ↓
┌─────────────────────────┐
│     ENCODER             │
│  (Bidirectional)        │
│  Processes full input   │
└───────────┬─────────────┘
            ↓
    [Encoded Representations]
            ↓
┌─────────────────────────┐
│     DECODER             │
│  (Autoregressive)       │
│  Cross-attends encoder  │
│  Generates output       │
└───────────┬─────────────┘
            ↓
Output: "Bonjour, comment allez-vous?"
```

## Architecture Deep Dive

```
                    ENCODER                              DECODER
              ┌─────────────────┐                 ┌─────────────────┐
              │ Self-Attention  │                 │ Masked Self-Attn│
              │ (Bidirectional) │                 │ (Causal)        │
              └────────┬────────┘                 └────────┬────────┘
                       ↓                                   ↓
              ┌─────────────────┐                 ┌─────────────────┐
              │ Feed Forward    │                 │ Cross-Attention │
              └────────┬────────┘                 │ (to encoder)    │
                       ↓                          └────────┬────────┘
              [Encoder Output] ──────────────────────────→ ↓
                                                  ┌─────────────────┐
                                                  │ Feed Forward    │
                                                  └────────┬────────┘
                                                           ↓
                                                    [Output Tokens]
```

## Popular Models

### T5 Family

| Model | Parameters | Best For |
|-------|-----------|----------|
| **T5-small** | 60M | Fast experimentation |
| **T5-base** | 220M | General tasks |
| **T5-large** | 770M | Higher quality |
| **T5-3B** | 3B | Best quality |
| **T5-11B** | 11B | Maximum performance |
| **Flan-T5** | 80M-11B | Instruction-tuned |
| **mT5** | 300M-13B | Multilingual (101 languages) |
| **LongT5** | 220M-3B | Long documents (16K tokens) |

### BART Family

| Model | Parameters | Best For |
|-------|-----------|----------|
| **BART-base** | 140M | General seq2seq |
| **BART-large** | 400M | Summarization |
| **mBART** | 610M | Multilingual translation |
| **BART-large-CNN** | 400M | News summarization |
| **BART-large-XSUM** | 400M | Abstractive summarization |

### Other Notable Models

| Model | Parameters | Specialty |
|-------|-----------|-----------|
| **PEGASUS** | 568M | Abstractive summarization |
| **MarianMT** | 74M-298M | Translation (1000+ pairs) |
| **NLLB** | 600M-54B | 200 languages translation |
| **LED (Longformer)** | 162M-460M | Long document (16K tokens) |

## Examples with Input/Output

### Example 1: Translation

```python
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="t5-base")

text = "Machine learning is transforming how we build software."
result = translator(text)

# Output: "L'apprentissage automatique transforme la façon dont nous construisons des logiciels."
```

### Example 2: Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Amazon rainforest, often referred to as the "lungs of the Earth," 
produces about 20% of the world's oxygen. It spans across nine countries 
in South America and is home to approximately 10% of all species on Earth.
The rainforest plays a crucial role in regulating the global climate by 
absorbing carbon dioxide. However, deforestation has accelerated in recent
decades, with an area the size of a football field being cleared every minute.
Scientists warn that continued destruction could push the ecosystem past a 
tipping point, transforming it from a carbon sink into a carbon source.
"""

summary = summarizer(article, max_length=60, min_length=20)

# Output: "The Amazon rainforest produces about 20% of the world's oxygen 
# and is home to 10% of all species on Earth. Deforestation has accelerated,
# with scientists warning of a potential tipping point."
```

### Example 3: Text-to-Text with T5

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# T5 treats everything as text-to-text
tasks = [
    "translate English to German: How are you?",
    "summarize: The quick brown fox jumps over the lazy dog multiple times.",
    "question: What is 2 + 2?",
    "sentiment: This product exceeded my expectations!",
]

for task in tasks:
    inputs = tokenizer(task, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(f"Input: {task}")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

# Outputs:
# "Wie geht es Ihnen?"
# "The quick brown fox jumps over the lazy dog."
# "4"
# "positive"
```

### Example 4: Grammar Correction

```python
from transformers import pipeline

corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

text = "I goes to the store yesterday and buyed some apple."
result = corrector(text)

# Output: "I went to the store yesterday and bought some apples."
```

### Example 5: Paraphrasing

```python
from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")

text = "The weather is extremely hot today, making it difficult to go outside."
results = paraphraser(text, num_return_sequences=3, num_beams=5)

# Outputs:
# "It's too hot to go outside today."
# "Today's scorching weather makes outdoor activities challenging."
# "The extreme heat today is keeping people indoors."
```

## Use Cases Comparison

| Task | Best Model | Quality | Speed |
|------|-----------|---------|-------|
| **Translation (En↔Fr,De,Es)** | MarianMT, NLLB | ★★★★★ | Fast |
| **Translation (Low-resource)** | NLLB, mBART | ★★★★☆ | Medium |
| **News Summarization** | BART-large-CNN | ★★★★★ | Medium |
| **Abstractive Summary** | PEGASUS, BART | ★★★★★ | Medium |
| **Long Doc Summary** | LED, LongT5 | ★★★★☆ | Slow |
| **Grammar Correction** | T5, GECToR | ★★★★☆ | Fast |
| **Paraphrasing** | PEGASUS, T5 | ★★★★☆ | Medium |
| **Instruction Following** | Flan-T5 | ★★★★★ | Medium |

## Encoder-Decoder vs Other Architectures

| Aspect | Encoder-Decoder | Encoder-Only (BERT) | Decoder-Only (GPT) |
|--------|----------------|---------------------|-------------------|
| **Translation** | ✅ Excellent | ❌ Poor | ★★★ Good |
| **Summarization** | ✅ Excellent | ❌ Poor | ★★★★ Very Good |
| **Classification** | ★★★ Good | ✅ Excellent | ★★★ Good |
| **Generation** | ★★★★ Very Good | ❌ Poor | ✅ Excellent |
| **QA (Extractive)** | ★★★ Good | ✅ Excellent | ★★★ Good |
| **QA (Generative)** | ✅ Excellent | ❌ Poor | ✅ Excellent |
| **Efficiency** | ★★★ Medium | ★★★★★ Fast | ★★★ Medium |

## Fine-Tuning Example

```python
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# Load model
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load summarization dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

def preprocess(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=150, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    fp16=True,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()
```

## Memory Requirements

| Model | Parameters | RAM (FP32) | RAM (FP16) | RAM (INT8) |
|-------|-----------|-----------|-----------|-----------|
| T5-small | 60M | 240 MB | 120 MB | 60 MB |
| T5-base | 220M | 880 MB | 440 MB | 220 MB |
| BART-base | 140M | 560 MB | 280 MB | 140 MB |
| BART-large | 400M | 1.6 GB | 800 MB | 400 MB |
| T5-large | 770M | 3.1 GB | 1.5 GB | 770 MB |
| Flan-T5-XL | 3B | 12 GB | 6 GB | 3 GB |
| T5-11B | 11B | 44 GB | 22 GB | 11 GB |

## Choosing the Right Model

```
What's your primary task?
├── Translation
│   ├── High-resource languages (En, Fr, De, Es, Zh) → MarianMT
│   ├── Low-resource/many languages → NLLB
│   └── Multilingual → mT5, mBART
│
├── Summarization
│   ├── News articles → BART-large-CNN
│   ├── Short abstractive → PEGASUS
│   ├── Long documents (>4K tokens) → LED, LongT5
│   └── General → Flan-T5
│
├── Text Transformation
│   ├── Grammar correction → T5-grammar
│   ├── Paraphrasing → PEGASUS-paraphrase
│   ├── Style transfer → T5
│   └── Data augmentation → BART
│
└── General Seq2Seq
    ├── Best quality → Flan-T5-XL
    ├── Balanced → Flan-T5-base
    └── Fast/light → T5-small
```

## Performance Benchmarks

### Summarization (ROUGE Scores on CNN/DailyMail)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| BART-large | 44.16 | 21.28 | 40.90 |
| PEGASUS | 44.17 | 21.47 | 41.11 |
| T5-large | 43.52 | 21.55 | 40.69 |
| LED-large | 44.40 | 21.50 | 40.80 |

### Translation (BLEU on WMT)

| Model | En→De | En→Fr | En→Zh |
|-------|-------|-------|-------|
| MarianMT | 41.2 | 43.8 | 35.5 |
| mBART | 38.5 | 41.2 | 33.8 |
| NLLB-200 | 42.1 | 44.5 | 37.2 |

## Common Patterns

### Prefix-based Task Specification (T5 Style)

```python
# T5 uses task prefixes
prefixes = {
    "translate": "translate English to French: ",
    "summarize": "summarize: ",
    "question": "question: ",
    "cola": "cola sentence: ",  # Grammaticality
}

text = "This is an example sentence."
input_text = prefixes["summarize"] + text
```

### Multi-task with Single Model

```python
from transformers import pipeline

# Flan-T5 handles multiple tasks without prefixes
flan = pipeline("text2text-generation", model="google/flan-t5-large")

# Different tasks, same model
flan("Translate to Spanish: Hello world")  # "Hola mundo"
flan("Summarize: [long text]")  # Summary
flan("Is this sentence grammatical? I goes store.")  # "No"
flan("What is the sentiment: I love this!")  # "positive"
```

## Related Models

- **[LLM](./LLM.md)** - Decoder-only, better for open-ended generation
- **[MLM](./MLM.md)** - Encoder-only, better for understanding tasks
- **[Speech Models](./SPEECH.md)** - Whisper uses encoder-decoder for ASR

## Resources

- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [BART Paper](https://arxiv.org/abs/1910.13461)
- [Flan-T5 Paper](https://arxiv.org/abs/2210.11416)
- [Hugging Face Seq2Seq Guide](https://huggingface.co/docs/transformers/tasks/summarization)
