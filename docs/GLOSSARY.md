# AI Glossary: Terms and Definitions

## A

**A2A (Agent-to-Agent)**
Communication protocol enabling direct interaction between AI agents for collaboration and task distribution.

**A2P (Agent-to-Person)**
Communication pattern defining how AI agents interact with human users through conversational and intuitive interfaces.

**A2S (Agent-to-System)**
Protocol for AI agents to interact with external systems, databases, APIs, and infrastructure.

**Adapter**
Small neural network modules that allow efficient fine-tuning of large pre-trained models by only training the adapter weights.

**Adversarial Training**
Training method where two neural networks compete (like GANs), one generating data and one discriminating real from fake.

**Agent**
An AI system capable of perceiving its environment, making decisions, and taking actions to achieve specific goals.

**Attention Mechanism**
Neural network component that allows models to focus on specific parts of the input when making predictions.

**Autoencoder**
Neural network trained to compress data into a lower-dimensional representation and then reconstruct it.

**Autoregressive Model**
Model that generates sequences one element at a time, using previously generated elements as context.

## B

**BERT (Bidirectional Encoder Representations from Transformers)**
Masked language model that understands context from both left and right directions simultaneously.

**Bidirectional**
Processing information in both forward and backward directions to capture full context.

**Byte-Pair Encoding (BPE)**
Tokenization method that breaks text into subword units based on frequency.

## C

**Checkpoint**
Saved snapshot of a model's weights during training, allowing resumption or evaluation.

**CLIP (Contrastive Language-Image Pre-training)**
Model trained to understand relationships between images and text through contrastive learning.

**Context Window**
Maximum amount of text (measured in tokens) a model can process at once.

**Contrastive Learning**
Training approach that learns by comparing similar (positive) and dissimilar (negative) examples.

**Cross-Attention**
Attention mechanism that allows one sequence to attend to another (e.g., image features attending to text).

## D

**Decoder**
Neural network component that generates output sequences from encoded representations.

**Diffusion Model**
Generative model that creates data by iteratively denoising random noise.

**Distillation**
Process of training a smaller "student" model to mimic a larger "teacher" model.

**DINO (Self-Distillation with No Labels)**
Self-supervised learning method for vision models.

## E

**Embedding**
Dense vector representation of data (text, images, etc.) in a continuous space where similar items are close together.

**Encoder**
Neural network component that transforms input into a compressed representation.

**Encoder-Decoder**
Architecture with separate encoding and decoding components, common in translation tasks.

**Few-Shot Learning**
Ability to learn from very few examples (typically 1-10 examples per class).

**Fine-Tuning**
Adapting a pre-trained model to a specific task by training on task-specific data.

## F

**Federated Learning**
Distributed training approach where models learn from decentralized data without data leaving source devices.

**Foundation Model**
Large-scale pre-trained model designed to be adapted for many downstream tasks.

**FID (Fréchet Inception Distance)**
Metric for evaluating quality of generated images by comparing feature distributions.

## G

**GAN (Generative Adversarial Network)**
Generative model with generator and discriminator networks trained adversarially.

**GNN (Graph Neural Network)**
Neural network designed to operate on graph-structured data.

**GPT (Generative Pre-trained Transformer)**
Family of autoregressive language models (GPT-2, GPT-3, GPT-4).

**Gradient**
Direction and magnitude of change in loss with respect to model parameters.

## H

**Hallucination**
When AI models generate plausible-sounding but factually incorrect information.

**Hidden State**
Internal representations within a neural network layer.

**Hugging Face**
Platform and organization providing AI models, datasets, and tools.

**Hyperparameter**
Configuration setting for training (e.g., learning rate, batch size) not learned from data.

## I

**In-Context Learning**
Model's ability to learn from examples provided in the prompt without weight updates.

**Inference**
Using a trained model to make predictions on new data.

**Instruction Tuning**
Fine-tuning models to follow natural language instructions.

## K

**Knowledge Distillation**
See Distillation.

**K-Shot Learning**
Learning from K examples per class.

## L

**LAM (Large Action Model)**
AI model designed to execute actions and interact with digital environments.

**Latent Space**
Lower-dimensional representation where similar items are close together.

**LCM (Latent Consistency Model)**
Fast image generation model requiring only 1-4 inference steps.

**LLM (Large Language Model)**
Large neural network trained on text for natural language understanding and generation.

**LMM (Large Multimodal Model)**
Model natively designed to process multiple modalities (text, image, audio, video).

**LoRA (Low-Rank Adaptation)**
Efficient fine-tuning method that adds small trainable matrices to frozen model weights.

**Loss Function**
Measure of how well model predictions match true values, used to guide training.

**LVM (Large Vision Model)**
Large model specialized for computer vision tasks.

## M

**MCP (Model Context Protocol)**
Standardized protocol for AI models to access external resources and tools.

**MLM (Masked Language Model)**
Model trained by predicting masked tokens in text (e.g., BERT).

**MLLM (Multimodal Large Language Model)**
LLM extended with additional modalities through adapters.

**MOE (Mixture of Experts)**
Architecture with multiple specialized sub-networks (experts) activated selectively.

**Multimodal**
Involving multiple types of data (text, images, audio, video).

## N

**NAS (Neural Architecture Search)**
Automated process of discovering optimal neural network architectures.

**Neurosymbolic AI**
Hybrid approach combining neural networks with symbolic reasoning.

**NLP (Natural Language Processing)**
Field of AI focused on understanding and generating human language.

**NLU (Natural Language Understanding)**
Subset of NLP focused on comprehending meaning and intent.

**NLG (Natural Language Generation)**
Subset of NLP focused on producing human-like text.

## O

**ONNX (Open Neural Network Exchange)**
Open format for representing machine learning models.

**Orchestration**
Coordinating multiple AI agents or models to work together.

**Overfitting**
When model learns training data too well and fails to generalize to new data.

## P

**Parameter**
Learnable weight in a neural network adjusted during training.

**Perplexity**
Metric measuring how well a probability model predicts text (lower is better).

**Pretraining**
Initial training on large datasets before task-specific fine-tuning.

**Prompt**
Input text or instruction given to a language model.

**Prompt Engineering**
Craft of designing effective prompts to get desired model behavior.

## Q

**Quantization**
Reducing numerical precision of model weights (e.g., 32-bit → 8-bit) to reduce size.

**Query**
In attention mechanisms, the representation asking for relevant information.

## R

**RAG (Retrieval-Augmented Generation)**
Combining information retrieval with generation for grounded, factual responses.

**Reinforcement Learning (RL)**
Training agents through trial and error with rewards and penalties.

**RLHF (Reinforcement Learning from Human Feedback)**
Training method using human preferences to improve model behavior.

**RNN (Recurrent Neural Network)**
Neural network processing sequential data with internal memory.

## S

**SAM (Segment Anything Model)**
Foundation model for image segmentation capable of zero-shot segmentation.

**Self-Attention**
Attention mechanism where a sequence attends to itself.

**Self-Supervised Learning**
Training without manual labels by creating tasks from data itself.

**Semantic Search**
Search based on meaning rather than keyword matching.

**SLM (Small Language Model)**
Efficient language model with fewer than 7B parameters, optimized for edge deployment.

**Supervised Learning**
Training with labeled input-output pairs.

**System Prompt**
Initial instruction defining an AI agent's behavior and personality.

## T

**Temperature**
Parameter controlling randomness in model outputs (higher = more random).

**Token**
Basic unit of text processing (word, subword, or character).

**Tokenization**
Breaking text into tokens for model processing.

**Top-K Sampling**
Sampling from K most probable next tokens.

**Top-P (Nucleus) Sampling**
Sampling from smallest set of tokens with cumulative probability ≥ P.

**Transfer Learning**
Applying knowledge from one task to another related task.

**Transformer**
Neural architecture based on self-attention, foundation of modern LLMs.

## U

**Underfitting**
When model is too simple to capture patterns in data.

**Unsupervised Learning**
Training without labeled data, finding patterns automatically.

## V

**VAE (Variational Autoencoder)**
Generative model learning probabilistic latent representations.

**Vector Database**
Database optimized for storing and searching high-dimensional vectors (embeddings).

**Vision Transformer (ViT)**
Transformer architecture adapted for computer vision.

**VLM (Vision Language Model)**
Model combining vision and language understanding.

**VQA (Visual Question Answering)**
Task of answering questions about images.

## W

**Weight**
Parameter in neural network determining connection strength between neurons.

**World Model**
AI system that builds internal representation of environment dynamics.

## Z

**Zero-Shot Learning**
Performing tasks without any task-specific training examples.

---

## Model Families

**GPT Family**: GPT-2, GPT-3, GPT-3.5, GPT-4, GPT-4V, GPT-4o
**Claude Family**: Claude 1, Claude 2, Claude 3 (Opus/Sonnet/Haiku)
**Llama Family**: Llama 1, Llama 2, Llama 3, Llama 3.1, Llama 3.2
**Gemini Family**: Gemini Nano, Gemini Pro, Gemini Ultra, Gemini 1.5
**BERT Family**: BERT, RoBERTa, DeBERTa, ALBERT, DistilBERT

## Common Acronyms

- **API**: Application Programming Interface
- **CUDA**: Compute Unified Device Architecture (NVIDIA GPU programming)
- **FP16/FP32**: 16-bit/32-bit floating point precision
- **GPU**: Graphics Processing Unit
- **HPC**: High-Performance Computing
- **INT4/INT8**: 4-bit/8-bit integer quantization
- **JSON**: JavaScript Object Notation
- **MLOps**: Machine Learning Operations
- **NPU**: Neural Processing Unit
- **REST**: Representational State Transfer
- **SDK**: Software Development Kit
- **TPU**: Tensor Processing Unit (Google)
- **VRAM**: Video RAM (GPU memory)

## Training Terms

**Epoch**: One complete pass through training dataset
**Batch**: Subset of data processed together
**Batch Size**: Number of examples per batch
**Learning Rate**: Step size for parameter updates
**Gradient Descent**: Optimization algorithm for training
**Backpropagation**: Algorithm for computing gradients
**Convergence**: When training loss stops improving
**Checkpoint**: Saved model state during training

## Evaluation Metrics

**Accuracy**: Percentage of correct predictions
**Precision**: True positives / (True positives + False positives)
**Recall**: True positives / (True positives + False negatives)
**F1 Score**: Harmonic mean of precision and recall
**BLEU**: Metric for machine translation quality
**ROUGE**: Metric for summarization quality
**Perplexity**: Language model quality metric
**FID**: Image generation quality metric

---

*For detailed explanations of specific models, see the [Professional Overview](./PROFESSIONAL_OVERVIEW.md).*
