# Part 3: NLP and Sequence Modeling Mini Project

## Project Overview
This project focuses on building an end-to-end Natural Language Processing (NLP) pipeline to classify the sentiment of incoming customer support tickets into three classes: `positive`, `neutral`, and `negative`. It provides a comparative analysis between a traditional machine learning approach (TF-IDF with Logistic Regression) and a deep learning approach using sequence modeling (LSTM).

## Repository Contents
- `notebook.ipynb`: The main Jupyter Notebook containing dataset exploration, preprocessing steps, text vectorization, baseline model evaluation, and the LSTM sequence model code.
- `requirements.txt`: List of Python packages required to run the pipeline.
- `results/`:
  - `model_evaluation.csv`: Contains the accuracy comparison between the Baseline model and the Sequence model.
  - `sample_predictions.txt`: Side-by-side textual output comparing predictions to true customer sentiment labels.

---

## Technical Analysis & Reflection

### 1. Why Text Must Be Converted Into Vectors (Task 3 Reflection)
Machine learning algorithms and neural networks are mathematical systems. They compute weights, gradients, dot products, and multi-dimensional matrix multiplications. They cannot natively interpret alphabetical string characters, sentences, or grammatical syntax. 

Text vectorization solves this by mapping text data into structured, numerical representations. Traditional vectorization methods like TF-IDF map documents into statistical word-frequency vectors, while modern Tokenizer-based sequences and Word Embeddings transform text into dense vectors that encapsulate semantic meanings. This transformation turns human language into a consistent mathematical matrix that machine learning algorithms can calculate and optimize.

---

### 2. Deep Learning & Transformer Reflection (Task 6 Reflection)

#### Why RNNs Struggle with Long-Term Dependencies
Recurrent Neural Networks (RNNs) process text sequences token by token (or word by word) chronologically, passing a hidden state vector forward through time. During the training phase, errors are backpropagated through time across these sequential steps. For long inputs, this involves repeated multiplication of the same matrix weights. 

Mathematically, if the weights are small, the gradients shrink exponentially until they collapse to zero (**Vanishing Gradient Problem**). If the weights are large, the gradients grow exponentially out of control (**Exploding Gradient Problem**). Consequently, information from the beginning of a long text document gets completely lost or overwritten before the network reaches the end of the text.

#### How LSTMs Help with Memory
Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem by introducing a specialized architecture called a **Cell State** ($C_t$), which acts like an internal information highway running straight down the entire text sequence. The flow of data through this cell state is regulated by three distinct mathematical filters called **Gates**:
1. **Forget Gate**: Decides what historical context from previous steps is irrelevant and should be discarded.
2. **Input Gate**: Decides what novel information from the current token should be added to the memory state.
3. **Output Gate**: Selects what specific details from the updated memory cell should be passed forward as the next hidden state.

Because information can flow smoothly through the cell state linearly with minimal matrix multiplications, LSTMs can retain long-term memory across extended sequences.

#### What Attention Solves in Sequence-to-Sequence Tasks
Even with advanced models like LSTMs, traditional encoder-decoder architectures suffer from an information bottleneck: they compress the entire source sequence into a single, fixed-size vector before passing it to the decoder. If the input text is a very long document, compressing it into a single vector causes severe data loss.

An **Attention Mechanism** removes this bottleneck entirely. Instead of forcing the model to rely on one summary vector, attention allows the decoder to dynamically "look back" at *all* the individual hidden states produced by the encoder for every single word in the input text. At each step of generating an output, it assigns numerical weights to different input words, allowing the network to focus strictly on the most contextually relevant tokens regardless of how far apart they are in the sequence.

#### Why Transformers Are Important in Modern NLP and Generative AI
Transformers completely revolutionized the field of NLP by discarding recurrence (RNNs and LSTMs) altogether and replacing it entirely with layers of **Self-Attention**. This shift introduced two major breakthroughs:
1. **Massive Parallelization**: Because there is no step-by-step sequential processing, an entire document can be processed by a GPU simultaneously rather than waiting for word after word. This allows models to be trained efficiently on billions or trillions of words of data.
2. **Global Contextual Understanding**: Self-attention evaluates the relationship of every word in a sentence with every other word at the exact same time. It captures complex, long-range semantic patterns and contextual nuances far better than sequential models.

This highly scalable and powerful architecture forms the absolute structural foundation for modern Large Language Models (LLMs) and Generative AI applications like GPT, BERT, and Claude.

---

## Setup and Execution Guide

### Local Installation
1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
