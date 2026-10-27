# The Complete AI Mastery Course

**Target:** From beginner → mathematically rigorous AI/ML engineer/researcher  
**Approach:** Mathematics + theory + implementation + experiments + projects  
**End goal:** You should be able to go from an idea such as *“I want an AI that predicts X”* to selecting an architecture, deriving the objective, implementing it, training it, evaluating it, debugging it, and deploying it.

---

## Course map

```text
                    ARTIFICIAL INTELLIGENCE
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
   MATHEMATICS                              COMPUTER SCIENCE
        │                                       │
        ├─ Linear Algebra                      ├─ Python
        ├─ Calculus                            ├─ Data Structures
        ├─ Probability                         ├─ Algorithms
        ├─ Statistics                          ├─ Systems
        ├─ Optimization                        └─ GPU Computing
        └─ Information Theory
                    │
                    ▼
              MACHINE LEARNING
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
   Supervised   Unsupervised   Reinforcement
   Learning       Learning       Learning
        │           │            │
        └───────────┼────────────┘
                    ▼
             DEEP LEARNING
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
      CNNs         RNNs       Transformers
       │            │             │
       └────────────┼─────────────┘
                    ▼
              FOUNDATION MODELS
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
      LLMs       Vision        Multimodal
       │
       ├─ Pretraining
       ├─ Fine-tuning
       ├─ PEFT / LoRA
       ├─ RLHF / preference optimization
       ├─ RAG
       ├─ Agents
       └─ Inference optimization
                    │
                    ▼
              AI ENGINEERING
                    │
       ├─ Evaluation
       ├─ Deployment
       ├─ Distributed training
       ├─ GPUs
       ├─ Quantization
       ├─ Safety
       └─ Production systems
                    │
                    ▼
              AI RESEARCH
```

I would divide the course into **18 phases**.

---

# Phase 0 — What AI Actually Is

Before mathematics, we establish the conceptual foundation.

### 0.1 What is intelligence?

- Intelligence vs computation
- Learning vs memorization
- Reasoning vs pattern recognition
- Symbolic vs statistical AI
- Narrow AI vs general AI
- Agents
- Environment, observations, actions and rewards

### 0.2 History of AI

You'll understand:

- Turing
- Dartmouth and the birth of AI
- Symbolic AI
- Expert systems
- AI winters
- Statistical machine learning
- Neural networks
- Deep learning
- AlexNet
- Transformers
- Foundation models
- Generative AI
- Modern reasoning systems

### 0.3 The central idea

We'll eventually reduce a huge amount of AI to:

\[
\text{Data} + \text{Model} + \text{Objective} + \text{Optimization}
\]

and understand every component deeply.

---

# Phase 1 — Python & Programming for AI

Before building AI, you need to be able to build software.

### Python

Learn:

- Variables
- Types
- Functions
- Classes
- Iterators
- Generators
- Decorators
- Exceptions
- Modules
- Packages
- Virtual environments
- Type hints
- Testing
- Profiling

### Scientific Python

Master:

- NumPy
- pandas
- Matplotlib
- SciPy

### Software engineering

Learn:

- Git
- GitHub
- APIs
- JSON
- command line
- Linux
- debugging
- logging
- unit tests
- environments
- reproducibility

### Project

Build a small numerical-computing library.

For example:

```python
x = Tensor([[1, 2], [3, 4]])

y = x @ x

print(y)
```

Eventually we'll build something resembling a tiny version of part of PyTorch.

---

# Phase 2 — Mathematics I: Linear Algebra

This is where the mathematical core begins.

## 2.1 Scalars, vectors and matrices

Understand:

\[
x \in \mathbb R
\]

\[
\mathbf{x}\in\mathbb R^n
\]

\[
A\in\mathbb R^{m\times n}
\]

Operations:

- addition
- multiplication
- transpose
- dot product
- matrix multiplication
- norms

### Geometry

You will understand:

- vectors as points
- vectors as directions
- projections
- distances
- angles
- orthogonality

For example:

\[
\mathbf{x}\cdot\mathbf{y}
=
\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta
\]

This becomes fundamental for embeddings and attention.

---

## 2.2 Linear transformations

Understand:

\[
f(\mathbf{x})=A\mathbf{x}
\]

and what matrices actually do geometrically.

Study:

- rotation
- scaling
- reflection
- projection
- shear

---

## 2.3 Vector spaces

- span
- basis
- dimension
- linear independence
- subspaces
- column space
- null space
- rank

---

## 2.4 Eigenvalues and eigenvectors

\[
A\mathbf v=\lambda\mathbf v
\]

You'll learn:

- eigendecomposition
- characteristic polynomial
- diagonalization
- spectral interpretation

---

## 2.5 Singular Value Decomposition

\[
A=U\Sigma V^T
\]

SVD is enormously important.

Applications:

- dimensionality reduction
- PCA
- compression
- embeddings
- numerical optimization
- recommender systems

---

## 2.6 Tensors

Understand:

\[
X\in\mathbb R^{B\times C\times H\times W}
\]

and why tensors dominate deep learning.

### Project

Implement:

- vector
- matrix
- matrix multiplication
- transpose
- norms
- SVD
- PCA

**without NumPy initially.**

Then implement them using NumPy.

---

# Phase 3 — Mathematics II: Calculus

Now we learn how models learn.

## 3.1 Functions

- limits
- continuity
- derivatives

## 3.2 Derivatives

Understand:

\[
\frac{df}{dx}
\]

geometrically and computationally.

---

## 3.3 Partial derivatives

For:

\[
f(x,y)
\]

learn:

\[
\frac{\partial f}{\partial x}
\]

and:

\[
\frac{\partial f}{\partial y}
\]

---

## 3.4 Gradients

\[
\nabla f=
\begin{bmatrix}
\frac{\partial f}{\partial x}\\
\frac{\partial f}{\partial y}\\
\vdots
\end{bmatrix}
\]

The gradient tells us the direction of greatest increase.

This becomes the foundation of gradient descent.

---

## 3.5 Jacobians

\[
J_{ij}=
\frac{\partial f_i}{\partial x_j}
\]

Important for:

- neural networks
- backpropagation
- transformations
- optimization

---

## 3.6 Hessians

\[
H_{ij}
=
\frac{\partial^2 f}
{\partial x_i\partial x_j}
\]

We'll use them to understand:

- curvature
- optimization
- Newton's method
- saddle points
- loss landscapes

---

## 3.7 Chain rule

The most important rule in deep learning:

\[
\frac{dz}{dx}
=
\frac{dz}{dy}
\frac{dy}{dx}
\]

We'll derive backpropagation from the chain rule.

---

# Phase 4 — Probability

AI is fundamentally probabilistic.

You'll learn:

### Probability fundamentals

- sample spaces
- events
- conditional probability
- Bayes' theorem
- independence

\[
P(A|B)=
\frac{P(B|A)P(A)}{P(B)}
\]

---

### Random variables

- discrete
- continuous
- probability mass functions
- probability density functions
- cumulative distributions

---

### Important distributions

- Bernoulli
- Binomial
- Gaussian
- Multivariate Gaussian
- Poisson
- Exponential
- categorical

---

### Expectation

\[
E[X]
\]

### Variance

\[
Var(X)=E[(X-\mu)^2]
\]

### Covariance

\[
Cov(X,Y)
=
E[(X-E[X])(Y-E[Y])]
\]

### Correlation

### Law of Large Numbers

### Central Limit Theorem

These concepts will eventually explain why many machine-learning methods work.

---

# Phase 5 — Statistics

You'll learn:

- populations
- samples
- estimators
- bias
- variance
- confidence intervals
- hypothesis testing
- p-values
- likelihood
- maximum likelihood estimation
- maximum a posteriori estimation
- Bayesian inference

We'll derive:

\[
\hat{\theta}_{MLE}
=
\arg\max_\theta P(D|\theta)
\]

and understand why maximizing likelihood leads to many familiar machine-learning objectives.

---

# Phase 6 — Information Theory

This is extremely important for modern AI.

Learn:

## Entropy

\[
H(X)
=
-\sum_xP(x)\log P(x)
\]

## Cross entropy

\[
H(P,Q)
=
-\sum_xP(x)\log Q(x)
\]

## KL divergence

\[
D_{KL}(P\|Q)
=
\sum_xP(x)
\log\frac{P(x)}{Q(x)}
\]

## Mutual information

\[
I(X;Y)
=
\sum_{x,y}
P(x,y)
\log
\frac{P(x,y)}
{P(x)P(y)}
\]

Applications:

- classification
- language models
- representation learning
- VAEs
- knowledge compression
- information bottleneck

---

# Phase 7 — Optimization

Now we can understand how AI models actually learn.

### Gradient descent

\[
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L(\theta)
\]

Study:

- batch gradient descent
- stochastic gradient descent
- mini-batch SGD

Then:

- Momentum
- AdaGrad
- RMSProp
- Adam
- AdamW

We'll understand what every term does.

---

## Convex optimization

Learn:

- convex functions
- convex sets
- local/global minima
- Lagrange multipliers
- constrained optimization
- KKT conditions

Then understand why deep learning is different from classical convex optimization.

---

# Phase 8 — Classical Machine Learning

Now we build actual ML models.

## Regression

### Linear regression

\[
\hat y=Xw+b
\]

Derive the closed-form solution:

\[
w=(X^TX)^{-1}X^Ty
\]

and then derive gradient descent.

---

### Logistic regression

\[
P(y=1|x)=\sigma(w^Tx+b)
\]

where:

\[
\sigma(x)=\frac1{1+e^{-x}}
\]

---

## Classification

Learn:

- k-nearest neighbors
- Naive Bayes
- decision trees
- random forests
- gradient boosting
- SVMs

Including the mathematics behind:

- entropy
- information gain
- Gini impurity
- margins
- kernels

---

# Phase 9 — Unsupervised Learning

Learn:

### Clustering

- K-means
- hierarchical clustering
- DBSCAN
- Gaussian mixture models

### Dimensionality reduction

- PCA
- SVD
- t-SNE
- UMAP

### Density estimation

### Anomaly detection

### Representation learning

---

# Phase 10 — ML Engineering

A model isn't useful just because it works on a notebook.

Learn:

- data collection
- data cleaning
- train/validation/test splitting
- data leakage
- preprocessing
- feature engineering
- feature selection
- normalization
- standardization
- class imbalance

---

## Evaluation

Understand:

### Regression

- MSE
- RMSE
- MAE
- \(R^2\)

### Classification

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrices

And critically:

**when each metric can mislead you.**

---

# Phase 11 — Neural Networks From Scratch

This is one of the most important phases.

We won't start with PyTorch.

We'll build a neural network ourselves.

Start with:

\[
z=Wx+b
\]

then:

\[
a=f(z)
\]

where \(f\) might be:

- sigmoid
- tanh
- ReLU
- GELU
- SiLU

---

## Forward propagation

Understand every operation.

## Loss functions

### MSE

\[
L=\frac1n\sum_i(y_i-\hat y_i)^2
\]

### Binary cross entropy

\[
L=
-[y\log p+(1-y)\log(1-p)]
\]

### Categorical cross entropy

\[
L=-\sum_i y_i\log p_i
\]

---

# Phase 12 — Backpropagation From First Principles

We derive backpropagation mathematically.

For a network:

\[
x
\rightarrow
W_1
\rightarrow
a_1
\rightarrow
W_2
\rightarrow
a_2
\rightarrow
L
\]

we calculate:

\[
\frac{\partial L}{\partial W_2}
\]

then:

\[
\frac{\partial L}{\partial a_1}
\]

then:

\[
\frac{\partial L}{\partial W_1}
\]

using the chain rule.

Then we'll implement **automatic differentiation from scratch.**

Something conceptually like:

```python
x = Value(2.0)
w = Value(3.0)

y = x * w
y.backward()

print(w.grad)
```

This teaches you what frameworks such as PyTorch are actually doing.

---

# Phase 13 — Deep Learning

Now move to PyTorch.

Learn:

- tensors
- computational graphs
- autograd
- modules
- optimizers
- datasets
- dataloaders
- GPU training
- checkpoints

Then study:

## Initialization

- Xavier
- He initialization

## Normalization

- BatchNorm
- LayerNorm
- RMSNorm

## Regularization

- dropout
- weight decay
- early stopping
- data augmentation

---

# Phase 14 — Computer Vision

Build vision systems from scratch.

## CNNs

Understand:

\[
Y_{i,j}
=
\sum_{m,n}
X_{i+m,j+n}K_{m,n}
\]

Learn:

- convolution
- padding
- stride
- pooling
- receptive fields
- channels

Architectures:

- LeNet
- AlexNet
- VGG
- ResNet
- EfficientNet

---

## Modern vision

Study:

- Vision Transformers
- object detection
- segmentation
- image embeddings
- CLIP-style contrastive learning
- diffusion models

### Projects

Build:

1. handwritten-digit classifier
2. image classifier
3. object detector
4. image segmentation model
5. image-text embedding system

---

# Phase 15 — Sequence Models

Study:

## RNNs

\[
h_t=f(W_xx_t+W_hh_{t-1}+b)
\]

Understand:

- hidden states
- sequence modeling
- teacher forcing
- vanishing gradients
- exploding gradients

---

## LSTM

Understand the gates mathematically:

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f)
\]

\[
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
\]

\[
o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
\]

and the cell state.

---

## GRU

Understand how it simplifies the LSTM.

---

# Phase 16 — Transformers

This deserves an entire course within the course.

Start from the problem with RNNs.

Then derive:

# Attention

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

We'll understand **every symbol**.

---

## Step-by-step

### Tokenization

Text:

```text
"The cat sat"
```

becomes tokens.

### Embeddings

\[
token\rightarrow vector
\]

### Positional information

Understand:

- sinusoidal embeddings
- learned positional embeddings
- rotary positional embeddings

### Query, Key, Value

Understand the intuition and mathematics.

### Multi-head attention

\[
MultiHead(Q,K,V)
=
Concat(head_1,\dots,head_h)W^O
\]

### Feed-forward network

\[
FFN(x)
=
W_2\sigma(W_1x+b_1)+b_2
\]

### Residual connections

\[
x'=x+F(x)
\]

### Layer normalization

Understand exactly why it is used.

---

# Phase 17 — Build a GPT From Scratch

This is a major milestone.

We'll build a small language model ourselves.

Pipeline:

```text
Raw text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embeddings
   ↓
Positional encoding
   ↓
Transformer blocks
   ↓
Linear projection
   ↓
Logits
   ↓
Softmax
   ↓
Next-token probability
```

We'll implement:

- tokenizer
- dataset
- batching
- causal masking
- embeddings
- attention
- multi-head attention
- MLP
- transformer block
- layer normalization
- optimizer
- training loop
- checkpointing
- text generation

Eventually:

```text
"Artificial intelligence is"
```

produces:

```text
"the..."
```

because the model estimates:

\[
P(x_{t+1}|x_1,\ldots,x_t)
\]

---

# Phase 18 — Large Language Models

Now move from toy GPTs to real LLMs.

Study:

### Scaling

- parameter count
- dataset size
- compute
- training tokens
- scaling laws

### Tokenization

- BPE
- WordPiece
- SentencePiece
- byte-level tokenization

### Training

Understand:

\[
L(\theta)
=
-\sum_t
\log P_\theta(x_t|x_{<t})
\]

---

# Phase 19 — Fine-Tuning

Now you'll learn how to modify existing models.

## Full fine-tuning

Train all parameters.

Understand:

- learning rates
- catastrophic forgetting
- overfitting
- dataset quality
- checkpointing

---

## Parameter-efficient fine-tuning

Study:

### LoRA

Instead of:

\[
W'=W+\Delta W
\]

represent:

\[
\Delta W=BA
\]

where:

\[
B\in\mathbb R^{d\times r}
\]

\[
A\in\mathbb R^{r\times k}
\]

with:

\[
r\ll d,k
\]

You'll understand why this dramatically reduces trainable parameters.

Then:

- QLoRA
- adapters
- prefix tuning
- prompt tuning

---

# Phase 20 — Instruction Tuning

Learn why pretrained language models aren't automatically useful assistants.

Study:

- supervised fine-tuning
- instruction datasets
- conversation formatting
- data quality
- mixture datasets
- synthetic data

You'll fine-tune a model to follow instructions.

---

# Phase 21 — Preference Optimization

Then we investigate how models are aligned with preferences.

Study:

- human preference datasets
- reward models
- RLHF
- PPO
- DPO
- preference optimization
- reward hacking
- overoptimization

Understand the mathematics behind:

\[
\max_\theta E[r(x,y)]
\]

and why preference learning is difficult.

---

# Phase 22 — Retrieval-Augmented Generation

Build RAG systems.

Pipeline:

```text
Documents
    ↓
Chunking
    ↓
Embedding model
    ↓
Vector database
    ↓
Similarity search
    ↓
Retrieved context
    ↓
LLM
    ↓
Answer
```

Learn:

- embeddings
- cosine similarity
- vector databases
- approximate nearest-neighbor search
- chunking strategies
- reranking
- hybrid search
- retrieval evaluation
- context construction

---

# Phase 23 — Embeddings

Deep dive into representation learning.

Study:

- word embeddings
- sentence embeddings
- contrastive learning
- cosine similarity
- metric learning
- triplet loss

Contrastive objective:

\[
L=-\log
\frac{
e^{sim(z_i,z_j)/\tau}
}{
\sum_k e^{sim(z_i,z_k)/\tau}
}
\]

We'll understand why this works.

---

# Phase 24 — Generative Models

Study the major families.

## Autoregressive models

\[
P(x)=
\prod_tP(x_t|x_{<t})
\]

## Autoencoders

## Variational Autoencoders

Learn:

- latent variables
- ELBO
- reparameterization trick

---

## GANs

Understand:

\[
\min_G\max_D
V(D,G)
\]

including:

- generator
- discriminator
- adversarial training
- mode collapse
- Wasserstein GAN

---

# Phase 25 — Diffusion Models

Deep dive.

Forward process:

\[
q(x_t|x_{t-1})
\]

Gradually adds noise.

Reverse process:

\[
p_\theta(x_{t-1}|x_t)
\]

learns to remove it.

Study:

- DDPM
- noise schedules
- score matching
- denoising
- latent diffusion
- conditioning
- classifier-free guidance

You'll implement a small diffusion model.

---

# Phase 26 — Reinforcement Learning

Learn the mathematical framework.

\[
(S,A,P,R,\gamma)
\]

Study:

- states
- actions
- rewards
- policies
- value functions
- returns
- Markov decision processes

---

## Bellman equations

\[
V(s)
=
E[
R_{t+1}
+
\gamma V(S_{t+1})
]
\]

Then:

- Q-learning
- SARSA
- DQN
- policy gradients
- actor-critic
- PPO

---

# Phase 27 — AI Agents

Understand agents as systems rather than simply models.

Architecture:

```text
          ┌───────────────┐
          │     Model     │
          └───────┬───────┘
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
    Memory      Tools     Planning
       │          │          │
       └──────────┼──────────┘
                  ↓
              Environment
                  ↓
              Observation
                  ↓
                Model
```

Study:

- tool use
- function calling
- planning
- memory
- reflection
- state management
- multi-agent systems
- agent evaluation

---

# Phase 28 — AI Systems & Infrastructure

Now become capable of training serious models.

Study:

### GPUs

Understand:

- CPU vs GPU
- CUDA
- GPU memory
- tensor cores
- FLOPs
- memory bandwidth

### Distributed training

- data parallelism
- model parallelism
- pipeline parallelism
- tensor parallelism
- distributed data parallelism
- ZeRO

### Mixed precision

- FP32
- FP16
- BF16
- FP8

### Memory optimization

- gradient checkpointing
- activation memory
- optimizer states

---

# Phase 29 — Model Compression

Learn:

- quantization
- pruning
- knowledge distillation
- weight sharing

Quantization:

\[
x_q
=
round(x/s)+z
\]

Study:

- INT8
- INT4
- GPTQ
- AWQ
- activation-aware quantization

---

# Phase 30 — Efficient Inference

Understand:

- batching
- dynamic batching
- KV cache
- speculative decoding
- continuous batching
- paged attention
- tensor parallel inference

And the fundamental tradeoffs:

\[
Latency
\leftrightarrow
Throughput
\leftrightarrow
Memory
\leftrightarrow
Cost
\]

---

# Phase 31 — AI Evaluation

This deserves serious attention.

Learn to evaluate:

### Traditional ML

- statistical significance
- cross-validation
- confidence intervals
- calibration

### LLMs

- accuracy
- exact match
- BLEU
- ROUGE
- perplexity
- human evaluation
- pairwise preference
- LLM-as-judge
- hallucination evaluation
- factuality
- robustness

Understand why **benchmark scores don't automatically mean a model is good for a particular application.**

---

# Phase 32 — AI Safety & Reliability

Study technically:

- adversarial examples
- distribution shift
- robustness
- hallucination
- calibration
- interpretability
- mechanistic interpretability
- alignment
- reward hacking
- specification gaming
- data poisoning
- model extraction
- prompt injection

---

# Phase 33 — Interpretability

Go inside neural networks.

Study:

- activation analysis
- feature visualization
- saliency
- probing
- attribution
- activation patching
- causal interventions
- circuits
- sparse autoencoders

The goal is to move from:

> "The model predicted X."

to:

> "Here is evidence about the internal computation associated with X."

---

# Phase 34 — Advanced Mathematics

For serious research, we'll eventually cover:

### Linear algebra

- spectral theory
- matrix calculus
- tensor algebra

### Calculus

- multivariable calculus
- differential equations
- vector calculus

### Probability

- measure-theoretic intuition
- stochastic processes
- Bayesian inference

### Optimization

- stochastic optimization
- constrained optimization
- second-order methods

### Information theory

- entropy
- mutual information
- rate-distortion theory

### Numerical methods

- numerical stability
- floating-point arithmetic
- conditioning

---

# Phase 35 — Research-Level Machine Learning

Then we study the research literature.

Topics:

- representation learning
- self-supervised learning
- meta-learning
- few-shot learning
- continual learning
- transfer learning
- multimodal learning
- world models
- reasoning
- test-time computation
- model scaling
- synthetic data

You'll learn how to read papers:

```text
Problem
 ↓
Prior work
 ↓
Hypothesis
 ↓
Method
 ↓
Mathematical formulation
 ↓
Experiments
 ↓
Ablations
 ↓
Results
 ↓
Limitations
```

---

# Phase 36 — Model Context Protocol (MCP) & Extensible Agent Ecosystems

Understand the open standard protocol connecting AI models to external tools, databases, and environments.

Architecture:

```text
┌────────────────────────────────────────────────────────┐
│                      Host Client                       │
│  (IDE, Desktop App, Agent Runtime, Orchestrator)       │
│                                                        │
│   ┌───────────────┐               ┌────────────────┐   │
│   │   LLM Core    │◄─────────────►│   MCP Client   │   │
│   └───────────────┘               └───────┬────────┘   │
└───────────────────────────────────────────┼────────────┘
                                            │
                     JSON-RPC 2.0 (stdio / SSE / HTTP)
                                            │
      ┌─────────────────────────────────────┴─────────────────────────────────────┐
      ↓                                     ↓                                     ↓
┌──────────────┐                      ┌──────────────┐                      ┌──────────────┐
│  MCP Server  │                      │  MCP Server  │                      │  MCP Server  │
│  (Database)  │                      │ (Filesystem) │                      │ (Third-Party)│
├──────────────┤                      ├──────────────┤                      ├──────────────┤
│ Resources    │                      │ Resources    │                      │ Tools        │
│ Tools        │                      │ Tools        │                      │ Prompts      │
└──────────────┘                      └──────────────┘                      └──────────────┘
```

Study:

### Protocol Foundations
- Client-Server-Host topology
- JSON-RPC 2.0 wire format (requests, notifications, responses, error codes)
- Transports: stdio vs HTTP with Server-Sent Events (SSE)
- Lifecycle: initialization, capability negotiation, ping, shutdown

### The Three Core Primitives
- **Resources**: URI-addressable read-only context (files, database records, API schemas)
- **Prompts**: User-controlled interactive workflows, parameterized prompt templates
- **Tools**: Model-controlled executable functions with side-effects and JSON Schema contracts

### Advanced Agent Integration
- Dynamic resource updates and subscriptions (`notifications/resources/updated`)
- Tool authorization, sandboxing, and security boundaries
- Multi-server aggregation and agent routing
- Building production MCP servers and clients from first principles

