# Phase 2: Mathematics I — Linear Algebra

Welcome to **Phase 2** of the Complete AI Mastery Curriculum!

If Artificial Intelligence is a skyscraper, **Linear Algebra is the bedrock foundation upon which it stands**. Every single thing in modern AI—from the pixel arrays of a computer vision model to the 12,288-dimensional token embeddings of GPT-4 and the query-key attention matrices of transformers—is formulated in the language of linear algebra.

---

## 🗺️ Phase Roadmap

This phase is divided into six comprehensive topic modules:

```text
phase_02_mathematics_linear_algebra/
├── index.html                                # Visual interactive phase hub
├── README.md                                 # Phase overview, roadmap, & objectives
│
├── 01_scalars_vectors_matrices/
│   ├── lecture.html & lecture.md             # Arrows, grids, dot products, norms & cosine similarity
│   ├── experiment.py                         # Scratch linear algebra engine & vector similarity
│   └── exercises_and_solutions.html & .md    # Hand calculations, dot products & norm derivations
│
├── 02_linear_transformations/
│   ├── lecture.html & lecture.md             # Space warpers, rotations, shearing & determinants
│   ├── experiment.py                         # 2D ASCII grid transformation visualizer
│   └── exercises_and_solutions.html & .md    # Matrix multiplication mechanics & geometric proofs
│
├── 03_vector_spaces/
│   ├── lecture.html & lecture.md             # Span, basis, linear independence & rank-nullity
│   ├── experiment.py                         # Gaussian elimination & rank detector from scratch
│   └── exercises_and_solutions.html & .md    # Subspace proofs, basis tests & column space drills
│
├── 04_eigenvalues_eigenvectors/
│   ├── lecture.html & lecture.md             # The invariant axes of the universe & diagonalization
│   ├── experiment.py                         # Power iteration algorithm to find eigenvalues from scratch
│   └── exercises_and_solutions.html & .md    # Characteristic polynomials & eigendecomposition by hand
│
├── 05_singular_value_decomposition/
│   ├── lecture.html & lecture.md             # A = U Σ V^T, low-rank approximation, LoRA & PCA
│   ├── experiment.py                         # Image compression via scratch SVD (ASCII/Matrix)
│   └── exercises_and_solutions.html & .md    # Full SVD derivations & Eckart-Young theorem proofs
│
└── 06_tensors_and_project/
    ├── lecture.html & lecture.md             # Multi-dimensional arrays (B, C, H, W), strides & broadcasting
    ├── experiment.py                         # Capstone project: Pure Python Mini-Tensor Library
    └── exercises_and_solutions.html & .md    # Tensor contraction, broadcasting arithmetic & shape logic
```

---

## 🎯 Learning Objectives

By the end of Phase 2, you will be able to:
1. **Think Geometrically & Algebraically:** Translate between geometric intuition (stretching, rotating, projecting) and algebraic operations (matrix multiplication, dot products, transposes).
2. **Deconstruct High-Dimensional Spaces:** Understand what it means when an LLM places words in a $4096$-dimensional vector space, and how cosine similarity measures semantic closeness.
3. **Master Matrix Factorization:** Understand eigendecomposition ($A = Q \Lambda Q^{-1}$) and Singular Value Decomposition ($A = U \Sigma V^T$) from first principles, and see why SVD powers LoRA (Low-Rank Adaptation) and PCA.
4. **Build a Tensor Engine from Scratch:** Implement vectors, matrices, dot products, matrix multiplication, strides, and tensor broadcasting in pure Python without relying on NumPy or PyTorch.

---

## 🧒 The "Explain Like I'm 12" Philosophy

Every concept begins with a concrete, physical analogy:
- Vectors are **treasure map arrows**.
- Matrices are **rubber sheet stretching machines**.
- Determinants are **area multipliers**.
- Eigenvectors are the **unstretching axes of a spinning globe**.
- SVD is the **three-step dance of rotation, scaling, and rotation**.

Every equation is paired with a **Symbol Decoder Table** defining every variable, pronunciation, and dimension.

