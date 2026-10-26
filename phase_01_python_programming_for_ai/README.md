# Phase 1: Python & Programming for AI

> *"Before building artificial brains, you must master the digital loom that weaves them: Python, clean software engineering, and the scientific computing stack."*

---

## 🧭 Phase Overview

Artificial intelligence is not just mathematical formulas on a chalkboard; it is runnable software executed on silicon microchips. To train models, handle gigabytes of training data, diagnose training instabilities, and build research-grade architectures, an AI practitioner must write clean, bug-free, reproducible, and blazing-fast Python.

Phase 1 provides a comprehensive, ground-up journey through modern Python, professional software engineering practices, the scientific computing stack (NumPy, pandas, Matplotlib, SciPy), and concludes with an end-to-end Capstone project: building a custom numerical tensor library from scratch.

---

## 📚 Curriculum Structure

| Topic Folder | Subject | Core Concepts | Deliverables |
| :--- | :--- | :--- | :--- |
| **[`01_python_core_and_advanced/`](./01_python_core_and_advanced/)** | **Core & Advanced Python** | Variables, types, mutability, functions, OOP & dunder methods, iterators, generators, decorators, and custom exceptions. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`02_python_tooling_and_engineering/`](./02_python_tooling_and_engineering/)** | **AI Software Engineering** | Type hints (`typing`), virtual environments, unit testing (`unittest`/`pytest`), profiling (`cProfile`), logging, debugging, Git/GitHub, and reproducibility. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`03_scientific_computing_stack/`](./03_scientific_computing_stack/)** | **The Scientific Stack** | NumPy (strides, vectorization, broadcasting), pandas (dataframes, indexing), Matplotlib (publication figures), SciPy (optimization, sparse matrices), APIs & JSON. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`04_project_micro_tensor_library/`](./04_project_micro_tensor_library/)** | **Capstone: Micro-Tensor Library** | Build a modular mini-PyTorch tensor library from scratch with strides, broadcasting, matrix multiplication, reduction, and a test suite. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |

---

## 🎯 Learning Objectives

By completing Phase 1, you will:
1. **Understand Memory & Mutability:** Explain exactly what happens in computer memory when variables are assigned, passed to functions, and mutated.
2. **Master Advanced Python Idioms:** Write clean generators that stream terabytes of data without RAM overflow, and construct decorators that monitor GPU latency.
3. **Write Production-Grade Code:** Use static type analysis, write thorough unit tests, profile bottlenecks, and maintain reproducible environments with locked dependencies.
4. **Harness Scientific Computing:** Manipulate multi-dimensional arrays with NumPy vectorization, clean messy real-world datasets with pandas, and run mathematical optimizations with SciPy.
5. **Build a Tensor Engine:** Understand how modern deep learning frameworks (PyTorch, JAX, TensorFlow) work under the hood by implementing your own tensor data structure.

---

## 🧪 Running the Code Labs

All experiments in this phase run under standard Python 3:
```bash
# Topic 1.1: Advanced Python patterns (generators, decorators, dunders)
python3 phase_01_python_programming_for_ai/01_python_core_and_advanced/experiment.py

# Topic 1.2: Software engineering (profiling, logging, unit tests)
python3 phase_01_python_programming_for_ai/02_python_tooling_and_engineering/experiment.py

# Topic 1.3: Scientific computing stack simulation & benchmarks
python3 phase_01_python_programming_for_ai/03_scientific_computing_stack/experiment.py

# Topic 1.4: Capstone Micro-Tensor library
python3 phase_01_python_programming_for_ai/04_project_micro_tensor_library/experiment.py
```

