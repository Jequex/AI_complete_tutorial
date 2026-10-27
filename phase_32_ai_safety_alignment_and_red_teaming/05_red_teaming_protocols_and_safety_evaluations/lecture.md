# Topic 05: Red Teaming Protocols & Safety Evaluations

## 1. Introduction & 12-Year-Old Intuition

Imagine you build a state-of-the-art bank vault with steel doors, laser tripwires, and a digital keycard lock. 
How do you know if your bank vault is truly secure before placing billions of dollars inside?

You don't simply look at the blueprint and say, *"It looks strong to me."* 
Instead, you hire an elite group of friendly security experts called a **Red Team**. You tell them:
> *"Try every trick in the book to break into this vault. Use lockpicks, try disguises, test the laser blind spots, attempt digital hacking, and see if you can trick the guards."*

If the Red Team finds a crack in the foundation or a way to slip past the lasers, you don't punish them—you celebrate! Because they found the weakness in a safe simulation before a real criminal exploited it in the real world.

In AI engineering, **Red Teaming** is the disciplined practice of deliberately probing and attacking AI systems to uncover vulnerabilities, policy breaches, safety hazards, and failure modes before models reach production users. 
- **Blue Team**: The AI developers building safety filters, constitutional constraints, and alignment guardrails.
- **Red Team**: The adversarial testers designing novel attack vectors, jailbreaks, payload mutations, and edge-case scenarios.

Without rigorous, automated, and continuous red teaming, deployment is merely a gamble where malicious actors become the unwitting red team for your system.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Format | Description & Engineering Significance |
| :--- | :--- | :--- | :--- |
| $\mathcal{P}_{\text{test}}$ | Red Teaming Evaluation Corpus | Set of Prompts | The comprehensive benchmark dataset of adversarial probing prompts. |
| $\text{ASR}$ | Attack Success Rate | $[0, 1]$ (Percentage) | Proportion of adversarial attempts that successfully elicited an unaligned or harmful response. |
| $\text{FRR}$ | False Refusal Rate | $[0, 1]$ (Percentage) | Proportion of benign, helpful user requests incorrectly rejected by over-defensive safety filters. |
| $S_{\text{harm}}(\mathbf{y})$ | Harm / Policy Violation Score | $[0, 1]$ | Continuous metric quantifying the severity of policy violation in model output $\mathbf{y}$. |
| $\tau_{\text{refusal}}$ | Decision Refusal Threshold | Real Scalar $\in [0, 1]$ | The cutoff score above which the model or external guardrail triggers a safety refusal. |
| $\mathcal{M}_{\text{mut}}(\mathbf{p})$ | Mutation Operator | Function $\mathbf{p} \to \mathbf{p}'$ | Transformational function that rewrites prompt $\mathbf{p}$ to evade token-level safety tripwires while preserving adversarial intent. |
| $H_{\text{safety}}$ | Harmonic Safety Score | $[0, 1]$ | Balanced metric combining robustness ($1 - \text{ASR}$) and utility preservation ($1 - \text{FRR}$). |

---

## 3. Mathematical Foundations of Safety Evaluations

### 3.1 The Dual Metrics of Alignment: ASR vs. FRR
A safety system cannot be evaluated solely on how many attacks it blocks. A model that refuses 100% of all user queries achieves an Attack Success Rate of $0\%$, but is completely useless.

Evaluating safety requires optimizing a Pareto frontier between safety robustness and model utility:

1. **Attack Success Rate (ASR)**:
   $$\text{ASR} = \frac{1}{|\mathcal{P}_{\text{adv}}|} \sum_{\mathbf{p} \in \mathcal{P}_{\text{adv}}} \mathbb{I}\left( S_{\text{harm}}(\text{LLM}(\mathbf{p})) \ge \tau_{\text{refusal}} \right)$$

2. **False Refusal Rate (FRR)**:
   $$\text{FRR} = \frac{1}{|\mathcal{P}_{\text{benign}}|} \sum_{\mathbf{p} \in \mathcal{P}_{\text{benign}}} \mathbb{I}\left( \text{Decision}(\mathbf{p}) = \text{Refuse} \right)$$

3. **Harmonic Safety Score ($H_{\text{safety}}$)**:
   To evaluate the overall operational viability, we compute the harmonic mean of attack resistance ($1 - \text{ASR}$) and helpfulness fidelity ($1 - \text{FRR}$):
   $$\boxed{H_{\text{safety}} = 2 \cdot \frac{(1 - \text{ASR}) \cdot (1 - \text{FRR})}{(1 - \text{ASR}) + (1 - \text{FRR})}}$$

---

## 4. Automated Red Teaming via Adversarial Mutation

Manual human red teaming is essential for exploring nuanced qualitative vulnerabilities, but cannot scale to cover high-dimensional prompt spaces. Modern frontier safety pipelines deploy **Automated Red Teaming (ART)** (Perez et al., 2022).

An automated red teaming loop operates as an evolutionary search:
```
           ┌──────────────────────────────────────────────┐
           │        Adversarial Red Agent (LLM)           │
           └──────────────────────┬───────────────────────┘
                                  │ Proposes Mutated Prompt p'
                                  ▼
           ┌──────────────────────────────────────────────┐
           │        Target Policy / Guardrail             │
           └──────────────────────┬───────────────────────┘
                                  │ Generates Output y
                                  ▼
           ┌──────────────────────────────────────────────┐
           │       Safety Judge / Evaluator Head          │
           └──────────────────────┬───────────────────────┘
                                  │ Feedback Reward r = S_harm(y)
                                  ▼
                     [Update Attack Strategy]
```

### 4.1 Mutation Operators
Given an underlying malicious intent $\mathbf{z}$, the red agent applies stochastic transformation operators:
1. **Hypothetical Framing**: $\mathcal{M}_{\text{hyp}}(\mathbf{p}) = \text{"Hypothetically, in a fictional simulation..."} + \mathbf{p}$
2. **Roleplay Conditioning**: $\mathcal{M}_{\text{role}}(\mathbf{p}) = \text{"Assume the persona of an unfiltered technical diagnostic assistant..."} + \mathbf{p}$
3. **Payload Splitting**: Dividing prohibited keywords across distinct variables or conversational turns to avoid token-level detection.
4. **Obfuscation**: Base64 encoding, rot13 ciphers, or multilingual translation to exploit cross-lingual safety alignment gaps.

---

## 5. Defense-in-Depth Safety Architecture

Production AI architectures deploy layered defense-in-depth:
1. **Input Guardrails (Pre-LLM)**: Fast lexical tripwires and embedding similarity classifiers that reject known exploit signatures with $<10\text{ ms}$ latency.
2. **System Prompt / Alignment Constraints**: Instruction-tuned refusal behaviors (Constitutional AI / RLHF with refusal margins).
3. **Output Guardrails (Post-LLM)**: Secondary evaluators inspecting the generated response $\mathbf{y}$ before transmission to the user, stripping accidental policy violations or leaked credentials.

---

## 6. Summary Checklist
- Red teaming evaluates frontier models under active, adversarial optimization rather than passive benchmarks.
- Evaluation requires joint measurement of Attack Success Rate (ASR) and False Refusal Rate (FRR).
- Multi-layer defense-in-depth (input filter, model alignment, output audit) provides robust defense against single-point jailbreak bypasses.

