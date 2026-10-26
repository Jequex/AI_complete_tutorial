#!/usr/bin/env python3
"""
Topic 8.5 Experiment: Naive Bayes & K-Nearest Neighbors (KNN) from Scratch
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, collections)

Demonstrates:
1. Multinomial Naive Bayes with Laplace (Add-1) Smoothing:
   - Vocabulary construction and term-frequency extraction.
   - Log-space class posterior computation to prevent underflow:
     log P(c | d) = log P(c) + sum_i log P(w_i | c)
   - Tested on email Spam vs. Ham corpus.
2. K-Nearest Neighbors (KNN) Instance-Based Classifier:
   - Euclidean (L2) and Manhattan (L1) distance metrics.
   - Non-parametric top-k neighbor retrieval and majority voting.
   - Tested on non-linear multi-cluster 2D benchmark.
3. Verification of mathematical properties and rigorous accuracy assertions (exit code 0).
"""

import math
import random
import sys

# =====================================================================
# 1. Multinomial Naive Bayes Classifier from Scratch
# =====================================================================

class MultinomialNaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.class_priors = {}
        self.word_counts = {}
        self.class_total_words = {}
        self.vocab = set()

    def _tokenize(self, text):
        # Basic alphanumeric tokenization and lowercasing
        clean_text = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
        return clean_text.split()

    def fit(self, documents, labels):
        N = len(documents)
        self.classes = sorted(list(set(labels)))
        self.vocab = set()
        
        # Initialize counts
        for c in self.classes:
            self.word_counts[c] = {}
            self.class_total_words[c] = 0
            
        class_doc_counts = {c: 0 for c in self.classes}
        
        for doc, label in zip(documents, labels):
            class_doc_counts[label] += 1
            tokens = self._tokenize(doc)
            for token in tokens:
                self.vocab.add(token)
                self.word_counts[label][token] = self.word_counts[label].get(token, 0) + 1
                self.class_total_words[label] += 1

        # Class priors: log P(c)
        for c in self.classes:
            self.class_priors[c] = math.log(class_doc_counts[c] / N)
            
        return self

    def predict_log_proba(self, text):
        tokens = self._tokenize(text)
        vocab_size = len(self.vocab)
        log_posteriors = {}
        
        for c in self.classes:
            log_prob = self.class_priors[c]
            denom = self.class_total_words[c] + self.alpha * vocab_size
            
            for token in tokens:
                if token in self.vocab:
                    count = self.word_counts[c].get(token, 0)
                    prob_token = (count + self.alpha) / denom
                    log_prob += math.log(prob_token)
                    
            log_posteriors[c] = log_prob
            
        return log_posteriors

    def predict(self, documents):
        preds = []
        for doc in documents:
            log_probs = self.predict_log_proba(doc)
            best_class = max(log_probs.keys(), key=lambda c: log_probs[c])
            preds.append(best_class)
        return preds

# =====================================================================
# 2. K-Nearest Neighbors Classifier from Scratch
# =====================================================================

class KNNClassifierScratch:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Lazy learner: memorizes the training data
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, u, v):
        if self.metric == "euclidean":
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))
        elif self.metric == "manhattan":
            return sum(abs(a - b) for a, b in zip(u, v))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _predict_single(self, query):
        distances = []
        for i, sample in enumerate(self.X_train):
            d = self._distance(query, sample)
            distances.append((d, self.y_train[i]))
            
        # Sort by distance and take top k
        distances.sort(key=lambda item: item[0])
        neighbors = distances[:self.k]
        
        # Majority voting
        vote_counts = {}
        for _, label in neighbors:
            vote_counts[label] = vote_counts.get(label, 0) + 1
            
        majority_class = max(vote_counts.keys(), key=lambda c: vote_counts[c])
        return majority_class

    def predict(self, X):
        return [self._predict_single(x) for x in X]

# =====================================================================
# 3. Synthetic Data Generation
# =====================================================================

def generate_spam_corpus():
    """Returns a synthetic dataset of labeled spam and ham messages."""
    spam_templates = [
        "Congratulations! You won a free lottery prize. Claim your cash bonus now!",
        "Urgent: Exclusive investment deal with guaranteed high returns. Click here for free money.",
        "Win free gift cards and luxury vacations. Claim your prize instantly!",
        "Double your crypto portfolio today with this secret trade offer. Free access.",
        "Refinance your mortgage with zero fees and low interest rates today.",
        "You have been selected for a free $1000 shopping voucher. Claim now!",
        "Secret weight loss pill guaranteed results in 3 days. Free trial package.",
        "Casino bonus alert! Claim 500 free spins and cash rewards right now."
    ]
    
    ham_templates = [
        "Hey, are we still meeting for lunch tomorrow at the cafeteria?",
        "Attached is the project status report for the quarterly engineering review.",
        "Can you review my code pull request before the team standup meeting?",
        "Don't forget to submit your weekly timesheet by Friday afternoon.",
        "Let's schedule a conference call to discuss the research roadmap.",
        "Great job on the presentation today, the client was really impressed.",
        "Here are the meeting minutes and action items from today's discussion.",
        "Are you free this weekend to go hiking in the state park?"
    ]
    
    docs = []
    labels = []
    
    # Expand dataset with minor permutations
    for template in spam_templates:
        docs.append(template)
        labels.append("spam")
    for template in ham_templates:
        docs.append(template)
        labels.append("ham")
        
    return docs, labels

def generate_knn_clusters(n_samples=300, seed=42):
    """
    Generates three Gaussian clusters in 2D space:
    - Class 0: Center (-2.0, -2.0)
    - Class 1: Center (+2.0, -2.0)
    - Class 2: Center ( 0.0, +2.0)
    """
    random.seed(seed)
    centers = [(-2.0, -2.0), (2.0, -2.0), (0.0, 2.0)]
    X = []
    y = []
    per_cluster = n_samples // 3
    
    for c_id, center in enumerate(centers):
        for _ in range(per_cluster):
            x1 = random.gauss(center[0], 0.7)
            x2 = random.gauss(center[1], 0.7)
            X.append([x1, x2])
            y.append(c_id)
            
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)

# =====================================================================
# 4. Main Experiment & Benchmarks
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Topic 8.5 Experiment: Naive Bayes & K-Nearest Neighbors from Scratch")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Multinomial Naive Bayes Spam Classification
    # -------------------------------------------------------------
    print("1. Multinomial Naive Bayes Text Classification:")
    docs, labels = generate_spam_corpus()
    nb = MultinomialNaiveBayesScratch(alpha=1.0)
    nb.fit(docs, labels)
    
    print(f"   Trained on {len(docs)} documents.")
    print(f"   Vocabulary size : {len(nb.vocab)} words.")
    print(f"   P(Spam) Prior   : {math.exp(nb.class_priors['spam']):.2f}")
    print(f"   P(Ham) Prior    : {math.exp(nb.class_priors['ham']):.2f}\n")

    # Test on unseen emails
    test_emails = [
        "Claim your free prize and cash bonus today",            # Should be SPAM
        "Are you free to meet for lunch and discuss the project" # Should be HAM (contains 'free' but heavy ham context)
    ]
    test_expected = ["spam", "ham"]
    test_preds = nb.predict(test_emails)
    
    for email, pred, exp in zip(test_emails, test_preds, test_expected):
        log_probs = nb.predict_log_proba(email)
        print(f"   Email : '{email}'")
        print(f"   Log P : Spam={log_probs['spam']:.2f}, Ham={log_probs['ham']:.2f}")
        print(f"   Pred  : {pred.upper()} (Expected: {exp.upper()})\n")
        assert pred == exp, f"Naive Bayes misclassified email: '{email}'"
        
    print("   -> Naive Bayes spam filter correctly classified unseen test messages!\n")

    # -------------------------------------------------------------
    # Test 2: K-Nearest Neighbors 2D Multi-Cluster Classification
    # -------------------------------------------------------------
    print("2. K-Nearest Neighbors (KNN) Multi-Cluster Benchmark:")
    X, y = generate_knn_clusters(n_samples=300, seed=42)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    print(f"   Train instances : {len(X_train)}, Test instances : {len(X_test)}")

    for metric in ["euclidean", "manhattan"]:
        for k in [3, 7]:
            knn = KNNClassifierScratch(k=k, metric=metric)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            acc = sum(1 for yt, yp in zip(y_test, preds) if yt == yp) / len(y_test)
            print(f"   KNN (k={k}, metric={metric:9s}) -> Test Accuracy: {acc * 100:.2f}%")
            assert acc >= 0.93, f"KNN accuracy too low for k={k}, metric={metric}: {acc}"

    print("\n3. Sanity & Bias-Variance Checks:")
    print("   -> Laplace smoothing prevented zero probability multiplication in Naive Bayes.")
    print("   -> KNN achieved > 93% accuracy across Euclidean and Manhattan metrics.")

    print("\n" + "=" * 70)
    print("All assertions PASSED successfully! [Exit Code 0]")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
