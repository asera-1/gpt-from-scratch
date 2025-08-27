# utils/ngram.py
# ======================================
# N-gram language model over BPE tokens (strings)
# - Laplace smoothing (k)
# - Generic interpolation for n=1..4
# - Perplexity
# - Generation with stop at </s> or max length
# - Save/load counts
# ======================================

import math
import json
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

class NGramModel:
    def __init__(self, n: int = 3):
        assert 1 <= n <= 4, "Supported n in [1..4]"
        self.n = n
        self.ngram_counts = [defaultdict(int) for _ in range(n)]   # index 0=unigram, 1=bigram, ...
        self.context_counts = [defaultdict(int) for _ in range(n)]
        self.vocab = set()

    # -------- Train --------
    def train(self, sequences: List[List[str]]):
        # sequences: each is a list of tokens, we prepend <s> * (n-1), append </s>
        for seq in sequences:
            tokens = ["<s>"] * (self.n - 1) + list(seq) + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens)):
                for order in range(1, self.n + 1):
                    if i - order + 1 < 0:
                        continue
                    ngram = tuple(tokens[i - order + 1 : i + 1])
                    context = ngram[:-1]
                    self.ngram_counts[order - 1][ngram] += 1
                    self.context_counts[order - 1][context] += 1

    # -------- Probabilities --------
    def laplace_prob(self, ngram: Tuple[str, ...], k: float = 1.0) -> float:
        order = len(ngram)
        context = ngram[:-1]
        V = len(self.vocab)
        c_ng = self.ngram_counts[order - 1][ngram]
        c_ctx = self.context_counts[order - 1][context]
        return (c_ng + k) / (c_ctx + k * V)

    def interpolated_prob(self, context: Tuple[str, ...], word: str, lambdas: List[float], k: float = 1.0) -> float:
        """
        Generic interpolation for n=1..self.n using Laplace-smoothed components.
        context: tuple of previous tokens (length up to n-1)
        word:    target token
        lambdas: weights for [unigram, bigram, trigram, 4gram] (truncate to n)
        """
        L = min(self.n, len(lambdas))
        weights = lambdas[:L]
        s = sum(weights)
        if s <= 0:
            # uniform fallback over available orders
            weights = [1.0 / L] * L
        else:
            weights = [w / s for w in weights]

        # assemble n-grams of different orders
        probs = []
        for order in range(1, L + 1):
            ctx = tuple(context[-(order - 1):]) if order > 1 else tuple()
            ng = ctx + (word,)
            probs.append(self.laplace_prob(ng, k=k))
        return sum(w * p for w, p in zip(weights, probs))

    # -------- Perplexity --------
    def perplexity(self, sequences: List[List[str]], lambdas: List[float], k: float = 1.0) -> float:
        log_sum = 0.0
        token_count = 0
        for seq in sequences:
            tokens = ["<s>"] * (self.n - 1) + list(seq) + ["</s>"]
            for i in range(self.n - 1, len(tokens)):
                ctx = tuple(tokens[i - (self.n - 1) : i])
                w = tokens[i]
                p = self.interpolated_prob(ctx, w, lambdas=lambdas, k=k)
                log_sum += -math.log(max(p, 1e-12))
                token_count += 1
        return math.exp(log_sum / max(token_count, 1))

    # -------- Generation --------
    def generate(self, context: List[str] = None, max_tokens: int = 30, method: str = "sample",
                 lambdas: List[float] = None, k: float = 1.0) -> List[str]:
        if context is None:
            context = ["<s>"] * (self.n - 1)
        else:
            context = ["<s>"] * max(0, self.n - 1 - len(context)) + context[: self.n - 1]
        out = []

        if lambdas is None:
            lambdas = [0.1, 0.3, 0.6, 0.0]  # reasonable default for up to 4-gram

        for _ in range(max_tokens):
            candidates = list(self.vocab)
            # Remove boundary markers from candidates if desired
            if "</s>" in candidates:
                candidates.remove("</s>")
            if "<s>" in candidates:
                candidates.remove("<s>")

            scores = []
            for tok in candidates:
                p = self.interpolated_prob(tuple(context[-(self.n - 1):]), tok, lambdas=lambdas, k=k)
                scores.append(p)

            if not scores:
                break

            if method == "argmax":
                idx = max(range(len(scores)), key=lambda i: scores[i])
                nxt = candidates[idx]
            else:
                # normalize and sample
                s = sum(scores)
                probs = [sc / s for sc in scores]
                nxt = random.choices(candidates, weights=probs, k=1)[0]

            if nxt == "</s>":
                break
            out.append(nxt)
            context.append(nxt)

        return out

    # -------- Save/Load --------
    def save(self, path_json: str):
        data = {
            "n": self.n,
            "ngram_counts": [{",".join(k): v for k, v in d.items()} for d in self.ngram_counts],
            "context_counts": [{",".join(k): v for k, v in d.items()} for d in self.context_counts],
            "vocab": sorted(list(self.vocab)),
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path_json: str):
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.n = int(data["n"])
        self.ngram_counts = [defaultdict(int) for _ in range(self.n)]
        self.context_counts = [defaultdict(int) for _ in range(self.n)]
        for i, d in enumerate(data["ngram_counts"]):
            for k, v in d.items():
                tup = tuple(k.split(",")) if k else tuple()
                self.ngram_counts[i][tup] = int(v)
        for i, d in enumerate(data["context_counts"]):
            for k, v in d.items():
                tup = tuple(k.split(",")) if k else tuple()
                self.context_counts[i][tup] = int(v)
        self.vocab = set(data["vocab"])
