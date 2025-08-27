# utils/bpe.py
# ======================================
# Byte Pair Encoding (BPE) tokenizer
# - Learn merges on normalized train text
# - Build/freeze vocab from train
# - Encode val/test with <unk>
# - Save/load merges + vocab
# ======================================

import re
import json
import unicodedata
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>"]

class BPETokenizer:
    def __init__(self, num_merges: int = 500):
        self.num_merges = num_merges
        self.merges = []  # List[Tuple[str, str]]
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._frozen: bool = False  # prevent val/test from expanding vocab

    # -------- Normalization --------
    def normalize(self, text: str) -> str:
        # NFKD + ASCII fold + lowercase + digits -> <num> + strip punctuation except spaces/word chars
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = text.lower()
        text = re.sub(r"\d+", "<num>", text)
        text = re.sub(r"[^\w\s]", " ", text)     # remove punctuation (you can relax this if desired)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -------- BPE Learning --------
    def _get_vocab(self, text: str) -> Dict[str, int]:
        vocab = defaultdict(int)
        words = re.findall(r"\b\w+\b", text)
        for w in words:
            token = " ".join(list(w)) + " </w>"
            vocab[token] += 1
        return vocab

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        pattern = re.escape(" ".join(pair))
        replacement = "".join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = re.sub(rf"(?<!\S){pattern}(?!\S)", replacement, word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def learn_bpe(self, text: str):
        """Learn merges from normalized train text"""
        text = self.normalize(text)
        vocab = self._get_vocab(text)
        self.merges = []
        for _ in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            vocab = self._merge_vocab(best, vocab)

    # -------- Segmentation --------
    def segment_word(self, word: str) -> List[str]:
        symbols = list(word) + ["</w>"]
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            merge_found = False
            for merge in self.merges:
                if merge in pairs:
                    j = pairs.index(merge)
                    symbols = symbols[:j] + ["".join(merge)] + symbols[j + 2:]
                    merge_found = True
                    break
            if not merge_found:
                break
        return symbols

    def segment_text(self, text: str) -> List[List[str]]:
        text = self.normalize(text)
        words = re.findall(r"\b\w+\b", text)
        return [self.segment_word(w) for w in words]

    # -------- Vocab Build / Freeze --------
    def build_vocab(self, train_text: str):
        """Populate token2id/id2token from the TRAIN text (frozen after)."""
        self.token2id.clear()
        self.id2token.clear()
        # add specials first with stable ids
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)

        segmented = self.segment_text(train_text)
        for word_toks in segmented:
            for t in word_toks:
                self._add_token(t)
        self._frozen = True  # lock after building from train

    def _add_token(self, tok: str) -> int:
        if tok not in self.token2id:
            idx = len(self.token2id)
            self.token2id[tok] = idx
            self.id2token[idx] = tok
        return self.token2id[tok]

    # -------- Encode / Decode --------
    def encode_tokens(self, tokens: Iterable[str], allow_new: bool = False) -> List[int]:
        ids = []
        for t in tokens:
            if t in self.token2id:
                ids.append(self.token2id[t])
            else:
                if allow_new and not self._frozen:
                    ids.append(self._add_token(t))
                else:
                    ids.append(self.token2id["<unk>"])
        return ids

    def encode(self, text: str, add_bos_eos: bool = False, allow_new: bool = False) -> List[int]:
        """Return FLAT list of token IDs."""
        segmented = self.segment_text(text)
        flat = []
        if add_bos_eos:
            flat.append(self.token2id["<bos>"])
        for word_toks in segmented:
            flat.extend(word_toks)
        if add_bos_eos:
            flat.append(self.token2id["<eos>"])
        return self.encode_tokens(flat, allow_new=allow_new)

    def decode(self, ids: List[int]) -> str:
        toks = [self.id2token.get(i, "<unk>") for i in ids]
        # remove </w>, keep words together
        text = "".join(tok.replace("</w>", " ") for tok in toks)
        return re.sub(r"\s+", " ", text).strip()

    def decode_token_id(self, i: int) -> str:
        return self.id2token.get(i, "<unk>")

    # -------- Save / Load --------
    def save(self, path_json: str):
        data = {
            "num_merges": self.num_merges,
            "merges": self.merges,
            "token2id": self.token2id,
            "id2token": {int(k): v for k, v in self.id2token.items()},
            "frozen": self._frozen,
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path_json: str):
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.num_merges = int(data["num_merges"])
        self.merges = [tuple(p) for p in data["merges"]]
        self.token2id = {k: v for k, v in data["token2id"].items()}
        self.id2token = {int(k): v for k, v in data["id2token"].items()}
        self._frozen = bool(data.get("frozen", True))
