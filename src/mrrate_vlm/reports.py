import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any


def clean_text(value: Any) -> str:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null", "n/a"}:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def duplicate_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", clean_text(text).casefold()))


def token_audit(tokenizer: Any, text: str, max_length: int = 64) -> dict:
    if not clean_text(text):
        raise ValueError("Report target is empty after cleaning")
    original = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    kept = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)[
        "input_ids"
    ]
    return {
        "original_tokens": len(original),
        "used_tokens": len(kept),
        "truncated": len(original) > max_length,
        "max_tokens": max_length,
    }


def duplicate_diagnostics(texts: list[str], threshold: float = 0.9) -> dict:
    keys = [duplicate_key(t) for t in texts]
    counts = Counter(keys)
    near_pairs = 0
    near_members = set()
    # Pilot-sized evaluation; intentionally O(N^2), no third-party NLP model.
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if (
                keys[i] != keys[j]
                and SequenceMatcher(None, keys[i], keys[j], autojunk=False).ratio() >= threshold
            ):
                near_pairs += 1
                near_members.update((i, j))
    return {
        "n": len(keys),
        "unique_normalized_impressions": len(counts),
        "exact_duplicate_members": sum(v for v in counts.values() if v > 1),
        "exact_duplicate_member_rate": sum(counts[k] > 1 for k in keys) / len(keys) if keys else 0,
        "near_duplicate_pairs_excluding_exact": near_pairs,
        "near_duplicate_members_excluding_exact": len(near_members),
        "near_duplicate_threshold": threshold,
        "near_duplicate_method": "SequenceMatcher on casefolded alphanumeric text",
    }
