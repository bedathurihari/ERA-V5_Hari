"""Lossless variable-length sparse Kronecker coordinates."""
import string
import torch

ALPHABET = string.ascii_lowercase + string.digits + " "
TO_ID = {c: i for i, c in enumerate(ALPHABET)}

def encode(text):
    if any(c not in TO_ID for c in text):
        raise ValueError("Unsupported character")
    return [(i, TO_ID[c]) for i, c in enumerate(text)]

def decode(coords, length):
    if length < 0 or len(coords) != length:
        raise ValueError("Expected one coordinate per character")
    chars = [None] * length
    for i, j in coords:
        if not (0 <= i < length and 0 <= j < len(ALPHABET)) or chars[i] is not None:
            raise ValueError("Duplicate or invalid coordinate")
        chars[i] = ALPHABET[j]
    return "".join(chars)

def dense(text):
    out = torch.zeros(len(text), len(ALPHABET))
    for i, j in encode(text):
        out[i, j] = 1
    return out.flatten()

def fixed32(text):
    return text[:32].ljust(32, " ")

def run_proofs():
    for n in (0, 1, 31, 32, 33, 64, 128, 1024):
        s = "".join(ALPHABET[(i * 7 + n) % len(ALPHABET)] for i in range(n))
        assert decode(encode(s), len(s)) == s
        assert int(dense(s).sum()) == n
    x, y = "a" * 32 + "a", "a" * 32 + "b"
    assert fixed32(x) == fixed32(y)
    assert encode(x) != encode(y)
    print("PASS: exact reconstruction up to 1024; fixed-32 collision avoided")

if __name__ == "__main__":
    run_proofs()
