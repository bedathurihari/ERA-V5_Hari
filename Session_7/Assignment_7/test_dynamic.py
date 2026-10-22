import unittest
from dynamic_kronecker import ALPHABET, encode, decode, dense, fixed32

class TestEncoding(unittest.TestCase):
    def test_round_trip(self):
        for n in (0, 1, 31, 32, 33, 64, 128, 1024):
            s = "".join(ALPHABET[i % len(ALPHABET)] for i in range(n))
            self.assertEqual(decode(encode(s), len(s)), s)

    def test_dense_equivalence(self):
        s = "apple 123"
        z = dense(s).view(len(s), len(ALPHABET))
        self.assertEqual(z.nonzero().tolist(), [list(p) for p in encode(s)])

    def test_fixed_collision(self):
        x, y = "z" * 32 + "a", "z" * 32 + "b"
        self.assertEqual(fixed32(x), fixed32(y))
        self.assertNotEqual(encode(x), encode(y))

    def test_reject_invalid(self):
        with self.assertRaises(ValueError):
            encode("A")
        with self.assertRaises(ValueError):
            decode([(0,0), (0,1)], 2)

if __name__ == "__main__":
    unittest.main()
