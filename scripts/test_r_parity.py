"""R-parity self-check for soc_master.py. Expected values were produced by
R 4.6.1 + caret 7.1.0 with set.seed(42). Run soc_master.py once first (needs output/data_clean.csv).
Usage: python test_r_parity.py"""
import pandas as pd
from soc_master import RRNG, OUT_DIR, create_data_partition

R_SPLIT_80 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28,
              29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 53, 56, 57, 59, 60,
              61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 80]
R_SPLIT_70 = [2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33,
              34, 40, 41, 43, 44, 45, 46, 47, 48, 50, 52, 54, 55, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69,
              70, 71, 72, 73, 74, 75, 77, 80]


def test_rng():
    r = RRNG(42)
    assert [round(r.unif_rand(), 15) for _ in range(3)] == [0.914806043496355, 0.937075413297862, 0.286139534786344]
    assert RRNG(42).sample(range(1, 11), 10) == [1, 5, 10, 8, 2, 4, 6, 9, 7, 3]
    assert RRNG(42).sample(range(37, 101), 7) == [85, 73, 37, 61, 46, 72, 54]


def test_caret():
    y = pd.read_csv(OUT_DIR / "data_clean.csv")["soc"].values
    assert list(create_data_partition(y, 0.8, 42) + 1) == R_SPLIT_80
    assert list(create_data_partition(y, 0.7, 42) + 1) == R_SPLIT_70


if __name__ == "__main__":
    test_rng()
    test_caret()
    print("R parity OK")
