"""Sanity tests for the CKNNA implementation in eval_cknna.py. CPU, seconds, no data.

    python tools/test_cknna.py

WHY THIS FILE EXISTS. The first implementation of `cknna` scored 0.975 for INDEPENDENT random
features and 0.978 for a row-shuffled copy -- numbers that would have looked like "our latent
is beautifully aligned with DINOv2" in a results table. It restricted the numerator and both
denominators to the same mutual-neighbour mask; masked pairs are the largest entries in both
kernels by construction, so that ratio tends to 1 regardless of any relationship. Nothing
about the failure was visible from the number alone.

A similarity metric must therefore be pinned to cases whose answer is known independently:
  * identical, rotated and rescaled features -> ~1 (CKA is invariant to rotation and scale)
  * independent features and shuffled rows    -> ~0
  * a partially related pair                  -> strictly between, and ordered correctly
  * the sparse cknna() (used at n=50k)         -> equal to cknna_dense() on every case above
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_cknna import cknna, cknna_dense  # noqa: E402


def main():
    torch.manual_seed(0)
    n, d, k = 512, 64, 10
    x = torch.randn(n, d)

    same = cknna(x, x, topk=k)
    rot = cknna(x, x @ torch.linalg.qr(torch.randn(d, d))[0], topk=k)
    scaled = cknna(x, 3.7 * x, topk=k)
    indep = cknna(x, torch.randn(n, d), topk=k)
    shuffled = cknna(x, x[torch.randperm(n)], topk=k)
    partial = cknna(x, 3.0 * x + torch.randn(n, d), topk=k)

    print(f"identical        {same:+.4f}   expect ~1")
    print(f"rotated          {rot:+.4f}   expect ~1 (rotation invariant)")
    print(f"rescaled         {scaled:+.4f}   expect ~1 (scale invariant)")
    print(f"partially related{partial:+.4f}   expect strictly between")
    print(f"independent      {indep:+.4f}   expect ~0")
    print(f"row-shuffled     {shuffled:+.4f}   expect ~0")

    assert same > 0.99, f"identical features scored {same}"
    assert rot > 0.99, f"a rotation changed the score ({rot}); CKA must be rotation invariant"
    assert scaled > 0.99, f"a rescaling changed the score ({scaled})"
    assert abs(indep) < 0.15, f"independent features scored {indep}; this is THE failure mode"
    assert abs(shuffled) < 0.15, f"row-shuffled features scored {shuffled}"
    assert indep < partial < same, f"ordering broken: {indep} < {partial} < {same}"

    # The sparse path must reproduce the dense reference, not just pass the same sanity bounds.
    cases = {"identical": x, "rotated": x @ torch.linalg.qr(torch.randn(d, d))[0],
             "independent": torch.randn(n, d), "partial": 3.0 * x + torch.randn(n, d),
             "different width": torch.randn(n, 2 * d)}
    for name, y in cases.items():
        for kk in (2, k, 50):
            sp, de = cknna(x, y, topk=kk), cknna_dense(x, y, topk=kk)
            assert abs(sp - de) < 1e-4, f"sparse != dense on {name}, k={kk}: {sp} vs {de}"
    print("sparse cknna matches cknna_dense on all cases")

    print("\nall CKNNA checks passed")


if __name__ == "__main__":
    main()
