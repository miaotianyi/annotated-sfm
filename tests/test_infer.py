import unittest
from functools import partial
import numpy as np
from sfm.generate import RandomSFM, RandomLinear, RandomCongruence, plot_dag
from sfm.inference import contrast_encode, contrast_decode,\
    vanilla_forward_infer, contrastive_forward_infer


class MyTestCase(unittest.TestCase):
    def test_contrast_coding(self):
        num_nodes = 10
        num_cases = 10
        for test_case in range(num_cases):
            key, val_1, val_2 = np.random.rand(3, num_nodes)
            # some values of val_2 are the same as those in val_1
            # so the contrast is nontrivial
            mask = np.random.rand(num_nodes) > 0.5
            val_2[mask] = val_1[mask]
            d1 = dict(zip(key, val_1))
            d2 = dict(zip(key, val_2))
            # contrastive encoding should only keep nodes with different valuation
            self.assertEqual(len(contrast_encode(d1, w_ref=d2)), sum(~mask))
            self.assertEqual(len(contrast_encode(d2, w_ref=d1)), sum(~mask))
            # contrastive encoding and decoding should be reverse operations
            self.assertEqual(contrast_decode(contrast_encode(d2, w_ref=d1), w_ref=d1), d2)
            self.assertEqual(contrast_decode(contrast_encode(d1, w_ref=d2), w_ref=d2), d1)

    def test_vanilla_total(self):
        n_cases = 10
        for test_case in range(n_cases):
            sfm = RandomSFM(20, 0.5, RandomLinear)
            w_exo = {u: np.random.randn() for u in sfm.exo_nodes}
            w_total = vanilla_forward_infer(sfm, w_exo)
            self.assertTrue(sfm.satisfied_by(w_total))

    def test_contrast_total(self):
        n_cases = 10
        m = 5
        prob_changed_exo = 0.5
        n = 20
        p = 0.2
        for test_case in range(n_cases):
            sfm = RandomSFM(n, p, partial(RandomCongruence, m=m))
            w_exo_1 = {u: int(np.random.randint(0, m)) for u in sfm.exo_nodes}
            # make a new exogenous valuation that differs in some values
            w_exo_2 = w_exo_1.copy()
            n_changed_exo = int(prob_changed_exo * len(w_exo_1))
            changed_exo = np.random.choice(list(w_exo_1.keys()), size=n_changed_exo, replace=False)
            print(f"{changed_exo=}")
            for u in changed_exo:
                # change the valuation of the selected nodes
                new_exo_vals = set(range(m))
                new_exo_vals.remove(w_exo_1[u])
                w_exo_2[u] = int(np.random.choice(list(new_exo_vals), size=1))
            w_ref = vanilla_forward_infer(sfm, w_exo_1)
            # ground truth using vanilla forward inference
            w_expected = vanilla_forward_infer(sfm, contrast_decode(w_exo_2, w_ref=w_exo_1))
            # result from contrastive forward inference
            w_actual = contrastive_forward_infer(sfm, w_exo_2, w_ref=w_ref)
            self.assertEqual(w_expected, w_actual)
            # plot_dag(sfm.graph)


if __name__ == '__main__':
    unittest.main()
