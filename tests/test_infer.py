import unittest
from functools import partial
import numpy as np
from sfm.generate import RandomSFM, RandomLinear, RandomCongruence, plot_dag
from sfm.inference import delta_encode, delta_decode,\
    vfi, cfi
from sfm.partial import partial_vfi, partial_cfi


class MyTestCase(unittest.TestCase):
    def test_delta_compression(self):
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
            self.assertEqual(len(delta_encode(d1, w0=d2)), sum(~mask))
            self.assertEqual(len(delta_encode(d2, w0=d1)), sum(~mask))
            # contrastive encoding and decoding should be reverse operations
            self.assertEqual(delta_decode(delta_encode(d2, w0=d1), w0=d1), d2)
            self.assertEqual(delta_decode(delta_encode(d1, w0=d2), w0=d2), d1)

    def test_vfi(self):
        n_cases = 10
        for test_case in range(n_cases):
            sfm = RandomSFM(20, 0.5, RandomLinear)
            w_exo = {u: np.random.randn() for u in sfm.exo_nodes}
            w = vfi(sfm, w_exo)
            self.assertTrue(sfm.satisfied_by(w))

    def test_cfi(self):
        n_cases = 10
        m = 5   # use congruence for non-injective functions
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
            w_ref = vfi(sfm, w_exo_1)
            # ground truth using vanilla forward inference
            w_expected = vfi(sfm, delta_decode(w_exo_2, w0=w_exo_1))
            # result from contrastive forward inference
            w_actual = cfi(sfm, w0=w_ref, w1_changed_exo=w_exo_2)
            self.assertEqual(w_expected, w_actual)
            # plot_dag(sfm.graph)

    def test_partial_vfi(self):
        # to test partial forward inference, the ground truth is generated by total forward inference
        n_cases = 10
        n_nodes = 20
        for test_case in range(n_cases):
            sfm = RandomSFM(n_nodes, 0.5, RandomLinear)
            w_exo = {u: np.random.randn() for u in sfm.exo_nodes}
            w_total = vfi(sfm, w_exo)
            # prepare partial inference
            target_size = np.random.randint(1, n_nodes + 1)
            print(f"{target_size=}")
            targets = np.random.choice(sfm.graph.nodes, size=target_size, replace=False)
            # ground truth w_partial
            expected = {u: w_total[u] for u in targets}
            actual = partial_vfi(sfm, w_exo=w_exo, target_nodes=targets)
            self.assertEqual(expected, actual)

    @staticmethod
    def tweak_exo(w0_exo: dict, prob_changed_exo: float, congruence_mod: int):
        # change some exo-nodes' values
        # domains are integer; structural functions are linear congruences
        w1_exo = w0_exo.copy()
        # number of changed nodes
        n_changed_exo = int(prob_changed_exo * len(w0_exo))
        print(f"{n_changed_exo=}")
        # randomly select changed nodes
        changed_exo = np.random.choice(list(w0_exo.keys()), size=n_changed_exo, replace=False)
        # assign new values using random integers
        for u in changed_exo:
            # change the valuation of the selected nodes
            new_exo_vals = set(range(congruence_mod))
            new_exo_vals.remove(w0_exo[u])
            w1_exo[u] = int(np.random.choice(list(new_exo_vals), size=1))
        # returns complete assignment (use delta_encode later to get w1_exo_change)
        return w1_exo

    def test_random_graph(self):
        for i in range(100):
            n = np.random.randint(1, 100)
            p = np.random.rand()
            from sfm.generate import random_dag
            self.assertEqual(len(random_dag(n, p).nodes), n)

    def test_partial_cfi_1(self):
        n_cases = 50
        m = 5   # congruence mod
        prob_changed_exo = 0.5
        n_nodes = 20
        p = 0.2
        for test_case in range(n_cases):
            print("-" * 50)
            sfm = RandomSFM(n_nodes, p, partial(RandomCongruence, m=m))
            w0_exo = {u: int(np.random.randint(0, m)) for u in sfm.exo_nodes}
            w1_exo = self.tweak_exo(w0_exo, prob_changed_exo=prob_changed_exo, congruence_mod=m)
            w1_changed_exo = delta_encode(w1=w1_exo, w0=w0_exo)

            w0 = vfi(sfm, w0_exo)
            # ground truth using vanilla forward inference
            w1_vfi = vfi(sfm, w1_exo)
            # result from contrastive forward inference
            w1_cfi_changed_exo = cfi(sfm, w0=w0, w1_changed_exo=w1_changed_exo)
            w1_cfi_full_exo = cfi(sfm, w0=w0, w1_changed_exo=w1_exo)
            self.assertEqual(w1_vfi, w1_cfi_full_exo)
            self.assertEqual(w1_vfi, w1_cfi_changed_exo)

            # plot_dag(sfm.graph)

            # select targets of interest
            target_size = np.random.randint(1, n_nodes + 1)
            print(f"target size: {target_size}/{sfm.graph.number_of_nodes()}")
            targets = np.random.choice(sfm.graph.nodes, size=target_size, replace=False)
            # ground truth w_targets
            w1t_vfi = {u: w1_vfi[u] for u in targets}
            # result from partial vfi
            w1t_partial_vfi = partial_vfi(sfm, w_exo=w1_exo, target_nodes=targets)
            self.assertEqual(w1t_vfi, w1t_partial_vfi)
            # result from partial cfi
            w1t_partial_cfi = partial_cfi(sfm, w0=w0, w1_c_exo=w1_changed_exo, target_nodes=targets)
            self.assertEqual(w1t_vfi, w1t_partial_cfi)



if __name__ == '__main__':
    unittest.main()
