import unittest
import numpy as np
from sfm.inference import contrast_encode, contrast_decode


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


if __name__ == '__main__':
    unittest.main()
