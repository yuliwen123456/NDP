import unittest
import numpy as np
import torch
from prototype_module import PrototypeAccumulator, Moments, prototype_scores, fuse


class PrototypeTests(unittest.TestCase):
    def test_original_point_gt_weighting_and_missing_class(self):
        features = torch.tensor([[2., 0.], [0., 4.]])
        inverse = torch.tensor([0, 0, 1, 1, 0])
        labels = torch.tensor([0, 1, 1, -1, 19])
        acc = PrototypeAccumulator(3, 2, 'cpu'); acc.update(features, inverse, labels)
        bank = acc.finish()
        self.assertEqual(bank['class_counts'].tolist(), [1, 2, 0])
        self.assertEqual(bank['valid_mask'].tolist(), [True, True, False])
        torch.testing.assert_close(bank['prototypes'][1], torch.tensor([1., 2.])/np.sqrt(5))
        p, nearest, sim = prototype_scores(torch.tensor([[-1., 0.], [1., 2.]]), bank)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.all(nearest != 2))  # Zero row cannot win against negative similarities.
        self.assertAlmostEqual(float(p[1]), 0., places=6)

    def test_moments_and_fusion(self):
        values = np.array([-3., -2., 10., 12.])
        m = Moments(); m.update(values[:1]); m.update(values[1:]); stats = m.result()
        self.assertAlmostEqual(stats['mean'], values.mean())
        self.assertAlmostEqual(stats['std'], values.std())
        both = {'S_ndp': stats, 'S_proto': stats}
        np.testing.assert_allclose(fuse(values, values, both, .5, 1e-8), (values-values.mean())/(values.std()+1e-8))
        np.testing.assert_allclose(fuse(values, values+100, both, 0, 1e-8), (values-values.mean())/(values.std()+1e-8))

    def test_empty_and_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            PrototypeAccumulator(2, 3, 'cpu').finish()
        with self.assertRaises(ValueError):
            Moments().update([float('nan')])
        with self.assertRaises(ValueError):
            fuse([0], [0], {}, 1.1, 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
