import sys
import unittest

import numpy as np

sys.path.append('..')
from graphsage.utils import load_data


class TestUtils(unittest.TestCase):
    def test01_load_data(self):
        from sklearn.preprocessing import StandardScaler
        prefix = '../example_data/toy-ppi'
        G, feats, id_map, walks, class_map = load_data(prefix, normalize=True, load_walks=True)
        G_feats = np.array([G.node[_node]['feature'] for _node in G.nodes()])
        scaler = StandardScaler()
        scaler.fit(G_feats)
        G_feats = scaler.transform(G_feats)
        for i, _node in enumerate(G.nodes()):
            isSame_feat = np.allclose(G_feats[_node], feats[_node], rtol=1e-1, atol=1e-2, equal_nan=False)
            isSame_label = np.array_equal(G.node[_node]['label'], class_map[_node])
            self.assertTrue(isSame_feat, 'feats[%i] is not same' % i)
            self.assertTrue(isSame_feat, 'labels[%i] is not same' % i)