import numpy as np
from mtgat.connectivity import plv_matrix

def test_plv_shape_bounds_symmetry():
    x=np.random.default_rng(2).normal(size=(4,1024))
    p=plv_matrix(x,256,{"alpha":(8,13),"beta":(13,30)})
    assert p.shape==(2,4,4)
    assert np.all((p>=0)&(p<=1))
    assert np.allclose(p,p.swapaxes(-1,-2),atol=1e-5)
