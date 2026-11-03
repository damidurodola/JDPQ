"""Write tests/fixture.npz with scipy.sparse.save_npz (uncompressed), the same
writer the converter uses, so the C++ reader is tested against the real format."""
import sys
import numpy as np, scipy.sparse as sp
X = sp.csr_matrix(np.array([[0.5, 0, 1e-10, 0], [0, 0, 0, 2.25], [1, 2, 3, 4]], dtype=np.float64))
X = sp.csr_matrix((X.data, X.indices.astype(np.int32), X.indptr.astype(np.int64)), shape=X.shape)
sp.save_npz(sys.argv[1], X, compressed=False)
