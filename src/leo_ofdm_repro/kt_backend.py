"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
from . import kt_lvd as r
BATCH=32


class Backend:

    def stage(self, c, v, ordinal, j, k):
        import torch
        with torch.no_grad():
            q = r.infer(c[ordinal * 20000:(ordinal + 1) * 20000], v[ordinal * 200:(ordinal + 1) * 200], nt=4000, batch=BATCH, time_transform='matrix')
        p = np.asarray(q['point'])
        return dict(j=j, k=k, point=p, rows=p[None], trace=None, score=None, processing_s=q['processing_s'])
