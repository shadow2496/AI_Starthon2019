import os

import faiss
import numpy as np


class BaseKNN(object):
    def __init__(self, database, method):
        if database.dtype != np.float32:
            database = database.astype(np.float32)
        self.N = len(database)
        self.D = database[0].shape[-1]
        self.database = database if database.flags['C_CONTIGUOUS'] \
            else np.ascontiguousarray(database)

    def add(self):
        self.index.add(self.database)

    def search(self, queries, k):
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)
        sims, ids = self.index.search(queries, k)
        return sims, ids


class KNN(BaseKNN):
    def __init__(self, database, method):
        super().__init__(database, method)
        self.index = {'cosine': faiss.IndexFlatIP,
                      'euclidean': faiss.IndexFlatL2}[method](self.D)
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.add()
