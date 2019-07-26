import os
import time

import numpy as np
import joblib
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm

from knn import KNN


trunc_ids = None
trunc_init = None
lap_alpha = None


def cache(filename):
    """Decorator to cache results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            path = os.path.join(self.cache_dir, filename)
            time0 = time.time()

            if os.path.exists(path):
                result = joblib.load(path)
                cost = time.time() - time0
                print("[cache] loading {} costs {:.2f}s".format(path, cost))
                return result

            result = func(*args, **kwargs)
            cost = time.time() - time0
            print("[cache] obtaining {} costs {:.2f}s".format(path, cost))
            joblib.dump(result, path)
            return result
        return wrapper
    return decorator


def get_offline_result(i):
    ids = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
    return scores


class Diffusion(object):
    def __init__(self, features, cache_dir):
        self.features = features
        self.N = len(self.features)
        self.cache_dir = cache_dir
        self.knn = KNN(self.features, method='cosine')

    @cache('offline.jbl')
    def get_offline_results(self, n_trunc, kd=50):
        """Get offline diffusion results for each feature"""
        print("[offline] starting offline diffusion")
        print("[offline] 1) prepare Laplacian and initial state")
        global trunc_ids, trunc_init, lap_alpha
        sims, ids = self.knn.search(self.features, n_trunc)
        trunc_ids = ids
        trunc_init = np.zeros(n_trunc)
        trunc_init[0] = 1
        lap_alpha = self.get_laplacian(sims[:, :kd], ids[:, :kd])

        print("[offline] 2) gallery-side diffusion")
        results = Parallel(n_jobs=-1, prefer='threads')(delayed(get_offline_result)(i)
                                                        for i in tqdm(range(self.N), desc='[offline] diffusion'))
        all_scores = np.concatenate(results)

        print("[offline] 3) merge offline results")
        rows = np.repeat(np.arange(self.N), n_trunc)
        offline = sparse.csr_matrix((all_scores, (rows, trunc_ids.reshape(-1))), shape=(self.N, self.N), dtype=np.float32)
        return offline

    def get_laplacian(self, sims, ids, alpha=0.99):
        affinity = self.get_affinity(sims, ids)

        n = affinity.shape[0]
        degrees = affinity @ np.ones(n) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix((degrees ** (-0.5), [0]), shape=(n, n), dtype=np.float32)
        stochastic = mat @ affinity @ mat

        sparse_eye = sparse.dia_matrix((np.ones(n), [0]), shape=(n, n), dtype=np.float32)
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    def get_affinity(self, sims, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset"""
        n = sims.shape[0]
        sims[sims < 0] = 0
        sims = sims ** gamma

        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(n):
            # Check reciprocity: i is in j's kNN and j is in i's kNN
            is_mutual = np.isin(ids[ids[i]], i).any(axis=1)
            if is_mutual.any():
                vec_ids.append(i * np.ones(is_mutual.sum(), dtype=int))
                mut_ids.append(ids[i, is_mutual])
                mut_sims.append(sims[i, is_mutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)), shape=(n, n), dtype=np.float32)
        affinity[range(n), range(n)] = 0
        return affinity
