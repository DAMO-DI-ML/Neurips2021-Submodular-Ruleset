import sys
import os
import tempfile
import subprocess as sp

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

binpath = os.path.dirname(os.path.abspath(__file__)) + '/bin/fastrule-linux-amd64'

class SubmodularRuleSet(BaseEstimator):
    def __init__(
        self, max_num_rules: int=16, time_limit=60,
        beta_pos=1.0, beta_neg=1.0, beta_diverse=0.1, beta_complex=0.1,
        parallelism=0, warmcache=0, bestsubset=0, exactdepth=0, allowrandom=0,
        verbose=False,
    ):
        self.max_num_rules = max_num_rules
        self.time_limit = time_limit
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
        self.beta_diverse = beta_diverse
        self.beta_complex = beta_complex
        self.parallelism = parallelism
        self.warmcache = warmcache
        self.bestsubset = bestsubset
        self.exactdepth = exactdepth
        self.allowrandom = allowrandom
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        data = np.concatenate((X, y[..., np.newaxis]), axis=-1)
        columns = ['f' + str(i) for i in range(X.shape[-1])]
        columns.append('label')
        df = pd.DataFrame(data, columns=columns)

        with tempfile.TemporaryFile(mode = "w+") as tmp:
            df.to_csv(tmp, index=False)
            tmp.flush()
            tmp.seek(0)
            stderr = sys.stderr.fileno() if self.verbose else sp.DEVNULL
            proc = sp.run([
                binpath,
                '-p', str(self.parallelism),
                '-d', '-',
                '-l', 'label',
                '-o', 'i',
                '-k', str(self.max_num_rules),
                '-t', str(self.time_limit) + 's',
                '-c', str(self.warmcache),
                '-b', str(self.bestsubset),
                '-e', str(self.exactdepth),
                '-r', str(self.allowrandom),
                '-pos', str(self.beta_pos),
                '-neg', str(self.beta_neg),
                '-complex', str(self.beta_complex),
                '-diverse', str(self.beta_diverse),
            ], stdin=tmp, stdout=sp.PIPE, stderr=stderr, check=True)
            lines = proc.stdout.decode("utf-8").splitlines()

        self.objval = float(lines[-1])
        self.itemsets = [
            [int(field) for field in line.split(' ')]
            if line.strip() != '' else []
            for line in lines[:-1]
        ]

    def predict(self, X: np.ndarray):
        if len(self.itemsets) == 0:
            return np.zeros(X.shape[0], dtype=int)
        predictions = [
            np.prod(X[..., itemset], axis=-1)
            for itemset in self.itemsets
        ]
        return np.greater(np.sum(predictions, axis=0), 0).astype(int)
    
    def overlap_on(self, X: np.ndarray):
        if len(self.itemsets) == 0:
            return 0
        predictions = [
            np.prod(X[..., itemset], axis=-1)
            for itemset in self.itemsets
        ]
        sump = np.sum(predictions, axis=0)
        pred = np.greater(sump, 0).astype(int)
        return np.sum(sump - pred).item() / X.shape[0]
