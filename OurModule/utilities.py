import typing as tp
from sklearn.preprocessing import OneHotEncoder


def oneHotEncoder(genome):
    enc = OneHotEncoder(sparse=False)
    enc.fit(np.array(genome).reshape(-1,1))
    # enc.categories_
    return enc.transform(np.array(genome).reshape(-1,1)), enc.categories_