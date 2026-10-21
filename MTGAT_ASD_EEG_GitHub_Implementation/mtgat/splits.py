import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def participant_split(participant_ids, test_size=0.2, val_size=0.1, seed=42):
    ids = np.asarray(participant_ids)
    rows = np.arange(len(ids))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(outer.split(rows, groups=ids))
    relative_val = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    tr, va = next(inner.split(trainval, groups=ids[trainval]))
    train, val = trainval[tr], trainval[va]
    assert set(ids[train]).isdisjoint(ids[val])
    assert set(ids[train]).isdisjoint(ids[test])
    assert set(ids[val]).isdisjoint(ids[test])
    return train, val, test
