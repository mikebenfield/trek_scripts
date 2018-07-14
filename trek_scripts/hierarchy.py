import numpy as np


def _classify(words, indices):
    """Classify the word vectors into two classes.

    `words`: a np array of shape `(word_count, dim)`
    `indices`: a np array of dtype int_, indicating which indices
        of `words` we are want to classify

    Returns
    `(class1_indices, class2_indices)`, which are each np arrays
        of dtype `int_`
    """

    array = words[indices]

    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(
        n_components=2, covariance_type='spherical', max_iter=20)

    gm.fit(array)

    predictions = gm.predict(array)

    class0 = predictions == 0
    class1 = predictions == 1

    return indices[class0], indices[class1]


def _word_hierarchy(words, indices, result_list):
    if len(indices) == 0:
        raise Exception("Can't happen")
    elif len(indices) == 1:
        return
    class0, class1 = _classify(words, indices)

    for index in class0:
        result_list[index].append(0)
    for index in class1:
        result_list[index].append(1)
    _word_hierarchy(words, class0, result_list)
    _word_hierarchy(words, class1, result_list)


def word_hierarchy(words):
    indices = np.arange(len(words), dtype=np.int_)
    result_list = [[] for _ in indices]
    _word_hierarchy(words, indices, result_list)
    return result_list


def read_hierarchy_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    result = [line.split()[1:] for line in lines]
    for i, strings in enumerate(result):
        result[i] = [int(s) for s in strings if s]
    return result
