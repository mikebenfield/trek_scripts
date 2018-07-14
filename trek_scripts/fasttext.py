import pathlib

import torch

import trek_scripts.opts as opts


def string_prep(s):
    """Prepare a transcript for fasttext to see it.

    Makes these changes:
    - Start with ' ~ ' and end with ' @ '
    - Put spaces around punctuation, so fasttext will consider
    punctuation as separate words.
    - Replace newlines with hashes, so that fasttext will see them
    as words and also so that it can learn across newlines.
    """
    s = ' ~ ' + s + ' @ '
    s = s.replace('[', ' [ ')
    s = s.replace(']', ' ] ')
    s = s.replace(':', ' : ')
    s = s.replace(';', ' ; ')
    s = s.replace('?', ' ? ')
    s = s.replace('/', ' / ')
    s = s.replace('.', ' . ')
    s = s.replace('-', ' - ')
    s = s.replace(',', ' , ')
    s = s.replace('+', ' + ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = s.replace('&', ' & ')
    s = s.replace('"', ' " ')
    s = s.replace('!', ' ! ')
    s = s.replace('\n', ' # ')
    return s


def files_prep(directories):
    """Prepare the text files in the given directories and concatenate.

    By 'prepare' we mostly mean to add some spacing so that fasttext will
    see punctuation as separate words.
    """

    result = []
    for directory in directories:
        path = pathlib.Path(directory)
        for child in path.iterdir():
            if child.suffix != '.txt':
                continue
            with open(child) as f:
                text = f.read()
            text = string_prep(text)
            result.append(text)
    return '\n'.join(result)


def read_nodes_file(filename):
    """Given a .nodes file, return a dict and a list containing the
    node tensors."""

    import numpy as np
    import torch

    with open(filename) as f:
        lines = f.readlines()

    embeddings = {}
    lst = []

    for i, line in enumerate(lines):
        word, rest = line.split(' ', maxsplit=1)
        array = np.fromstring(rest, sep=' ')
        tensor = torch.from_numpy(array)
        if opts.cuda:
            tensor = tensor.cuda()
        embeddings[word] = i
        lst.append(tensor)
    max_len = max(len(tensor) for tensor in lst)
    lst2 = []
    for tensor in lst:
        tensor2 = torch.zeros([2 * max_len])
        if opts.cuda:
            tensor2 = tensor2.cuda()
        for i, j in enumerate(tensor):
            tensor2[2 * i + int(j.item())] = 1
        lst2.append(tensor2)

    return embeddings, lst2


def read_embeddings_file(filename):
    '''Given a .vec file produced by fasttext, return a dict and a
    list containing the embeddings.
    
    The dict will map word -> index, and the list will have the nth
    vector in the nth place.'''

    import numpy as np
    import torch

    with open(filename) as f:
        lines = f.readlines()
    embeddings = {}
    lst = []
    for i, line in enumerate(lines[1:]):
        word, rest = line.split(' ', maxsplit=1)
        array = np.fromstring(rest, sep=' ')
        tensor = torch.from_numpy(array)
        if opts.cuda:
            tensor = tensor.cuda()
        embeddings[word] = i
        lst.append(tensor)

    return embeddings, lst


def embed_directory(data_directory, embedding_directory, indices, vectors,
                    nodes):
    """For every .txt file in `data_directory`, write a .index file, a .node
    file, and a .vector file in `encoding_directory`.

    `data_directory`: contains show directories with .txt files
    `indices`: dict word -> index
    `vectors`: map index -> tensor
    `nodes`: map index -> node (as tensor)
    """
    path = pathlib.Path(data_directory)
    for child in path.iterdir():
        if child.suffix != '.txt':
            continue
        with open(child) as f:
            text = f.read()
        text = string_prep(text)
        words = text.split(' ')
        # remove empty strings
        words = [word for word in words if word]

        text_indices = torch.zeros([len(words)], dtype=torch.long)
        text_vectors = torch.zeros([len(words), len(vectors[0])])
        text_nodes = torch.zeros([len(words), len(nodes[0])])

        for i, word in enumerate(words):
            index = indices[word]
            text_indices[i] = index
            text_vectors[i] = vectors[index]
            text_nodes[i] = nodes[index]

        filename_indices = child.name[:-4] + '.indices'
        filepath_indices = pathlib.Path(embedding_directory, filename_indices)
        torch.save(text_indices, filepath_indices)

        filename_vectors = child.name[:-4] + '.vectors'
        filepath_vectors = pathlib.Path(embedding_directory, filename_vectors)
        torch.save(text_vectors, filepath_vectors)

        filename_nodes = child.name[:-4] + '.nodes'
        filepath_nodes = pathlib.Path(embedding_directory, filename_nodes)
        torch.save(text_nodes, filepath_nodes)
