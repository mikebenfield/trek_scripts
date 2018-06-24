import pathlib

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

def data_prep(directories):
    '''prepare all the text files in the given directory and concatenate'''
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
            

def read_embeddings_file(filename):
    '''Given a .vec file produced by fasttext, return a dict and a
    list containing the embeddings.
    
    The dic will map word -> tensor, and the list will have the nth
    word in the nth place.'''

    import numpy as np
    import torch

    with open(filename) as f:
        lines = f.readlines()
    embeddings = {}
    lst = []
    for line in lines[1:]:
        word, rest = line.split(' ', maxsplit=1)
        array = np.fromstring(rest, sep=' ')
        tensor = torch.from_numpy(array)
        if opts.cuda:
            tensor = tensor.cuda()
        embeddings[word] = tensor
        lst.append(word)

    return embeddings, lst
