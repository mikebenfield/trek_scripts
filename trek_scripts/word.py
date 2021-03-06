import pathlib

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import trek_scripts.opts as opts
from trek_scripts.util import batch_iter


class WordMap:
    """Maps between the various representations of words.

    Specifically, a word has 4 representations:
    - a string
    - an integer index
    - a vector (as a 1-D tensor)
    - a sequence of classes (as a tuple of 0s and 1s)
    """

    @staticmethod
    def from_directory(directory):
        """Given a directory in which are found a `words.vec` file
        and a `words.nodes` file, produce a `WordMap`.
        """
        import pathlib

        vec_file = pathlib.Path(directory, 'words.vec')
        with open(vec_file) as f:
            lines = f.readlines()[1:]

        first_array = np.fromstring(
            lines[0].split(
                ' ', maxsplit=1)[1], sep=' ')
        dim = len(first_array)

        vectors = torch.zeros([len(lines), dim])
        if opts.cuda:
            vectors = vectors.cuda()
        indices = {}

        for i, line in enumerate(lines):
            word, rest = line.split(' ', maxsplit=1)
            array = np.fromstring(rest, sep=' ')
            tensor = torch.from_numpy(array)
            if opts.cuda:
                tensor = tensor.cuda()
            indices[word] = i
            vectors[i] = tensor

        class_file = pathlib.Path(directory, 'words.nodes')

        with open(class_file) as f:
            lines = f.readlines()
        classes = [line.split()[1:] for line in lines]
        for i, strings in enumerate(classes):
            classes[i] = tuple(int(s) for s in strings if s)

        return WordMap(indices, vectors, classes)

    def __init__(self, indices, vectors, classes):
        """

        `indices`: dict mapping word -> index
        `vectors`: tensor mapping index -> vector
        `classes`: list mapping index -> tuple
        """
        tuple_to_index = {}

        for i, tupl in enumerate(classes):
            tuple_to_index[tupl] = i

        strings = [() for _ in range(len(vectors))]
        for string in indices:
            i = indices[string]
            strings[i] = string

        self.strings = strings
        self.tuple_to_index = tuple_to_index
        self.string_to_index = indices
        self.vectors = vectors
        self.classes = classes

    def dim(self):
        return len(self.vectors[0])

    def longest_tuple(self):
        return max(len(clas) for clas in self.classes)

    def word_count(self):
        return len(self.vectors)

    def index(self, x):
        """Return the index representation of the string or tuple x."""
        if isinstance(x, str):
            return self.string_to_index[x]
        elif isinstance(x, tuple):
            return self.tuple_to_index[x]
        else:
            raise ValueError('x should be tuple or string; received {}'.format(
                x))

    def string(self, x):
        """Return the string representation of the index or tuple x."""
        if isinstance(x, tuple):
            index = self.tuple_to_index[x]
        elif isinstance(x, int):
            index = x
        else:
            raise ValueError('x should be tuple or int; received {}'.format(x))
        return self.strings[index]

    def vector(self, x):
        """Return the vector representation of the index, tuple, or string x."""
        if isinstance(x, tuple):
            index = self.tuple_to_index[x]
        elif isinstance(x, str):
            index = self.string_to_index[x]
        else:
            index = x

        return self.vectors[index]

    def clas(self, x):
        """Return the class (tuple) representation of the index or string x."""
        if isinstance(x, tuple):
            index = self.tuple_to_index[x]
        elif isinstance(x, str):
            index = self.string_to_index[x]
        else:
            raise ValueError('x should be string or int; received {}'.format(
                x))
        return self.classes[index]


class WordRnn(nn.Module):
    """Word-level model.

    The model works like this: the last word embedding `x` and previous `hidden`
    output go into an LSTM: `output, new_hidden = LSTM(x, hidden)`. The other
    input is `classes`. Maybe this would be better called `clusters`. But in any
    case, it's the sequence of 0s and 1s that defines the hierarchy of clusters
    into which the word being predicted falls.
    """

    def __init__(self, input_size, hidden_size, num_layers, hierarchy_depth,
                 dropout):
        """
        Create a word-level model.

        `hidden_size` and `num_layers` refer to the LSTM cells.

        `input_size` is the dimension of the word embedding.

        `hierarchy_depth` refers to the dimension of the cluster
        hierarchy.
        """
        super().__init__()

        self.hierarchy_depth = hierarchy_depth
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)
        self.linear1 = nn.Linear(hidden_size + 2 * hierarchy_depth - 2, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, classes, x, hidden=None, full=True):
        sz = x.size()
        x = x.view(sz[0], 1, sz[1])
        x, hidden = self.lstm(x, hidden)
        sz = x.size()
        x = x.view(sz[0], sz[2])
        if full:
            results = []
            for i in range(0, 2 * self.hierarchy_depth, 2):
                classes_ = classes.clone()
                classes_[:, i:] = 0
                y = torch.cat([x, classes_], dim=-1)
                y = self.linear1(y)
                y = F.relu(y)
                y = self.linear2(y)
                y = F.relu(y)
                y = self.linear3(y)
                results.append(y)
            return torch.cat(results, dim=-1), hidden
        else:
            x = torch.cat([x, classes], dim=-1)
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            x = F.relu(x)
            x = self.linear3(x)
            return x, hidden


def cross_entropy_loss1(prediction, classes):
    classes = -2 * classes + 1
    signed_prediction = classes.float() * prediction
    loss = torch.log(1 + torch.exp(signed_prediction))
    return loss.mean(dim=-1)


def cross_entropy_loss2(prediction, nodes):
    sz = list(nodes.size())
    sz[-1] /= 2
    sz.append(2)
    nodes = nodes.view(*sz)
    sign = torch.empty(sz)
    sign = nodes[..., 0] - nodes[..., 1]
    loss = torch.log(1 + torch.exp(sign * prediction))
    return loss.mean(dim=-1).mean(dim=-1)


def hallucinate(model, max_len, word_map, rand):
    model.eval()
    output = ['~']

    hidden = None

    for _ in range(max_len):
        inp = word_map.vector(output[-1])
        inp = inp.view(1, -1)
        nodes = []
        nodes_tensor = torch.zeros([1, 2 * word_map.longest_tuple() - 2])
        if opts.cuda:
            nodes_tensor = nodes_tensor.cuda()
        while True:
            out, hidden_ = model(nodes_tensor, inp, hidden, False)
            val = out[0, 0].item()
            prob = 1 / (1 + np.exp(-val))
            val = rand.uniform()
            if output[-1] == '~':
                print('prob val', prob, val, nodes)
            if val <= prob:
                nodes.append(1)
            else:
                nodes.append(0)
            try:
                word = word_map.string(tuple(nodes))
                if output[-1] == '~':
                    print(tuple(nodes))
                output.append(word)
                break
            except KeyError:
                pass
            nodes_tensor[0, 2 * len(nodes) - 2 + nodes[-1]] = 1
        hidden = hidden_
        if output[-1] == '@':
            break

    return ' '.join(output)


def _format_tensors(chunk_size, nodes, vectors):
    """Put `indices` and `vectors` into a format appropriate for training.

    `indices` is a list of 1-D long tensors, each of length `word_count`,
    representing a transcript.

    `vectors` is a list of 2-D float tensors, of size
    `(word_count, dim)`.

    Returns `ret_indices`, `ret_vectors`.

    `ret_indices` is a 2-D long tensor of shape `(transcript_len, batch_size)`

    `ret_vectors` is a 3-D float tensor of shape `(transcript_len, batch_size, dim)`

    Here `transcript_len` is the
    smallest integer at least as large as every `word_count` and such that
    `transcript_len % chunk_size == 1`. The excess entries are filled with
    the last word in the first tensor.
    """
    batch_size = len(nodes)
    final_word_nodes = nodes[0][-1]
    final_word_vector = vectors[0][-1]
    dim = len(final_word_vector)
    node_count = len(final_word_nodes)
    transcript_len = max(len(transcript) for transcript in nodes)

    mod = transcript_len % chunk_size
    if mod != 1:
        diff = chunk_size - mod + 1
        # now transcript_len % chunk_size == 1
        transcript_len += diff

    ret_nodes = torch.zeros([transcript_len, batch_size, node_count])
    ret_vectors = torch.zeros([transcript_len, batch_size, dim])

    if opts.cuda:
        ret_nodes = ret_nodes.cuda()
        ret_vectors = ret_vectors.cuda()

    for i in range(batch_size):
        tensor = nodes[i]
        length = len(tensor)
        ret_nodes[0:length, i] = tensor
        if length < len(ret_nodes):
            ret_nodes[length:, i] = final_word_nodes

        tensor_vector = vectors[i]
        ret_vectors[0:length, i, :] = tensor_vector
        if length < len(ret_vectors):
            ret_vectors[length:, i, :] = final_word_vector

    return ret_nodes, ret_vectors


def train(
        model,
        optimizer,
        chunk_size,
        transcripts_nodes,
        transcripts_vectors, ):
    model.train()

    nodes, vectors = _format_tensors(chunk_size, transcripts_nodes,
                                     transcripts_vectors)

    last_hidden = None

    if opts.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    total_loss = 0

    for i in range(0, len(nodes) - 1, chunk_size):
        if last_hidden is not None:
            last_hidden = (last_hidden[0].detach(), last_hidden[1].detach())
        optimizer.zero_grad()

        loss = torch.zeros([1], requires_grad=True, device=device)

        for j in range(i, i + chunk_size):
            output, last_hidden = model(nodes[j + 1, :, :-2], vectors[j],
                                        last_hidden)
            loss = loss.clone()
            loss += cross_entropy_loss2(output, nodes[j + 1])

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    size = nodes.size()
    denominator = (size[0] - 1) * size[1]
    return total_loss / denominator


def test(model, transcripts_nodes, transcripts_vectors):
    nodes, vectors = _format_tensors(1, transcripts_nodes, transcripts_vectors)
    model.eval()

    last_hidden = None

    total_loss = 0

    for i in range(0, len(nodes) - 1):
        if last_hidden is not None:
            last_hidden = (last_hidden[0].detach(), last_hidden[1].detach())
        output, last_hidden = model(nodes[i + 1, :, :-2], vectors[i],
                                    last_hidden)
        total_loss += cross_entropy_loss2(output, nodes[i + 1]).item()

    return total_loss / (len(nodes) - 1)


def full_train(
        model,
        optimizer,
        rand,
        epochs,
        train_episodes,
        test_episodes,
        batch_size,
        chunk_size,
        word_map,
        model_directory, ):
    """
    Train the word-level model for the given number of epochs.

    After each epoch, the model will be saved in `model_directory` and a small
    sample hallucinated script will be printed to the screen.
    """
    for epoch in range(epochs):
        print('Beginning epoch {}'.format(epoch))

        total_train_loss = 0
        for i, train_batch in enumerate(
                batch_iter(rand, batch_size, train_episodes)):
            as_nodes = [
                torch.load(ep.with_suffix('.nodes')) for ep in train_batch
            ]
            as_vectors = [
                torch.load(ep.with_suffix('.vectors')) for ep in train_batch
            ]

            if opts.cuda:
                as_nodes = [tensor.cuda() for tensor in as_nodes]
                as_vectors = [tensor.cuda() for tensor in as_vectors]

            loss = train(model, optimizer, chunk_size, as_nodes, as_vectors)
            print('Batch {}, loss {}'.format(i, loss))
            total_train_loss += len(as_nodes) * loss

        total_test_loss = 0
        for test_batch in batch_iter(rand, batch_size, test_episodes):
            as_nodes = [
                torch.load(ep.with_suffix('.nodes')) for ep in test_batch
            ]
            as_vectors = [
                torch.load(ep.with_suffix('.vectors')) for ep in test_batch
            ]
            if opts.cuda:
                as_nodes = [tensor.cuda() for tensor in as_nodes]
                as_vectors = [tensor.cuda() for tensor in as_vectors]
            loss = test(model, as_nodes, as_vectors)
            total_test_loss += len(as_nodes) * loss

        average_test_loss = total_test_loss / len(test_episodes)
        average_train_loss = total_train_loss / len(train_episodes)

        print('mean training loss for epoch {}: {}'.format(epoch,
                                                           average_train_loss))
        print('mean testing loss for epoch {}: {}'.format(epoch,
                                                          average_test_loss))

        text = hallucinate(model, 100, word_map, rand)
        print(text)
        path = pathlib.Path(model_directory, 'model_{:0>4}'.format(epoch))
        torch.save(model, path)
