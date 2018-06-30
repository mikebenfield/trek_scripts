
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
        and a `words.hierarchy` file, produce a `WordMap`.
        """
        import pathlib

        vec_file = pathlib.Path(directory, 'words.vec')
        with open(vec_file) as f:
            lines = f.readlines()[1:]

        first_array = np.fromstring(lines[0].split(' ', maxsplit=1)[1], sep=' ')
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

        class_file = pathlib.Path(directory, 'words.hierarchy')

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
            raise ValueError('x should be tuple or string; received {}'.format(x))

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
            raise ValueError('x should be string or int; received {}'.format(x))
        return self.classes[index]


class WordRnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 top_layer_size, hierarchy_depth):
        super().__init__()

        self.hierarchy_depth = hierarchy_depth
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear_x = nn.Linear(hidden_size, top_layer_size)
        self.linear_w = nn.Linear(2 * hierarchy_depth - 2, top_layer_size)
        self.linear_final = nn.Linear(top_layer_size, 1)

    def forward(self, classes, x, hidden=None, full=True):
        sz = x.size()
        x = x.view(sz[0], 1, sz[1])
        x, hidden = self.lstm(x, hidden)
        sz = x.size()
        x = x.view(sz[0], sz[2])
        if full:
            results = [] * self.hierarchy_depth
            for i in range(0, 2*self.hierarchy_depth, 2):
                classes_ = classes.clone()
                classes_[i:] = 0
                y = self.linear_x(x) + self.linear_w(classes_)
                y = F.relu(y)
                y = self.linear_final(y)
                results.append(y)
            return torch.cat(results, dim=-1), hidden
        else:
            x = self.linear_x(x) + self.linear_w(classes)
            x = F.relu(x)
            x = self.linear_final(x)
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
    loss1 = torch.log(1 + torch.exp(prediction))
    loss2 = torch.log(1 + torch.exp(-prediction))
    loss = nodes[..., 0] * loss1 + nodes[..., 1] * loss2
    return loss.mean(dim=-1).mean(dim=-1)

def hallucinate(model, max_len, word_map, rand):
    model.eval()
    output = []

    last_index = word_map.index('~')
    final = word_map.index('@')

    hidden = None

    output = []

    for _ in range(max_len):
        inp = torch.zeros([1, word_map.dim()])
        if opts.cuda:
            inp = inp.cuda()
        inp[0, :] = word_map.vector(last_index)
        out, hidden = model(inp, hidden)
        nparray = out[0].detach().cpu().numpy()
        probabilities = 1 / (1 + np.exp(-nparray))
        classes = [rand.choice(2, p=np.array([1-p, p])) for p in probabilities]
        for i in range(len(classes)):
            tupl = tuple(classes[:i])
            try:
                word = word_map.string(tupl)
                output.append(word)
                break
            except KeyError:
                pass
    print(len(output))
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
        transcripts_vectors,
):
    model.train()

    nodes, vectors = _format_tensors(chunk_size, transcripts_nodes, transcripts_vectors)

    last_hidden = None

    if opts.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    total_loss = 0

    for i in range(0, len(nodes) - 1, chunk_size):
        if not last_hidden is None:
            last_hidden = (last_hidden[0].detach(), last_hidden[1].detach())
        optimizer.zero_grad()

        loss = torch.zeros([1], requires_grad=True, device=device)

        for j in range(i, i + chunk_size):
            output, last_hidden = model(nodes[j+1, :, :-2], vectors[j], last_hidden)
            loss = loss.clone()
            loss += cross_entropy_loss2(output, nodes[j+1])

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    size = nodes.size()
    denominator = (size[0] - 1) * size[1]
    return total_loss / denominator


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
    model_directory,
):
    for epoch in range(epochs):
        print('Beginning epoch {}'.format(epoch))

        total_train_loss = 0
        for i, train_batch in enumerate(batch_iter(rand, batch_size, train_episodes)):
            print('Batch {}'.format(i))
            as_nodes = [torch.load(ep.with_suffix('.nodes')) for ep in train_batch]
            as_vectors = [torch.load(ep.with_suffix('.vectors')) for ep in train_batch]

            if opts.cuda:
                as_nodes = [tensor.cuda() for tensor in as_nodes]
                as_vectors = [tensor.cuda() for tensor in as_vectors]

            print('ok')

            loss = train(model, optimizer,
                         chunk_size, as_nodes, as_vectors)
            print('loss ', loss)
            total_train_loss += loss
        print('total loss', total_train_loss)
            # print(hallucinate(model, 500, word_map, rand))
