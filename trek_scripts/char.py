"""Character level models.

Testing and training functions accept `tensors` arguments that are
lists of encoded 1-D tensors indexed by string index.
"""

import torch
from torch import nn
import torch.nn.functional as F

import trek_scripts.opts as opts

class CharRnnTop(nn.Module):
    def __init__(self, io_size, hidden_size, layer_size, num_layers):
        super().__init__()
        self.io_size = io_size
        self.gru = nn.GRU(io_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, layer_size)
        self.linear2 = nn.Linear(layer_size, io_size)

    def forward(self, x, hidden=None):
        sz = x.size()
        x = x.view(sz[0], 1, sz[1])
        x, hidden = self.gru(x, hidden)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=2)
        sz = x.size()
        return (x.view(sz[0], sz[2]), hidden)

class CharRnnNoTop(nn.Module):
    def __init__(self, io_size, hidden_size, num_layers):
        super().__init__()
        self.io_size = io_size
        self.gru = nn.GRU(io_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, io_size)

    def forward(self, x, hidden=None):
        sz = x.size()
        x = x.view(sz[0], 1, sz[1])
        x, hidden = self.gru(x, hidden)
        x = self.linear1(x)
        x = F.log_softmax(x, dim=2)
        sz = x.size()
        return (x.view(sz[0], sz[2]), hidden)

N_CODEPOINTS = 91

def code_to_char(index):
    if index == 0:
        return '\n'
    if 1 <= index <= 3:
        return chr(index + 0x1F)
    if 4 <= index <= 88:
        return chr(index + 0x22)
    if index == N_CODEPOINTS - 2:
        # beginning of file
        return '~'
    if index == N_CODEPOINTS - 1:
        # end of file
        return '@'
    raise ValueError('Invalid index: {}'.format(index))

def char_to_code(char):
    code = ord(char)
    if code == 0xA:
        return 0
    if 0x20 <= code <= 0x22:
        return code - 0x1F
    if 0x26 <= code <= 0x7A:
        return code - 0x22
    if char == '~':
        return N_CODEPOINTS-2
    if char == '@':
        return N_CODEPOINTS-1
    raise ValueError('Invalid char: {} with codepoint {}'.format(char, code))

def encode_string(string):
    encoded = torch.zeros([len(string)], dtype=torch.long)
    for i, char in enumerate(string):
        encoded[i] = char_to_code(string[i])
    return encoded

def encode_directory(directory):
    '''For every .txt file in this directory, write
    a .encode file according to our scheme.'''
    import torch

    path = pathlib.Path(directory)
    for child in path.iterdir():
        if child.suffix != '.txt':
            continue
        with open(child) as f:
            text = f.read()
        text = '~' + text + '@'
        tensor = encode_string(text)
        new_file_name = child.with_suffix('.encode')
        torch.save(tensor, new_file_name)

def _format_tensors(chunk_size, tensors):
    """Take the list `tensors` of 1-D tensors and put them into a
    format appropriate for training.

    `tensors` is a list of 1-D tensors, each of which is a sequence of
    codepoints encoding a transcript. The result is 
    
    `onehot, encoded`

    `encoded` is a 2-D tensor of shape `(transcript_len, batch_size)`, where
    `batch_size` is the length of `tensors`, and `transcript_len` is the
    smallest integer at least as large as every tensor and
    such that `transcript_len % chunk_size == 1`.
    The excess codepoints are filled in with the code for `@`.
    
    `onehot` is similar, but it's a 3-D tensor of shape
    `(transcript_len, batch_size, N_CODEPOINTS)`. `onehot[i, j]` is the onehot
    encoding of `encoding[i, j]`.
    """
    batch_size = len(tensors)
    transcript_len = max(len(tensor) for tensor in tensors)
    mod = transcript_len % chunk_size
    if mod != 1:
        diff = chunk_size - mod + 1
        # now transcript_len % chunk_size == 1
        transcript_len += diff

    encoded = torch.zeros([transcript_len, batch_size], dtype=torch.long)
    onehot = torch.zeros([transcript_len, batch_size, N_CODEPOINTS])
    indices = torch.arange(max_len, dtype=torch.long)

    if opts.cuda:
        encoded = encoded.cuda()
        onehot = onehot.cuda()
        indices = indices.cuda()

    for i in range(batch_size):
        tensor = tensors[i]
        length = len(tensor)
        encoded[0:length, i] = tensor
        encoded[length:, i] = char_to_code('@')
        onehot[indices, i, encoded[:, i]] = 1

    return onehot, encoded

def test(model, loss_f, tensors):
    """Return the average loss suffered as `model` tries to predict
    each character.


    `tensors` is a list of 1-D `long` tensors, each representing
    a transcript.
    """

    onehot, encoded = format_tensors(1, tensors)

    model.eval()

    last_hidden = None

    total_loss = 0
    for i in range(0, len(encoded) - 1):
        if not last_hidden is None:
            last_hidden = last_hidden.detach()
        output, last_hidden = model(onehot[i, :, :], last_hidden)
        total_loss += loss_f(output, encoded[i+1]).item()

    return total_loss / (len(encoded) - 1)

def train(model, loss_f, optimizer, chunk_size, tensors):
    """Train the model on `tensors`.

    `tensors` is a list of 1-D `long` tensors, each representing
    a transcript.

    Return the average loss suffered as `model` tries to predict
    each character.
    """
    model.train()

    onehot, encoded = format_tensors(chunk_size, tensors)

    last_hidden = None

    total_loss = 0
    for i in range(0, len(encoded) - 1, chunk_size):
        if not last_hidden is None:
            last_hidden = last_hidden.detach()
        optimizer.zero_grad()
        loss = torch.zeros([1], requires_grad=True)
        if opts.cuda:
            loss = loss.cuda()
        for j in range(i, i + chunk_size):
            output, last_hidden = model(onehot[j, :, :], last_hidden)
            loss = loss.clone()
            new_loss = loss_f(output, encoded[j+1])
            loss += new_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / (len(encoded) - 1)

def full_train(
    model,
    optimizer,
    epochs,
    train_episodes,
    test_episodes,
    loss_f,
    model_directory,
):
    for epoch 