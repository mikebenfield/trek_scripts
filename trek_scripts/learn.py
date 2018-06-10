
N_CODEPOINTS = 91

import torch
from torch import nn
import torch.nn.functional as F

import trek_scripts.opts as opts

class CharRnn(nn.Module):
    def __init__(self, io_size, hidden_size):
        super().__init__()
        self.io_size = io_size
        self.gru = nn.GRUCell(io_size, hidden_size)
        self.linear = nn.Linear(hidden_size, io_size)

    def forward(self, x, hidden):
        hidden = self.gru(x, hidden)
        x = F.log_softmax(x, dim=1)
        return (x, hidden)

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

def encode_strings(chunk_size, strings):
    '''Returns (onehot, encoded_strings).
    
    index as onehot[string_index, batch_index, codepoint]
    and      encoded_strings[string_index, batch_index]
    '''
    count = len(strings)
    max_len = 1 + max([len(s) for s in strings])
    diff = chunk_size - max_len % chunk_size + 1
    # now max_len % chunk_size == 1
    max_len += diff

    encoded_strings = torch.zeros([max_len, count], dtype=torch.long, requires_grad=False)
    onehot = torch.zeros([max_len, count, N_CODEPOINTS], requires_grad=False)
    # N_CODEPOINTS - 2 will indicate beginning of file
    onehot[0, :, N_CODEPOINTS-2] = 1
    encoded_strings[0, :] = N_CODEPOINTS - 2

    for string_index in range(1, max_len):
        for j, string in enumerate(strings):
            if string_index > len(string):
                # N_CODEPOINTS - 1 will indicate end of file
                code = N_CODEPOINTS - 1
            else:
                code = char_to_code(string[string_index-1])
            onehot[string_index, j, code] = 1
            encoded_strings[string_index, j] = code

    if opts.cuda:
        onehot = onehot.cuda()
        encoded_strings = encoded_strings.cuda()

    return (onehot, encoded_strings)

def hallucinate(model, hidden_size, max_len, rand):
    import numpy as np

    model.eval()
    output = []

    last_code = char_to_code('~')

    hidden = torch.zeros([1, hidden_size])
    if opts.cuda:
        hidden = last_hidden.cuda()

    for _ in range(max_len):
        inp = torch.zeros([1, N_CODEPOINTS])
        inp[0, last_code] = 1
        if opts.cuda:
            inp = inp.cuda()
        out, hidden = model(inp, hidden)
        nparray = out.detach().numpy()
        nparray = np.exp(nparray)
        last_code = rand.choice(N_CODEPOINTS, p=nparray[0])
        char = code_to_char(last_code)
        if char == '@':
            break
        output.append(char)

    return ''.join(output)

def test(model, hidden_size, loss_f, strings):
    onehot, encoded_strings = encode_strings(chunk_size, strings)
    max_len, count = encoded_strings.size()
    del strings

    print('Test strings encoded')

    model.eval()

    last_hidden = torch.zeros([count, hidden_size])
    if opts.cuda:
        last_hidden = last_hidden.cuda()

    total_loss = 0
    for i in range(0, encoded_strings.size()[0] - 1):
        output, last_hidden = model(onehot[j, :, :], last_hidden)
        total_loss += loss_f(output, encoded_strings[j+1])

    return total_loss

def train(model, hidden_size, loss_f, optimizer, chunk_size, strings):
    onehot, encoded_strings = encode_strings(chunk_size, strings)
    max_len, count = encoded_strings.size()
    del strings

    print('Strings encoded')

    model.train()

    last_hidden = torch.zeros([count, hidden_size])

    if opts.cuda:
        last_hidden = last_hidden.cuda()

    total_loss = 0
    for i in range(0, max_len - 1, chunk_size):
        last_hidden.detach_()
        optimizer.zero_grad()
        loss = torch.zeros([1], requires_grad=True)
        if opts.cuda:
            loss = loss.cuda()
        for j in range(i, min(i+chunk_size, max_len)):
            output, last_hidden = model(onehot[j, :, :], last_hidden)
            loss = loss.clone()
            new_loss = loss_f(output, encoded_strings[j+1])
            loss += new_loss
            print('found loss ', new_loss.data[0])
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return total_loss / (max_len - 1)
