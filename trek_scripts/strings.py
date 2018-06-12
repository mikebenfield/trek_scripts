
import torch

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
