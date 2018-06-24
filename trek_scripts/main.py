
import argparse
import html
import os
import pathlib
import re
import urllib
import urllib.error
import urllib.request

from trek_scripts import char
from trek_scripts import opts
from trek_scripts import util

def die(msg):
    import sys
    print(msg)
    sys.exit(1)

def download_show(url, url_directory, start_n, last_n, destination,
                  pad_leading_zeros):
    base_url = urllib.parse.urljoin(url, url_directory)
    if not base_url.endswith('/'):
        base_url += '/'
    try:
        os.mkdir(destination)
    except FileExistsError:
        pass
    if not os.path.isdir(destination):
        die("Couldn't make directory at {}".format(destination))

    for n in range(start_n, last_n + 1):
        if pad_leading_zeros:
            url_filename = '{:0>2}.htm'.format(n)
        else:
            url_filename = str(n) + '.htm'
        full_url = urllib.parse.urljoin(base_url, url_filename)
        filename = '{:0>3}'.format(str(n)) + '.html'
        filepath = pathlib.PurePath(destination, filename)
        try:
            urllib.request.urlretrieve(full_url, filepath)
        except urllib.error.HTTPError:
            # a few show numbers are skipped; we don't care if some URLs are not
            # found
            pass

def download(url, directory):
    if not url.endswith('/'):
        url += '/'
    download_show(url, "StarTrek", 1, 79, pathlib.PurePath(directory, "TOS"), False)
    download_show(url, "NextGen", 101, 277, pathlib.PurePath(directory, "TNG"), False)
    download_show(url, "DS9", 401, 575, pathlib.PurePath(directory, "DS9"), False)
    download_show(url, "Voyager", 101, 722, pathlib.PurePath(directory, "VOY"), False)
    download_show(url, "Enterprise", 1, 98, pathlib.PurePath(directory, "ENT"), True)

def strip_html(directory):
    '''For each file .html file in this directory, create a .txt version
        with all html tags removed.'''
    path = pathlib.Path(directory)
    regex_br = re.compile(r'(?:<br>|</p>|<p>)')
    regex_tag = re.compile(r'<.*?>')
    regex_curly_square = re.compile(r'\{(.*?)\]')
    regex_square_curly = re.compile(r'\[(.*?)\}')
    regex_curly_paren = re.compile(r'\{(.*?)\)')
    regex_paren_curly = re.compile(r'\(.*?\}')
    regex_curly_curly = re.compile(r'\{.*?\}')
    regex_leading_space = re.compile(r'^\s+')
    regex_trailing_space = re.compile(r'\s+$')
    regex_multiple_space = re.compile(r'  +')
    regex_space_only = re.compile(r'^\s*$')

    for child in path.iterdir():
        if child.suffix == '.html':
            with open(child, encoding='Latin-1') as f:
                lines = f.readlines()

            # remove newlines
            lines = [line.replace('\n', ' ') for line in lines]

            # concatenate
            text = ''.join(lines)

            # bogus tags
            text = text.replace('&lt;br&gt;', '<br>')
            text = text.replace('br&gt;', '<br>')

            # <br> tags and </p> tags to newlines
            text = regex_br.sub('\n', text)

            # remove HTML tags
            text = regex_tag.sub('', text)

            # unescape HTML
            text = html.unescape(text)

            # from malformed tags
            text = text.replace('\n>', '\n')

            # correct braces
            text = text.replace('[OC}', '[OC]')
            text = text.replace('{OC]', '[OC]')
            text = text.replace('{OC}', '[OC]')

            text = regex_curly_square.sub('[\1]', text)
            text = regex_square_curly.sub('[\1]', text)
            text = regex_curly_curly.sub('[\1]', text)
            text = regex_curly_paren.sub('(\1)', text)
            text = regex_paren_curly.sub('(\1)', text)

            # tabs
            text = text.replace('\t', ' ')

            # oddball characters
            text = text.replace('\u0001', ' ')
            text = text.replace('\u0091', ' ')
            text = text.replace('\u0092', ' ')
            text = text.replace('\u0096', ' ')
            text = text.replace('\u00A0', ' ')

            # multiple spaces
            text = regex_multiple_space.sub(' ', text)

            # typos
            text = text.replace('Later -_', 'Later.)')
            text = text.replace('_', ')')
            text = text.replace('|', '')
            text = text.replace('@', ':')
            text = text.replace('>', '.')
            text = text.replace('=', '-')

            # more bogus tags
            text = text.replace(' <', '')

            # remove backticks
            text = text.replace('`', '')

            # remove hash
            text = text.replace('#', '')

            # undo some censorship
            text = text.replace('****', 'bloody')

            # non-Latin characters
            text = text.replace('ß', 'ss')
            text = text.replace('ä', 'a')
            text = text.replace('à', 'a')
            text = text.replace('ç', 'c')
            text = text.replace('è', 'e')
            text = text.replace('é', 'e')
            text = text.replace('ê', 'e')
            text = text.replace('ï', 'i')
            text = text.replace('ó', 'o')

            # Cyrillic text
            text = text.replace('что случилось', "chto sluchilos'")

            # remove empty lines
            lines = text.split('\n')

            lines = [line for line in lines if not regex_space_only.match(line)]
            # remove leading spaces
            lines = [regex_trailing_space.sub('', line) for line in lines]
            lines = [regex_leading_space.sub('', line) for line in lines]

            # new title
            if lines[0].startswith('The Star Trek'):
                lines[0] = 'Star Trek'
            elif lines[0].startswith('The Next Generation'):
                lines[0] = 'Star Trek: The Next Generation'
            elif lines[0].startswith('The Deep'):
                lines[0] = 'Star Trek: Deep Space Nine'
            elif lines[0].startswith('The Voyager'):
                lines[0] = 'Star Trek: Voyager'
            elif lines[0].startswith('The Enterprise'):
                lines[0] = 'Star Trek: Enterprise'

            # remove ending text
            lines0 = []
            for i in range(0, len(lines)):
                line = lines[i]
                if line.startswith('<Back') or line.startswith('< Back'):
                    # transcript is over
                    break
                if line.startswith('Star Trek ®'):
                    # transcript is over
                    break
                lines0.append(line)
            lines = lines0

            new_file_name = child.with_suffix('.txt')
            with open(new_file_name, 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

def arg_download(args):
    download(args.url, args.dir)

def arg_strip(args):
    for name in ["TOS", "TNG", "DS9", "VOY", "ENT"]:
        path = pathlib.PurePath(args.dir, name)
        strip_html(path)

def arg_encode_char(args):
    for name in ["TOS", "TNG", "DS9", "VOY", "ENT"]:
        path = pathlib.PurePath(args.dir, name)
        char.encode_directory(path)

def arg_hallucinate(args):
    import torch
    import numpy.random as random
    import trek_scripts.learn as learn
    rand = random.RandomState(args.seed)
    dict_ = torch.load(args.model)
    hidden_size = dict_['hidden_size']
    layer_size = dict_['layer_size']
    num_layers = dict_['num_layers']
    model = learn.CharRnn(91, hidden_size=hidden_size, layer_size=layer_size, num_layers=num_layers)
    model.load_state_dict(dict_['model'])
    s = learn.hallucinate(model, args.max_len, rand)
    print(s)
    
def arg_train_word(args):
    import numpy.random as random
    import torch.nn as nn

    import trek_scripts.opts as opts
    import trek_scripts.fasttext as fasttext
    import trek_scripts.word as word
    
    opts.cuda = args.cuda

    rand = random.RandomState(args.seed)

    directory = args.directory
    shows = args.shows.split(',')
    shows = [show.strip() for show in shows]
    test_episodes, train_episodes = util.train_test_split(
        rand, 
        '.encode',
        args.directory,
        shows,
        args.test_size)
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    dropout = args.dropout
    batch_size = args.batch_size
    embedding_file = pathlib.Path(directory, args.embedding_file)

    words, lst = fasttext.read_embeddings_file(embedding_file)

    model = WordRnn(input_size=len(words[0]), hidden_size=hidden_size,
                    num_layers=num_layers, output_size = len(lst))

    loss_f = nn.NLLoss()


def arg_train_char(args):
    from torch import optim
    from torch import nn

    from numpy import random

    opts.cuda = args.cuda

    rand = random.RandomState(args.seed)

    shows = args.shows.split(',')
    shows = [show.strip() for show in shows]
    test_episodes, train_episodes = util.train_test_split(
        rand, 
        '.encode',
        args.directory,
        shows,
        args.test_size)
    if args.model_name == 'top':
        model = char.CharRnnTop(91, args.hidden_size, args.layer_size, args.num_layers)
    else:
        model = char.CharRnnNoTop(91, args.hidden_size, args.num_layers)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.epsilon
    )

    if opts.cuda:
        model.cuda()

    loss_f = nn.NLLLoss()

    train_paths = [pathlib.Path(args.directory, ep) for ep in train_episodes]
    test_paths = [pathlib.Path(args.directory, ep) for ep in test_episodes]

    char.full_train(model, optimizer, rand, args.epochs,
                    train_paths, test_paths, args.batch_size, args.chunk_size, loss_f,
                    args.model_directory)

    print('Done')

def arg_fasttext_prep(args):
    import trek_scripts.fasttext as fasttext
    directory = args.directory
    shows = args.shows.split(',')
    dirs = [pathlib.Path(directory, show) for show in shows]
    result = fasttext.data_prep(dirs)
    result_path = pathlib.Path(directory, 'trek.txt')
    with open(result_path, 'w') as f:
        f.write(result)

def arg_fasttext_embed_parser(args):
    import subprocess

    directory = args.directory
    dim = args.dim
    out_filename = args.filename
    epochs = args.epochs
    in_filename = pathlib.Path(directory, 'trek.txt')
    out_filename = pathlib.Path(directory, out_filename)

    subprocess.run(['fasttext', 'cbow', '-input', in_filename,
                    '-output', out_filename,
                    '-epoch', epochs],
                   check=True)

def main():
    parser = argparse.ArgumentParser(description='Generate some scripts')

    subparsers = parser.add_subparsers()

    download_parser = subparsers.add_parser('download')
    download_parser.add_argument('--url', dest='url', required=True)
    download_parser.add_argument('--directory', dest='dir', required=True)
    download_parser.set_defaults(func=arg_download)

    strip_parser = subparsers.add_parser('strip')
    strip_parser.add_argument('--directory', dest='dir', required=True)
    strip_parser.set_defaults(func=arg_strip)

    encode_char_parser = subparsers.add_parser(
        'encode_char',
        help='Encode all transcripts for character-level models.'
    )
    encode_char_parser.add_argument('--directory', dest='dir', required=True)
    encode_char_parser.set_defaults(func=arg_encode_char)

    fasttext_prep_parser = subparsers.add_parser(
        'fasttext_prep',
        help='Prepare and concatenate the shows for fasttext; save in the file trek.txt in the directory specified.'
    )
    fasttext_prep_parser.add_argument(
        '--directory', required=True,
        help='Data directory containing transcript directories'
    )
    fasttext_prep_parser.add_argument(
        '--shows',
        default='TOS,TNG,DS9,VOY,ENT',
        help='Comma separated list of series to train on'
    )
    fasttext_prep_parser.set_defaults(func=arg_fasttext_prep)

    fasttext_embed_parser = subparsers.add_parser(
        'fasttext_embed',
        help='Use fasttext to construct word embeddings.'
    )
    fasttext_embed_parser.add_argument(
        '--directory', required=True,
        help='Data directory containing trek.txt output from fasttext_prep'
    )
    fasttext_embed_parser.add_argument(
        '--epochs', required=True,
        help='How many epochs should fasttext train?'
    )
    fasttext_embed_parser.add_argument(
        '--dim', required=True,
        help='How many dimensions should the space of word vectors be?'
    )
    fasttext_embed_parser.add_argument(
        '--filename', required=True,
        help='Name of file prefix in data directory to save word embeddings'
    )
    fasttext_embed_parser.set_defaults(func=arg_fasttext_embed_parser)

    train_word_parser = subparsers.add_parser(
        'train_word',
        help='Train a word level model based on embeddings constructed by fasttext.'
    )
    train_word_parser.add_argument(
        '--directory', required=True,
        help='Data directory containing transcript directories and word embeddings'
    )
    train_word_parser.add_argument(
        '--embedding_file', required=True,
        help='Filename in data directory of embeddings file'
    )
    train_word_parser.add_argument(
        '--num_layers', required=True,
        help='How many LSTM layers to use.'
    )
    train_word_parser.add_argument(
        '--shows',
        default='TOS,TNG,DS9,VOY,ENT',
        help='Comma separated list of series to train on'
    )
    train_word_parser.add_argument(
        '--model_directory', required=True,
        help='Directory in which to save models'
    )
    train_word_parser.add_argument(
        '--hidden_size', type=int, default=256,
        help='Size of the hidden layer in the LSTM cell'
    )
    train_word_parser.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout probability (0 for no dropout)'
    )
    train_word_parser.add_argument(
        '--cuda', action='store_true',
    )
    train_word_parser.add_argument(
        '--test_size', type=float, required=True,
        help='Proportion of transcripts to use for testing'
    )
    train_word_parser.add_argument(
        '--batch_size', type=int, default=10,
    )
    train_word_parser.add_argument(
        '--epochs', type=int, required=True,
        help='Number of training epochs'
    )
    train_word_parser.add_argument(
        '--seed', type=int,
        help='Random number seed'
    )
    train_word_parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate for Adam optimizer'
    )
    train_word_parser.set_defaults(func=arg_train_word)

    train_char_parser = subparsers.add_parser(
        'train_char',
        help='Train a character level model.'
    )
    train_char_parser.add_argument(
        '--directory', required=True,
        help='Data directory containing transcript directories'
    )
    train_char_parser.add_argument(
        '--model_directory', required=True,
        help='Directory in which to save models'
    )
    train_char_parser.add_argument(
        '--model', required=False,
        help='Saved model file to begin training with'
    )
    train_char_parser.add_argument(
        '--shows',
        default='TOS,TNG,DS9,VOY,ENT',
        help='Comma separated list of series to train on'
    )
    train_char_parser.add_argument(
        '--hidden_size', type=int, default=128,
        help='Size of the hidden layer in the GRU cell'
    )
    train_char_parser.add_argument(
        '--num_layers', type=int, default=1,
        help='Numer of GRU layers'
    )
    train_char_parser.add_argument(
        '--layer_size', type=int, default=128,
        help='Size of the layer in the GRU cell'
    )
    train_char_parser.add_argument(
        '--chunk_size', required=True, type=int,
    )
    train_char_parser.add_argument(
        '--cuda', action='store_true',
    )
    train_char_parser.add_argument(
        '--test_size', type=float, required=True,
        help='Proportion of transcripts to use for testing'
    )
    train_char_parser.add_argument(
        '--batch_size', type=int, default=10,
    )
    train_char_parser.add_argument(
        '--epochs', type=int, required=True,
        help='Number of training epochs'
    )
    train_char_parser.add_argument(
        '--seed', type=int,
        help='Random number seed'
    )
    train_char_parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate for Adam optimizer'
    )
    train_char_parser.add_argument(
        '--epsilon', type=float, default=1e-4,
        help='Epsilon parameter for Adam optimizer'
    )
    train_char_parser.add_argument(
        '--model_name', default='notop',
        help='`Top` to use a model with an extra linear layer on top.'
    )
    train_char_parser.set_defaults(func=arg_train_char)

    hallucinate_parser = subparsers.add_parser('hallucinate')
    hallucinate_parser.add_argument(
        '--model', required=True,
        help='Saved model file to use'
    )
    hallucinate_parser.add_argument(
        '--max_len', required=True, type=int,
    )
    hallucinate_parser.add_argument(
        '--seed', type=int,
        help='Random number seed'
    )
    hallucinate_parser.set_defaults(func=arg_hallucinate)

    args = parser.parse_args()
    args.func(args)
