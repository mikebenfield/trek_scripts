
import argparse
import html
import os
import pathlib
import re
import urllib
import urllib.error
import urllib.request

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

def select_from_list(rand, lst, count):
    if count >= len(lst):
        return (lst, [])
    indices = rand.choice(len(lst), size=count, replace=False)
    indices = set(indices)
    selected = []
    others = []
    for i, item in enumerate(lst):
        if i in indices:
            selected.append(item)
        else:
            others.append(item)
    return (selected, others)

def arg_hallucinate(args):
    import torch
    import numpy.random as random
    import trek_scripts.learn as learn
    rand = random.RandomState(args.seed)
    dict_ = torch.load(args.model)
    hidden_size = dict_['hidden_size']
    model = learn.CharRnn(91, hidden_size=hidden_size)
    model.load_state_dict(dict_['model'])
    s = learn.hallucinate(model, args.hidden_size, args.max_len, rand)
    print(s)

def train_test_split(rand, data_directory, shows, test_proportion):
    '''Returns (train_episodes, test_episodes).
    
    These are paths relative to the data directory.
    '''
    episodes = []
    for show in shows:
        for child in pathlib.Path(data_directory, show).iterdir():
            if child.suffix == '.txt':
                episodes.append(pathlib.Path(*child.parts[-2:]))
    test_size = int(test_proportion * len(episodes))
    test_episodes, train_episodes = select_from_list(rand, episodes, test_size)
    return test_episodes, train_episodes
    
def batch_iter(rand, batch_size, lst):
    visited = []
    not_visited = lst
    use = []
    while not_visited:
        visited += use
        (use, not_visited) = select_from_list(rand, not_visited, batch_size)
        yield use

def arg_train(args):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    import numpy.random as random

    from trek_scripts import learn
    from trek_scripts import opts

    opts.cuda = args.cuda

    rand = random.RandomState(args.seed)

    if args.model:
        dict_ = torch.load(args.model)
        hidden_size = dict_['hidden_size']
        test_episodes = dict_['test_episodes']
        train_episodes = dict_['train_episodes']
        model = learn.CharRnn(91, hidden_size=hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(dict_['model'])
        optimizer.load_state_dict(dict_['optimizer'])
        del dict_
    else:
        shows = args.shows.split(',')
        shows = [show.strip() for show in shows]
        test_episodes, train_episodes = train_test_split(
            rand, 
            args.directory,
            shows,
            args.test_size)
        hidden_size = args.hidden_size
        model = learn.CharRnn(91, hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if opts.cuda:
        model.cuda()

    loss_f = nn.NLLLoss()

    train_paths = [pathlib.Path(args.directory, ep) for ep in train_episodes]
    test_paths = [pathlib.Path(args.directory, ep) for ep in test_episodes]
    for epoch in range(args.epochs):
        print('beginning epoch {}'.format(epoch))

        total_train_loss = 0
        for train_batch in batch_iter(rand, args.batch_size, train_paths):
            print('batch')
            strings = [open(ep).read() for ep in train_batch]
            loss = learn.train(model, hidden_size, loss_f, optimizer,
                               args.chunk_size, strings)
            total_train_loss += len(strings) * loss

        average_loss = total_train_loss / len(train_episodes)
        print('average training loss for epoch {}: {}'.format(epoch, average_loss))
        print('saving model for epoch {}'.format(epoch))
        path = pathlib.Path(args.model_directory, 'model_{:0>4}'.format(epoch))

        total_test_loss = 0
        for test_batch in batch_iter(rand, args.batch_size, test_paths):
            strings = [open(ep).read() for ep in test_batch]
            loss = learn.test(model, hidden_size, loss_f, strings)
            total_test_loss += len(strings) * loss
        average_loss = total_test_loss / len(test_episodes)
        print('average test loss for epoch {}: {}'.format(epoch, average_loss))

        torch.save({
            'test_episodes': test_episodes,
            'train_episodes': train_episodes,
            'hidden_size': hidden_size,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)

    print('Done')
            
def main():
    print(-1)
    parser = argparse.ArgumentParser(description='Generate some scripts')

    subparsers = parser.add_subparsers()

    download_parser = subparsers.add_parser('download')
    download_parser.add_argument('--url', dest='url', required=True)
    download_parser.add_argument('--directory', dest='dir', required=True)
    download_parser.set_defaults(func=arg_download)

    strip_parser = subparsers.add_parser('strip')
    strip_parser.add_argument('--directory', dest='dir', required=True)
    strip_parser.set_defaults(func=arg_strip)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument(
        '--directory', required=True,
        help='Data directory containing transcript directories'
    )
    train_parser.add_argument(
        '--model_directory', required=True,
        help='Directory in which to save models'
    )
    train_parser.add_argument(
        '--model', required=False,
        help='Saved model file to begin training with'
    )
    train_parser.add_argument(
        '--shows',
        default='TOS,TNG,DS9,VOY,ENT',
        help='Comma separated list of series to train on'
    )
    train_parser.add_argument(
        '--hidden_size', type=int, default=128,
        help='Size of the hidden layer in the GRU cell'
    )
    train_parser.add_argument(
        '--chunk_size', required=True, type=int,
    )
    train_parser.add_argument(
        '--cuda', action='store_true',
    )
    train_parser.add_argument(
        '--test_size', type=float, required=True,
        help='Proportion of transcripts to use for testing'
    )
    train_parser.add_argument(
        '--batch_size', type=int, default=10,
    )
    train_parser.add_argument(
        '--epochs', type=int, required=True,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--seed', type=int,
        help='Random number seed'
    )
    train_parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate for Adam optimizer'
    )
    train_parser.set_defaults(func=arg_train)

    hallucinate_parser = subparsers.add_parser('hallucinate')
    hallucinate_parser.add_argument(
        '--model', required=True,
        help='Saved model file to use'
    )
    hallucinate_parser.add_argument(
        '--hidden_size', required=True, type=int,
    )
    hallucinate_parser.add_argument(
        '--max_len', required=True, type=int,
    )
    hallucinate_parser.add_argument(
        '--seed', type=int,
        help='Random number seed'
    )
    hallucinate_parser.set_defaults(func=arg_hallucinate)

    print(-2)
    args = parser.parse_args()
    print(-3)
    args.func(args)
