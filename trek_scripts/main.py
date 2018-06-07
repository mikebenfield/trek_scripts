
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
    # for name in ["DS9"]:
        path = pathlib.PurePath(args.dir, name)
        strip_html(path)

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

    args = parser.parse_args()
    args.func(args)
