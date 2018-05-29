
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
    regex_br = re.compile(r'<br>')
    regex = re.compile(r'<.*?>')
    regex2 = re.compile(r'^ +')
    regex3 = re.compile(r' +$')
    regex4 = re.compile(r'^\s*$')
    regex_dialogue = re.compile(r"^[A-Z' \[\]]+:")
    for child in path.iterdir():
        if child.suffix == '.html':
            with open(child, encoding='Latin-1') as f:
                lines = f.readlines()
            # <br> tags to newlines
            lines = [regex_br.sub('\n', line) for line in lines]
            lines = ''.join(lines)
            lines = lines.split('\n')
            # remove html tags
            lines = [regex.sub('', line) for line in lines]
            # remove leading spaces
            lines = [regex2.sub('', line) for line in lines]
            # remove trailing spaces
            lines = [regex3.sub('', line) for line in lines]
            # remove double spaces
            lines = [line.replace('  ', ' ') for line in lines]
            # unescape HTML
            lines = [html.unescape(line) for line in lines]
            # remove empty lines
            lines = [line for line in lines if
                     regex4.match(line) is None]

            # some odd formatting errors
            lines = [line.replace('[OC}', '[OC]') for line in lines]

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

            # get rid of unnecessary newlines
            lines0 = []
            for i in range(0, len(lines)):
                line = lines[i]
                if line.startswith('<Back'):
                    # transcript is over
                    break
                if i <= 1 or line.startswith('[') or line.startswith('(') \
                   or line.startswith('Original Airdate') \
                   or regex_dialogue.match(line):
                    # it should indeed be on its own line
                    lines0.append(line)
                else:
                    # it should be appended to the previous line
                    lines0[-1] += (' ' + line)
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
