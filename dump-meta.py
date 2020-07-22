#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pickle


def process(filename, args):

    if os.path.isdir(filename):
        filenames = [
            x for x in os.listdir(filename)
            if x.endswith('.meta')
        ]
        if not filenames:
            return
        filename = os.path.join(
            filename,
            max(filenames, key=lambda x: int(x.split('.')[0])),
        )

    if not os.path.exists(filename):
        return

    with open(filename, 'rb') as fin:
        data = pickle.load(fin)

    print('#' * 5, filename, '#' * 5)
    if args.verbose:
        for k in sorted(data):
            print('=' * 5, k, '=' * 5)
            print(data[k])
            print()
    else:
        for k in sorted(data):
            if k.startswith('best'):
                print('{}: {}'.format(k, data[k]))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('infile', nargs='+')
    args = parser.parse_args()

    for filename in args.infile:
        process(filename, args)
    

if __name__ == '__main__':
    main()

