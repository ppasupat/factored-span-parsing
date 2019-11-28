#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit job with nlprun (Stanford internal).
"""
from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import sys


BASEDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'out',
))


def main():
    # Usage: ./submit.py NAME ARG1 ARG2 ...
    execs = [
        int(filename.split('.')[0])
        for filename in os.listdir(BASEDIR)
        if filename.split('.')[0].isdigit()
    ]
    outname = str(0 if not execs else max(execs) + 1) + '.' + sys.argv[1]
    outdir = os.path.join(BASEDIR, outname)
    command = ['./main.py', '-o', outname] + sys.argv[2:]
    wrapped_command = [
        'nlprun',
        '-a', 'ppasupat-yay',
        '-q', 'jag',
        '-o', os.path.join(outdir, 'nlprun.out'),
        ' '.join(command),
    ]
    print(wrapped_command)
    if input('Is this OK? (y/N): ').lower() != 'y':
        print('Bye!')
        exit(1)
    os.makedirs(outdir)
    subprocess.run(wrapped_command)
    

if __name__ == '__main__':
    main()

