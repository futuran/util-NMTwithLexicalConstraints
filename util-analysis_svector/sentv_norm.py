#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE='文埋め込みのノルムと文の長さに相関があるかを調査'

import argparse
from cgitb import text
import numpy as np
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-e', '--emb')
    parser.add_argument('-t', '--text')
    parser.add_argument('-d', '--dimension', default=768, type=int,
                        help='Dimension of all saved vectors. Default: 512')
    parser.add_argument('-dtype', default='float32',
                        help='Default data-type of the saved vectors. Default: float32')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    print('Reading text...')
    with open(args.text, 'r') as f:
        texts = [l.strip() for l in f]

    print('Reading embeddings...')
    embs = np.fromfile(args.emb, dtype=args.dtype).reshape(-1, args.dimension)

    print(f'{len(texts)=}')
    print(f'{len(embs)=}')
    assert len(texts) == len(embs)

    for text, emb in zip(texts, embs):
        # print(len(text.split()))
        # print(type(emb))
        print(np.linalg.norm(emb))


if __name__ == '__main__':
    main()