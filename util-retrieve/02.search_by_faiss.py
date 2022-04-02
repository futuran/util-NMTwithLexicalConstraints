#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    Read a query matrix cached in the specified file.
    Calculate k-best similar vectors in TMS and TMT vectors according to inner product.
    Generate a match file recording the indices and similarity scores for every query vector.
    NOTE.
    Since the default search index is IndexFlatIP, any vectors input must be normalized ones in order to
    take the inner product as the cosine similarity.
'''

import argparse
import numpy as np
try:
    import faiss
except ImportError as e:
    raise e

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-q', '--query_emb',
                        help='Path to the embedding file where query information is saved')
    parser.add_argument('-qt', '--query_text',
                        help='Path to the query text file.')
    parser.add_argument('-tms',
                        help='Path to the embedding file where TMS is saved.')
    parser.add_argument('-tmst', '--tms_text',
                        help='Path to the TMS text file.')
    parser.add_argument('-o', '--output',
                        help='Path to the output match file.')

    parser.add_argument('-tmt', default=None,
                        help='Path to the embedding file where TMT is saved. Default: None.'
                             '\nIf set to None, only TMS will be searched.')
    parser.add_argument('-k', '--kbest', default=10, type=int,
                        help='K best. Default: 10')

    parser.add_argument('-d', '--dimension', default=512, type=int,
                        help='Dimension of all saved vectors. Default: 512')
    parser.add_argument('-dtype', default='float32',
                        help='Default data-type of the saved vectors. Default: float32')

    # parser.add_argument('--faiss-index-type', default='IndexFlatIP',
    #                     help='Index of faiss used to do k-neighbour search. Default: IndexFlatIP')

    parser.add_argument('--include-perfect-match', action='store_true',
                        help='Identical candidate is excluded in default. If you want to include it, add this option.')

    args = parser.parse_args()

    return args

def generate_match_lines(res_list):
    for res in res_list:
        # [(i1, s1), (i2, s2), ...]
        yield ' ||| '.join([f'{i} {s}' for i,s in res])

def main():

    args = parse_args()

    print('Reading text lines...')
    with open(args.query_text, 'r') as f:
        query_lines = [l.strip() for l in f]
    with open(args.tms_text, 'r') as f:
        tms_lines = [l.strip() for l in f]

    print('Reading TMS embeddings...')
    tms_emb = np.fromfile(args.tms, dtype=args.dtype).reshape(-1, args.dimension)

    if args.tmt is not None:
        print('Reading TMT embeddings...')
        tmt_emb = np.fromfile(args.tmt, dtype=args.dtype).reshape(-1, args.dimension)
    else:
        tmt_emb = None

    print('Reading query embeddings...')
    q_emb = np.fromfile(args.query_emb, dtype=args.dtype).reshape(-1, args.dimension)


    # core
    print('Building faiss GPU Index.')
    index = faiss.IndexFlatIP(args.dimension)
    index = faiss.index_cpu_to_all_gpus(index)

    print('Calculating K-Best in TMS & TMT')
    total_emb = np.vstack((tms_emb, tmt_emb)) if tmt_emb is not None else tms_emb
    index.add(total_emb)
    scores, indices = index.search(q_emb, args.kbest * 2 if tmt_emb is not None else args.kbest)

    print('Collecting results...')
    q_num = q_emb.shape[0]
    tms_num = tms_emb.shape[0]
    indices[indices>=tms_num] -= tms_num    # modify the tmt indices
    all_line_res = []

    # extract fuzzy matches for every query
    for row_i in tqdm(range(q_num), mininterval=0.5, ncols=50):
        query_text = query_lines[row_i]
        score = scores[row_i, :]
        indice = indices[row_i, :]

        res_i, res_s = [], []
        for i, match_i in enumerate(indice):
            match_s = score[i]
            if match_i in res_i:
                continue

            if args.include_perfect_match or tms_lines[match_i] != query_text:
                res_i.append(match_i)
                res_s.append(match_s)

            if len(res_i) >= args.kbest:
                break

        res = list(zip(res_i, res_s))
        all_line_res.append(res)

    with open(args.output, 'w') as f:
        for l in generate_match_lines(all_line_res):
            f.write(l + '\n')

if __name__ == '__main__':
    main()