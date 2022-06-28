#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import argparse
from SetSimilaritySearch import SearchIndex
from tqdm import tqdm
from multiprocessing import Pool, Lock

def generate_match_lines(res_list):
    for res in res_list:
        # [(i1, s1), (i2, s2), ...]
        yield ' ||| '.join([f'{i} {s}' for i, s in res])


def extract(ql, tgt_index, tgt_lines, include_perfect_match, kbest):
    # print(ql)

    # extract fuzzy matches for every query
    # for ql in tqdm(q_lines, mininterval=0.5, ncols=50):
    results = tgt_index.query(ql.split())
    # print(results)

    indice = [x[0] for x in results]
    score  = [x[1] for x in results]

    res_i, res_s = [], []
    for i, (match_i, match_s) in enumerate(zip(indice, score)):
        if match_i in res_i:
            continue

        if include_perfect_match or ql != tgt_lines[match_i]:
            res_i.append(match_i)
            res_s.append(match_s)

        if len(res_i) >= kbest:
            break

    res = list(zip(res_i, res_s))
    return res

def wrap_extract(arg):
    return extract(*arg)

def main():

    args = parse_args()

    print('Reading data...')
    with open(args.tms, 'r') as f:
        src_lines = [l.strip() for l in f]
        if args.remove_bpe_in_corpus:
            src_lines = [l.replace('@@ ', '') for l in src_lines]

    with open(args.tmt, 'r') as f:
        tgt_lines = [l.strip() for l in f]
        if args.remove_bpe_in_corpus:
            tgt_lines = [l.replace('@@ ', '') for l in tgt_lines]

    with open(args.query, 'r') as f:
        q_lines = [l.strip() for l in f]
        if args.remove_bpe_in_corpus:
            q_lines = [l.replace('@@ ', '') for l in q_lines]

    assert len(src_lines) == len(tgt_lines)

    print('Indexing...')
    tgt_index = SearchIndex([l.split() for l in tgt_lines],
                            similarity_func_name="containment", similarity_threshold=0.5)

    print('Searching...')
    all_line_res = []

    # def extract(x):
    #     ql = q_lines[x]
    #     # extract fuzzy matches for every query
    #     # for ql in tqdm(q_lines, mininterval=0.5, ncols=50):
    #     results = tgt_index.query(ql.split())

    #     indice = [x[0] for x in results]
    #     score  = [x[1] for x in results]

    #     res_i, res_s = [], []
    #     for i, (match_i, match_s) in enumerate(zip(indice, score)):
    #         if match_i in res_i:
    #             continue

    #         if args.include_perfect_match or ql != tgt_lines[match_i]:
    #             res_i.append(match_i)
    #             res_s.append(match_s)

    #         if len(res_i) >= args.kbest:
    #             break

    #     res = list(zip(res_i, res_s))
    #     return res

    p = Pool(64)
    # all_line_res = p.map(wrap_extract, [(ql, tgt_index, tgt_lines, args.include_perfect_match, args.kbest) for ql in q_lines])
    imap = p.map(wrap_extract, [(ql, tgt_index, tgt_lines, args.include_perfect_match, args.kbest) for ql in q_lines])
    result = list(tqdm(imap, total=len(q_lines)))
    
    print(all_line_res)

    with open(args.output, 'w') as f:
        for l in generate_match_lines(all_line_res):
            f.write(l + '\n')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--tms', required=True,
                        help='Path to the source corpus file.')
    parser.add_argument('-t', '--tmt', required=True,
                        help='Path to the target corpus file.')

    parser.add_argument('--remove-bpe-in-corpus', action='store_true',
                        help='If there are BPE marks in corpus, remove them before input the sentences into SBERT.')

    parser.add_argument('-q', '--query', required=True,
                        help='Path to the embedding file where query information is saved')

    parser.add_argument('-k', '--kbest', default=10, type=int,
                        help='K best. Default: 10')

    parser.add_argument('-o', '--output',
                        help='Path to the output match file.')

    parser.add_argument('--include-perfect-match', action='store_true',
                        help='Identical candidate is excluded in default. If you want to include it, add this option.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
