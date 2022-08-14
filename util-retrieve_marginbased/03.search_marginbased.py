#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from audioop import reverse
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s2t', '--src_to_tgt_retrieve',
                        help='Path to the match file that holds the results of retrieving tgt sentence using src sentence query')
    parser.add_argument('-t2s', '--tgt_to_src_retrieve',
                        help='Path to the match file that holds the results of retrieving src sentence using tgt sentence query')

    parser.add_argument('-o', '--output',
                        help='Path to the output match file.')

    parser.add_argument('-k', '--kbest', default=10, type=int,
                        help='K best. Default: 10')

    args = parser.parse_args()

    return args

def generate_match_lines(res_list):
    for res in res_list:
        # [(i1, s1), (i2, s2), ...]
        yield ' ||| '.join([f'{i} {s}' for i,s in res])


def main():

    args = parse_args()

    def load_match_file(path):
        # format of the return is as follows
        # [ [[id, score], [id, score], ... [id, score]], # sentence No.1
        #   [[id, score], [id, score], ... [id, score]], # sentence No.2
        #   ...
        #   [[id, score], [id, score], ... [id, score]], # Last sentence
        # ]
        with open(path, 'r') as f:
            out_list = []
            for l in f:
                tmp = []
                for ll in l.strip().split(' ||| '):
                    idx, score = (ll.split())
                    idx = int(idx)
                    score = float(score)
                    tmp.append((idx, score))
                out_list.append(tmp)
            return out_list

    print('Loading match files...')
    src_match = load_match_file(args.src_to_tgt_retrieve)
    tgt_macth = load_match_file(args.tgt_to_src_retrieve)

    print(len(src_match))
    print(len(tgt_macth))

    new_src_match = []
    new_src_match_args = []

    # score = A / ( B + C )
    for idx_s, Ns in enumerate(tqdm(src_match)):      # Ns stands for N_x in the paper
        B = np.sum([float(knn_s[1]) for knn_s in Ns]) / len(Ns)

        tmp = []
        for knn_s in Ns:
            # print(knn_s)
            A = knn_s[1]
            C = np.sum([float(knn_t[1]) for knn_t in tgt_macth[knn_s[0]]]) / len(tgt_macth[knn_s[0]])

            score = A / ( B + C )

            tmp.append((knn_s[0], score))

        new_src_match_args.append(np.argsort(-np.array([x[1] for x in tmp])))
        y = np.argsort(-np.array([x[1] for x in tmp]))
        import matplotlib.pyplot as plt
        x = [i for i in range(len(y))]
        plt.scatter(x, y, s=1)
        # exit()
        if idx_s == 20:
            break

        tmp.sort(key=lambda x: x[1], reverse=True)
        new_src_match.append(tmp)
    

    plt.xlabel("Original Rank")
    plt.ylabel("Reranked Rank")
    plt.grid(True)
    plt.savefig("fig.png",  dpi=300)
    import csv
    with open('./data.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(new_src_match_args)


    # with open(args.output, 'w') as f:
    #     for l in generate_match_lines(new_src_match):
    #         f.write(l + '\n')


if __name__ == '__main__':
    main()