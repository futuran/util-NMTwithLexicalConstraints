# 類似文の類似度の各順位ごとの平均の一覧を調査

import argparse
import numpy as np

def similarity(args):

    # matchファイルの解析
    with open(args.match_file, 'r') as f:
        orig_matches = f.readlines()
    matches = []
    for one_line in orig_matches:
        tmp = []
        for x in one_line.split('|||'):
            tmp.append(np.float64(x.split()[1]))
        matches.append(tmp)

    similarities = np.array(matches)
    mean_similarities = np.mean(similarities, axis=0)
    for x in mean_similarities:
        print(x)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-match_file')
    parser.add_argument('-topk', default=10)

    args = parser.parse_args()

    similarity(args)


main()


#--match-file ${DIR}.sbert_top100/match/${prefix}.match