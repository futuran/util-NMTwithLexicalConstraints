#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器


# Setting of Logger
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
logger = getLogger(__name__)
logger.setLevel(DEBUG)

sh = StreamHandler()
logger.addHandler(sh)
fh = FileHandler('vector.log')
logger.addHandler(fh)
formatter = Formatter(
    '%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)

def summary(arr):
    return pd.DataFrame(pd.Series(arr.ravel()).describe()).transpose()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-en_q', '--en_emb')
    parser.add_argument('-ja_q', '--ja_emb')

    parser.add_argument('-d', '--dimension', default=768, type=int,
                        help='Dimension of all saved vectors. Default: 768')
    parser.add_argument('-dtype', default='float32',
                        help='Default data-type of the saved vectors. Default: float32')

    args = parser.parse_args()

    return args

def pca(df):
    # dfs = df.iloc[:, :-1].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    dfs = df.iloc[:, :-1]#.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    print(dfs)
    pca = PCA()
    pca.fit(dfs)
    # データを主成分空間に写像
    feature = pca.transform(dfs)
    # 主成分得点
    # pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]).head()
    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(6, 6))
    # plt.scatter(feature[:, 383], feature[:, 384], alpha=0.8, c=list(df.iloc[:, -1]))
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, -1]))
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    plt.savefig('pca.png')

    # PCA の固有値
    koyuchi = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    # print(koyuchi.values)
    # koyuchi.values.to_csv('eigenvalue.csv')
    np.savetxt('eigenvalue.csv', koyuchi.values)

    # plt.grid()
    # plt.bar(range(len(koyuchi)), koyuchi, width=1)
    # plt.xlabel("rank")
    # plt.ylabel("size")
    # plt.show()
    # plt.savefig('pcaed_size.png')



def main():

    args = parse_args()
    logger.debug('Reading English embeddings...')
    en_emb = np.fromfile(
        args.en_emb, dtype=args.dtype).reshape(-1, args.dimension)
    ja_emb = np.fromfile(
        args.ja_emb, dtype=args.dtype).reshape(-1, args.dimension)

    # 🐼
    df_en_emb = pd.DataFrame(en_emb)
    df_ja_emb = pd.DataFrame(ja_emb)
    df_en_emb['lang'] = 0
    df_ja_emb['lang'] = 1

    # ノルムに関する基本統計量
    print('En/Norm:\n',summary(np.linalg.norm(en_emb, axis=1)))
    print('Ja/Norm:\n',summary(np.linalg.norm(ja_emb, axis=1)))


    print(np.linalg.norm(np.mean(en_emb, axis=0)))
    print(np.linalg.norm(np.mean(ja_emb, axis=0)))
    print(np.linalg.norm(np.mean(en_emb, axis=0) - np.mean(ja_emb, axis=0)))

    print('En/Norm:\n',summary(np.mean(en_emb, axis=0)))
    print('Ja/Norm:\n',summary(np.mean(ja_emb, axis=0)))


    # PCA
    pca(pd.concat([df_en_emb, df_ja_emb]))
    # pca(df_ja_emb)




if __name__ == '__main__':
    main()
