# 語彙制約で参照訳文をどれだけカバーできるかを調査

import argparse
from itertools import count
from statistics import multimode
import spacy
import numpy as np

from matplotlib import pyplot as plt

en = spacy.load('en_core_web_sm')
ja = spacy.load('ja_core_news_sm')
en_stopwords = en.Defaults.stop_words
ja_stopwords = ja.Defaults.stop_words

sw_general = set([  'について',
                    'として', 
                    'による', 'により', 'によって', 
                    'これら',
                    'における', 'において',
                    'に関する', 'に関して',
                    'に対する', 'に対して',
                    '及び', 'および'])

sw_one_bite = set(list('1234567890!"#$%&\'()*+-.,/:;<=>?@[]^_`{|}~¥ '))
sw_two_bite = set(list('１２３４５６７８９０！”＃＄％＆’（）＊＋ー。、＼：；＜＝＞？＠「」＾＿｀『｜』〜￥　，．・／‐［］'))

additional_stopword = sw_general | sw_one_bite | sw_two_bite


# OracleとProposedの重複率
def duplicate_rate(oracle_list, proposed_list):
    counts = [0,0,0]
    # [Oracleにのみ含まれる単語数, どちらにも含まれる単語数, Proposedにのみ含まれる単語数]
    # 1文の中で複数回同じ単語が出てきてもそれは１回として数える
    for oracle, proposed in zip(oracle_list, proposed_list):
        counts[0] += len(set(oracle) - set(proposed))
        counts[1] += len(set(oracle) & set(proposed))
        counts[2] += len(set(proposed) - set(oracle))

    # print("[Oracleにのみ含まれる単語数, どちらにも含まれる単語数, Proposedにのみ含まれる単語数]")
    out = np.array([x/len(oracle_list) for x in counts])
    return out

# 制約語彙によるリファレンス文のカバー率
def cover_rate_lexiconst(ref_list, op_list):
    counts = [0,0,0]
    # [refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]
    for ref, op in zip(ref_list, op_list):
        ref_vocab = set(ref) - set(ja_stopwords) - set(additional_stopword)
        counts[0] += len(ref_vocab - set(op))
        counts[1] += len(ref_vocab & set(op))
        counts[2] += len(set(op) - ref_vocab)
    
    # print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")
    out = np.array([x/len(ref_list) for x in counts])
    return out

# 類似文によるリファレンス文のカバー率
def cover_rate_with_sim(ref_list, sim_list):
    counts = [0,0,0]
    # [refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]
    for ref, sim in zip(ref_list, sim_list):
        ref_vocab = set(ref) - set(ja_stopwords) - set(additional_stopword)
        sim_vocab = set(sim) - set(ja_stopwords) - set(additional_stopword)
        counts[0] += len(ref_vocab - sim_vocab)
        counts[1] += len(ref_vocab & sim_vocab)
        counts[2] += len(sim_vocab - ref_vocab)
    
    # print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")
    out = np.array([x/len(ref_list) for x in counts])
    return out


# 制約語彙によるリファレンス文のカバー率と、それらがベースラインで翻訳できたか否かを分類
def componet_raio_of_ref(ref_list, op_list, out_list):
    counts = [0,0,0,0]
    # refの単語うち...
    # [outにもopにも含まれる単語数, opに含まれoutに含まれない単語数, outに含まれopに含まれない単語数, どちらにも含まれない単語数]
    for ref, op, out in zip(ref_list, op_list, out_list):
        ref_vocab = set(ref) - set(ja_stopwords) - set(additional_stopword)
        out_vocab = set(out) - set(ja_stopwords) - set(additional_stopword)
        counts[0] += len(ref_vocab & out_vocab & set(op))
        counts[1] += len(ref_vocab & set(op) - out_vocab)
        counts[2] += len(ref_vocab & out_vocab - set(op))
        counts[3] += len(ref_vocab - set(op) - out_vocab)

    out = np.array([x/len(ref_list) for x in counts])
    return out


def load(args):
    with open(args.ref) as f:
        ref_list = f.readlines()
    ref_list = [x.strip().split() for x in ref_list]

    if args.out != None:
        with open(args.out) as f:
            out_list = f.readlines()
        out_list = [x.strip().split() for x in out_list]
    else:
        out_list = None


    if args.sim != None:
        with open(args.sim) as f:
            sim_list = f.readlines()
        multiple_sim_list = []
        for i in range(int(args.topk)):
            tmp_sim_list =[]
            for sim in sim_list:
                tmp = ' '.join(sim.strip().split('|')[1:i+2]).split()
                tmp_sim_list.append(tmp)
            multiple_sim_list.append(tmp_sim_list)

        sim_list = [x[x.index('|')+1:].strip().split() for x in sim_list]
    else:
        multiple_sim_list = None

    if args.oracle != None:
        with open(args.oracle) as f:
            oracle_list = f.readlines()
        oracle_list = [x[x.index('|')+1:].replace('|','').strip().split() for x in oracle_list]
    else:
        oracle_list = None

    if args.proposed != None:
        with open(args.proposed) as f:
            proposed_list = f.readlines()
        proposed_list = [x[x.index('|')+1:].strip().split() for x in proposed_list]
    else:
        proposed_list = None

    return ref_list, out_list, multiple_sim_list, oracle_list, proposed_list

def load_multiple_proposed_list(args):
    multiple_proposed_list = []

    for i in range(100):
        with open(args.multiple_proposed + str(i+1)) as f:
            proposed_list = f.readlines()
        multiple_proposed_list.append([x[x.index('|')+1:].strip().split() for x in proposed_list])

    return multiple_proposed_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ref')
    parser.add_argument('-out', default=None)
    parser.add_argument('-sim', default=None)
    parser.add_argument('-oracle', default=None)
    parser.add_argument('-proposed',  default=None)
    parser.add_argument('-multiple_proposed',  default=None)
    parser.add_argument('-topk', default=10)

    args = parser.parse_args()

    ref_list, out_list, multiple_sim_list, oracle_list, proposed_list = load(args)
    # multiple_proposed_list = load_multiple_proposed_list(args)

    # if args.oracle != None and args.proposed != None:
    #     print('OracleとProposedの重複率：')
    #     print("[Oracleにのみ含まれる単語数, どちらにも含まれる単語数, Proposedにのみ含まれる単語数]")
    #     out = duplicate_rate(oracle_list, proposed_list)
    #     print(np.round(out, 2))

    # # ReferenceとOracleの重複率
    # if args.oracle != None:
    #     print('\nReferenceとOracleの重複率：')
    #     print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")
    #     out = cover_rate_lexiconst(ref_list, oracle_list)
    #     print(np.round(out, 2))

    # # ReferenceとProposedの重複率
    # if args.proposed != None:
    #     print('\nReferenceとProposedの重複率：')
    #     print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")
    #     out = cover_rate_lexiconst(ref_list, proposed_list)
    #     print(np.round(out, 2))
    # if args.multiple_proposed != None:
    #     print('\nReferenceとProposedの重複率：')
    #     print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")    
    #     for i in range(int(args.topk)):
    #         out = cover_rate_lexiconst(ref_list, multiple_proposed_list[i])
    #         print(np.round(out, 2))


    # # Referenceと第１類似文〜第i類似文の重複率
    if args.sim != None:   
        coverages_ref_sim = np.zeros((int(args.topk),3))
        for i in range(int(args.topk)):
            # print('\nReferenceと第１類似文〜第{}類似文の重複率：'.format(i+1))
            # print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")
            coverages_ref_sim[i] = cover_rate_with_sim(ref_list, multiple_sim_list[i])
            # print(np.round(out, 2))
        print('\nReferenceと第１類似文〜第100類似文の重複率：')
        print("[refにのみ含まれる単語数, どちらにも含まれる単語数, opにのみ含まれる単語数]")
        print(np.round(coverages_ref_sim,2))


    # 制約語彙によるリファレンス文のカバー率と、それらがベースラインで翻訳できたか否かを分類
    # if args.out != None and args.multiple_proposed != None:
    #     componet_raio_of_ref_list = np.zeros((int(args.topk),4))
    #     for i in range(int(args.topk)):
    #         componet_raio_of_ref_list[i] = componet_raio_of_ref(ref_list,multiple_proposed_list[i], out_list)
            
    #     print(np.round(componet_raio_of_ref_list,2))
    #     np.savetxt('tmp1.csv', componet_raio_of_ref_list, delimiter=',', fmt='%f')

    # if args.out != None and args.multiple_proposed != None:
    #     componet_raio_of_ref_list = np.zeros((int(args.topk),4))
    #     for i in range(int(args.topk)):
    #         componet_raio_of_ref_list[i] = componet_raio_of_ref(ref_list,multiple_sim_list[i], out_list)
            
    #     print(np.round(componet_raio_of_ref_list,2))
    #     np.savetxt('tmp2.csv', componet_raio_of_ref_list, delimiter=',', fmt='%f')

main()