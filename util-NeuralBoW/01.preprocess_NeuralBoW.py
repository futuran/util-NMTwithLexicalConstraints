import argparse
from dataclasses import dataclass, field
import spacy
import re

def get_stopword():
    spacy_ja = spacy.load('ja_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    sw_en = set(spacy.lang.en.stop_words.STOP_WORDS)
    sw_ja = set(spacy.lang.ja.stop_words.STOP_WORDS)
    return spacy_en, sw_en, spacy_ja, sw_ja


class Sentence():
    @dataclass
    class SentenceData:
        # sent: 文そのもの
        sent: str = ''
        # setsp: 単語ごとに分割したもの
        sentsp: list = field(default_factory=list)
        # len: 単語数＝文長
        len: int = -1

        def __init__(self, sent) -> None:
            self.sent = sent
            self.sentsp = self.sent.split()
            self.len = len(self.sentsp)


    def __init__(self, ssr ,language, sentence):
        """
            Args:
                ssr str: SRC, SIM or REF
                language: en or ja
                sentence: sentence you want to deal with.
        """
        self.language = language

        sentence = sentence.replace(' 　 ',' _ ')
        self.tkn   = self.SentenceData(sentence)

        # ssr
        if ssr not in ['SRC', 'SIM', 'REF']:
            print('warn')
        self.ssr = ssr
        if self.ssr == 'SIM':
            self.tkn_tag_list = ['O'] * self.tkn.len
            self.tkn_assist_list = [''] * self.tkn.len


def compare_sim_and_ref(sim: Sentence, ref: Sentence, stopwords):
    """
        refに含まれる単語を見て、simの各tknに参考になるか否か(=refに含まれている単語、かつ、stopwordではない)のタグを付与する関数
        Args:
            sim Sentence: similar sentence
            ref Sentence: reference
            stopword set: set of stopword 言語には無関係。
    """
    if sim.ssr != 'SIM' or ref.ssr != 'REF':
        print('warn')
        exit()

    ref_vocab_set = set(ref.tkn.sentsp)
    for i, sim_vocab in enumerate(sim.tkn.sentsp):
        if sim_vocab in ref_vocab_set and sim_vocab not in stopwords:
            if i == 0 or sim.tkn_tag_list[i-1] == 'O':
                sim.tkn_tag_list[i] = 'B'  # Assist
            else:
                sim.tkn_tag_list[i] = 'I'  # Assist
            sim.tkn_assist_list[i] = sim_vocab

        else:
            sim.tkn_tag_list[i] = 'O'  # Rubbish
            sim.tkn_assist_list[i] = '|'


    assist_pharases = ' '.join(sim.tkn_assist_list).split('|')
    assist_pharases = set([x.strip() for x in assist_pharases]) - set([''])

    return assist_pharases


def transform_format(args):

    # Load data
    src_file = args.in_dir + args.prefix + '.' + args.in_suffix_src
    with open(src_file, 'r') as f:
        src_sims_orig =  f.readlines()
    trg_file = args.in_dir + args.prefix + '.' + args.in_suffix_trg
    with open(trg_file, 'r') as f:
        ref_orig =  f.readlines()

    spacy_en, sw_en, spacy_ja, sw_ja = get_stopword()   # ストップワード作成
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

    stopwords = sw_en | sw_ja | sw_general | sw_one_bite | sw_two_bite
    # print(f'{stopwords}')

    in4oracle = []      # input for oracle
    ref4oracle = []     # tag
    in4nbow = []        # input for BERT
    ref4nbow = []       # tag
    numofsim = []       # 類似文の数のリスト

    # 単語の頻度（上位1000単語）表示
    # import collections
    # import pprint
    # tmp=[]
    # for x in ref_orig:
    #     tmp += x.strip().split()
    # tmp = collections.Counter(tmp)
    # print(len(tmp))
    # pprint.pprint(tmp.most_common()[:1000])

    for i, (src_sims, ref) in enumerate(zip(src_sims_orig, ref_orig)):
        # 類似文と参照訳文を比較しながら、
        # 類似文中の単語で、参照訳文にも含まれる単語については参考になることを表すタグを付与する。
        
        tmp = src_sims.strip().split(' {} '.format(args.src_split_token))
        src  =  Sentence('SRC', 'en', tmp[0])
        sims = [Sentence('SIM', 'en', tmp[i]) for i in range(1,len(tmp))][:int(args.topk)]
        ref  =  Sentence('REF', 'en', ref.strip())

        #print(len(sims))

        if i % 1000 == 0:
            print('i:{}'.format(i))

        j = 0
        assist_vocabs = set()
        for j, sim in enumerate(sims):

            assist_vocabs |= set(compare_sim_and_ref(sim, ref, stopwords))

            in4nbow.append('{} {} {}\n'.format(src.tkn.sent, args.out_split_token, sim.tkn.sent))
            ref4nbow.append('{} O {}\n'.format(' '.join(['O'] * src.tkn.len), ' '.join(sim.tkn_tag_list)))

        # 部分的な重複を排除
        # あるフレーズが別のフレーズの部分集合になっている時は削除
        assist_vocabs = list(assist_vocabs)
        assist_vocabs.sort(key=len)
        uniq_assist_vocabs = []
        for current_id in range(len(assist_vocabs)):
            for biggers_id in range(current_id+1, len(assist_vocabs)):
                #print(assist_vocabs[current_id], assist_vocabs[biggers_id])
                if assist_vocabs[current_id] in assist_vocabs[biggers_id]:
                    break
            else:
                # forがbreakで抜けなかったとき
                uniq_assist_vocabs.append(assist_vocabs[current_id])

        # print(uniq_assist_vocabs)
        # if i == 3:
        #     exit()

        in4oracle.append('{} {} {}\n'.format(src.tkn.sent, args.out_split_token, ' {} '.format(args.out_split_token).join(uniq_assist_vocabs)))
        ref4oracle.append('{}\n'.format(ref.tkn.sent))
        
        numofsim.append('{}\n'.format(j+1))



    out_oracle_path = args.out_oracle_dir + args.prefix
    with open(out_oracle_path + '.' + args.out_suffix_src, 'w') as f:
        f.writelines(in4oracle)
    with open(out_oracle_path + '.' + args.out_suffix_trg, 'w') as f:
        f.writelines(ref4oracle)

    out_NBoW_path = args.out_NBoW_dir + args.prefix
    with open(out_NBoW_path + '.in4nbow', 'w') as f:
        f.writelines(in4nbow)
    with open(out_NBoW_path + '.ref4nbow', 'w') as f:
        f.writelines(ref4nbow)
    with open(out_NBoW_path + '.numofsim', 'w') as f:
        f.writelines(numofsim)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prefix')

    # input
    parser.add_argument('-in_dir')
    parser.add_argument('-in_suffix_src')
    parser.add_argument('-in_suffix_trg')

    # output
    parser.add_argument('-out_oracle_dir')
    parser.add_argument('-out_NBoW_dir')
    parser.add_argument('-out_suffix_src')
    parser.add_argument('-out_suffix_trg')

    parser.add_argument('-topk')

    parser.add_argument('-src_split_token', default='|')
    parser.add_argument('-out_split_token', default='|')

    args = parser.parse_args()

    transform_format(args)


main()