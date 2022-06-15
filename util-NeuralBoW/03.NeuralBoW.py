from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
import logging
import spacy

import argparse
import itertools
import pprint


def get_stopword():
    spacy_ja = spacy.load('ja_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    sw_en = set(spacy.lang.en.stop_words.STOP_WORDS)
    sw_ja = set(spacy.lang.ja.stop_words.STOP_WORDS)
    return spacy_en, sw_en, spacy_ja, sw_ja


def load_data(args, only_test=False):
    def load_each_data(in4nbow, ref4nbow, shuffle=False):
        with open(in4nbow, 'r') as f:
            input = f.readlines()
        with open(ref4nbow, 'r') as f:
            ref = f.readlines()
        data = []
        for i, (in_sent, ref_sent) in enumerate(zip(input, ref)):
            if len(in_sent.split()) != len(ref_sent.split()):
                print('warning')
                exit()
            else:
                for j, (in_vocab, ref_vocab) in enumerate(zip(in_sent.split(), ref_sent.split())):
                    data.append([i, in_vocab, ref_vocab])

        return data

    print('loading data...')

    train_df = None
    dev_df = None

    if only_test == False:
        train_data = load_each_data(args.train_in4nbow, args.train_ref4nbow)
        dev_data = load_each_data(args.dev_in4nbow, args.dev_ref4nbow)

        train_df = pd.DataFrame(train_data, columns=[
                                'sentence_id', 'words', 'labels'])
        dev_df = pd.DataFrame(
            dev_data, columns=['sentence_id', 'words', 'labels'])

    test_data = load_each_data(args.test_in4nbow, args.test_ref4nbow)
    test_df = pd.DataFrame(test_data, columns=[
                           'sentence_id', 'words', 'labels'])

    return train_df, dev_df, test_df


def train(args, train_df, dev_df, test_df):
    print('start training...')

    # ラベル設定
    model_args = NERArgs()
    model_args.labels_list = ['B', 'I', 'O']
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.use_cuda = True
    model_args.max_seq_length = 512
    model_args.output_dir = str(args.output_dir) + '/outputs'
    model_args.best_model_dir = str(args.output_dir) + '/outputs/best_model'
    model_args.tensorboard_dir = str(args.output_dir) + '/'
    model_args.cache_dir = str(args.output_dir) + '/cache_dir'
    model_args.train_batch_size = 16
    print(f'{model_args=}')

    # モデルの作成
    model = NERModel(
        'xlmroberta',
        'xlm-roberta-large',
        args=model_args,
        use_cuda=True,
        cuda_device=1)

    # 学習
    model.train_model(train_df)

    # 評価
    result, model_outputs, predictions = model.eval_model(dev_df)
    print(f'dev /{result=}')

    return 0


def eval(args):
    with open(args.test_in4nbow, 'r') as f:
        input = f.readlines()
    with open(args.test_ref4nbow, 'r') as f:
        ref = f.readlines()

    # ラベル設定
    model_args = NERArgs()
    model_args.labels_list = ['B', 'I', 'O']
    model_args.reprocess_input_data = True
    model_args.use_cuda = False
    model_args.max_seq_length = 512

    # モデルの作成
    model = NERModel(
        'xlmroberta',
        str(args.output_dir) + '/outputs',
        args=model_args)

    predictions, raw_outputs = model.predict(input)

    # 分析
    bio_counts = [[0] * 3 for i in range(3)]
    # 各正解と予測の組について、その実数をカウント。
    # BIOの順で格納。0行目は、正解がBで、予測が順にBIO。
    # ref\pred | B | I | O
    #    B     |   |   |
    #    I     |   |   |
    #    O     |   |   |

    for i, (input_sent, ref_tag, preds) in enumerate(zip(input, ref, predictions)):
        pred_vocab_list = [list(x.keys())[0] for x in preds]
        pred_tag_list = [list(x.values())[0] for x in preds]
        pred_length = len(pred_vocab_list)

        # BERTの出力は後半が一部欠損するので、Outputと同じ長さだけ切り出す。
        in_vocab_list = input_sent.strip().split()[:pred_length]
        ref_tag_list = ref_tag.strip().split()[:pred_length]

        for pred_vocab, pred_tag, ref_tag in zip(pred_vocab_list, pred_tag_list, ref_tag_list):
            ref_tag_index = ['B', 'I', 'O'].index(ref_tag)
            pred_tag_index = ['B', 'I', 'O'].index(pred_tag)
            bio_counts[ref_tag_index][pred_tag_index] += 1

    pprint.pprint(bio_counts)

    return 0


def predict(args):
    def predidct_one(in4nbow, ref4nbow, numofsim, out_name, topk=10):
        with open(in4nbow, 'r') as f:
            input = f.readlines()
        with open(ref4nbow, 'r') as f:
            ref = f.readlines()
        with open(numofsim, 'r') as f:
            numofsim = f.readlines()
        numofsim_list = [int(x.strip()) for x in numofsim]

        # for debug
        # input = input[:1000]
        # ref = ref[:1000]
        # numofsim_list = numofsim_list[:1000]
        ##############

        # ラベル設定
        model_args = NERArgs()
        model_args.labels_list = ['B', 'I', 'O']
        model_args.reprocess_input_data = True
        model_args.use_cuda = True
        model_args.max_seq_length = 512
        model_args.eval_batch_size = 128

        model_args.use_multiprocessing_for_evaluation = True

        # モデルの作成
        model = NERModel(
            'xlmroberta',
            str(args.output_dir) + '/outputs',
            args=model_args,
            use_cuda=True)

        predictions = []
        chunk_size = 10000
        div_num = len(input) // chunk_size + 1
        print('# of Division: {}'.format(div_num))
        for i in range(div_num):
            print('No.{}/{}'.format(i+1, div_num))
            _predictions, _raw_outputs = model.predict(
                input[chunk_size * i:chunk_size * (i+1)])
            predictions += _predictions

        # 分析
        bio_counts = [[0] * 3 for i in range(3)]
        # 各正解と予測の組について、その実数をカウント。
        # BIOの順で格納。0行目は、正解がB/Iで、予測が順にBIO。
        # ref\pred | B | I | O
        #    B     |   |   |
        #    I     |   |   |
        #    O     |   |   |

        assist_vocabs = []
        out_sent_list = []

        # 分析用
        out_sent_list_with_arbitral_simnum = [[] for _ in range(int(topk))]

        # ストップワード作成
        spacy_en, sw_en, spacy_ja, sw_ja = get_stopword()
        sw_general = set(['について',
                          'として',
                          'による', 'により', 'によって',
                          'これら',
                          'における', 'において',
                          'に関する', 'に関して',
                          'に対する', 'に対して',
                          '及び', 'および'])
        sw_one_bite = set(list('1234567890!"#$%&\'()*+-.,/:;<=>?@[]^_`{|}~¥ '))
        sw_two_bite = set(
            list('１２３４５６７８９０！”＃＄％＆’（）＊＋ー。、＼：；＜＝＞？＠「」＾＿｀『｜』〜￥　，．・／‐［］'))
        stopwords = sw_en | sw_ja | sw_general | sw_one_bite | sw_two_bite

        sentence_no = 0  # 文番号。1文当たり約10文の類似文があるが、ものによって異なるためこの変数で管理。
        sim_count = 0   # 類似文の数

        assist_phrases = ''

        for i, (input_sent, ref_tag, preds) in enumerate(zip(input, ref, predictions)):
            pred_vocab_list = [list(x.keys())[0] for x in preds]
            pred_tag_list = [list(x.values())[0] for x in preds]
            pred_length = len(pred_vocab_list)

            # BERTの出力は後半が一部欠損するので、Outputと同じ長さだけ切り出す。
            in_vocab_list = input_sent.strip().split()[:pred_length]
            ref_tag_list = ref_tag.strip().split()[:pred_length]

            out_sent = ' '.join(in_vocab_list).split(' | ')[0] + ' | '

            for pred_vocab, pred_tag, ref_tag in zip(pred_vocab_list, pred_tag_list, ref_tag_list):
                ref_tag_index = ['B', 'I', 'O'].index(ref_tag)
                pred_tag_index = ['B', 'I', 'O'].index(pred_tag)
                bio_counts[ref_tag_index][pred_tag_index] += 1

                pred_vocab = pred_vocab.strip()
                if pred_tag in list('BI') and pred_vocab not in stopwords:
                    assist_phrases += pred_vocab + ' '
                else:
                    assist_phrases += '| '

            # 利用した類似文の数をカウントアップ
            sim_count += 1

            # 分析用 ##########################################################################
            # 利用する類似文の数ごとに分けて処理。##################################################

            assist_phrases_for_analyze = assist_phrases.split('|')
            assist_phrases_for_analyze = list(
                set([x.strip() for x in assist_phrases_for_analyze]) - set(['']))

            # 部分的な重複を排除
            # あるフレーズが別のフレーズの部分集合になっている時は削除
            assist_phrases_for_analyze.sort(key=len)
            uniq_assist_phrases_for_analyze = []
            for current_id in range(len(assist_phrases_for_analyze)):
                for biggers_id in range(current_id+1, len(assist_phrases_for_analyze)):
                    if assist_phrases_for_analyze[current_id] in assist_phrases_for_analyze[biggers_id]:
                        break
                else:
                    # forがbreakで抜けなかったとき
                    uniq_assist_phrases_for_analyze.append(
                        assist_phrases_for_analyze[current_id])

            out_sent_for_analyze = out_sent + \
                ' {} '.format('|').join(uniq_assist_phrases_for_analyze)

            if sim_count == numofsim_list[sentence_no]:
                for j in range(sim_count-1, int(topk)):
                    out_sent_list_with_arbitral_simnum[j].append(
                        out_sent_for_analyze + '\n')
            else:
                out_sent_list_with_arbitral_simnum[sim_count -
                                                   1].append(out_sent_for_analyze + '\n')
            ##################################################################################

            # 類似文の数(sim_count)が、numofsim_listに格納されている文数に一致すると、
            # assis_vocabが参考情報としてまとめられ、リセットされる
            if sim_count == numofsim_list[sentence_no]:
                assist_phrases = assist_phrases.split('|')
                assist_phrases = set([x.strip()
                                     for x in assist_phrases]) - set([''])

                # 部分的な重複を排除
                # あるフレーズが別のフレーズの部分集合になっている時は削除
                assist_phrases = list(assist_phrases)
                assist_phrases.sort(key=len)
                uniq_assist_phrases = []
                for current_id in range(len(assist_phrases)):
                    for biggers_id in range(current_id+1, len(assist_phrases)):
                        #print(assist_phrases[current_id], assist_phrases[biggers_id])
                        if assist_phrases[current_id] in assist_phrases[biggers_id]:
                            break
                    else:
                        # forがbreakで抜けなかったとき
                        uniq_assist_phrases.append(assist_phrases[current_id])

                out_sent += ' {} '.format('|').join(uniq_assist_phrases)
                out_sent_list.append(out_sent + '\n')

                # Reset
                assist_phrases = ''
                sentence_no += 1
                sim_count = 0

        pprint.pprint(bio_counts)

        with open(str(args.output_dir) + '/' + str(out_name), 'w') as f:
            f.writelines(out_sent_list)

        # 分析用 ##########################################################################
        for i in range(int(args.topk_of_sim)):
            with open(str(args.output_dir) + '/topk/' + str(out_name) + '.with_upto_No.' + str(i+1), 'w') as f:
                f.writelines(out_sent_list_with_arbitral_simnum[i])
        ##################################################################################

        return 0

    predidct_one(args.test_in4nbow, args.test_ref4nbow, args.test_numofsim, str(
        args.corpus_name)+'_test' + args.output_file_suffix, args.topk_of_sim)
    # predidct_one(args.dev_in4nbow, args.dev_ref4nbow, args.dev_numofsim, 'aspec_dev.assisted.src.tkn')
    # predidct_one(args.train_in4nbow, args.train_ref4nbow, args.train_numofsim, 'aspec_train_h40000.assisted.src.tkn')

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_in4nbow')
    parser.add_argument('-train_ref4nbow')
    parser.add_argument('-train_numofsim')

    parser.add_argument('-dev_in4nbow')
    parser.add_argument('-dev_ref4nbow')
    parser.add_argument('-dev_numofsim')

    parser.add_argument('-test_in4nbow')
    parser.add_argument('-test_ref4nbow')
    parser.add_argument('-test_numofsim')

    parser.add_argument('-corpus_name')
    parser.add_argument('-output_dir')

    #parser.add_argument('-split_token', default='|')

    #parser.add_argument('-model', default='bert-base-multilingual-cased')

    parser.add_argument('-only_predict', default=False)

    parser.add_argument('-big_boundary', default='|')
    parser.add_argument('-topk_of_sim', default=10)
    parser.add_argument('-output_file_suffix',  default='.assisted.src.tkn')

    args = parser.parse_args()

    # ログの設定
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    if args.only_predict == False:
        train_df, dev_df, test_df = load_data(args)
        train(args, train_df, dev_df, test_df)
        eval(args)
        pass
    predict(args)


main()
