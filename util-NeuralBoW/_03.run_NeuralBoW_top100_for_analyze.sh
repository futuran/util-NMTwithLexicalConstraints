# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}.sbert_top100.NBoW

# ${CORPUS}のみ
python ../../NeuralBoW/NeuralBoW.py  -train_in4nbow  $DIR/${CORPUS}_train_h40000.in4nbow \
                                    -train_ref4nbow $DIR/${CORPUS}_train_h40000.ref4nbow \
                                    -train_numofsim $DIR/${CORPUS}_train_h40000.numofsim \
                                    -dev_in4nbow    $DIR/${CORPUS}_dev.in4nbow \
                                    -dev_ref4nbow   $DIR/${CORPUS}_dev.ref4nbow \
                                    -dev_numofsim   $DIR/${CORPUS}_dev.numofsim \
                                    -test_in4nbow   $DIR/${CORPUS}_test.in4nbow \
                                    -test_ref4nbow  $DIR/${CORPUS}_test.ref4nbow \
                                    -test_numofsim  $DIR/${CORPUS}_test.numofsim \
                                    -corpus_name ${CORPUS} \
                                    -output_dir ../../experiments/nbow.${CORPUS}_h40000.sbert_top100_for_analyze \
                                    -only_predict True \
                                    -big_boundary '||'

# testの結果
# ASPEC
# [[136680, 2177, 74127], [2908, 13602, 10450], [47177, 4456, 9459481]]
# KFTT
# 