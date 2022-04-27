# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}.sbert_trgonly.NBoW

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
                                    -output_dir ../../experiments/nbow.${CORPUS}_h40000.sbert_trgonly \
                                    #-only_predict True

# testの結果
# ASPEC
# [[17290, 266, 5397], [199, 1837, 489], [5242, 538, 940338]]
# KFTT
# [[0, 0, 6866], [0, 0, 781], [0, 0, 505241]]