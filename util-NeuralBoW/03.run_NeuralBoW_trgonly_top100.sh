# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}.sbert_trgonly_top100.NBoW

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
                                    -only_predict True \
                                    -output_file_suffix '.assisted_top100.src.tkn' \
                                    -topk_of_sim 100

# testの結果
# ASPEC
# [[126174, 1488, 41148], [1060, 9324, 2651], [42074, 3242, 9538231]]
# KFTT
# 