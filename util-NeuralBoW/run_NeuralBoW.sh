# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}.sbert.NBoW

# ${CORPUS}のみ
python ../../NeuralBoW/NeuralBoW.py  -train_in4nbow  $DIR/${CORPUS}_dev.in4nbow \
                                    -train_ref4nbow $DIR/${CORPUS}_dev.ref4nbow \
                                    -train_numofsim $DIR/${CORPUS}_dev.numofsim \
                                    -dev_in4nbow    $DIR/${CORPUS}_dev.in4nbow \
                                    -dev_ref4nbow   $DIR/${CORPUS}_dev.ref4nbow \
                                    -dev_numofsim   $DIR/${CORPUS}_dev.numofsim \
                                    -test_in4nbow   $DIR/${CORPUS}_test.in4nbow \
                                    -test_ref4nbow  $DIR/${CORPUS}_test.ref4nbow \
                                    -test_numofsim  $DIR/${CORPUS}_test.numofsim \
                                    -corpus_name ${CORPUS} \
                                    -output_dir ../../experiments/nbow.aspec_h40000.sbert
                                    #-only_predict

# testの結果



