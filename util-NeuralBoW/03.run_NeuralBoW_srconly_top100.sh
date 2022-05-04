# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}.sbert_srconly_top100.NBoW

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
                                    -output_dir ../../experiments/nbow.${CORPUS}_h40000.sbert_srconly \
                                    -only_predict True \
                                    -output_file_suffix '.assisted_top100.src.tkn.for_analyze' \
                                    -big_boundary '||'


OUTDIR=../../experiments/nbow.${CORPUS}_h40000.sbert_srconly/
sed 's/||/|/g' $OUTDIR/${CORPUS}_test.assisted_top100.src.tkn.for_analyze > $OUTDIR/${CORPUS}_test.assisted_top100.src.tkn

# testの結果
# ASPEC
# [[157474, 3009, 52501], [2105, 18593, 6262], [50794, 6645, 9453675]]
# KFTT
# 