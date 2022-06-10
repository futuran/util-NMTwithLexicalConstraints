CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi
MODEL=$2
if [ "$MODEL" = "" ]; then
    echo "Please designate MODEL!!!"
    exit
fi

# top100のみ
mkdir -p ../../experiments/nbow.${CORPUS}_h40000.$MODEL.srctrg_top100/topk/
DIR=../../data/${CORPUS}.$MODEL.NBoW/srctrg_top100/
python 03.NeuralBoW.py  -train_in4nbow  $DIR/${CORPUS}_train_h40000.in4nbow \
                        -train_ref4nbow $DIR/${CORPUS}_train_h40000.ref4nbow \
                        -train_numofsim $DIR/${CORPUS}_train_h40000.numofsim \
                        -dev_in4nbow    $DIR/${CORPUS}_dev.in4nbow \
                        -dev_ref4nbow   $DIR/${CORPUS}_dev.ref4nbow \
                        -dev_numofsim   $DIR/${CORPUS}_dev.numofsim \
                        -test_in4nbow   $DIR/${CORPUS}_test.in4nbow \
                        -test_ref4nbow  $DIR/${CORPUS}_test.ref4nbow \
                        -test_numofsim  $DIR/${CORPUS}_test.numofsim \
                        -corpus_name ${CORPUS} \
                        -output_dir ../../experiments/nbow.${CORPUS}_h40000.$MODEL.srctrg_top100 \
                        # -only_predict True \
                        # -topk_of_sim 100

# testの結果
# source-source
# top10: 

# source-target
# top10:
# top100: 