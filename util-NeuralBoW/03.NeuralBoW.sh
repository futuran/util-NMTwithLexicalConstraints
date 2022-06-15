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

# top10のみ
mkdir -p ../../experiments/nbow.${CORPUS}_h40000.$MODEL.srcsrc_top10/topk/
DIR=../../data/${CORPUS}.$MODEL.NBoW/srcsrc_top100/
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
                        -output_dir ../../experiments/nbow.${CORPUS}_h40000.$MODEL.srcsrc_top10 \
                        -only_predict True \
                        -topk_of_sim 100

# testの結果
# LaBSE
# source-source
# top10: [[18005, 349, 6502], [216, 2382, 762], [5266, 652, 942061]]

# source-target
# top10: [[21021, 314, 7898], [300, 2367, 844], [6372, 630, 965501]]
# top100: [[163775, 1841, 64929], [1807, 12872, 5135], [50455, 3786, 9796943]]


# SBERT
# source-source
# top10:[[21256, 598, 6379], [421, 3938, 1218], [6076, 1187, 938675]]
# top100:[[161415, 3494, 51481], [2489, 21415, 7129], [50821, 7725, 9558251]]