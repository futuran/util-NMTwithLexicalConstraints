# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi
DIR=../../data/${CORPUS}
MODEL=$2
if [ "$MODEL" = "" ]; then
    echo "Please designate MODEL!!!"
    exit
fi

mkdir -p ${DIR}.$MODEL.oracle
mkdir -p ${DIR}.$MODEL.NBoW

for topk in `seq 1 100`; do
    echo $topk
    mkdir ${DIR}.$MODEL.oracle/srcsrc_top$topk/
    mkdir ${DIR}.$MODEL.NBoW/srcsrc_top$topk/
    for tvt in dev test train_h40000
    do
        python 01.preprocess_NeuralBoW.py   -prefix         ${CORPUS}_$tvt \
                                            -in_dir         ${DIR}.$MODEL/merge_srcsrc.top100/ \
                                            -in_suffix_src  en.with_match.tkn \
                                            -in_suffix_trg  ja.with_match.tkn \
                                            -out_oracle_dir ${DIR}.$MODEL.oracle/srcsrc_top$topk/ \
                                            -out_NBoW_dir   ${DIR}.$MODEL.NBoW/srcsrc_top$topk/ \
                                            -out_suffix_src oracle.src.tkn \
                                            -out_suffix_trg oracle.trg.tkn \
                                            -topk           $topk &
    done
done
