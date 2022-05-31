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

mkdir ${DIR}.$MODEL/match_srctrg
mkdir ${DIR}.$MODEL/match_srcsrc
# 2 search by faiss
for prefix in dev test train_h40000
do
    if [ $prefix = 'test' ]; then
        PM='--include-perfect-match'
    else
        PM=''
    fi

    # 検索対象は目的言語側のみ
    python 02.search_by_faiss.py \
        -q    ${DIR}.$MODEL/emb/${prefix}.en.emb \
        -qt   ${DIR}/${CORPUS}_$prefix.en.tkn \
        -tms  ${DIR}.$MODEL/emb/train.ja.emb \
        -tmst ${DIR}/${CORPUS}_train.en.tkn \
        -o    ${DIR}.$MODEL/match_srctrg/${prefix}.match \
        -k 100 -d 768 $PM
    wait

    # 検索対象は原言語側のみ
    python 02.search_by_faiss.py \
        -q    ${DIR}.$MODEL/emb/${prefix}.en.emb \
        -qt   ${DIR}/${CORPUS}_$prefix.en.tkn \
        -tms  ${DIR}.$MODEL/emb/train.en.emb \
        -tmst ${DIR}/${CORPUS}_train.en.tkn \
        -o    ${DIR}.$MODEL/match_srcsrc/${prefix}.match \
        -k 100 -d 768 $PM
    wait

done

