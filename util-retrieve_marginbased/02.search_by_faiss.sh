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

# mkdir ${DIR}.$MODEL/match_srctgt
mkdir ${DIR}.$MODEL/match_srctgt_inv

# 2 search by faiss
for prefix in valid test train
do
    if [ $prefix = 'test' ]; then
        PM='--include-perfect-match'
    else
        PM=''
    fi

    # # 検索対象は目的言語側のみ
    # python 02.search_by_faiss.py \
    #     -q    ${DIR}.$MODEL/emb/${prefix}.en.emb \
    #     -qt   ${DIR}/${CORPUS}_$prefix.en.tkn \
    #     -tms  ${DIR}.$MODEL/emb/train.fr.emb \
    #     -tmst ${DIR}/${CORPUS}_train.en.tkn \
    #     -o    ${DIR}.$MODEL/match_srctgt/${prefix}.match \
    #     -k 100 -d 768 $PM
    # wait

    # # 検索対象は原言語側のみ（本来の方法）
    # python 02.search_by_faiss.py \
    #     -q    ${DIR}.$MODEL/emb/train.fr.emb \
    #     -qt   ${DIR}/${CORPUS}_train.fr.tkn \
    #     -tms  ${DIR}.$MODEL/emb/${prefix}.en.emb \
    #     -tmst ${DIR}/${CORPUS}_${prefix}.en.tkn \
    #     -o    ${DIR}.$MODEL/match_srctgt_inv/${prefix}.match \
    #     -k 100 -d 768 $PM
    # wait

    # 検索対象は原言語側のみ（検索の大きさを揃えるため、train.enを検索）
    python 02.search_by_faiss.py \
        -q    ${DIR}.$MODEL/emb/train.fr.emb \
        -qt   ${DIR}/${CORPUS}_train.fr.tkn \
        -tms  ${DIR}.$MODEL/emb/train.en.emb \
        -tmst ${DIR}/${CORPUS}_train.en.tkn \
        -o    ${DIR}.$MODEL/match_srctgt_inv/${prefix}.match \
        -k 100 -d 768 $PM
    wait

done

