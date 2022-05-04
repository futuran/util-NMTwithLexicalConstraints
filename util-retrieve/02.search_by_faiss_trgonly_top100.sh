CORPUS=$1
DIR=../../data/${CORPUS}

mkdir -p ${DIR}.sbert_trgonly_top100/match
# 2 search by faiss
for prefix in dev test train_h40000
do
    if [ $prefix = 'test' ]; then
        PM='--include-perfect-match'
    else
        PM=''
    fi

    # 検索対象は原言語側のみ
    python 02.search_by_faiss.py \
        -q    ${DIR}.sbert/emb/${prefix}.ja.emb \
        -qt   ${DIR}/${CORPUS}_$prefix.en.tkn \
        -tms  ${DIR}.sbert/emb/train.ja.emb \
        -tmst ${DIR}/${CORPUS}_train.en.tkn \
        -o    ${DIR}.sbert_trgonly_top100/match/${prefix}.match \
        -k 100 -d 768 $PM
    wait
done

