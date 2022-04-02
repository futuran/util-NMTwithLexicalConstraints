CORPUS=$1
DIR=../../data/${CORPUS}

mkdir ${DIR}.sbert/match
# 2 search by faiss
for prefix in dev test train_h40000 train
do
    if [ $prefix = 'test' ]; then
        PM='--include-perfect-match'
    else
        PM=''
    fi

    python 02.search_by_faiss.py \
        -q    ${DIR}.sbert/emb/${prefix}.en.emb \
        -qt   ${DIR}/${CORPUS}_$prefix.en.tkn \
        -tms  ${DIR}.sbert/emb/train.en.emb \
        -tmst ${DIR}/${CORPUS}_train.en.tkn \
        -tmt  ${DIR}.sbert/emb/train.ja.emb \
        -o    ${DIR}.sbert/match/${prefix}.match \
        -k 10 -d 768 $PM
    wait
done

