CORPUS=$1
DIR=../../data/${CORPUS}
SIZE=40000

mkdir -p ${DIR}/tmp

# シャッフル
paste -d '|' ${DIR}/${CORPUS}_train.en.tkn ${DIR}/${CORPUS}_train.ja.tkn > ${DIR}/tmp/${CORPUS}_train.enja.tkn
shuf --random-source=${DIR}/tmp/${CORPUS}_train.enja.tkn ${DIR}/tmp/${CORPUS}_train.enja.tkn | head -n ${SIZE} > ${DIR}/tmp/${CORPUS}_train_h${SIZE}.enja.tkn

# 指定したサイズだけ切り出し
cut -d '|' -f1 ${DIR}/tmp/${CORPUS}_train_h${SIZE}.enja.tkn > ${DIR}/${CORPUS}_train_h${SIZE}.en.tkn
cut -d '|' -f2 ${DIR}/tmp/${CORPUS}_train_h${SIZE}.enja.tkn > ${DIR}/${CORPUS}_train_h${SIZE}.ja.tkn
