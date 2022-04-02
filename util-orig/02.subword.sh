CORPUS=$1
DIR=../../data/${CORPUS}
OPNUM=32000

# subword用のコードを切り出す前のtrainデータ全体から作成
cat ${DIR}/${CORPUS}_train.en.tkn ${DIR}/${CORPUS}_train.ja.tkn > ${DIR}/tmp/${CORPUS}_train.all.tkn
subword-nmt learn-bpe -s ${OPNUM} < ${DIR}/tmp/${CORPUS}_train.all.tkn > ${DIR}/code.all

# ${CORPUS}
for tvt in train_h40000 train dev test; do
    subword-nmt apply-bpe -c ${DIR}/code.all < ${DIR}/${CORPUS}_${tvt}.en.tkn > ${DIR}/${CORPUS}_${tvt}.en.tkn.bpe
    subword-nmt apply-bpe -c ${DIR}/code.all < ${DIR}/${CORPUS}_${tvt}.ja.tkn > ${DIR}/${CORPUS}_${tvt}.ja.tkn.bpe
done