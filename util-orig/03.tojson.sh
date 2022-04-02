CORPUS=$1
DIR=../../data/${CORPUS}

for tvt in train_h40000 dev test;
do
    python ./03.tojson.py -src ${DIR}/${CORPUS}_$tvt.en.tkn \
                          -trg ${DIR}/${CORPUS}_$tvt.ja.tkn \
                          -out ${DIR}/${CORPUS}_$tvt.tkn.json
done