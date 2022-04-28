TYPE="_srconly"
CORPUS=$1
dir_oracle=../../data/${CORPUS}.sbert$TYPE.oracle/
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.sbert$TYPE

for tvt in train_h40000 dev test;
do
    python ./05.tojson.py -src ${dir_oracle}/${CORPUS}_$tvt.oracle.src.tkn \
                          -trg ${dir_oracle}/${CORPUS}_$tvt.oracle.trg.tkn \
                          -out ${dir_oracle}/${CORPUS}_$tvt.oracle.tkn.json
done

python ./05.tojson.py -src ../../experiments/nbow.${CORPUS}_h40000.sbert$TYPE/${CORPUS}_test.assisted.src.tkn \
                      -trg ${dir_oracle}/${CORPUS}_$tvt.oracle.trg.tkn \
                      -out ${dir_oracle}/${CORPUS}_$tvt.assisted.tkn.json