
CORPUS=$1
dir_oracle=../../data/${CORPUS}.sbert_top1.oracle/
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.sbert/for_analyze/

for tvt in train_h40000 dev test;
do
    python ./05.tojson.py -src ${dir_oracle}/${CORPUS}_$tvt.oracle.src.tkn \
                          -trg ${dir_oracle}/${CORPUS}_$tvt.oracle.trg.tkn \
                          -out ${dir_oracle}/${CORPUS}_$tvt.oracle.tkn.json
done

python ./05.tojson.py -src ${dir_assisted}/${CORPUS}_test.assisted_top100.src.tkn.with_upto_No.1 \
                      -trg ${dir_oracle}/${CORPUS}_test.oracle.trg.tkn \
                      -out ${dir_oracle}/${CORPUS}_test.assisted.tkn.json