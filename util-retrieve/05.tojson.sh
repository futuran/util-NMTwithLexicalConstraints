CORPUS=$1
DIR=../../data/${CORPUS}
dir=${DIR}.sbert/merge.top1/

for tvt in train_h40000 dev test;
do
    python ./05.tojson.py -src $dir/${CORPUS}_$tvt.en.with_match.tkn \
                          -trg $dir/${CORPUS}_$tvt.ja.with_match.tkn \
                          -out $dir/${CORPUS}_$tvt.tkn.json
done