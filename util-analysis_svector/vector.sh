CORPUS=$1
DIR=../../data/${CORPUS}

python vector.py --en_emb ${DIR}.sbert/emb/test.en.emb \
                 --ja_emb ${DIR}.sbert/emb/test.ja.emb