CORPUS=$1
DIR=../../data/${CORPUS}

# python vector.py --en_emb ${DIR}.msbert/orig_emb/dev.en.emb \
#                  --ja_emb ${DIR}.msbert/orig_emb/dev.ja.emb

python vector.py --en_emb ${DIR}.labse/orig_emb/dev.en.emb \
                 --ja_emb ${DIR}.labse/orig_emb/dev.ja.emb