import json
import argparse

def tojson(args):

    with open(args.src, 'r') as f:
        src_list = f.readlines()
    with open(args.trg, 'r') as f:
        trg_list = f.readlines()

    sentences = []
    for i, (src, trg) in enumerate(zip(src_list, trg_list)):
        src = src.strip().replace('\ ','').replace('"','`')
        trg = trg.strip().replace('\ ','').replace('"','`')

        sentences.append(r'{"translation": {"en": "' + src + '" ,"ja": "' + trg + '"}}\n')

    with open(args.out, 'w') as f:
        f.writelines(sentences)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src')
    parser.add_argument('-trg')
    parser.add_argument('-out')

    args = parser.parse_args()

    tojson(args)

main()
