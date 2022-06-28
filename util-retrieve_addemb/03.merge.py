#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os


def merge_match_lines(concat, *args):
    '''
    if there are multiple match files input, merge then into one for the sake of convenience
    in the following process
    '''
    # make sure that lines number in every match file is equivalent
    std_len = len(args[0])
    for lines in args:
        assert len(lines) == std_len

    new_lines = []
    for line_no in range(std_len):
        new_line = concat.join([lines[line_no] for lines in args])
        new_lines.append(new_line)

    return new_lines


def process(opt):
    def read_f(fn):
        with open(fn, 'r') as f:
            return [l.strip() for l in f]

    src_lines = read_f(opt.src)
    tgt_lines = read_f(opt.tgt)
    tmt_lines = read_f(opt.tmt)

    match_lines_set = []
    for fn in opt.match_file:
        match_lines_set.append(read_f(fn))

    merged_match_lines = merge_match_lines(opt.match_line_concat_symbol, *match_lines_set)

    src_withmatch_res, src_nonmatch_res = [], []
    tgt_withmatch_res, tgt_nonmatch_res = [], []
    for i, src_line in enumerate(src_lines):
        tgt_line = tgt_lines[i]
        match_info_line = merged_match_lines[i]
        if match_info_line.strip() == '':
            match_info = []
        else:
            match_info = [(int(p.split()[0]), float(p.split()[1])) for p in
                          match_info_line.split(opt.match_line_concat_symbol)]
            match_info.sort(key=lambda x: x[1], reverse=True)

        aug_query = []
        for match_i, match_v in match_info:
            if match_v <= opt.threshold:
                break
            tmt_line = tmt_lines[match_i]
            if opt.permit_duplicate_match or tmt_line not in aug_query:
                aug_query.append(tmt_line)
                if len(aug_query) == opt.topk:
                    break

        if len(aug_query) == 0:  # non_match records
            src_nonmatch_res.append(src_line)
            tgt_nonmatch_res.append(tgt_line)
        else:  # with_match records
            if opt.add_extra_blank:
                while len(aug_query) < opt.topk:
                    aug_query.append(opt.blank_symbol)
            aug_query.insert(0, src_line)
            src_withmatch_res.append(opt.concat_symbol.join(aug_query))
            tgt_withmatch_res.append(tgt_line)

    def write_in(dir, fn, type, res):
        '''
        :param dir: output directory
        :param fn: the path of the original raw corpus file
        :param type: with_match or non_match
        :param res: container lines in which will be written
        '''
        base = os.path.basename(fn)
        pre, suf = base.rsplit('.', 1)
        wf = open(os.path.join(dir, f'{pre}.{type}.{suf}'), 'w')
        for l in res:
            wf.write(l.strip() + '\n')
        wf.close()

    write_in(opt.output_dir, opt.src, 'with_match', src_withmatch_res)
    write_in(opt.output_dir, opt.tgt, 'with_match', tgt_withmatch_res)
    write_in(opt.output_dir, opt.src, 'non_match', src_nonmatch_res)
    write_in(opt.output_dir, opt.tgt, 'non_match', tgt_nonmatch_res)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', required=True,
                        help='Path to the original raw source corpus. Expected to be $SPLIT.$LANG')
    parser.add_argument('-t', '--tgt', required=True,
                        help='Path to the original raw target corpus. Expected to be $SPLIT.$LANG')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Path to the output directory.')
    parser.add_argument('--match-file', nargs='+', required=True,
                        help='Specify the match files. Multiple inputs are supported.')
    parser.add_argument('-tmt', required=True,
                        help='Target side of Translation Memory.')

    parser.add_argument('--topk', default=3, type=int,
                        help='DEFAULT: 3')
    parser.add_argument('--threshold', default=0.75, type=float,
                        help='Only match whose similarity score is above the threshold will be considered as valid '
                             'ones and concatenated. DEFAULT: 0.75')
    parser.add_argument('--permit-duplicate-match', action='store_true', default=False,
                        help='If this flag is false, all same fuzzy matches translations will only be concatenated '
                             'once even if there may be multiple candidates.')
    parser.add_argument('--add-extra-blank', action='store_true', default=False,
                        help='Set the flag to append extra blank symbols until topk fuzzy matches (including blanks) '
                             'are concatenated.')

    # some default symbol representations
    parser.add_argument('--concat-symbol', default=' @@@ ')
    parser.add_argument('--blank-symbol', default='[BLANK]')
    parser.add_argument('--match-line-concat-symbol', default=' ||| ')

    opt = parser.parse_args()
    process(opt)


if __name__ == '__main__':
    main()
