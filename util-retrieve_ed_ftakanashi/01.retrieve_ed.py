#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
This script can accept a query file and build a augmented source corpus file by using SetSimilaritySearch and EditDistance.
'''

import argparse
import editdistance
import queue
import time
from tqdm import tqdm
from SetSimilaritySearch import SearchIndex as SI
from multiprocessing import Process, Lock, JoinableQueue, Manager


def distance_score(a, b):
    '''
    The original editdistance library only return an absolute distance figure.
    Change it into a edit distance score which measures how similar between a & b by evaluating the edit distance
    FOR CHANGING a INTO b.
    reference: https://arxiv.org/pdf/1505.05841.pdf
    '''
    distance = editdistance.eval(a, b)
    set_len = len(list(set(a)))
    ed = max(1 - float(distance) / set_len, 0)
    return ed


def match_fuzzy(query, search_index, opt):
    '''
    Do a primary search by SetSimilaritySearch(SSS)
    :param query:
    :param search_index:
    :param opt:
    :return:
    '''
    if type(query) is not list:
        query = query.strip().split()
    query = set(query)    # SSS requires query to be unique

    if opt.include_perfect_match:
        flag = 99999
    else:
        flag = 1.0
    cand_indices = [t[0] for t in list(sorted(search_index.query(query), key=lambda x: x[1], reverse=True)) if
                    t[1] < flag]

    return cand_indices[:opt.sss_nbest]


def calc_edit_distance(query, cand_indices, tms_lines, opt):
    candidates = [tms_lines[i].strip().split() for i in cand_indices]
    res = {}
    if type(query) is not list:
        query = query.strip().split()
    for i, c in enumerate(candidates):
        ed_score = distance_score(query, c)
        if ed_score >= opt.ed_lambda:
            res[cand_indices[i]] = distance_score(query, c)

    return res


class QueryWorker(Process):
    def __init__(self, p_name, queue, lock, **kwargs):
        super(QueryWorker, self).__init__()
        self.p_name = p_name
        self.queue = queue
        self.lock = lock
        self.search_index = kwargs.get('search_index')
        self.opt = kwargs.get('opt')
        self.tms_lines = kwargs.get('tms_lines')
        self.cache = kwargs.get('cache')
        self.res_container = kwargs.get('res_container')

    def concat_process(self, q_i, query):
        if query not in self.cache:
            cand_indices = match_fuzzy(query, self.search_index, self.opt)
            cand_info = calc_edit_distance(query, cand_indices, self.tms_lines, self.opt)
            appendix = []
            for cand_i in sorted(cand_info, key=lambda x:cand_info[x], reverse=True):
                cand_v = cand_info[cand_i]
                appendix.append((cand_i, cand_v))
                if len(appendix) >= self.opt.format_n:
                    break

            self.lock.acquire()
            self.cache[query] = appendix
            self.lock.release()

        else:
            appendix = self.cache[query]

        self.lock.acquire()
        self.res_container[q_i] = appendix
        self.lock.release()

    def nbest_process(self, q_i, query):
        pass

    def run(self):
        while True:
            try:
                q_i, query = self.queue.get(
                    timeout=self.opt.subprocess_timeout)  # block=False may be dangerous within multi-threading
            except queue.Empty as e:
                print(
                    f'Process [{self.p_name}] terminated for empty queue. Current queue length [{self.queue.qsize()}]')
                self.queue.join()
                break

            if self.opt.format_mode == 'series':
                self.concat_process(q_i, query)
            elif self.opt.format_mode == 'parallel':
                self.nbest_process(q_i, query)
            else:
                raise Exception(f'Invalid format mode {self.opt.format_mode}')

            self.queue.task_done()
            current_queue_size = self.queue.qsize()
            if current_queue_size % self.opt.report_every == 0 and current_queue_size > 0:
                print(f'{current_queue_size} pairs left', flush=True)


class WorkerPool():
    def __init__(self, queue, size, search_index, opt, tms_lines, cache, res_container):
        self.queue = queue
        self.pool = []
        self.lock = Lock()
        for i in range(size):
            print(f'Starting subprocess[{i}]')
            self.pool.append(QueryWorker(i, self.queue, self.lock,
                                         search_index=search_index,
                                         opt=opt,
                                         tms_lines=tms_lines,
                                         cache=cache,
                                         res_container=res_container))

    def startWork(self):
        for p in self.pool:
            p.start()

    def joinAll(self):
        for p in self.pool:
            p.join()


def process(opt):
    print('Reading query data...')
    with open(opt.query, 'r') as f:
        query_lines = f.readlines()

    print('Reading source side translation memory data...')
    with open(opt.tms, 'r') as f:
        tms_lines = f.readlines()

    print('Building Search Index...')
    search_index = SI([set(l.strip().split()) for l in tms_lines],    # SSS requires each entry to be a set
                      similarity_threshold=opt.sss_lambda,
                      similarity_func_name='containment_min')    # paper indicates the func name is called containment_max

    print('Building Task Queue...')
    _queue = JoinableQueue()
    i = 0
    for q in tqdm(query_lines, mininterval=1.0, ncols=100, leave=True):
        _queue.put((i, q))
        i += 1

    manager = Manager()
    cache = manager.dict()
    res_container = manager.dict()
    print('Initializing Workers Pool...')
    workers_pool = WorkerPool(_queue, opt.workers, search_index, opt, tms_lines, cache, res_container)
    workers_pool.startWork()
    workers_pool.joinAll()

    print('Writing result file...')
    wf = open(opt.output, 'w')
    for q_i in tqdm(sorted(res_container), mininterval=1, ncols=50):
        info = res_container[q_i]
        s = ' ||| '.join([f'{i} {v}' for i,v in info])
        wf.write(f'{s}\n')
    wf.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--query', required=True)
    parser.add_argument('-tms', required=True)
    parser.add_argument('-o', '--output', required=True,
                        help='Path to the match file.')

    parser.add_argument('-sss-nbest', default=2000, type=int,
                        help='N Best similar sentences selected by SetSimilaritySearch.\n'
                             'DEFAULT: 2000')
    parser.add_argument('-sss_lambda', default=0.5, type=float,
                        help='Threshold for SetSimilaritySearch.\n'
                             'DEFAULT: 0.5')
    parser.add_argument('--include-perfect-match', action='store_true', default=False,
                        help='Perfect match will be included during sss.\n'
                             'DEFAULT: False')
    parser.add_argument('--ed_lambda', default=0.5, type=float,
                        help='Threshold for edit distance score.\n'
                             'DEFAULT: 0.5')
    parser.add_argument('--format-mode', default='series',
                        help='Format mode. "series" and "parallel" are available.\nDEFAULT: series')
    parser.add_argument('--format-n', default=3, type=int,
                        help='Deciding format N.\nDEFAULT: 3')

    parser.add_argument('--workers', default=16, type=int,
                        help='Multi-Processing.\nDEFAULT: 16')
    parser.add_argument('--subprocess-timeout', default=60, type=int,
                        help='Timeout for a subprocess getting task from queue.\n'
                             'DEFAULT: 60')

    parser.add_argument('--report-every', default=100000, type=int,
                        help='Log interval.\nDEFAULT: 100000')

    opt = parser.parse_args()

    start_time = time.time()
    process(opt)
    end_time = time.time()
    print(f'Time Cosumed:{end_time - start_time}')


if __name__ == '__main__':
    main()
