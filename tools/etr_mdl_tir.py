# -*- coding: utf-8 -*-
"""§12.2: MDL-суффиксы × ретийский (TIR) — тирренская морфология с
безучительной стороны.

Вопросы: (1) пересекается ли выученный БЕЗ УЧИТЕЛЯ суффиксный лексикон S
(§9.4, results/mdl_suffixes_v1.csv) с морфемами TIR; (2) кончаются ли
ретийские словоформы на суффиксы S чаще, чем на случайные наборы той же
формы. ОГОВОРКА ЦИРКУЛЯРНОСТИ (регистрируется): морфемы TIR выделены
исследователями отчасти ПО СРАВНЕНИЮ с этрусским — тест (1) не есть
независимое свидетельство родства; тест (2) на сырых словоформах чище.

Нуль: R=1000 случайных суффикс-наборов того же мультимножества длин из
пула словоконечных подстрок train-типов (конструкция §9.4). seed=42.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_mdl_tir.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
MIN_STEM = 3
OUT_LOG = os.path.join('logs', 'etr_mdl_tir.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


TIR_MAP = {'þ': 'th', 'χ': 'ch', 'φ': 'ph', 'ś': 's', 'š': 's',
           'θ': 'th', 'ϑ': 'th'}


def norm_tir(w):
    w = (w or '').lower()
    w = ''.join(TIR_MAP.get(c, c) for c in
                __import__('unicodedata').normalize('NFD', w)
                if not __import__('unicodedata').combining(c))
    return re.sub(r'[^a-z]', '', w)


def main():
    os.makedirs('logs', exist_ok=True)
    log('=== §12.2: MDL-суффиксы × TIR (ретийский) ===')
    S = []
    with open(os.path.join('results', 'mdl_suffixes_v1.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            S.append(r['suffix'])
    S_set = set(S)
    log(f'MDL-суффиксов: {len(S)} (выучены без учителя, §9.4)')

    # --- TIR-морфемы ---------------------------------------------------------
    tir_m = set()
    with open(os.path.join('data', 'external', 'tir', 'tir_morphemes.csv'),
              encoding='utf-8') as f:
        rows_m = list(csv.DictReader(f))
    for r in rows_m:
        t = r['title'].strip().lstrip('-')
        m = re.fullmatch(r'([a-zþ]+)\(([a-zþ]+)\)', t)
        if m:
            tir_m.add(norm_tir(m.group(1)))
            tir_m.add(norm_tir(m.group(1) + m.group(2)))
        else:
            tir_m.add(norm_tir(t))
    tir_m = {x for x in tir_m if x}
    log(f'TIR-морфем (с вариантами скобок): {len(tir_m)}: '
        + ' '.join('-' + x for x in sorted(tir_m)))

    # --- ретийские словоформы -------------------------------------------------
    raet = set()
    with open(os.path.join('data', 'external', 'tir', 'tir_words.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'Raetic' in (r['language'] or ''):
                w = norm_tir(r['title'])
                if len(w) >= MIN_STEM + 1:
                    raet.add(w)
    raet = sorted(raet)
    log(f'ретийских словоформ (чистых, len>={MIN_STEM + 1}): {len(raet)}')

    # --- пул для нуля (как в §9.4: словоконечные подстроки train-типов) ------
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    freq = Counter(t['ascii'] for r in view for t in r['toks']
                   if t['kind'] == 'W' and '-' not in t['ascii']
                   and len(t['ascii']) >= 3)
    cand = Counter()
    for w in freq:
        for k in range(1, 5):
            if len(w) - k >= MIN_STEM:
                cand[w[-k:]] += 1
    pool = sorted(c for c, n in cand.items() if n >= 15)
    pool_by_len = {k: [c for c in pool if len(c) == k] for k in (1, 2, 3, 4)}
    need = Counter(len(s) for s in S)
    rng = np.random.default_rng(SEED)

    def null_sets():
        for _ in range(R):
            cur = set()
            for k in sorted(need):
                idx = rng.choice(len(pool_by_len[k]), size=need[k],
                                 replace=False)
                cur.update(pool_by_len[k][i] for i in idx)
            yield cur

    # --- тест 1: |S ∩ TIR-морфемы| -------------------------------------------
    obs1 = len(S_set & tir_m)
    sims1 = np.array([len(ns & tir_m) for ns in null_sets()])
    p1 = float((1 + (sims1 >= obs1).sum()) / (R + 1))
    log(f'\nтест 1 (ЦИРКУЛЯРНОСТЬ ЗАРЕГИСТРИРОВАНА): |S∩TIR| = {obs1} '
        f'({" ".join("-" + x for x in sorted(S_set & tir_m))}); '
        f'нуль {sims1.mean():.2f}±{sims1.std():.2f}; p={p1:.4f}')

    # --- тест 2: ретийские окончания ------------------------------------------
    def end_share(sset):
        sbl = {k: {s for s in sset if len(s) == k} for k in (1, 2, 3, 4)}
        n = 0
        for w in raet:
            for k in (4, 3, 2, 1):
                if len(w) - k >= MIN_STEM and w[-k:] in sbl[k]:
                    n += 1
                    break
        return n / max(len(raet), 1)

    obs2 = end_share(S_set)
    rng = np.random.default_rng(SEED + 1)
    sims2 = np.array([end_share(ns) for ns in null_sets()])
    p2 = float((1 + (sims2 >= obs2).sum()) / (R + 1))
    log(f'тест 2 (чище): доля ретийских словоформ с окончанием из S: '
        f'{100 * obs2:.1f}%; нуль {100 * sims2.mean():.1f}%'
        f'±{100 * sims2.std():.1f}%; p={p2:.4f}')

    # --- тест 2b (v2, §19): top-K против потолка -----------------------------
    K = 10
    topk = set(S[:K])  # S в порядке DL-выигрыша из csv
    need_k = Counter(len(s) for s in topk)
    rng = np.random.default_rng(SEED + 2)

    def null_sets_k():
        for _ in range(R):
            cur = set()
            for k in sorted(need_k):
                idx = rng.choice(len(pool_by_len[k]), size=need_k[k],
                                 replace=False)
                cur.update(pool_by_len[k][i] for i in idx)
            yield cur

    obs2b = end_share(topk)
    sims2b = np.array([end_share(ns) for ns in null_sets_k()])
    p2b = float((1 + (sims2b >= obs2b).sum()) / (R + 1))
    log(f'тест 2b (top-{K} по DL-выигрышу: '
        + ' '.join('-' + s for s in S[:K]) + f'): доля ретийских '
        f'словоформ {100 * obs2b:.1f}%; нуль {100 * sims2b.mean():.1f}%'
        f'±{100 * sims2b.std():.1f}%; p={p2b:.4f}')
    log('\nчтение: разведочный слой; тест 1 не независим от этрускологии '
        '(морфемы TIR выделялись со взглядом на этрусский); тест 2 — '
        'слабое независимое свидетельство общей суффиксальной фонотактики/'
        'морфологии, если p мал.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
