# -*- coding: utf-8 -*-
"""§31 (серия 4, цикл 4): лемносская загадка §27 + TTR-точки для жанровой
таблицы LA (их запрос из мостика 2026-07-11).

1. ЛЕМНОС НИЖЕ КОНТРОЛЯ (§27: 63.0% < 73.3%) — гипотеза вокализма:
   в лемносском алфавите ЕСТЬ o (в этрусском нет); соответствие
   лемн. o ~ этр. u известно из литературы. Тесты: (а) покрытие top-10
   на подмножестве слов без o; (б) покрытие после замены o→u; нуль тот
   же (форм-согласованные наборы, R=1000).
2. Профили финалей (дескриптив): последний символ LEMN/RAET/ETR.
3. TTR ПЕРВОГО СЛОТА для таблицы LA: ретийские мультисловные чтения TIR
   (деление интерпунктом ·; страты по type_object: rock / прочее);
   лемносские мультисловные записи корпуса. Точки шлются в мостик той
   же метрикой, что «пятая точка» LA.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_lemnos_endings.py
"""
import csv
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
K = 10
MIN_STEM = 3
OUT_LOG = os.path.join('logs', 'etr_lemnos_endings.log')
LOG = []

TIR_MAP = {'þ': 'th', 'χ': 'ch', 'φ': 'ph', 'ś': 's', 'š': 's',
           'θ': 'th', 'ϑ': 'th'}


def log(m=''):
    print(m)
    LOG.append(m)


def norm_any(w):
    w = (w or '').lower()
    w = ''.join(TIR_MAP.get(c, c) for c in unicodedata.normalize('NFD', w)
                if not unicodedata.combining(c))
    return re.sub(r'[^a-z]', '', w)


def main():
    os.makedirs('logs', exist_ok=True)
    log('=== §31: Лемнос (вокализм) + TTR-точки для таблицы LA ===')
    S = []
    with open(os.path.join('results', 'mdl_suffixes_v1.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            S.append(r['suffix'])
    topk = set(S[:K])
    sbl = {k: {s for s in topk if len(s) == k} for k in (1, 2, 3, 4)}
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    lemn_recs = [r for r in corpus['records'] if r['lang'] == 'lemn'
                 and r.get('variant_of') is None]
    lemn = sorted({t['ascii'] for r in lemn_recs for t in r['toks']
                   if t['kind'] == 'W' and '-' not in t['ascii']
                   and len(t['ascii']) >= MIN_STEM + 1})

    def covered(w):
        for k in (4, 3, 2, 1):
            if len(w) - k >= MIN_STEM and w[-k:] in sbl[k]:
                return 1
        return 0

    def cover_share(words):
        return sum(covered(w) for w in words) / max(len(words), 1)

    # пул нуля (§19)
    freq = Counter(t['ascii'] for r in view for t in r['toks']
                   if t['kind'] == 'W' and '-' not in t['ascii']
                   and len(t['ascii']) >= 3)
    cand = Counter()
    for w in freq:
        for k in range(1, 5):
            if len(w) - k >= MIN_STEM:
                cand[w[-k:]] += 1
    pool_by_len = {k: sorted(c for c, n in cand.items()
                             if n >= 15 and len(c) == k)
                   for k in (1, 2, 3, 4)}
    need = Counter(len(s) for s in topk)
    rng = np.random.default_rng(SEED)

    def null_p(words, obs):
        sims = np.zeros(R)
        for i in range(R):
            cur = set()
            for k in sorted(need):
                idx = rng.choice(len(pool_by_len[k]), size=need[k],
                                 replace=False)
                cur.update(pool_by_len[k][j] for j in idx)
            cbl = {kk: {s for s in cur if len(s) == kk} for kk in (1, 2, 3, 4)}

            def cov2(w):
                for kk in (4, 3, 2, 1):
                    if len(w) - kk >= MIN_STEM and w[-kk:] in cbl[kk]:
                        return 1
                return 0
            sims[i] = sum(cov2(w) for w in words) / max(len(words), 1)
        return float((1 + (sims >= obs - 1e-12).sum()) / (R + 1)), sims

    log(f'\n--- 1. лемносский вокализм (слов: {len(lemn)}) ---')
    with_o = [w for w in lemn if 'o' in w]
    no_o = [w for w in lemn if 'o' not in w]
    log(f'слов с o: {len(with_o)} ({len(with_o) / len(lemn):.0%}); '
        f'покрытие: с o {cover_share(with_o):.1%}, без o '
        f'{cover_share(no_o):.1%}, всё {cover_share(lemn):.1%}')
    lemn_ou = [w.replace('o', 'u') for w in lemn]
    obs_ou = cover_share(lemn_ou)
    p_ou, sims = null_p(lemn_ou, obs_ou)
    log(f'после замены o→u: покрытие {obs_ou:.1%} '
        f'(нуль {sims.mean():.1%}±{sims.std():.1%}, p={p_ou:.4f})')

    # --- 2. профили финалей ---------------------------------------------------
    log('\n--- 2. финальные символы (доли топ-5) ---')
    raet = set()
    with open(os.path.join('data', 'external', 'tir', 'tir_words.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'Raetic' in (r['language'] or ''):
                w = norm_any(r['title'])
                if len(w) >= MIN_STEM + 1:
                    raet.add(w)
    etr_w = sorted(freq)
    for name, words in (('ETR', etr_w), ('RAET', sorted(raet)),
                        ('LEMN', lemn)):
        fc = Counter(w[-1] for w in words)
        tot = sum(fc.values())
        log(f'  {name:<5} ' + ' '.join(f'{c}:{n / tot:.0%}'
                                       for c, n in fc.most_common(5)))

    # --- 3. TTR первого слота (точки для таблицы LA) --------------------------
    log('\n--- 3. TTR первого слота (метрика LA) ---')
    strata = defaultdict(list)
    with open(os.path.join('data', 'external', 'tir',
                           'tir_inscriptions.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['language'] != 'Raetic':
                continue
            reading = (r['reading'] or '').strip()
            if '·' not in reading:
                continue
            first = norm_any(reading.split('·')[0])
            if len(first) >= 2 and '?' not in reading.split('·')[0][:2]:
                st = 'rock' if r['type_object'] == 'rock' else 'portable'
                strata[st].append(first)
    for st, firsts in sorted(strata.items()):
        ttr = len(set(firsts)) / len(firsts)
        log(f'  RAET {st:<9} n={len(firsts):>3} TTR={ttr:.2f}')
    lemn_multi = []
    for r in lemn_recs:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W'
              and len(t['ascii']) >= 2]
        if len(ws) >= 2:
            lemn_multi.append(ws[0])
    if lemn_multi:
        log(f'  LEMN мультисловных записей {len(lemn_multi)}, '
            f'TTR={len(set(lemn_multi)) / len(lemn_multi):.2f} '
            f'(мало — точка условная)')
    log('\nчтение: 1 — вокализм объясняет/не объясняет лемносский провал '
        'покрытия; 3 — точки для межтрадиционной таблицы LA (слать в '
        'мостик с оговорками n).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
