# -*- coding: utf-8 -*-
"""§27 (серия 3, цикл 8): тирренская таблица окончаний с контролем.

Вопрос-проверка §19: top-10 этрусских MDL-суффиксов покрывали 93.1%
ретийских исходов — тирренский ли это сигнал или ареальная фонотактика?
Таблица покрытия тем же top-K на:
  ETR-holdout — словоформы Клузиума (held-out §17);
  RAET        — ретийские словоформы TIR;
  LEMN        — лемносские словоформы (корпус, lang='lemn');
  LAT/FAL     — КОНТРОЛЬ: латинские/фалискские словоформы корпуса
                (Бурман-разметка) — генетически чужие соседи.
Нуль для каждой строки: R=1000 форм-согласованных случайных наборов
окончаний из этрусского пула (§19). Если контроль покрывается так же —
сигнал ареальный; если тирренские строки выше контроля — родственный.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_tyrsenian_table.py
"""
import csv
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
K = 10
MIN_STEM = 3
OUT_LOG = os.path.join('logs', 'etr_tyrsenian_table.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


TIR_MAP = {'þ': 'th', 'χ': 'ch', 'φ': 'ph', 'ś': 's', 'š': 's',
           'θ': 'th', 'ϑ': 'th'}


def norm_any(w):
    w = (w or '').lower()
    w = ''.join(TIR_MAP.get(c, c) for c in unicodedata.normalize('NFD', w)
                if not unicodedata.combining(c))
    return re.sub(r'[^a-z]', '', w)


def main():
    os.makedirs('logs', exist_ok=True)
    log('=== §27: тирренская таблица окончаний (top-10) с контролем ===')
    S = []
    with open(os.path.join('results', 'mdl_suffixes_v1.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            S.append(r['suffix'])
    topk = S[:K]
    log(f'top-{K}: ' + ' '.join('-' + s for s in topk))
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    allrec = corpus['records']
    view = [r for r in allrec
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]

    def words_of(recs, min_len=MIN_STEM + 1):
        out = set()
        for r in recs:
            for t in r['toks']:
                if t['kind'] == 'W' and '-' not in t['ascii'] \
                        and len(t['ascii']) >= min_len:
                    out.add(t['ascii'])
        return sorted(out)

    sets = {}
    cl_arts = {r['artifact_id'] for r in view
               if (r.get('region') or '') == 'Cl'}
    sets['ETR-holdout(Cl)'] = words_of(
        [r for r in view if r['artifact_id'] in cl_arts])
    raet = set()
    with open(os.path.join('data', 'external', 'tir', 'tir_words.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'Raetic' in (r['language'] or ''):
                w = norm_any(r['title'])
                if len(w) >= MIN_STEM + 1:
                    raet.add(w)
    sets['RAET(TIR)'] = sorted(raet)
    sets['LEMN'] = words_of([r for r in allrec if r['lang'] == 'lemn'
                             and r.get('variant_of') is None])
    sets['LAT/FAL(контроль)'] = words_of(
        [r for r in allrec if r['lang'] in ('lat', 'fal')
         and r.get('variant_of') is None])

    # пул нуля — из этрусского канонического вида (§19)
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

    def coverage(words, sset):
        sbl = {k: {s for s in sset if len(s) == k} for k in (1, 2, 3, 4)}
        n = 0
        for w in words:
            for k in (4, 3, 2, 1):
                if len(w) - k >= MIN_STEM and w[-k:] in sbl[k]:
                    n += 1
                    break
        return n / max(len(words), 1)

    rng = np.random.default_rng(SEED)
    log(f'\n{"срез":<20} {"n слов":>7} {"top-K":>7} {"нуль":>13} {"p":>8}')
    for name, words in sets.items():
        if len(words) < 20:
            log(f'{name:<20} {len(words):>7} — мало слов')
            continue
        obs = coverage(words, set(topk))
        sims = np.zeros(R)
        for i in range(R):
            cur = set()
            for k in sorted(need):
                idx = rng.choice(len(pool_by_len[k]), size=need[k],
                                 replace=False)
                cur.update(pool_by_len[k][j] for j in idx)
            sims[i] = coverage(words, cur)
        p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
        log(f'{name:<20} {len(words):>7} {obs:>7.1%} '
            f'{sims.mean():>6.1%}±{sims.std():.1%} {p:>8.4f}')
    log('\nчтение: если LAT/FAL-контроль покрыт так же и с тем же p — '
        'сигнал §19 ареальный (общая фонотактика Италии), не '
        'специфически тирренский; расщепление строк — родственный.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
