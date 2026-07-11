# -*- coding: utf-8 -*-
"""Предрегистрация №4: ретийское покрытие top-10 выше ареального контроля.

Правила заморожены в validation/prereg4_raetic_vs_control.md (коммит ДО
прогона; прогон ОДИН). H1: coverage(RAET) - coverage(LAT/FAL) > 0 сверх
нуля перестановки языковых меток (R=10000, seed=42), p<=0.05.

Запуск (единственный):
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_prereg4_raetic.py
"""
import csv
import os
import pickle
import re
import sys
import unicodedata

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
K = 10
MIN_STEM = 3
OUT_LOG = os.path.join('logs', 'etr_prereg4_raetic.log')
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
    log('=== Предрегистрация №4: Raetic > LAT/FAL по top-10 покрытию ===')
    log('правила: validation/prereg4_raetic_vs_control.md (до прогона)')
    S = []
    with open(os.path.join('results', 'mdl_suffixes_v1.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            S.append(r['suffix'])
    topk = set(S[:K])
    log(f'top-{K}: ' + ' '.join('-' + s for s in sorted(topk)))
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    raet = set()
    with open(os.path.join('data', 'external', 'tir', 'tir_words.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'Raetic' in (r['language'] or ''):
                w = norm_any(r['title'])
                if len(w) >= MIN_STEM + 1:
                    raet.add(w)
    latfal = set()
    for r in corpus['records']:
        if r['lang'] in ('lat', 'fal') and r.get('variant_of') is None:
            for t in r['toks']:
                if t['kind'] == 'W' and '-' not in t['ascii'] \
                        and len(t['ascii']) >= MIN_STEM + 1:
                    latfal.add(t['ascii'])
    raet, latfal = sorted(raet), sorted(latfal)
    log(f'слов: RAET {len(raet)}, LAT/FAL {len(latfal)}')

    sbl = {k: {s for s in topk if len(s) == k} for k in (1, 2, 3, 4)}

    def covered(w):
        for k in (4, 3, 2, 1):
            if len(w) - k >= MIN_STEM and w[-k:] in sbl[k]:
                return 1
        return 0

    cov = np.array([covered(w) for w in raet + latfal])
    n_r = len(raet)
    obs = cov[:n_r].mean() - cov[n_r:].mean()
    log(f'покрытие: RAET {cov[:n_r].mean():.1%}, LAT/FAL '
        f'{cov[n_r:].mean():.1%}; разность {obs:+.1%}')
    rng = np.random.default_rng(SEED)
    sims = np.zeros(R)
    for i in range(R):
        p_ = rng.permutation(len(cov))
        sims[i] = cov[p_[:n_r]].mean() - cov[p_[n_r:]].mean()
    p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
    log(f'нуль (перестановка меток): {sims.mean():+.1%}±{sims.std():.1%}; '
        f'p={p:.4f}')
    log(f'H1 {"ПОДТВЕРЖДЕНА" if p <= 0.05 else "НЕ ПОДТВЕРЖДЕНА"} '
        f'(критерий p<=0.05, единственный прогон)')
    log('\nоговорки: латынь — один контроль, не вся Италия; контраст '
        '«сверх ареала», не доказательство родства сам по себе.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
