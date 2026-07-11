# -*- coding: utf-8 -*-
"""§33 (серия 4, цикл 6): развязка конфаундов дрейфа генитива §30.

§30 нашёл рост доли -al 2.9%→38.8%→54.5% по вековым бинам (p=0.0002) с
двумя названными конфаундами. Развязки:
1. БЕЗ АДРИИ: дрейф на записях, датированных до v0.10 (Адрия — поздняя
   и могла тащить поздний бин).
2. СТРАТИФИКАЦИЯ ПО РОДУ: -al тяготеет к женским формулярам; тест
   повторяется отдельно на записях с мужским контекстом (муж. преномены
   ETP_POS masc, без женских) и с женским.
3. ВНУТРИРЕГИОНАЛЬНЫЙ НУЛЬ: перестановка бинов ВНУТРИ региона (страты
   сохраняются) — время развязывается от географии.
Статистика везде §30: разброс доли -al между бинами; R=10000, seed=42.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_genitive_drift2.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
OUT_LOG = os.path.join('logs', 'etr_genitive_drift2.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def century_bin(y):
    if y >= 550:
        return 0
    if y >= 300:
        return 1
    return 2


def to_ascii_word(w):
    w = re.sub(r"[^a-zθχφσςśšê']", '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def gender_lex():
    masc, fem = set(), set()
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = to_ascii_word(row.get('Etruscan'))
            if len(w) < 3:
                continue
            if (row.get('masc') or '').strip() == '1':
                masc.add(w)
            if (row.get('fem') or '').strip() == '1':
                fem.add(w)
    return masc - fem, fem - masc


def drift(events, rng):
    """events: [(bin, is_al, stratum)] -> (spread, p) с нулём перестановки
    is_al-меток внутри страт."""
    bins = np.array([b for b, _, _ in events])
    labs = np.array([x for _, x, _ in events])
    strata = np.array([s for _, _, s in events])
    kbins = sorted(set(bins.tolist()))

    def spread(lb):
        vals = [lb[bins == b].mean() for b in kbins if (bins == b).any()]
        return max(vals) - min(vals) if len(vals) >= 2 else np.nan

    obs = spread(labs)
    sims = np.zeros(R)
    idx_by_str = {s: np.where(strata == s)[0]
                  for s in sorted(set(strata.tolist()))}
    for i in range(R):
        lb = labs.copy()
        for s, idx in idx_by_str.items():
            lb[idx] = labs[idx][rng.permutation(len(idx))]
        sims[i] = spread(lb)
    p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
    return obs, sims, p


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None
            and r.get('y_from') is not None]
    masc, fem = gender_lex()
    log('=== §33: развязка конфаундов дрейфа генитива ===')

    def gen_events(recs, stratum_of):
        out = []
        for r in recs:
            b = century_bin(r['y_from'])
            st = stratum_of(r)
            ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W'
                  and '-' not in t['ascii']]
            for w in ws:
                if len(w) < 5:
                    continue
                if w.endswith('al'):
                    out.append((b, 1, st))
                elif w.endswith('s') and not w.endswith(('is', 'us', 'es')):
                    out.append((b, 0, st))
        return out

    rng = np.random.default_rng(SEED)
    tests = []
    tests.append(('полная выборка (§30)', view, lambda r: 0))
    no_adria = [r for r in view
                if '[дата: CIE IV.1.1]' not in (r.get('note') or '')]
    tests.append(('без Адрии', no_adria, lambda r: 0))
    tests.append(('внутрирегиональный нуль', view,
                  lambda r: r.get('region') or '?'))

    def has_gender(r, lex):
        ws = {t['ascii'] for t in r['toks'] if t['kind'] == 'W'}
        return bool(ws & lex)

    male_recs = [r for r in view if has_gender(r, masc)
                 and not has_gender(r, fem)]
    fem_recs = [r for r in view if has_gender(r, fem)
                and not has_gender(r, masc)]
    tests.append((f'мужской контекст (n_rec={len(male_recs)})',
                  male_recs, lambda r: 0))
    tests.append((f'женский контекст (n_rec={len(fem_recs)})',
                  fem_recs, lambda r: 0))

    for name, recs, strat in tests:
        ev = gen_events(recs, strat)
        if len(ev) < 30:
            log(f'{name:<32} событий {len(ev)} — мало')
            continue
        by_bin = defaultdict(Counter)
        for b, x, _ in ev:
            by_bin[b][x] += 1
        prof = ' '.join(
            f'бин{b}:{by_bin[b][1]}/{by_bin[b][0] + by_bin[b][1]}'
            f'({by_bin[b][1] / max(by_bin[b][0] + by_bin[b][1], 1):.0%})'
            for b in sorted(by_bin))
        obs, sims, p = drift(ev, rng)
        log(f'{name:<32} n={len(ev):>3} {prof}')
        log(f'{"":<32} разброс {obs:.3f} (нуль {sims.mean():.3f}'
            f'±{sims.std():.3f}) p={p:.4f}')
    log('\nчтение: дрейф устойчив, если держится без Адрии, при '
        'внутрирегиональном нуле и в обоих родовых срезах (род может '
        'законно нести часть сигнала — поздние женские формуляры).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
