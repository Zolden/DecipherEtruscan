# -*- coding: utf-8 -*-
"""§14.2: тройная батарея нулей (рекомендация LA-ветки, CROSS_PROJECT
2026-07-11) для ядра позиционной типологии: mi (инициальность) и
lupu (финальность).

Нуль A — длина×позиция: перестановка порядка слов внутри записи
(совместная для семейства из двух статистик, min-p; R=5000) — корректный
W-Y §8.8, воспроизводится здесь для полноты батареи.
Нуль B — марковский биграммный: цепь ТОЛЬКО на внутренних биграммах
(w→w', без BOS/EOS — граничные переходы кодировали бы позиционность
тавтологично); старт равномерно по словарю, длины записей реальные;
R=200 синтетических корпусов; p = доля корпусов, где позиционная доля
цели >= наблюдаемой (пропуски при <5 вхождениях цели логируются).
Нуль C — плацебо-специфичность: не-операторные слова частотной полосы
цели (×0.5..×2, n>=8) — их позиционные доли образуют плацебо-полосу;
p = (1 + #{плацебо >= цель}) / (K + 1).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_null_battery.py
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R_A = 5000
R_B = 200
OUT_LOG = os.path.join('logs', 'etr_null_battery.log')
LOG = []

OPS_ALL = {'clan', 'sec', 'sech', 'puia', 'ruva', 'ati', 'apa', 'papa',
           'teta', 'nefts', 'turce', 'turuce', 'muluvanice', 'lupu',
           'lupuce', 'svalce', 'amce', 'ame', 'zilath', 'zilch', 'zilc',
           'camthi', 'maru', 'purth', 'mi', 'mini', 'mine', 'ta', 'ca',
           'cn', 'itun', 'eca', 'avils', 'avil', 'ril', 'suthi',
           'suthina', 'mlach', 'flere', 'fler'}


def log(m=''):
    print(m)
    LOG.append(m)


def share(recs, word, kind):
    n_occ = n_pos = 0
    for ws in recs:
        for i, w in enumerate(ws):
            if w == word:
                n_occ += 1
                n_pos += (i == 0) if kind == 'init' else (i == len(ws) - 1)
    return (n_pos / n_occ if n_occ else float('nan')), n_occ


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    recs = []
    for r in view:
        ws = [t['ascii'] for t in r['toks']
              if t['kind'] == 'W' and '-' not in t['ascii']
              and len(t['ascii']) >= 2]
        if len(ws) >= 2:
            recs.append(ws)
    log('=== §14.2: батарея нулей LA для ядра типологии ===')
    log(f'мультисловных записей: {len(recs)}')

    TARGETS = [('mi', 'init'), ('lupu', 'fin')]
    obs = {}
    for w, kind in TARGETS:
        s, n = share(recs, w, kind)
        obs[w] = s
        log(f'наблюдение: {w} {kind}-доля {s:.1%} (вхождений {n})')

    # --- нуль A: длина×позиция (совместная перестановка, min-p) -------------
    rng = np.random.default_rng(SEED)
    rel = [ws for ws in recs if any(w in ('mi', 'lupu') for w in ws)]
    ge = {w: 0 for w, _ in TARGETS}
    ge_fam = 0
    for _ in range(R_A):
        hits = {}
        sim_recs = [list(np.array(ws)[rng.permutation(len(ws))])
                    for ws in rel]
        for w, kind in TARGETS:
            s, _ = share(sim_recs, w, kind)
            hits[w] = s >= obs[w] - 1e-12
            ge[w] += hits[w]
        ge_fam += any(hits.values())
    log('\n--- нуль A (длина×позиция, совместный) ---')
    for w, kind in TARGETS:
        log(f'  {w}: raw p={(ge[w] + 1) / (R_A + 1):.4f}')
    log(f'  семейный min-p (общая перестановка): p̃<={(ge_fam + 1) / (R_A + 1):.4f}')

    # --- нуль B: внутренние биграммы -----------------------------------------
    log('\n--- нуль B (марковский, внутренние биграммы, старт равномерный) ---')
    big = defaultdict(Counter)
    vocab = sorted({w for ws in recs for w in ws})
    vi = {w: i for i, w in enumerate(vocab)}
    for ws in recs:
        for a, b in zip(ws, ws[1:]):
            big[a][b] += 1
    nxt = {}
    for a, cnt in big.items():
        ws_, ns = zip(*sorted(cnt.items()))
        p = np.array(ns, float)
        nxt[a] = (list(ws_), p / p.sum())
    lens = [len(ws) for ws in recs]
    rng_b = np.random.default_rng(SEED + 1)
    sims = {w: [] for w, _ in TARGETS}
    skipped = Counter()
    for _ in range(R_B):
        syn = []
        for L in lens:
            seq = [vocab[int(rng_b.integers(len(vocab)))]]
            for _ in range(L - 1):
                a = seq[-1]
                if a in nxt:
                    ws_, p = nxt[a]
                    seq.append(ws_[int(rng_b.choice(len(ws_), p=p))])
                else:
                    seq.append(vocab[int(rng_b.integers(len(vocab)))])
            syn.append(seq)
        for w, kind in TARGETS:
            s, n = share(syn, w, kind)
            if n >= 5:
                sims[w].append(s)
            else:
                skipped[w] += 1
    for w, kind in TARGETS:
        arr = np.array(sims[w])
        if len(arr) < 50:
            log(f'  {w}: НЕДОСТАТОЧНО синтетических вхождений '
                f'(валидных корпусов {len(arr)}/{R_B}) — нуль B неинформативен')
            continue
        p = float((1 + (arr >= obs[w] - 1e-12).sum()) / (len(arr) + 1))
        log(f'  {w}: синтетическая {kind}-доля {arr.mean():.1%}±{arr.std():.1%} '
            f'(валидных корпусов {len(arr)}, пропущено {skipped[w]}); '
            f'p={p:.4f}')

    # --- нуль C: плацебо-специфичность ---------------------------------------
    log('\n--- нуль C (плацебо: не-операторы частотной полосы цели) ---')
    freq = Counter(w for ws in recs for w in ws)
    for w, kind in TARGETS:
        f0 = freq[w]
        band = [x for x in freq
                if x not in OPS_ALL and freq[x] >= 8
                and 0.5 * f0 <= freq[x] <= 2.0 * f0]
        shares = []
        for x in sorted(band):
            s, n = share(recs, x, kind)
            if n >= 8:
                shares.append((s, x))
        if not shares:
            log(f'  {w}: полоса пуста — плацебо неинформативно')
            continue
        arr = np.array([s for s, _ in shares])
        p = float((1 + (arr >= obs[w] - 1e-12).sum()) / (len(arr) + 1))
        worst = max(shares)
        log(f'  {w} (freq {f0}): плацебо K={len(shares)}, доля '
            f'{arr.mean():.1%}±{arr.std():.1%}, max {worst[0]:.1%} '
            f'({worst[1]}); p={p:.4f}')

    log('\nчтение: батарея по рекомендации LA (2026-07-11). A — позиция не '
        'случайна при данных длинах; B — не выводится из внутренних биграмм; '
        'C — специфична цели, а не её частотному классу.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
