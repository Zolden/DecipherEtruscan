# -*- coding: utf-8 -*-
"""§24 (серия 3, цикл 5): лексическая когерентность секций LL — секции
как тематические блоки.

После негативов позиционной грамматики (§16 T2/T3, §19 T4) — другая
единица: СЛОВАРЬ. Вопрос: концентрируются ли повторы типов внутри
секций (датное членение §10)? Статистика: доля пар вхождений одного
типа, попадающих в одну секцию (типы с 2..12 вхождениями; сверхчастотная
формула топ-15 исключена и учтена отдельно). Нуль — циклический сдвиг
вектора секционных меток по последовательности строк (R=10000, seed=42):
сохраняет размеры секций, порядок строк и СМЕЖНОСТЬ (повторы соседних
строк не дают ложного сигнала). Дескриптив: секционно-специфичные слова
(все вхождения в одной секции, n>=3) — заготовка карты глав (§25).
Карта: results/ll_scroll_map_v2.csv.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ll_lexsections.py
"""
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
OUT_LOG = os.path.join('logs', 'etr_ll_lexsections.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    rows = []
    with open(os.path.join('results', 'll_scroll_map_v2.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['position']),
                         set(r['classes'].split()) - {'—'},
                         [w for w in r['text'].split() if len(w) >= 3]))
    rows.sort()
    n = len(rows)
    cal_idx = [i for i, (_, c, _) in enumerate(rows) if 'CAL' in c]
    runs = {i for j, i in enumerate(cal_idx)
            if j == 0 or cal_idx[j - 1] != i - 1}
    sec = np.zeros(n, int)
    cur = 0
    for i in range(n):
        if i in runs and i > 0:
            cur += 1
        sec[i] = cur
    n_sec = cur + 1
    log('=== §24: лексическая когерентность секций LL ===')
    log(f'строк: {n}; секций: {n_sec}')

    occ = defaultdict(list)   # word -> [line_idx]
    for i, (_, _, ws) in enumerate(rows):
        for w in ws:
            occ[w].append(i)
    freq = {w: len(v) for w, v in occ.items()}
    formulaic = {w for w, _ in sorted(freq.items(),
                                      key=lambda kv: -kv[1])[:15]}
    tested = sorted(w for w, c in freq.items()
                    if 2 <= c <= 12 and w not in formulaic)
    log(f'типов с 2–12 вхождениями (без топ-15 формулы '
        f'{sorted(formulaic)[:6]}…): {len(tested)}')

    def share(sec_vec):
        tot = same = 0
        for w in tested:
            idx = occ[w]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    tot += 1
                    same += sec_vec[idx[a]] == sec_vec[idx[b]]
        return same / max(tot, 1)

    obs = share(sec)
    rng = np.random.default_rng(SEED)
    sims = np.zeros(R)
    for r_i in range(R):
        sh = int(rng.integers(1, n))
        sims[r_i] = share(np.roll(sec, sh))
    p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
    log(f'доля одно-секционных пар повторов: {obs:.1%}; нуль сдвигов '
        f'{sims.mean():.1%}±{sims.std():.1%}; p={p:.4f}')

    # формульная лексика — отдельно (ожидаемо размазана)
    tested_f = sorted(formulaic & {w for w, c in freq.items() if c >= 2})
    tf = tested
    tested = tested_f
    obs_f = share(sec)
    tested = tf
    log(f'контроль (формульная топ-15): {obs_f:.1%} (та же статистика)')

    log('\n--- секционно-специфичные слова (все вхождения в одной секции, n>=3) ---')
    spec = []
    for w, idx in occ.items():
        if len(idx) >= 3 and len({sec[i] for i in idx}) == 1:
            spec.append((int(sec[idx[0]]), w, len(idx)))
    for s, w, c in sorted(spec):
        log(f'  секция {s:>2}: {w:<14} ×{c}')
    log('\nчтение: p мал => повторы концентрируются в границах секций '
        '(секции — тематические блоки, а не только датные); '
        'секционно-специфичные слова — заготовка карты глав (§25).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
