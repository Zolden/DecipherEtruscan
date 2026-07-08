# -*- coding: utf-8 -*-
"""§2.7: комплементарность падежных маркеров — фонотактика или грамматика?

Комплементарные пары §2.3 (например -s/-al: основа берёт один маркер)
могли бы объясняться фонотактикой конца основы (после гласной — один
маркер, после согласной — другой), а не грамматическими классами.
Контроль: тот же тест депрессии пересечения, но СТРАТИФИЦИРОВАННО по
финальной букве основы (и, вторым контролем, по длине основы): внутри
страты фонотактическое условие константно, значит остаточная депрессия —
не от финали.

Нуль: независимые гипергеометрические розыгрыши в каждой страте
(перестановка принадлежности к A/B внутри страты), MC R=10000, seed=42;
p_lo = P(Σ пересечений ≤ наблюдаемого). Ожидание E_strat = Σ|A_f||B_f|/|U_f|
— «фонотактически честное»; сравнение E_unstrat → E_strat → obs
раскладывает эффект на фонотактическую и грамматическую части.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_morph_controls.py
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_morph_controls.log')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def view_of(corpus):
    return [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]


PAIRS = [('s', 'al'), ('s', 'l'), ('l', 'al'), ('s', 'ial'), ('l', 'ial'),
         ('al', 'sa'), ('s', 'na'), ('l', 'sa'), ('ial', 'sa'), ('s', 'isa')]
CAND = ['s', 'l', 'al', 'ial', 'us', 'sa', 'isa', 'la', 'na', 'ce',
        'thi', 'c', 'm', 'ei']


def strat_test(U_by, A, B, rng):
    """Стратифицированная депрессия: страты — ключи U_by (основа→страта)."""
    strata = {}
    for st in A | B | set(U_by):
        strata.setdefault(U_by[st], {'U': 0, 'A': [], 'B': []})
    for st, f in U_by.items():
        s = strata[f]
        s['U'] += 1
        if st in A:
            s['A'].append(st)
        if st in B:
            s['B'].append(st)
    obs = len(A & B)
    E = 0.0
    sims = np.zeros(R)
    for f, s in sorted(strata.items()):
        nU, nA, nB = s['U'], len(s['A']), len(s['B'])
        if nA and nB:
            E += nA * nB / nU
            sims += rng.hypergeometric(nA, nU - nA, nB, size=R)
    p_lo = float(((sims <= obs).sum() + 1) / (R + 1))
    return obs, E, p_lo


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.4'
    view = view_of(corpus)
    vocab = {w for w in Counter(t['ascii'] for r in view for t in r['toks']
                                if t['kind'] == 'W')
             if '-' not in w and len(w) >= 2}
    log('=== §2.7: стратифицированный контроль комплементарности ===')
    log(f'R={R}, seed={SEED}; словарь: {len(vocab)} типов')

    stems_of = {}
    for s in CAND:
        stems_of[s] = {w[:-len(s)] for w in vocab
                       if w.endswith(s) and len(w) >= len(s) + 3}
    U = set()
    for s in CAND:
        U |= stems_of[s]
    log(f'универсум основ: {len(U)}')

    rng = np.random.default_rng(SEED)
    log()
    log(f'{"пара":<10} {"∩":>4} {"E_нестрат":>10} {"E_финаль":>10} '
        f'{"p(финаль)":>10} {"E_фин+длина":>12} {"p(ф+д)":>10}')
    m = len(PAIRS)
    n_sig = 0
    for s1, s2 in PAIRS:
        A, B = stems_of[s1], stems_of[s2]
        E_un = len(A) * len(B) / len(U)
        by_final = {st: st[-1] for st in U}
        obs, E_f, p_f = strat_test(by_final, A, B, rng)
        by_fl = {st: (st[-1], min(len(st), 8)) for st in U}
        _, E_fl, p_fl = strat_test(by_fl, A, B, rng)
        star = ' *' if min(1.0, p_fl * m) < 0.05 else ''
        n_sig += min(1.0, p_fl * m) < 0.05
        log(f'-{s1}/-{s2:<6} {obs:>4} {E_un:>10.1f} {E_f:>10.1f} '
            f'{p_f:>10.2e} {E_fl:>12.1f} {p_fl:>10.2e}{star}')
    log(f'* — p×{m} < 0.05 (Бонферрони по {m} парам, строгая страта '
        f'финаль×длина)')
    log(f'пар, комплементарных ПОСЛЕ контроля финали и длины: {n_sig}/{m}')
    log()
    log('чтение таблицы: если E_финаль намного меньше E_нестрат — часть '
        'эффекта фонотактическая; если p при стратах мал — остаток '
        'грамматический (классы основ), финалью не объясним.')

    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
