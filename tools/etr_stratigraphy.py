# -*- coding: utf-8 -*-
"""§4.1: поколенческая стратиграфия — основы имён по векам (датированная
ETP-часть, v0.4).

Вопрос: устойчивы ли ономастические основы во времени (роды живут
веками)? Тест: MI(основа, вековой бин) по датированным вхождениям против
перестановки бинов (R=10000, seed=42) — «привязаны ли основы к векам»;
НИЗКАЯ MI против нуля означала бы вековую устойчивость, ВЫСОКАЯ —
вековую специфичность. Плюс дескриптивно: основы, живущие в ≥2 бинах.
Оговорка: датированы только ~300 записей — мощность мала, результат
может быть неинформативным (публикуется в любом случае).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_stratigraphy.py
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
R = 10000
OUT_LOG = os.path.join('logs', 'etr_stratigraphy.log')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


CASE_ENDS = ('isa', 'ial', 'al', 'us', 'sa', 's', 'l')


def stem_of(w):
    for e in CASE_ENDS:
        if w.endswith(e) and len(w) - len(e) >= 3:
            return w[:-len(e)]
    return w


def century_bin(y):
    if y >= 550:
        return 'до -550'
    if y >= 300:
        return '-550…-300'
    return 'после -300'


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None
            and r['y_from'] is not None]
    log('=== §4.1: стратиграфия основ по векам ===')
    log(f'датированных записей вида: {len(view)}')
    occ = []
    for r in view:
        b = century_bin(r['y_from'])
        for t in r['toks']:
            if t['kind'] == 'W' and '-' not in t['ascii'] \
                    and len(t['ascii']) >= 4:
                occ.append((stem_of(t['ascii']), b,
                            r['region'] or r['city'] or '?'))
    stems = Counter(s for s, _, _ in occ)
    tested = {s for s, c in stems.items() if c >= 3}
    occ_t = [(s, b, g) for s, b, g in occ if s in tested]
    log(f'вхождений: {len(occ)}; основ с n≥3: {len(tested)} '
        f'({len(occ_t)} вхождений)')

    bins = sorted({b for _, b, _ in occ_t})
    s_list = sorted(tested)
    si = {s: i for i, s in enumerate(s_list)}
    bi = {b: i for i, b in enumerate(bins)}
    sa = np.array([si[s] for s, _, _ in occ_t])
    ba = np.array([bi[b] for _, b, _ in occ_t])
    N = len(occ_t)

    def mi_of(barr):
        M = np.zeros((len(s_list), len(bins)))
        np.add.at(M, (sa, barr), 1)
        P = M / N
        pr = P.sum(1, keepdims=True)
        pc = P.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = P * np.log(P / (pr @ pc))
        return float(np.nansum(t))

    mi = mi_of(ba)
    rng = np.random.default_rng(SEED)
    sims = np.array([mi_of(ba[rng.permutation(N)]) for _ in range(R)])
    p_hi = float(((sims >= mi).sum() + 1) / (R + 1))
    p_lo = float(((sims <= mi).sum() + 1) / (R + 1))
    log(f'MI(основа, век) = {mi:.4f}; нуль {sims.mean():.4f}±{sims.std():.4f}; '
        f'p(специфичность)={p_hi:.4f}, p(устойчивость)={p_lo:.4f}')

    log()
    log('--- основы, живущие в ≥2 вековых бинах (дескриптивно) ---')
    span = {}
    for s, b, g in occ_t:
        span.setdefault(s, set()).add(b)
    multi = {s: bs for s, bs in span.items() if len(bs) >= 2}
    log(f'таких основ: {len(multi)} из {len(tested)}')
    for s in sorted(multi, key=lambda x: -stems[x])[:10]:
        regs = Counter(g for s2, _, g in occ_t if s2 == s)
        log(f'  {s:<12} n={stems[s]:>3} бины={sorted(multi[s])} '
            f'регионы={dict(regs.most_common(3))}')

    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
