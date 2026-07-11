# -*- coding: utf-8 -*-
"""Предрегистрация №2: региональная инвариантность MDL-морфологии (Cl).

Правила заморожены в validation/prereg2_region_morphology.md (коммит ДО
прогона; прогон ОДИН; публикация как вышло). H1: доля новых TCap-типов,
разложимых как train-основа + суффикс из S (train = канонический вид
v0.8 БЕЗ ART:TCap), выше нуля случайных суффикс-наборов (R=200, тот же
мультинабор длин), критерий p<=0.05. H2 (вторичная): bits/token модели
на TCap < посимвольной 3-граммы B1.

MDL-машинерия ИМПОРТИРУЕТСЯ из tools/etr_mdl_morph.py (analyses_of,
counts_of, assign_round, total_dl, approx_gain и константы) — расхождение
процедур исключено по построению; жадный цикл повторён построчно.

Запуск (единственный):
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_prereg2_region.py
"""
import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etr_mdl_morph as M  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
OUT_LOG = os.path.join('logs', 'etr_prereg2_region.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== Предрегистрация №2: MDL-морфология → регион Cl (Clusium) ===')
    log('правила: validation/prereg2_region_morphology.md (заморожены до прогона)')

    # памятник в test, если хотя бы одна его запись имеет region 'Cl'
    cl_arts = {r['artifact_id'] for r in view if (r.get('region') or '') == 'Cl'}
    tr_freq, te_freq = Counter(), Counter()
    n_tr = n_te = 0
    for r in view:
        is_te = r['artifact_id'] in cl_arts
        tgt = te_freq if is_te else tr_freq
        if is_te:
            n_te += 1
        else:
            n_tr += 1
        for t in r['toks']:
            if t['kind'] == 'W' and '-' not in t['ascii'] \
                    and len(t['ascii']) >= 3:
                tgt[t['ascii']] += 1
    N = sum(tr_freq.values())
    N_te = sum(te_freq.values())
    types_sorted = sorted(tr_freq)
    log(f'train (без Cl-памятников): {n_tr} записей, {N} токенов, '
        f'{len(types_sorted)} типов; Cl: {n_te} записей, {N_te} токенов, '
        f'{len(te_freq)} типов')

    alphabet = sorted({ch for w in types_sorted for ch in w})
    A = len(alphabet)
    H0 = math.log2(A + 1)
    cand = {}
    for w in types_sorted:
        for k in range(1, M.MAX_SUF_LEN + 1):
            if len(w) - k >= M.MIN_STEM:
                cand.setdefault(w[-k:], []).append(w)
    pool = {c: ws for c, ws in cand.items() if len(ws) >= M.MIN_TYPES}
    log(f'кандидатов: {len(pool)}; H0={H0:.4f}')

    # --- жадный рост S (построчно как в etr_mdl_morph.main) -----------------
    rng = np.random.default_rng(M.SEED)
    S = {''}
    assign = {w: (w, '') for w in types_sorted}
    cs, cu, users = M.counts_of(assign, tr_freq)
    dl_cur, _, _ = M.total_dl(assign, tr_freq, S, N, H0)
    added = []
    banned = set()
    while True:
        best_g, best_c, best_sw = 0.0, None, None
        for c in sorted(pool):
            if c in S or c in banned:
                continue
            g, sw = M.approx_gain(c, pool[c], assign, tr_freq,
                                  cs, cu, users, S, N, H0)
            if g > best_g + 1e-9:
                best_g, best_c, best_sw = g, c, sw
        if best_c is None:
            break
        snap = dict(assign)
        S.add(best_c)
        for w in best_sw:
            assign[w] = (w[:-len(best_c)], best_c)
        cs, cu, users = M.counts_of(assign, tr_freq)
        for _ in range(3):
            assign = M.assign_round(types_sorted, tr_freq, S, assign,
                                    cs, cu, users, N, H0)
            cs, cu, users = M.counts_of(assign, tr_freq)
        dl_new, _, _ = M.total_dl(assign, tr_freq, S, N, H0)
        if dl_cur - dl_new <= 1e-6:
            S.discard(best_c)
            banned.add(best_c)
            assign = snap
            cs, cu, users = M.counts_of(assign, tr_freq)
            continue
        added.append(best_c)
        dl_cur = dl_new
        if len(S) - 1 >= M.MAX_SUF:
            break
    S_non = sorted(s for s in S if s)
    log(f'|S| = {len(S_non)}; первые 12: '
        + ' '.join('-' + c for c in added[:12]))
    stems_train = set(cs)

    # --- H1: продуктивность на новых TCap-типах ------------------------------
    te_types = sorted(te_freq)
    oov = [w for w in te_types if w not in tr_freq]
    log(f'\nH1: новых TCap-типов (нет в train): {len(oov)} из {len(te_types)}')
    sets_by_len = {k: {s for s in S_non if len(s) == k}
                   for k in range(1, M.MAX_SUF_LEN + 1)}

    def decomposable(w, sbl):
        for k in range(1, M.MAX_SUF_LEN + 1):
            if len(w) - k >= M.MIN_STEM and w[-k:] in sbl[k] \
                    and w[:-k] in stems_train:
                return True
        return False

    obs_dec = sum(1 for w in oov if decomposable(w, sets_by_len))
    obs = obs_dec / max(len(oov), 1)
    pool_by_len = {k: sorted(c for c in pool if len(c) == k)
                   for k in range(1, M.MAX_SUF_LEN + 1)}
    need = Counter(len(s) for s in S_non)
    nulls = []
    guard = 0
    while len(nulls) < M.R_NULL and guard < 100 * M.R_NULL:
        guard += 1
        cur = set()
        for k in sorted(need):
            idx = rng.choice(len(pool_by_len[k]), size=need[k],
                             replace=False)
            cur.update(pool_by_len[k][i] for i in idx)
        if cur == set(S_non):
            continue
        nbl = {k: {s for s in cur if len(s) == k}
               for k in range(1, M.MAX_SUF_LEN + 1)}
        nulls.append(sum(1 for w in oov if decomposable(w, nbl))
                     / max(len(oov), 1))
    nulls = np.array(nulls)
    p1 = float((1 + (nulls >= obs - 1e-12).sum()) / (len(nulls) + 1))
    log(f'разложимы {obs_dec}/{len(oov)} = {100 * obs:.1f}%; нуль '
        f'{100 * nulls.mean():.1f}% (max {100 * nulls.max():.1f}%); '
        f'p = {p1:.4f}')
    log(f'H1 {"ПОДТВЕРЖДЕНА" if p1 <= 0.05 else "НЕ ПОДТВЕРЖДЕНА"} '
        f'(критерий p<=0.05, единственный прогон)')

    # --- H2: bits/token на TCap ---------------------------------------------
    Vs, Vu = len(cs), len(S)
    char_cnt = Counter()
    for w in types_sorted:
        f = tr_freq[w]
        for ch in w:
            char_cnt[ch] += f
    char_denom = sum(char_cnt.values()) + N + (A + 1)

    def char_cost(sq):
        b = sum(-math.log2((char_cnt.get(ch, 0) + 1) / char_denom)
                for ch in sq)
        return b + -math.log2((N + 1) / char_denom)

    def model_cost(w):
        best = None
        for st, su in M.analyses_of(w, S):
            sb = (-math.log2((cs.get(st, 0) + 1) / (N + Vs))
                  if st in stems_train else char_cost(st))
            c = sb + -math.log2((cu.get(su, 0) + 1) / (N + Vu))
            if best is None or c < best - 1e-12:
                best = c
        return best

    tri, ctx = Counter(), Counter()
    for w in types_sorted:
        f = tr_freq[w]
        seq = '^^' + w + '$'
        for i in range(2, len(seq)):
            tri[(seq[i - 2:i], seq[i])] += f
            ctx[seq[i - 2:i]] += f
    V1 = A + 1

    def b1_cost(w):
        seq = '^^' + w + '$'
        return sum(-math.log2((tri.get((seq[i - 2:i], seq[i]), 0) + 0.5)
                              / (ctx.get(seq[i - 2:i], 0) + 0.5 * V1))
                   for i in range(2, len(seq)))

    bits_m = sum(te_freq[w] * model_cost(w) for w in te_types) / N_te
    bits_b1 = sum(te_freq[w] * b1_cost(w) for w in te_types) / N_te
    log(f'\nH2: биты/токен на TCap: модель {bits_m:.3f} | B1 {bits_b1:.3f} '
        f'({bits_b1 - bits_m:+.3f})')
    log(f'H2 {"ПОДТВЕРЖДЕНА" if bits_m < bits_b1 else "НЕ ПОДТВЕРЖДЕНА"} '
        f'(зарегистрированное направление: модель < B1)')
    log('\nоговорки предрегистрации: OCR-шум TCap консервативен к H1; '
        'жанровый сдвиг — часть утверждения. Результат публикуется как есть.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
