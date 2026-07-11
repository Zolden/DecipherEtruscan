# -*- coding: utf-8 -*-
"""§30 (серия 4, цикл 2): стратиграфия v2 — датированный корпус после
вливания датировок Адрии (v0.10, §29).

Три вопроса:
1. Сколько теперь датированного и как оно распределено по векам/регионам
   (дескриптив; вклад Адрии отдельной строкой).
2. Повторение теста §4.1 (MI(основа, вековой бин), перестановка бинов,
   R=10000) на расширенной выборке — устойчивы ли основы во времени.
3. НОВОЕ — хронологический профиль морфологии: доля -s против -al
   генитивов по вековым бинам (перестановочный нуль бинов, R=10000):
   есть ли дрейф падежного выбора во времени (архаический/поздний).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_stratigraphy2.py
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
OUT_LOG = os.path.join('logs', 'etr_stratigraphy2.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


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
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    dated = [r for r in view if r.get('y_from') is not None]
    adria = [r for r in dated if '[дата: CIE IV.1.1]' in (r.get('note') or '')]
    log('=== §30: стратиграфия v2 (датировки Адрии влиты) ===')
    log(f'датированных записей вида: {len(dated)} '
        f'(из них Адрия v0.10: {len(adria)})')
    bins = Counter(century_bin(r['y_from']) for r in dated)
    log('по бинам: ' + ', '.join(f'{b}: {n}' for b, n in sorted(bins.items())))
    regs = Counter((r.get('region') or '?') for r in dated)
    log('по регионам: ' + ', '.join(f'{g}:{n}' for g, n in
                                    regs.most_common(8)))

    # --- 2. MI(основа, бин) v2 ------------------------------------------------
    occ = []
    for r in dated:
        b = century_bin(r['y_from'])
        for t in r['toks']:
            if t['kind'] == 'W' and '-' not in t['ascii'] \
                    and len(t['ascii']) >= 4:
                occ.append((stem_of(t['ascii']), b))
    stems = Counter(s for s, _ in occ)
    tested = {s for s, c in stems.items() if c >= 3}
    occ_t = [(s, b) for s, b in occ if s in tested]
    s_list = sorted(tested)
    si = {s: i for i, s in enumerate(s_list)}
    b_list = sorted({b for _, b in occ_t})
    bi = {b: i for i, b in enumerate(b_list)}
    sa = np.array([si[s] for s, _ in occ_t])
    ba = np.array([bi[b] for _, b in occ_t])
    N = len(occ_t)
    log(f'\n--- MI(основа, век) v2: вхождений {N}, основ n>=3: {len(tested)} ---')

    def mi_of(barr):
        M = np.zeros((len(s_list), len(b_list)))
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
    log(f'MI = {mi:.4f}; нуль {sims.mean():.4f}±{sims.std():.4f}; '
        f'p(специфичность)={((sims >= mi).sum() + 1) / (R + 1):.4f}, '
        f'p(устойчивость)={((sims <= mi).sum() + 1) / (R + 1):.4f}')

    # --- 3. хронология генитивов -s vs -al ------------------------------------
    log('\n--- хронология генитивов: -s против -al по бинам ---')
    ev = []
    for r in dated:
        b = century_bin(r['y_from'])
        for t in r['toks']:
            w = t['ascii']
            if t['kind'] != 'W' or '-' in w or len(w) < 5:
                continue
            if w.endswith('al'):
                ev.append((b, 1))
            elif w.endswith('s') and not w.endswith(('is', 'us', 'es')):
                ev.append((b, 0))
    cnt = defaultdict(Counter)
    for b, is_al in ev:
        cnt[b][is_al] += 1
    for b in b_list:
        n_al, n_s = cnt[b][1], cnt[b][0]
        if n_al + n_s:
            log(f'  {b:<11} -al {n_al:>3} / -s {n_s:>3} '
                f'(доля -al {n_al / (n_al + n_s):.1%})')
    labs = np.array([x for _, x in ev])
    bins_e = np.array([bi.get(b, -1) for b, _ in ev])
    obs_tab = np.array([labs[bins_e == i].mean() if (bins_e == i).any()
                        else np.nan for i in range(len(b_list))])
    obs_spread = np.nanmax(obs_tab) - np.nanmin(obs_tab)
    sims2 = np.zeros(R)
    for r_i in range(R):
        pb = bins_e[rng.permutation(len(bins_e))]
        tab = np.array([labs[pb == i].mean() if (pb == i).any() else np.nan
                        for i in range(len(b_list))])
        sims2[r_i] = np.nanmax(tab) - np.nanmin(tab)
    p2 = float((1 + (sims2 >= obs_spread - 1e-12).sum()) / (R + 1))
    log(f'разброс доли -al между бинами: {obs_spread:.3f}; нуль '
        f'{sims2.mean():.3f}±{sims2.std():.3f}; p={p2:.4f}')
    log('\nчтение: п.2 — вековая привязка основ; п.3 — дрейф падежного '
        'выбора во времени; оговорка: датированная выборка смещена к '
        'ETP+Адрия, интерпретировать с регионами из дескриптива.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
