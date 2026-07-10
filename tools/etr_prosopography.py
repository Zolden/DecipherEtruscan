# -*- coding: utf-8 -*-
"""§4: просопографический каркас — роды × регионы с нуль-моделью.

Идея: ономастические основы (роды/gentilicia) должны быть локальны;
проверяем это как статистическую гипотезу и получаем каркас для
датировок и семантики.

Данные: вхождения типов, размеченных NAME-M/F (метки v2 = Хилл+ETP_POS),
в записях вида v0.3 с известным регионом. Основа = слово минус один слой
падежного окончания (isa/ial/al/us/sa/s/l — длиннейшее подходящее;
детерминированно).

Тесты (R=10000, seed=42):
  1. ГЛОБАЛЬНО: взаимная информация MI(основа, регион) по вхождениям
     против перестановки регионов — «роды локальны как класс».
  2. ПО ОСНОВАМ (n≥5 вхождений): статистика = максимум вхождений в одном
     регионе; нуль — та же перестановка; семейный контроль Westfall–Young
     min-p по всем тестируемым основам.
Дескриптивно: со-встречаемость основ в записях (пары n≥3), формулы
филиации (генитив + clan/sec).

Выход: results/prosopography_v1.csv, logs/etr_prosopography.log.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_prosopography.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R = 10000
SEED = 42
MIN_OCC = 5
OUT_LOG = os.path.join('logs', 'etr_prosopography.log')
OUT_CSV = os.path.join('results', 'prosopography_v1.csv')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def view_of(corpus):
    return [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]


def gloss_class(g):
    g2 = g.replace('mrs-', '###-').replace('ms-', '###-')
    i_f = min([i for i in (g.find('mrs-'), g.find('ms-')) if i != -1],
              default=-1)
    i_m = g2.find('mr-')
    if i_f != -1 and (i_m == -1 or i_f < i_m):
        return 'NAME-F'
    if i_m != -1:
        return 'NAME-M'
    return 'OTHER'


def to_ascii_word(w):
    w = re.sub(r'[^a-zθχφσςśšê\']', '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


CASE_ENDS = ('isa', 'ial', 'al', 'us', 'sa', 's', 'l')


def stem_of(w):
    for e in CASE_ENDS:
        if w.endswith(e) and len(w) - len(e) >= 3:
            return w[:-len(e)]
    return w


# не-имена, просочившиеся в разметку (операторы/формульные слова) — вон
NOT_NAMES = set(('mi mini mine clan clens sec sech puia ati apa lupu lupuce '
                 'svalce avils avil ril turce turuce muluvanice mulvanice '
                 'zinace zilath zilc suthi thui cver tular ame amce itun ita '
                 'ica eca ca cn cen etnam vacl fler naper tiur').split())


def name_types(view):
    """Типы NAME-M/F: глоссы Хилла (одно-словные) + ETP_POS masc/fem."""
    names = set()
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        if r['trs'] and len(ws) == 1 and '-' not in ws[0] and len(ws[0]) >= 3:
            if gloss_class(' '.join(r['trs']).lower()).startswith('NAME'):
                names.add(ws[0])
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = to_ascii_word(row.get('Etruscan'))
            if len(w) >= 3 and ((row.get('masc') or '').strip() == '1'
                                or (row.get('fem') or '').strip() == '1'):
                if (row.get('theo') or '').strip() != '1':
                    names.add(w)
    return names - NOT_NAMES


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    view = view_of(corpus)
    names = name_types(view)
    log('=== Просопографический каркас (§4) ===')
    log(f'вид: {len(view)} записей; типов-имён (метки v2): {len(names)}; '
        f'R={R}, seed={SEED}')

    # вхождения имя-основа × регион
    occ_stem = []
    occ_region = []
    rec_stems = []
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        stems_here = []
        for w in ws:
            if w in names:
                st = stem_of(w)
                stems_here.append(st)
                if r['region']:
                    occ_stem.append(st)
                    occ_region.append(r['region'])
        if len(set(stems_here)) >= 2:
            rec_stems.append(sorted(set(stems_here)))
    log(f'вхождений имён в записях с регионом: {len(occ_stem)}; '
        f'основ: {len(set(occ_stem))}; записей с ≥2 основами: '
        f'{len(rec_stems)}')

    regions = sorted(set(occ_region))
    reg_idx = np.array([regions.index(x) for x in occ_region])
    stems_all = sorted(set(occ_stem))
    stem_idx = {s: i for i, s in enumerate(stems_all)}
    st_arr = np.array([stem_idx[s] for s in occ_stem])
    nS, nR_ = len(stems_all), len(regions)
    N = len(occ_stem)

    def contingency(perm_regions):
        M = np.zeros((nS, nR_), dtype=np.int32)
        np.add.at(M, (st_arr, perm_regions), 1)
        return M

    M_obs = contingency(reg_idx)

    def mi_of(M):
        P = M / N
        pr = P.sum(axis=1, keepdims=True)
        pc = P.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = P * np.log(P / (pr @ pc))
        return float(np.nansum(t))

    mi_obs = mi_of(M_obs)
    rng = np.random.default_rng(SEED)
    tested = [i for i in range(nS) if M_obs[i].sum() >= MIN_OCC]
    max_obs = M_obs[tested].max(axis=1)
    n_test = len(tested)
    mi_sim = np.zeros(R)
    max_sim = np.zeros((R, n_test), dtype=np.int32)
    for r_i in range(R):
        pr = reg_idx[rng.permutation(N)]
        M = contingency(pr)
        mi_sim[r_i] = mi_of(M)
        max_sim[r_i] = M[tested].max(axis=1)
    p_mi = float(((mi_sim >= mi_obs).sum() + 1) / (R + 1))
    log()
    log(f'--- 1. глобальный тест локальности ---')
    log(f'MI(основа, регион) = {mi_obs:.4f} нат; нуль: среднее '
        f'{mi_sim.mean():.4f}, max {mi_sim.max():.4f}; p={p_mi:.4f}')

    # по основам: W–Y
    p_raw = ((max_sim >= max_obs[None, :]).sum(axis=0) + 1) / (R + 1)
    p_sim = np.zeros_like(max_sim, dtype=np.float64)
    for j in range(n_test):
        col = max_sim[:, j]
        order = np.sort(col)
        idx = np.searchsorted(order, col, side='left')
        p_sim[:, j] = (R - idx + 1) / (R + 1)
    minp = p_sim.min(axis=1)
    p_adj = np.array([((minp <= p).sum() + 1) / (R + 1) for p in p_raw])
    n_sig = int((p_adj < 0.05).sum())
    log()
    log(f'--- 2. по основам (n≥{MIN_OCC}: {n_test} основ; W–Y) ---')
    log(f'регионально сконцентрированы после семейного контроля: {n_sig}')
    order2 = np.argsort(p_adj, kind='stable')
    log(f'{"основа":<12} {"n":>4} {"топ-регион":>10} {"доля":>6} '
        f'{"p":>8} {"p̃_сем":>8}')
    rows_csv = []
    for j in order2[:15]:
        i = tested[j]
        st = stems_all[i]
        n = int(M_obs[i].sum())
        k = int(M_obs[i].argmax())
        share = M_obs[i, k] / n
        log(f'{st:<12} {n:>4} {regions[k]:>10} {share:>6.0%} '
            f'{p_raw[j]:>8.4f} {p_adj[j]:>8.4f}')
    for j in range(n_test):
        i = tested[j]
        st = stems_all[i]
        n = int(M_obs[i].sum())
        k = int(M_obs[i].argmax())
        rows_csv.append({'stem': st, 'n_occ': n,
                         'top_region': regions[k],
                         'top_share': f'{M_obs[i, k] / n:.3f}',
                         'p': f'{p_raw[j]:.5f}',
                         'p_adj': f'{p_adj[j]:.5f}'})

    # --- дескриптивный каркас ---------------------------------------------
    log()
    log('--- 3. каркас: со-встречаемость основ (дескриптивно) ---')
    pair_cnt = Counter()
    for sts in rec_stems:
        for a_i in range(len(sts)):
            for b_i in range(a_i + 1, len(sts)):
                pair_cnt[(sts[a_i], sts[b_i])] += 1
    top_pairs = [(p, c) for p, c in pair_cnt.most_common(15) if c >= 3]
    for (a, b), c in top_pairs:
        log(f'  {a} + {b}: ×{c}')
    filia = 0
    KIN = {'clan', 'sec', 'sech', 'puia'}
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        for i in range(len(ws) - 1):
            if ws[i + 1] in KIN and stem_of(ws[i]) != ws[i]:
                filia += 1
    log(f'формул филиации «X-ген + clan/sec/puia»: ×{filia}')

    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'n_occ', 'top_region',
                                          'top_share', 'p', 'p_adj'], lineterminator='\n')
        w.writeheader()
        for row in sorted(rows_csv, key=lambda x: float(x['p_adj'])):
            w.writerow(row)
    log()
    log(f'каркас записан: {OUT_CSV}')
    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
