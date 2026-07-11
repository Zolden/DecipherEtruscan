# -*- coding: utf-8 -*-
"""Ответ на вопрос LA-треда (CROSS_PROJECT, обратная связь): есть ли в
этрусском операторы «ВТОРОЙ позиции» (после дейктического открытия)?
LA нашёл TE и SA-RA2 значимо вторыми (p̃_сем=0.001–0.003).

Тест: для операторов §1 — обогащение позиции i==1 в записях с ≥3
словоформами; нуль — равномерная позиция (как §1), R=10000, seed=42,
Westfall–Young по семейству.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_second_position.py
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_second_position.log')
LOG = []
OPS = {
    'mi': ('mi',), 'mini': ('mini', 'mine'), 'clan': ('clan',),
    'sec': ('sec', 'sech'), 'puia': ('puia',),
    'lupu': ('lupu', 'lupuce'), 'svalce': ('svalce',),
    'avils': ('avils',), 'turce': ('turce', 'turuce', 'turice'),
    'muluvanice': ('muluvanice', 'muluvanike', 'mulvanice', 'mulvenice'),
    'zinace': ('zinace', 'zinake'),
    'zilath': ('zilath', 'zilc', 'zilci', 'zilachnce'),
    'suthi': ('suthi', 'suti'), 'thui': ('thui',),
    'ame': ('ame', 'amce'), 'cver': ('cver', 'cvera'),
    'itun': ('itun', 'ita', 'ica', 'eca', 'ca', 'cn', 'cen'),
    'etnam': ('etnam',), 'vacl': ('vacl',), 'fler': ('fler',),
}


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    log('RETRACT 2026-07-10: семейные p̃ ниже невалидны; turce не отличается '
        'от других dedicatory verbs (Fisher p=.358). См. §8 отчёта.')
    log()
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    occ = {}
    for rec in view:
        ws = [t['ascii'] for t in rec['toks'] if t['kind'] == 'W']
        if len(ws) < 3:
            continue
        for name, forms in OPS.items():
            fs = set(forms)
            for i, w in enumerate(ws):
                if w in fs:
                    occ.setdefault(name, []).append((i, len(ws)))
    log('=== Операторы второй позиции (ответ LA-треду) ===')
    log(f'записей ≥3 словоформ: '
        f'{sum(1 for r in view if sum(t["kind"] == "W" for t in r["toks"]) >= 3)}; '
        f'R={R}, seed={SEED}')
    tests = [(n, o) for n, o in sorted(occ.items()) if len(o) >= 5]
    rng = np.random.default_rng(SEED)
    # СОВМЕСТНЫЙ нуль (исправление 2026-07-10): общая перестановка
    # порядка слов записи на replicate, статистика — позиция 1 (вторая).
    R_WY = 5000
    op_ix = {n: k for k, (n, _) in enumerate(tests)}
    rec_labels = []
    for rec in view:
        ws = [t['ascii'] for t in rec['toks'] if t['kind'] == 'W']
        if len(ws) < 3:
            continue
        lab = np.full(len(ws), -1, dtype=np.int16)
        hit = False
        for n, forms in OPS.items():
            if n not in op_ix:
                continue
            for i, w in enumerate(ws):
                if w in forms:
                    lab[i] = op_ix[n]
                    hit = True
        if hit:
            rec_labels.append(lab)
    sims = np.zeros((R_WY, len(tests)), np.int32)
    obs = np.zeros(len(tests), int)
    for j, (n, o) in enumerate(tests):
        obs[j] = sum(1 for i, _ in o if i == 1)
    for r_i in range(R_WY):
        for lab in rec_labels:
            perm = rng.permutation(len(lab))
            second = lab[perm[1]]
            if second >= 0:
                sims[r_i, second] += 1
    p_raw = ((sims >= obs[None, :]).sum(axis=0) + 1) / (R_WY + 1)
    p_sim = np.zeros_like(sims, float)
    for j in range(len(tests)):
        col = np.sort(sims[:, j])
        idx = np.searchsorted(col, sims[:, j], side='left')
        p_sim[:, j] = (R_WY - idx + 1) / (R_WY + 1)
    minp = p_sim.min(axis=1)
    log(f'{"оператор":<12} {"n":>4} {"2-я поз.":>8} {"p":>8} {"p̃_сем":>8}')
    for j, (n, o) in enumerate(tests):
        padj = ((minp <= p_raw[j]).sum() + 1) / (R_WY + 1)
        mark = ' *' if padj < 0.05 else ''
        log(f'{n:<12} {len(o):>4} {obs[j]:>4} ({obs[j] / len(o):>4.0%}) '
            f'{p_raw[j]:>8.4f} {padj:>8.4f}{mark}')
    # --- открытость инвентарей позиций (ответ на §BW LA) --------------------
    log()
    log('--- открытость инвентарей: позиция 1 vs 2 (записи ≥3 словоформ) ---')
    DEIX = {'mi', 'mini', 'mine', 'itun', 'ita', 'ica', 'eca', 'ca', 'cn',
            'cen', 'cehen', 'thui'}
    p1, p2 = [], []
    for rec in view:
        ws = [t['ascii'] for t in rec['toks'] if t['kind'] == 'W']
        if len(ws) < 3:
            continue
        seq = [w for w in ws]
        if seq[0] in DEIX:  # после дейктического открытия сдвиг
            seq = seq[1:]
        if len(seq) >= 2:
            p1.append(seq[0])
            p2.append(seq[1])
    ttr1 = len(set(p1)) / len(p1)
    ttr2 = len(set(p2)) / len(p2)
    # нуль: перестановка слов между позициями внутри пар
    rng3 = np.random.default_rng(SEED)
    diffs = np.zeros(R)
    arr = np.array(list(zip(p1, p2)))
    for r_i in range(R):
        flip = rng3.random(len(arr)) < 0.5
        a = np.where(flip, arr[:, 1], arr[:, 0])
        b = np.where(flip, arr[:, 0], arr[:, 1])
        diffs[r_i] = (len(set(a.tolist())) - len(set(b.tolist()))) / len(arr)
    obs_d = ttr1 - ttr2
    p_open = float(((diffs >= obs_d).sum() + 1) / (R + 1))
    log(f'пар позиций: {len(p1)}; TTR(поз.1)={ttr1:.2f}, '
        f'TTR(поз.2)={ttr2:.2f}; Δ={obs_d:+.2f}; '
        f'нуль (обмен внутри пар): p={p_open:.4f}')
    log(f'топ слов позиции 2: {Counter(p2).most_common(8)}')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
