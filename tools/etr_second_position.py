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
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.4'
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
    sims = np.zeros((R, len(tests)), np.int32)
    obs = np.zeros(len(tests), int)
    for j, (n, o) in enumerate(tests):
        Ls = np.array([L for _, L in o])
        obs[j] = sum(1 for i, _ in o if i == 1)
        u = rng.random((R, len(o)))
        sims[:, j] = ((u >= 1.0 / Ls) & (u < 2.0 / Ls)).sum(axis=1)
    p_raw = ((sims >= obs[None, :]).sum(axis=0) + 1) / (R + 1)
    p_sim = np.zeros_like(sims, float)
    for j in range(len(tests)):
        col = np.sort(sims[:, j])
        idx = np.searchsorted(col, sims[:, j], side='left')
        p_sim[:, j] = (R - idx + 1) / (R + 1)
    minp = p_sim.min(axis=1)
    log(f'{"оператор":<12} {"n":>4} {"2-я поз.":>8} {"p":>8} {"p̃_сем":>8}')
    for j, (n, o) in enumerate(tests):
        padj = ((minp <= p_raw[j]).sum() + 1) / (R + 1)
        mark = ' *' if padj < 0.05 else ''
        log(f'{n:<12} {len(o):>4} {obs[j]:>4} ({obs[j] / len(o):>4.0%}) '
            f'{p_raw[j]:>8.4f} {padj:>8.4f}{mark}')
    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
