# -*- coding: utf-8 -*-
"""§19 (серия 2, цикл 5): обобщение позиционной грамматики секций LL —
leave-one-section-out (semi-Markov-lite, вторая часть плана Sol №3).

Вопрос: устойчив ли ВНУТРИсекционный позиционный профиль класса между
секциями? Для класса X и отложенной секции s: строки s ранжируются
плотностью относительных позиций X в ОСТАЛЬНЫХ секциях (гауссово ядро,
bw=0.15); качество — pooled AUC против фактического присутствия X в s.
Нуль — совместный циклический сдвиг классового вектора по строкам
(R=2000, один сдвиг на replicate для всех классов; min-p по семейству).
Карта — results/ll_scroll_map_v2.csv (140 строк, секции по CAL §10/§16).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ll_sections2.py
"""
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 2000
BW = 0.15
CLASSES = ['VACL', 'THEO', 'OFFER', 'VERB-R', 'CONJ']
OUT_LOG = os.path.join('logs', 'etr_ll_sections2.log')
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
                         set(r['classes'].split()) - {'—'}))
    rows.sort()
    cls_sets = [c for _, c in rows]
    n = len(rows)
    cal_idx = [i for i, c in enumerate(cls_sets) if 'CAL' in c]
    runs = {i for j, i in enumerate(cal_idx)
            if j == 0 or cal_idx[j - 1] != i - 1}
    sec_id = np.zeros(n, int)
    cur = 0
    for i in range(n):
        if i in runs and i > 0:
            cur += 1
        sec_id[i] = cur
    n_sec = cur + 1
    sizes = Counter(sec_id.tolist())
    relpos = np.zeros(n)
    for s in range(n_sec):
        idx = np.where(sec_id == s)[0]
        relpos[idx] = (np.linspace(0, 1, len(idx))
                       if len(idx) >= 2 else 0.5)
    ok_lines = np.array([sizes[sec_id[i]] >= 4 for i in range(n)])
    log('=== §19: LOSO-обобщение позиций классов в секциях LL ===')
    log(f'строк: {n}; секций: {n_sec}; строк в секциях >=4: '
        f'{int(ok_lines.sum())}')

    def pooled_auc(cs):
        aucs = []
        for c in CLASSES:
            has = np.array([c in s for s in cs])
            scores = np.zeros(n)
            for s in range(n_sec):
                te = (sec_id == s) & ok_lines
                tr = (sec_id != s) & ok_lines & has
                if not te.any() or tr.sum() < 3:
                    continue
                mus = relpos[tr][:, None]
                scores[te] = np.exp(
                    -0.5 * ((relpos[te][None, :] - mus) / BW) ** 2
                ).mean(axis=0)
            pos = scores[ok_lines & has]
            neg = scores[ok_lines & ~has]
            if len(pos) >= 5 and len(neg) >= 5:
                gt = (pos[:, None] > neg[None, :]).mean()
                eq = (pos[:, None] == neg[None, :]).mean()
                aucs.append(gt + 0.5 * eq)
            else:
                aucs.append(np.nan)
        return np.array(aucs)

    obs = pooled_auc(cls_sets)
    rng = np.random.default_rng(SEED)
    ge = np.zeros(len(CLASSES))
    any_hit = 0
    valid = ~np.isnan(obs)
    for _ in range(R):
        sh = int(rng.integers(1, n))
        cs = cls_sets[sh:] + cls_sets[:sh]
        sim = pooled_auc(cs)
        hit = np.where(np.isnan(sim) | ~valid, False, sim >= obs)
        ge += hit
        any_hit += hit.any()
    log(f'\n{"класс":<8} {"AUC":>6} {"raw p":>8}')
    for c, o, g in zip(CLASSES, obs, ge):
        if np.isnan(o):
            log(f'{c:<8} — мало данных')
        else:
            log(f'{c:<8} {o:>6.3f} {(g + 1) / (R + 1):>8.4f}')
    log(f'семейный min-p (общий сдвиг): p̃<={(any_hit + 1) / (R + 1):.4f}')
    log('\nчтение: AUC>0.5 + малый p значили бы, что класс занимает '
        'УСТОЙЧИВОЕ место внутри секции (позиционная грамматика ритуала); '
        'негатив публикуем как границу применимости датного членения.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
