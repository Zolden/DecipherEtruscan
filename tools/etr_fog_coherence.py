# -*- coding: utf-8 -*-
"""§7.1: когерентность смыслового тумана внутри надписей + связка с жанрами.

Вопрос владельца проекта: соседствуют ли туманные значения осмысленно?
Тест 1 (КОГЕРЕНТНОСТЬ): в мультисловных записях пары туман-слов чаще ли
делят семантический ДОМЕН (trade/vessel/animal/…), чем при случайной
раздаче доменов? Нуль — перестановка доменов между туман-словами
(R=1000, seed=42).
Тест 2 (СВЯЗКА СО СТРУКТУРОЙ): домен vessel/trade у туман-слов чаще ли
встречается в записях «говорящих предметов» (mi/mini — жанр §5.7)?
Гипотеза мотивирована известными грецизмами-сосудами (qutum, aska…).
Нуль — перестановка жанров записей (R=1000).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_fog_coherence.py
"""
import csv
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
OUT_LOG = os.path.join('logs', 'etr_fog_coherence.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    fog = {}
    for r in csv.DictReader(open(os.path.join('results',
                                              'concept_fog_v1.csv'),
                                 encoding='utf-8')):
        if not r['known_gloss']:  # только непереведённые — честный туман
            fog[r['word']] = r['domain']
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== §7.1: когерентность тумана ===')
    log(f'туман-слов (непереведённых): {len(fog)}; R={R}, seed={SEED}')

    # --- 1. когерентность доменов внутри записей ---------------------------
    rec_words = []
    for r in view:
        ws = sorted({t['ascii'] for t in r['toks'] if t['kind'] == 'W'
                     and t['ascii'] in fog})
        if len(ws) >= 2:
            rec_words.append(ws)
    words_all = sorted({w for ws in rec_words for w in ws})
    log(f'записей с ≥2 туман-словами: {len(rec_words)}; '
        f'вовлечено слов: {len(words_all)}')

    def same_dom_pairs(dom_of):
        s = 0
        for ws in rec_words:
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    s += dom_of[ws[i]] == dom_of[ws[j]]
        return s

    obs = same_dom_pairs(fog)
    rng = np.random.default_rng(SEED)
    doms = [fog[w] for w in words_all]
    sims = np.zeros(R)
    for r_i in range(R):
        perm = rng.permutation(len(words_all))
        dmap = {w: doms[perm[i]] for i, w in enumerate(words_all)}
        sims[r_i] = same_dom_pairs(dmap)
    p1 = float(((sims >= obs).sum() + 1) / (R + 1))
    log(f'однодоменных пар в записях: {obs} (нуль {sims.mean():.1f}'
        f'±{sims.std():.1f}); p={p1:.4f}')

    # --- 2. vessel/trade × «говорящие предметы» -----------------------------
    MI = {'mi', 'mini', 'mine'}
    n11 = n10 = n01 = n00 = 0
    labels = []
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        if len(ws) < 2:
            continue
        speaking = bool(set(ws) & MI)
        has_vt = any(fog.get(w) in ('vessel', 'trade') for w in ws)
        labels.append((speaking, has_vt))
        n11 += speaking and has_vt
        n10 += speaking and not has_vt
        n01 += (not speaking) and has_vt
        n00 += (not speaking) and not has_vt
    log()
    log('--- туман vessel/trade × жанр «говорящий предмет» ---')
    log(f'таблица: mi&vt={n11}, mi&нет={n10}, не-mi&vt={n01}, '
        f'не-mi&нет={n00}')
    obs2 = n11
    spk = np.array([a for a, _ in labels])
    vt = np.array([b for _, b in labels])
    sims2 = np.zeros(R)
    for r_i in range(R):
        sims2[r_i] = int((spk[rng.permutation(len(spk))] & vt).sum())
    p2 = float(((sims2 >= obs2).sum() + 1) / (R + 1))
    rate_mi = n11 / max(n11 + n10, 1)
    rate_no = n01 / max(n01 + n00, 1)
    log(f'vt-доля в mi-записях: {rate_mi:.1%} против {rate_no:.1%} вне; '
        f'нуль (перестановка жанров): p={p2:.4f}')
    log()
    log('вывод: p1 — есть ли внутренняя согласованность тумана; p2 — '
        'согласуется ли туман со структурным жанром (слабый приор §7 '
        'работает, только если хотя бы один из них мал).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
