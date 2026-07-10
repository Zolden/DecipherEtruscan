# -*- coding: utf-8 -*-
"""Жанровая стратификация позиционного синтаксиса (таблица для LA-треда).

Жанры (правила по тексту, §1): ritual = строки LL/Капуи (CIEW 9001/7002);
dedic = есть глагол посвящения/дара (turce/muluvanice/zinace/cver);
speaking = есть mi/mini (и не dedic); epitaph = есть операторы
родства/витальности/гробницы (и не выше); other — прочее.

Для каждого жанра (записи ≥3 словоформ, дейктическое открытие сдвинуто):
TTR инвентарей позиций 1 и 2 + нуль обмена внутри пар (R=10000, seed=42);
доля глаголов посвящения на позиции 2. Вывод — сравнимая таблица (ответ
на §BW LA: «открытость позиции — свойство жанра»).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_genres.py
"""
import os
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
R = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_genres.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


DEIX = {'mi', 'mini', 'mine', 'itun', 'ita', 'ica', 'eca', 'ca', 'cn',
        'cen', 'cehen', 'thui'}
DEDIC = {'turce', 'turuce', 'turice', 'muluvanice', 'muluvanike',
         'mulvanice', 'mulvenice', 'zinace', 'cver'}
EPIT = {'clan', 'clens', 'sec', 'sech', 'puia', 'ati', 'apa', 'lupu',
        'lupuce', 'svalce', 'avils', 'ril', 'suthi'}


def genre_of(r, ws):
    if r['src'] == 'CIEW' and r['eid'] in ('9001', '7002'):
        return 'ritual'
    st = set(ws)
    if st & DEDIC:
        return 'dedic'
    if st & {'mi', 'mini', 'mine'}:
        return 'speaking'
    if st & EPIT:
        return 'epitaph'
    return 'other'


def main():
    os.makedirs('logs', exist_ok=True)
    log('AUDIT OVERRIDE 2026-07-10: ritual raw p=.038 не переживает '
        'Bonferroni по пяти жанрам (p=.190); это тренд, не открытие.')
    log()
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    rng = np.random.default_rng(SEED)
    log('=== Жанровая стратификация позиций (для LA-треда) ===')
    buckets = {}
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        if len(ws) < 3:
            continue
        g = genre_of(r, ws)
        seq = ws[1:] if ws[0] in DEIX else ws
        if len(seq) >= 2:
            buckets.setdefault(g, []).append((seq[0], seq[1]))
    log(f'{"жанр":<10} {"пар":>5} {"TTR1":>6} {"TTR2":>6} {"Δ":>7} '
        f'{"p(Δ>0)":>8} {"посвящ.@2":>10}')
    for g in ['epitaph', 'dedic', 'speaking', 'ritual', 'other']:
        pairs = buckets.get(g, [])
        if len(pairs) < 20:
            log(f'{g:<10} {len(pairs):>5}  — мало данных')
            continue
        p1 = [a for a, _ in pairs]
        p2 = [b for _, b in pairs]
        t1 = len(set(p1)) / len(p1)
        t2 = len(set(p2)) / len(p2)
        arr = np.array(pairs)
        diffs = np.zeros(R)
        for i in range(R):
            flip = rng.random(len(arr)) < 0.5
            a = np.where(flip, arr[:, 1], arr[:, 0])
            b = np.where(flip, arr[:, 0], arr[:, 1])
            diffs[i] = (len(set(a.tolist())) - len(set(b.tolist()))) \
                / len(arr)
        obs = t1 - t2
        p_pos = float(((diffs >= obs).sum() + 1) / (R + 1))
        tr2 = sum(1 for _, b in pairs if b in DEDIC) / len(pairs)
        log(f'{g:<10} {len(pairs):>5} {t1:>6.2f} {t2:>6.2f} {obs:>+7.2f} '
            f'{p_pos:>8.4f} {tr2:>10.1%}')
    log()
    log('чтение: Δ>0 при малом p — позиция 1 ОТКРЫТЕЕ (LA-схема); Δ<0 — '
        'закрытое открытие (ономастическая формула).')
    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
