# -*- coding: utf-8 -*-
"""§15 (цикл 1): жанровая стратификация инициальности — следствие батареи
§14.2 (плацебо: имена-заголовки инициальны не хуже mi) и ответа LA
(механика «заголовки конкурируют с частицей» подтверждена на LB).

Тесты:
A. ПРЯМАЯ КОНКУРЕНЦИЯ: записи, содержащие И mi/mini, И размеченное имя
   (ETP_POS NAME-M/F, механизм §8.2): кто занимает позицию 1? Нуль —
   совместная перестановка порядка слов записи (R=5000, seed=42; как
   §8.8): P(mi первым | состав записи).
B. СТРАТИФИКАЦИЯ: маркеры жанра, НЕЗАВИСИМЫЕ от целей: эпитафные
   (родство/возраст/смерть) и посвятительные (глаголы дарения). Доля
   инициальности mi и имён ВНУТРИ страт; плацебо §14.2 переигрывается в
   нужной страте.
C. TTR ПЕРВОГО СЛОТА по жанрам — тот же статистик, что «пятая точка»
   LA (TTR = типов/вхождений первой позиции; открытый слот — высокий
   TTR): наш вклад в межтрадиционную таблицу «открытость позиции =
   свойство жанра». Кластерный bootstrap по памятникам (R=1000) для CI.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_genre_initial.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R_PERM = 5000
R_BOOT = 1000
OUT_LOG = os.path.join('logs', 'etr_genre_initial.log')
LOG = []

MI = {'mi', 'mini', 'mine'}
EPIT_MARK = {'clan', 'sec', 'sech', 'puia', 'ruva', 'ati', 'apa', 'nefts',
             'lupu', 'lupuce', 'avils', 'avil', 'ril', 'svalce', 'tenu'}
DED_MARK = {'turce', 'turuce', 'muluvanice', 'mulveni', 'mulu', 'alice',
            'aliqu', 'acasce'}


def log(m=''):
    print(m)
    LOG.append(m)


def to_ascii_word(w):
    w = re.sub(r"[^a-zθχφσςśšê']", '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def etp_names():
    votes = defaultdict(Counter)
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = to_ascii_word(row.get('Etruscan'))
            if len(w) < 3:
                continue
            if (row.get('theo') or '').strip() == '1':
                votes[w]['THEO'] += 1
            elif (row.get('masc') or '').strip() == '1':
                votes[w]['NAME'] += 1
            elif (row.get('fem') or '').strip() == '1':
                votes[w]['NAME'] += 1
            elif (row.get('TAG') or '').strip() == 'VERB':
                votes[w]['VERB'] += 1
    out = {}
    for w, cnt in votes.items():
        top = cnt.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            out[w] = top[0][0]
    return out


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    lab = etp_names()
    names = {w for w, c in lab.items() if c == 'NAME'}
    recs = []
    for r in view:
        ws = [t['ascii'] for t in r['toks']
              if t['kind'] == 'W' and '-' not in t['ascii']
              and len(t['ascii']) >= 2]
        if len(ws) >= 2:
            recs.append((r['artifact_id'], ws))
    log('=== §15: жанровая стратификация инициальности ===')
    log(f'мультисловных записей: {len(recs)}; размеченных имён: {len(names)}')

    # --- A. прямая конкуренция mi ↔ имя -------------------------------------
    comp = [(a, ws) for a, ws in recs
            if set(ws) & MI and set(ws) & names]
    n_mi1 = sum(ws[0] in MI for _, ws in comp)
    n_nm1 = sum(ws[0] in names for _, ws in comp)
    log(f'\n--- A. конкуренция: записи с mi/mini И именем: {len(comp)} ---')
    log(f'mi первым: {n_mi1} ({n_mi1 / max(len(comp), 1):.1%}); '
        f'имя первым: {n_nm1} ({n_nm1 / max(len(comp), 1):.1%})')
    rng = np.random.default_rng(SEED)
    ge = 0
    for _ in range(R_PERM):
        s = 0
        for _, ws in comp:
            perm = rng.permutation(len(ws))
            s += ws[int(perm[0])] in MI
        ge += s >= n_mi1
    p_a = (ge + 1) / (R_PERM + 1)
    exp = np.mean([sum(1 for w in ws if w in MI) / len(ws)
                   for _, ws in comp]) if comp else float('nan')
    log(f'нуль (перестановка внутри записи): ожидание mi-первым '
        f'{exp:.1%}; p={p_a:.4f}')

    # --- B. стратификация по независимым маркерам ---------------------------
    log('\n--- B. инициальность внутри жанровых страт ---')

    def stratum(ws):
        sw = set(ws)
        ep = bool(sw & EPIT_MARK)
        de = bool(sw & DED_MARK)
        if ep and not de:
            return 'эпитафный'
        if de and not ep:
            return 'посвятительный'
        if ep and de:
            return 'смешанный'
        return 'прочий'

    strat_recs = defaultdict(list)
    for a, ws in recs:
        strat_recs[stratum(ws)].append((a, ws))
    log(f'{"страта":<14} {"n":>5} {"mi нач/встр":>12} {"имя нач/встр":>13}')
    for st in ('эпитафный', 'посвятительный', 'смешанный', 'прочий'):
        rr = strat_recs[st]
        mi_in = [ws for _, ws in rr if set(ws) & MI]
        nm_in = [ws for _, ws in rr if set(ws) & names]
        mi_i = sum(ws[0] in MI for ws in mi_in)
        nm_i = sum(ws[0] in names for ws in nm_in)
        log(f'{st:<14} {len(rr):>5} '
            f'{mi_i:>4}/{len(mi_in):<7} {nm_i:>5}/{len(nm_in):<7}')
    ded = strat_recs['посвятительный'] + strat_recs['прочий']
    mi_ded = [ws for _, ws in ded if set(ws) & MI]
    nm_ded = [ws for _, ws in ded if set(ws) & names]
    s_mi = sum(ws[0] in MI for ws in mi_ded) / max(len(mi_ded), 1)
    s_nm = sum(ws[0] in names for ws in nm_ded) / max(len(nm_ded), 1)
    log(f'вне эпитафной страты: mi инициальна {s_mi:.1%} '
        f'({len(mi_ded)} записей), имена инициальны {s_nm:.1%} '
        f'({len(nm_ded)}) — плацебо §14.2 в правильной страте')

    # --- C. TTR первого слота по жанрам (вклад в таблицу LA) ----------------
    log('\n--- C. TTR первого слота (открытость позиции; статистик LA) ---')
    rng2 = np.random.default_rng(SEED + 1)
    for st in ('эпитафный', 'посвятительный', 'смешанный', 'прочий'):
        rr = strat_recs[st]
        if len(rr) < 30:
            log(f'  {st:<14} n={len(rr)} — мало для TTR')
            continue
        firsts = [ws[0] for _, ws in rr]
        ttr = len(set(firsts)) / len(firsts)
        arts = sorted({a for a, _ in rr})
        by_art = defaultdict(list)
        for a, ws in rr:
            by_art[a].append(ws[0])
        boots = np.zeros(R_BOOT)
        for i in range(R_BOOT):
            samp = [by_art[arts[j]] for j in
                    rng2.integers(0, len(arts), size=len(arts))]
            fl = [w for grp in samp for w in grp]
            boots[i] = len(set(fl)) / max(len(fl), 1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        log(f'  {st:<14} n={len(firsts):>4} TTR={ttr:.2f} '
            f'[CI по памятникам {lo:.2f}–{hi:.2f}]')
    log('\nчтение: A — специфичность дейксиса в лоб (конкуренция в одной '
        'записи); B — уточнение формулировки §14.2 по стратам; C — строка '
        'Этрурии для межтрадиционной таблицы LA «открытость позиции = '
        'свойство жанра».')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
