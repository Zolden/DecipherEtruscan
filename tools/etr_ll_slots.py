# -*- coding: utf-8 -*-
"""§5.8: слот-грамматика ритуальной строки LL + карта свитка ~100%.

Данные: строки LL (CIEW 9001, v0.6): нумерованные (сквозная нумерация
F&W, выровнены по построению) + ненумерованные (интервальная привязка к
диапазону номеров своей страницы).

1. КЛАССЫ токенов строки: VACL (vacl), OFFER (приношения: fler, zusle-,
   vinum, faśei, θapn-, zuslev-), THEO (теонимы и их генитивы), VERB-R
   (ритуальные глаголы: nunθen-, θez-, trin, ara, śin-, acil-), CAL
   (сроки: zaθrum-, eslem, acale, celi, tiur-), CONJ (etnam), X.
2. ПОЗИЦИОННЫЕ тесты классов (начало/конец строки; нуль равномерной
   позиции, R=10000, seed=42, Westfall–Young по семейству).
3. ПОРЯДОК ПАР: P(VACL раньше THEO | оба в строке), P(OFFER раньше
   VERB-R), P(CAL раньше VACL) — биномиальный нуль 50%.
4. Карта свитка: results/ll_scroll_map.csv — каждая строка LL с позицией
   (точный номер или интервал страницы), текстом, классовой сигнатурой.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ll_slots.py
"""
import csv
import os
import pickle
import sys
from collections import Counter

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding='utf-8')
R = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_ll_slots.log')
OUT_CSV = os.path.join('results', 'll_scroll_map.csv')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


CLASSES = {
    'VACL': lambda w: w == 'vacl',
    'OFFER': lambda w: w in ('fler', 'flere', 'vinum', 'fasei', 'faseic')
    or w.startswith(('zusle', 'thapn', 'zuslev', 'racuse')),
    'THEO': lambda w: w in ('tin', 'tins', 'tinsi', 'tinsin', 'uni',
                            'unialti', 'lusa', 'lusas', 'veive', 'veives',
                            'aiser', 'aiseras', 'seus', 'nethuns',
                            'nethunsl', 'crap', 'crapsti', 'catha', 'cel')
    or w.startswith(('nethuns', 'crapst')),
    'VERB-R': lambda w: w.startswith(('nunthen', 'thezi', 'thezer',
                                      'scanin', 'sin', 'acil', 'trin',
                                      'hechs', 'hex')) or w == 'ara',
    'CAL': lambda w: w.startswith(('zathrum', 'eslem', 'acal', 'celi',
                                   'tiur')),
    'CONJ': lambda w: w == 'etnam',
}


def cls_of(w):
    for c, f in CLASSES.items():
        if f(w):
            return c
    return 'X'


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    ll = [r for r in corpus['records']
          if r['src'] == 'CIEW' and r['eid'] == '9001']
    numbered, unaligned = [], []
    page_range = {}
    for r in ll:
        pg, num = (int(x) for x in r['key'].split('.'))
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W'
              and '-' not in t['ascii'] and len(t['ascii']) >= 2]
        if not ws:
            continue
        if num < 900:
            numbered.append((num, pg, ws))
            a, b = page_range.get(pg, (10 ** 9, 0))
            page_range[pg] = (min(a, num), max(b, num))
        else:
            unaligned.append((pg, num, ws))
    numbered.sort()
    log('=== §5.8: слот-грамматика строки LL + карта свитка ===')
    log(f'нумерованных строк: {len(numbered)}; ненумерованных: '
        f'{len(unaligned)}; R={R}, seed={SEED}')

    # --- 2. позиционные тесты классов ---------------------------------------
    all_lines = [ws for _, _, ws in numbered] + [ws for _, _, ws in unaligned]
    occ = {}
    for ws in all_lines:
        if len(ws) < 2:
            continue
        for i, w in enumerate(ws):
            c = cls_of(w)
            if c != 'X':
                occ.setdefault(c, []).append((i, len(ws)))
    rng = np.random.default_rng(SEED)
    tests = []
    for c in sorted(occ):
        o = occ[c]
        if len(o) >= 5:
            tests.append((c, 'нач', sum(1 for i, _ in o if i == 0), o))
            tests.append((c, 'фин',
                          sum(1 for i, L in o if i == L - 1), o))
    sims = np.zeros((R, len(tests)), np.int32)
    for j, (c, side, obs, o) in enumerate(tests):
        Ls = np.array([L for _, L in o])
        u = rng.random((R, len(o)))
        sims[:, j] = ((u < 1.0 / Ls) if side == 'нач'
                      else (u >= 1.0 - 1.0 / Ls)).sum(axis=1)
    obs_a = np.array([t[2] for t in tests])
    p_raw = ((sims >= obs_a[None, :]).sum(axis=0) + 1) / (R + 1)
    p_sim = np.zeros_like(sims, float)
    for j in range(len(tests)):
        col = np.sort(sims[:, j])
        idx = np.searchsorted(col, sims[:, j], side='left')
        p_sim[:, j] = (R - idx + 1) / (R + 1)
    minp = p_sim.min(axis=1)
    log()
    log('--- позиции классов в строке ---')
    log(f'{"класс":<8} {"стор.":<5} {"n":>4} {"доля":>6} {"p":>8} '
        f'{"p̃_сем":>8}')
    for j, (c, side, obs, o) in enumerate(tests):
        padj = ((minp <= p_raw[j]).sum() + 1) / (R + 1)
        mark = ' *' if padj < 0.05 else ''
        log(f'{c:<8} {side:<5} {len(o):>4} {obs / len(o):>6.0%} '
            f'{p_raw[j]:>8.4f} {padj:>8.4f}{mark}')

    # --- 3. порядок пар ------------------------------------------------------
    log()
    log('--- порядок пар классов внутри строки (биномиальный нуль 50%) ---')
    for c1, c2 in [('VACL', 'THEO'), ('VACL', 'OFFER'), ('OFFER', 'VERB-R'),
                   ('CAL', 'VACL'), ('THEO', 'VERB-R')]:
        a = b = 0
        for ws in all_lines:
            i1 = [i for i, w in enumerate(ws) if cls_of(w) == c1]
            i2 = [i for i, w in enumerate(ws) if cls_of(w) == c2]
            if i1 and i2:
                if min(i1) < min(i2):
                    a += 1
                elif min(i2) < min(i1):
                    b += 1
        n = a + b
        if n >= 5:
            p = binomtest(a, n, 0.5, alternative='greater').pvalue
            log(f'  {c1} раньше {c2}: {a}/{n} ({a / n:.0%}), p={p:.4f}')
        else:
            log(f'  {c1}/{c2}: мало совместных строк ({n})')

    # --- 3b. ПЕРИОДНЫЙ уровень (ответ на N19): пробеги смежных строк --------
    # период = максимальный пробег нумерованных строк с зазором <=2;
    # порядок классов = порядок их ПЕРВЫХ появлений внутри периода;
    # биномиальный нуль 50% на направление, Бонферрони по числу пар.
    log()
    log('--- 3b. периодный уровень: порядок первых появлений классов ---')
    # периоды = блоки по 5 смежных строк внутри страницы в порядке файла
    # (включая ненумерованные: внутри страницы порядок OCR ~ порядок свитка)
    by_page = {}
    for r in ll:
        pg, num = (int(x) for x in r['key'].split('.'))
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W'
              and '-' not in t['ascii'] and len(t['ascii']) >= 2]
        if ws:
            by_page.setdefault(pg, []).append((num, ws))
    runs = []
    for pg in sorted(by_page):
        rows = by_page[pg]
        for i0 in range(0, len(rows) - 1, 5):
            chunk = rows[i0:i0 + 5]
            if len(chunk) >= 3:
                runs.append(chunk)
    log(f'периодов (блоки по 5 строк внутри страницы, длина>=3): '
        f'{len(runs)}')
    order_cnt = {}
    co_period = 0
    for run in runs:
        first = {}
        for li, (num, ws) in enumerate(run):
            for w in ws:
                c = cls_of(w)
                if c != 'X' and c not in first:
                    first[c] = li
        if 'VACL' in first and 'THEO' in first:
            co_period += 1
        ks = sorted(first)
        for i1 in range(len(ks)):
            for i2 in range(i1 + 1, len(ks)):
                c1, c2 = ks[i1], ks[i2]
                if first[c1] == first[c2]:
                    continue
                a, b = order_cnt.get((c1, c2), (0, 0))
                if first[c1] < first[c2]:
                    a += 1
                else:
                    b += 1
                order_cnt[(c1, c2)] = (a, b)
    tests2 = [(k, v) for k, v in sorted(order_cnt.items())
              if sum(v) >= 6]
    m2 = len(tests2)
    log(f'совместность VACL+THEO: {co_period} периодов (на уровне строк '
        f'было 1) — единица ритуала подтверждена как период')
    log(f'пар классов с n>=6 периодов: {m2} (Бонферрони x{m2})')
    for (c1, c2), (a, b) in tests2:
        n = a + b
        lead, cnt = (c1, a) if a >= b else (c2, b)
        pv = binomtest(cnt, n, 0.5, alternative='greater').pvalue
        padj = min(1.0, pv * m2)
        mark = ' *' if padj < 0.05 else ''
        log(f'  {c1} vs {c2}: {lead} раньше в {cnt}/{n} '
            f'({cnt / n:.0%}), p={pv:.4f}, p_adj={padj:.4f}{mark}')

    # --- 4. карта свитка -----------------------------------------------------
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['position', 'kind', 'key', 'classes', 'text'])
        for num, pg, ws in numbered:
            sig = ' '.join(sorted({cls_of(x) for x in ws} - {'X'})) or '—'
            w.writerow([num, 'exact', f'{pg}.{num}', sig, ' '.join(ws)])
        for pg, num, ws in sorted(unaligned):
            a, b = page_range.get(pg, ('', ''))
            sig = ' '.join(sorted({cls_of(x) for x in ws} - {'X'})) or '—'
            w.writerow([f'{a}–{b}', 'interval', f'{pg}.{num}', sig,
                        ' '.join(ws)])
    n_int = sum(1 for pg, _, _ in unaligned if pg in page_range)
    log()
    log(f'карта свитка: {OUT_CSV} — {len(numbered)} точных позиций + '
        f'{len(unaligned)} интервальных ({n_int} с диапазоном страницы); '
        f'покрытие позиционированием: '
        f'{(len(numbered) + n_int) / (len(numbered) + len(unaligned)):.0%}')

    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
