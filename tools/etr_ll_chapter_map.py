# -*- coding: utf-8 -*-
"""§28 (серия 3, цикл 9): карта глав Liber Linteus — структурное чтение.

ДЕСКРИПТИВНЫЙ синтез подтверждённых слоёв (никаких новых p): по каждой
датной секции (правило §10, карта v2) — диапазон позиций, зачин
(CAL-строка), теонимы (ETP_POS theo=1 ∩ словарь секции + известные
боги LL), ритуальные глаголы (VERB-R-строки), сигнатурная лексика
(§24: все вхождения в одной секции, n>=2), рукописные пометы Herbig
(красные линии §9.1). Выход — results/ll_chapter_map.csv + читаемая
сводка в логе: «что происходит в свитке» на уровне, который данные
реально поддерживают (§24: обряд циклический, секции = датные проходы).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ll_chapter_map.py
"""
import csv
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
OUT_CSV = os.path.join('results', 'll_chapter_map.csv')
OUT_LOG = os.path.join('logs', 'etr_ll_chapter_map.log')
LOG = []

RED_LINES = {111.5: 'красная линия писца (Herbig VI 8/9)',
             230.5: 'красная линия писца (Herbig XI 13/14)'}


def log(m=''):
    print(m)
    LOG.append(m)


def to_ascii_word(w):
    w = re.sub(r"[^a-zθχφσςśšê']", '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    theo = set()
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if (row.get('theo') or '').strip() == '1':
                w = to_ascii_word(row.get('Etruscan'))
                if len(w) >= 3:
                    theo.add(w)
    rows = []
    with open(os.path.join('results', 'll_scroll_map_v2.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['position']),
                         set(r['classes'].split()) - {'—'},
                         [w for w in r['text'].split() if len(w) >= 3]))
    rows.sort()
    n = len(rows)
    cal_idx = [i for i, (_, c, _) in enumerate(rows) if 'CAL' in c]
    runs = {i for j, i in enumerate(cal_idx)
            if j == 0 or cal_idx[j - 1] != i - 1}
    sec = []
    cur = 0
    for i in range(n):
        if i in runs and i > 0:
            cur += 1
        sec.append(cur)
    n_sec = cur + 1
    occ = defaultdict(list)
    for i, (_, _, ws) in enumerate(rows):
        for w in ws:
            occ[w].append(i)
    signature = defaultdict(list)
    for w, idx in occ.items():
        if len(idx) >= 2 and len({sec[i] for i in idx}) == 1:
            signature[sec[idx[0]]].append((w, len(idx)))

    log('=== §28: карта глав Liber Linteus (структурное чтение) ===')
    log('единица — датная секция (§10); LL читается как циклический обряд '
        '(§24), карта фиксирует датные проходы и их наполнение\n')
    out = []
    for s in range(n_sec):
        idx = [i for i in range(n) if sec[i] == s]
        p0, p1 = rows[idx[0]][0], rows[idx[-1]][0]
        opener = ' '.join(rows[idx[0]][2][:7]) if 'CAL' in rows[idx[0]][1] \
            else '(до первой точной даты)'
        theos = sorted({w for i in idx for w in rows[i][2]
                        if w in theo})
        verbs = sorted({w for i in idx for w in rows[i][2]
                        if 'VERB-R' in rows[i][1]
                        and w.endswith(('ri', 'th', 'ne', 'nth'))})[:6]
        sig = ' '.join(f'{w}×{c}' for w, c in
                       sorted(signature.get(s, []), key=lambda x: -x[1])[:5])
        marks = [v for k, v in RED_LINES.items() if p0 <= k <= p1]
        out.append([s, f'{p0}–{p1}', len(idx), opener,
                    ' '.join(theos), ' '.join(w for w in verbs),
                    sig, '; '.join(marks)])
        log(f'секция {s:>2} (поз. {p0}–{p1}, строк {len(idx)})')
        log(f'  зачин: {opener}')
        if theos:
            log(f'  теонимы: {" ".join(theos)}')
        if sig:
            log(f'  сигнатура: {sig}')
        if marks:
            log(f'  рукопись: {"; ".join(marks)}')
        log()
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['section', 'positions', 'n_lines', 'opener_cal',
                     'theonyms', 'ritual_verbs', 'signature_vocab',
                     'manuscript_marks'])
        wr.writerows(out)
    log(f'карта записана: {OUT_CSV}')
    log('чтение: дескриптив по подтверждённым слоям; интерпретации '
        'уровня «глава посвящена богу X» НЕ делаются (§24: секции не '
        'тематические); карта — навигационный инструмент.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
