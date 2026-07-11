# -*- coding: utf-8 -*-
"""§11.2: аудит свидетелей LL — расхождения чтений F&W (корпус) ↔ Herbig.

По выровненным строкам (§9.1, results/herbig_fw_alignment.csv, kind=exact)
сравниваем словоформы F&W-слоя (results/ll_scroll_map.csv) с чтениями
Herbig (data/external/cie_online/herbig_ll_index.csv). Классы:
  exact       — точное совпадение нормализованных форм;
  soft        — совпадение по префиксу>=4 / скелету согласных (обрезки);
  disagree    — пара остатков с расстоянием Левенштейна <=2 — КАНДИДАТ
                расхождения свидетелей (OCR одного из них; sample50 #9
                suo↔suth именно такого рода);
  fw_only / herbig_only — слово видит только один свидетель (порча или
                неполнота индекса; НЕ ошибка по построению).
Ничего в корпусе не правим: выход — реестр флагов для ручной сверки и
для будущих эпистемических флагов заморозки.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_herbig_audit.py
"""
import csv
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
OUT_LOG = os.path.join('logs', 'etr_herbig_audit.log')
OUT_CSV = os.path.join('results', 'herbig_witness_flags.csv')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


SKEL = str.maketrans('', '', 'aeiou')


def soft_eq(a, b):
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    sa, sb = a.translate(SKEL), b.translate(SKEL)
    return len(sa) >= 3 and sa == sb


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log('=== §11.2: аудит свидетелей F&W ↔ Herbig (LL) ===')
    fw_words = {}
    with open(os.path.join('results', 'll_scroll_map.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'exact':
                fw_words[int(r['position'])] = [w for w in r['text'].split()
                                                if len(w) >= 2]
    hb_words = {}
    with open(os.path.join('data', 'external', 'cie_online',
                           'herbig_ll_index.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'gamma' not in r['flags'] and r['col'] != 'FN':
                hb_words.setdefault((r['col'], int(r['line'])),
                                    set()).add(r['word_norm'])
    pairs = []
    with open(os.path.join('results', 'herbig_fw_alignment.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'exact':
                pairs.append((int(r['fw_position']), r['fw_key'],
                              r['herbig_col'], int(r['herbig_line'])))
    log(f'выровненных пар строк: {len(pairs)}')

    rows = []
    stat = Counter()
    for pos, key, col, line in pairs:
        fw = list(fw_words.get(pos, []))
        hb = set(hb_words.get((col, line), set()))
        # 1) точные
        for w in list(fw):
            if w in hb:
                rows.append((pos, key, col, line, w, w, 'exact'))
                stat['exact'] += 1
                fw.remove(w)
                hb.discard(w)
        # 2) мягкие
        for w in list(fw):
            hit = next((v for v in sorted(hb) if soft_eq(w, v)), None)
            if hit:
                rows.append((pos, key, col, line, w, hit, 'soft'))
                stat['soft'] += 1
                fw.remove(w)
                hb.discard(hit)
        # 3) кандидаты расхождений (lev<=2 на остатках)
        for w in list(fw):
            best, bd = None, 3
            for v in sorted(hb):
                d = lev(w, v)
                if d < bd:
                    best, bd = v, d
            if best is not None and bd <= 2:
                rows.append((pos, key, col, line, w, best, 'disagree'))
                stat['disagree'] += 1
                fw.remove(w)
                hb.discard(best)
        for w in fw:
            rows.append((pos, key, col, line, w, '', 'fw_only'))
            stat['fw_only'] += 1
        for v in sorted(hb):
            rows.append((pos, key, col, line, '', v, 'herbig_only'))
            stat['herbig_only'] += 1

    n_tok = stat['exact'] + stat['soft'] + stat['disagree'] + stat['fw_only']
    log(f'токенов F&W в выровненных строках: {n_tok}')
    for k in ('exact', 'soft', 'disagree', 'fw_only', 'herbig_only'):
        log(f'  {k:<12} {stat[k]:>4}')
    agree = (stat['exact'] + stat['soft']) / max(
        stat['exact'] + stat['soft'] + stat['disagree'], 1)
    log(f'согласие свидетелей на сопоставимых словах: {agree:.1%}')
    log('\nкандидаты расхождений (реестр для ручной сверки):')
    for pos, key, col, line, w, v, kind in rows:
        if kind == 'disagree':
            log(f'  поз.{pos:<4} {key:<8} {col} {line:<3} F&W={w:<14} '
                f'Herbig={v}')

    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['fw_position', 'fw_key', 'herbig_col', 'herbig_line',
                     'fw_word', 'herbig_word', 'kind'])
        for r in rows:
            wr.writerow(r)
    log(f'\nреестр записан: {OUT_CSV}')
    log('чтение: fw_only/herbig_only — неполнота свидетелей (норма); '
        'disagree — очередь ручной сверки; правки корпуса ТОЛЬКО через '
        'supplement/erratum-механизм после сверки с изданием.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
