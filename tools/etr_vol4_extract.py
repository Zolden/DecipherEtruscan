# -*- coding: utf-8 -*-
"""§26 (серия 3, цикл 7): извлечение фактов CIE Vol. IV.1.1 (Atria).

Источник: data/external/cie_online/CIE-IV.1.1_Atria_20001-20422.pdf
(born-digital, полный текстовый слой; Gaucci 2017 — В КОПИРАЙТЕ:
извлекаются только ФАКТЫ — номера титулов, конкордансы Rix/Meiser ET,
датировочные формулы, наличие надписи; критический текст НЕ копируется).

Выход: data/external/cie_online/vol4_1_tituli.csv — по титулу: номер,
Rix-конкорданс(ы), датировка (латинская формула '…saec. a.Ch.n.'),
inscriptio/littera-признак, уже-в-корпусе (CIEP 20xxx). Слой для
будущего supplement-решения; в корпус НЕ вливается.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_vol4_extract.py
"""
import csv
import os
import pickle
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
PDF = os.path.join('data', 'external', 'cie_online',
                   'CIE-IV.1.1_Atria_20001-20422.pdf')
OUT_CSV = os.path.join('data', 'external', 'cie_online',
                       'vol4_1_tituli.csv')
OUT_LOG = os.path.join('logs', 'etr_vol4_extract.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    import fitz
    doc = fitz.open(PDF)
    full = '\n'.join(doc[i].get_text() for i in range(len(doc)))
    log('=== §26: факты CIE Vol. IV.1.1 (Atria, tit. 20001-20422) ===')

    # сегментация по титулам: строка, начинающаяся '200xx '
    parts = re.split(r'\n(20[0-4]\d\d)\s', full)
    tituli = {}
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        if 20001 <= num <= 20422 and num not in tituli:
            tituli[num] = parts[i + 1][:2000]
    log(f'сегментировано титулов: {len(tituli)} из 422')

    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    in_corpus = {int(r['eid']) for r in corpus['records']
                 if r['src'] == 'CIEP' and r['eid'].isdigit()
                 and 20001 <= int(r['eid']) <= 20422}

    rows = []
    n_rix = n_date = n_inscr = 0
    for num in sorted(tituli):
        t = tituli[num]
        rix = sorted({m.replace(' ', '') for m in re.findall(
            r'ET2?,?\s*(Ad\s*[\d.]+\d)', t)})
        m_date = re.search(
            r'((?:[IVX]+\s*[-–]\s*)?(?:ineunti|exeunti|medio)?\s*[IVX]+'
            r'\s*saec\.\s*a\.Ch\.n\.)', t)
        date = re.sub(r'\s+', ' ', m_date.group(1)).strip() if m_date else ''
        has_inscr = bool(re.search(r'\bInscriptio\b|\bLitterae?\b', t))
        n_rix += bool(rix)
        n_date += bool(date)
        n_inscr += has_inscr
        rows.append([num, ';'.join(rix), date, int(has_inscr),
                     int(num in in_corpus)])
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['titulus', 'rix_et', 'dating_latin', 'has_inscriptio',
                     'in_corpus_ciep'])
        wr.writerows(rows)
    log(f'с Rix/Meiser ET-конкордансом: {n_rix}; с датировочной формулой: '
        f'{n_date}; с Inscriptio/Littera: {n_inscr}; уже в корпусе (CIEP): '
        f'{sum(r[4] for r in rows)}')
    log(f'csv записан: {OUT_CSV}')
    log('дисциплина: только факты (номера, конкордансы, датировки, '
        'признаки) — критический текст Gaucci 2017 не копируется; '
        'чтения надписей в томе преимущественно факсимиле (§9.3).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
