# -*- coding: utf-8 -*-
"""§18 (серия 2, цикл 4): errata-реестр LL v1 — сверка расхождений
свидетелей по аппарату Herbig + алфавитная систематика.

Три источника вердиктов:
1. АППАРАТ: кандидат из results/herbig_witness_flags.csv (kind=disagree)
   подтверждается, если herbig-слово (в θ-орфографии) находится в
   Adnotationes/Textus PDF (второй+третий свидетели: сам Herbig и
   цитируемые им Krall/Torp).
2. АЛФАВИТ: в этрусской транслитерации НЕТ букв g и o — токены LL/TCap
   с g/o суть OCR-обломки θ (F&W печатает θ, djvu-OCR даёт 6/9/0/8/G/o;
   парсер уже чинит цифры, буквы G/o оставались). Развёртка g→th, o→th
   даёт кандидата.
3. СЛОВАРЬ: поправка валидируется, если исправленная форма аттестована
   в корпусе (вне данной записи) или в Herbig-индексе, а исходная — нет.

Выход: data/supplements/errata_ll_v1.csv (key, token, corrected,
evidence, status: confirmed/candidate) — реестр; корпус НЕ правится
(интеграция erratum-флагами — отдельная заморозка v0.9).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_errata_ll.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
PDF = os.path.join('data', 'external', 'cie_online', 'Supplementum_I.pdf')
OUT_CSV = os.path.join('data', 'supplements', 'errata_ll_v1.csv')
OUT_LOG = os.path.join('logs', 'etr_errata_ll.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def to_theta(w):
    """ascii-форма → θ-орфография Herbig для поиска в PDF-тексте."""
    return w.replace('th', 'ϑ').replace('ch', 'χ').replace('ph', 'φ')


def main():
    os.makedirs('logs', exist_ok=True)
    log('=== §18: errata LL v1 (аппарат Herbig + алфавитная систематика) ===')
    import fitz
    doc = fitz.open(PDF)
    herbig_txt = '\n'.join(doc[p].get_text() for p in range(20, 42)).lower()
    herbig_flat = re.sub(r'[^a-zϑχφśçθα-ω]', '', herbig_txt)

    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.8'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    vocab = Counter(t['ascii'] for r in view for t in r['toks']
                    if t['kind'] == 'W')
    hb_words = set()
    with open(os.path.join('data', 'external', 'cie_online',
                           'herbig_ll_index.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            hb_words.add(r['word_norm'])

    rows = []

    # --- 1. кандидаты из аудита свидетелей -----------------------------------
    n_conf = 0
    with open(os.path.join('results', 'herbig_witness_flags.csv'),
              encoding='utf-8') as f:
        dis = [r for r in csv.DictReader(f) if r['kind'] == 'disagree']
    log(f'кандидатов расхождений из аудита: {len(dis)}')
    for r in dis:
        fw, hb = r['fw_word'], r['herbig_word']
        th = to_theta(hb)
        in_app = th in herbig_flat or hb in herbig_flat
        v_fw, v_hb = vocab.get(fw, 0), vocab.get(hb, 0)
        # присутствие hb в PDF Herbig ЦИРКУЛЯРНО (индекс оттуда же);
        # confirmed требует: исходное почти неаттестовано, поправка
        # частотна в корпусе, аппарат совпал. Частотное исходное — оба
        # свидетеля видят РАЗНЫЕ реальные слова: не erratum.
        if v_fw >= 3:
            status = 'both-attested'
        elif v_fw <= 1 and v_hb >= 3 and in_app:
            status = 'confirmed'
        else:
            status = 'candidate'
        n_conf += status == 'confirmed'
        rows.append((r['fw_key'], f"{r['herbig_col']} {r['herbig_line']}",
                     fw, hb,
                     ('apparatus' if in_app else 'index-only')
                     + (f'+vocab{v_hb}' if v_hb >= 3 else ''),
                     status))

    # --- 2. алфавитная систематика: g/o в CIEW-токенах -----------------------
    log('\n--- алфавитная развёртка: g/o (букв нет в этрусском) ---')
    n_alpha = 0
    for r in view:
        if r['src'] != 'CIEW' or r['eid'] not in ('9001', '7002'):
            continue
        key = r.get('key') or r['rid']
        for t in r['toks']:
            if t['kind'] != 'W':
                continue
            w = t['ascii']
            if not re.search(r'[go]', w) or '-' in w:
                continue
            # кандидаты замены: каждое g/o -> th
            cand = re.sub(r'[go]', 'th', w)
            better = ((vocab.get(cand, 0) >= 2 or cand in hb_words)
                      and vocab.get(w, 0) <= 1)
            ev = []
            if to_theta(cand) in herbig_flat:
                ev.append('apparatus')
            if vocab.get(cand, 0) >= 2:
                ev.append('vocab')
            if cand in hb_words:
                ev.append('herbig-index')
            if better and ev:
                n_alpha += 1
                rows.append((r['rid'], key, w, cand,
                             'alphabet+' + '+'.join(ev), 'confirmed'
                             if len(ev) >= 2 else 'candidate'))
    log(f'алфавитных кандидатов с поддержкой: {n_alpha}')

    # дедуп по (token, corrected)
    seen = set()
    uniq = []
    for row in rows:
        k = (row[2], row[3])
        if k not in seen:
            seen.add(k)
            uniq.append(row)
    st_cnt = Counter(r[5] for r in uniq)
    log(f'\nреестр: {len(uniq)} уникальных строк; статусы: '
        + ', '.join(f'{k}:{v}' for k, v in sorted(st_cnt.items())))
    log('confirmed (все):')
    for r in [x for x in uniq if x[5] == 'confirmed']:
        log(f'  {r[2]:<14} → {r[3]:<14} [{r[4]}] {r[0]}')
    log('both-attested (свидетели видят разные реальные слова, не erratum):')
    for r in [x for x in uniq if x[5] == 'both-attested'][:8]:
        log(f'  {r[2]:<14} ↔ {r[3]:<14} {r[0]}')

    # --- 3. валидация реестра -------------------------------------------------
    n_att_corr = sum(1 for r in uniq if vocab.get(r[3], 0) >= 2
                     or r[3] in hb_words)
    n_att_orig = sum(1 for r in uniq if vocab.get(r[2], 0) >= 2)
    log(f'\nвалидация: исправленная форма аттестована (корпус>=2 или '
        f'Herbig-индекс) у {n_att_corr}/{len(uniq)}; исходная — у '
        f'{n_att_orig}/{len(uniq)} (поправки уводят в ИЗВЕСТНЫЙ словарь)')

    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['rid_or_key', 'locus', 'token', 'corrected',
                     'evidence', 'status'])
        for r in sorted(uniq):
            wr.writerow(r)
    log(f'реестр записан: {OUT_CSV}')
    log('дисциплина: корпус не правится; интеграция — erratum-флагами '
        'при заморозке v0.9 (только confirmed).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
