# -*- coding: utf-8 -*-
"""§7.2: каскад арбитров для тумана (рекомендация LA-ветки, §CP их отчёта).

LA-урок: туман работает только там, где есть материальный арбитр.
Каскад: (1) СТРОГАЯ фонетика — sim=1.0 и конвергентный ярус A (не
контактный B); (2) РЕГИСТР — жанровый профиль вхождений кандидата
против корпусного базлайна; (3) ЧИСЛОВОЕ ОКРУЖЕНИЕ — доля вхождений
рядом с числительными (аналог метрологии LA; у нас NUM = avils-числа и
арабские F&W). Выход — короткий список кандидатов с диагностикой
(разведочный: n мал, p не заявляем, зависимость документов отмечаем).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_fog_cascade.py
"""
import csv
import os
import pickle
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
OUT_LOG = os.path.join('logs', 'etr_fog_cascade.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


MI = {'mi', 'mini', 'mine'}
EPIT = {'clan', 'sec', 'puia', 'lupu', 'avils', 'ril', 'suthi'}


def main():
    os.makedirs('logs', exist_ok=True)
    fog_rows = list(csv.DictReader(open(
        os.path.join('results', 'concept_fog_v1.csv'), encoding='utf-8')))
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.7'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== §7.2: каскад арбитров тумана (по рекомендации LA) ===')

    # --- арбитр 1: строгая фонетика ---------------------------------------
    cand = [r for r in fog_rows if float(r['sim']) >= 0.999
            and r['tier'] == 'A' and not r['known_gloss']]
    log(f'арбитр 1 (sim=1.0, ярус A, непереведённые): '
        f'{len(cand)} кандидатов из {len(fog_rows)} туман-строк')

    # индексы вхождений
    occ = {}
    for rec in view:
        ws = [t['ascii'] for t in rec['toks'] if t['kind'] == 'W']
        kinds = [t['kind'] for t in rec['toks']]
        wpos = [i for i, t in enumerate(rec['toks']) if t['kind'] == 'W']
        genre = ('ritual' if rec['src'] == 'CIEW'
                 and rec['eid'] in ('9001', '7002')
                 else 'mi' if set(ws) & MI
                 else 'epit' if set(ws) & EPIT else 'other')
        has_num = any(k == 'N' for k in kinds)
        for w in set(ws):
            occ.setdefault(w, []).append((genre, has_num, rec['src'],
                                          rec['eid']))
    base_genre = Counter()
    base_num = [0, 0]
    for w, os_ in occ.items():
        for g, hn, _, _ in os_:
            base_genre[g] += 1
            base_num[hn] += 1
    n_base = sum(base_genre.values())
    log(f'базлайн жанров (по вхождениям слов): '
        + ', '.join(f'{g}:{c / n_base:.0%}' for g, c in
                    base_genre.most_common())
        + f'; рядом с NUM: {base_num[1] / n_base:.1%}')

    # --- арбитры 2–3 на кандидатах -----------------------------------------
    log()
    log(f'{"слово":<14} {"n":>3} {"концепт":<18} {"домен":<9} '
        f'{"жанры":<24} {"NUM":>5} {"док.завис."}')
    keep = []
    for r in sorted(cand, key=lambda x: -int(x['freq'])):
        w = r['word']
        os_ = occ.get(w, [])
        if len(os_) < 2:
            continue
        gcnt = Counter(g for g, _, _, _ in os_)
        num_rate = sum(hn for _, hn, _, _ in os_) / len(os_)
        eids = {(s, e) for _, _, s, e in os_}
        dep = 'один-док!' if len(eids) == 1 else f'{len(eids)} док.'
        gs = '+'.join(f'{g}:{c}' for g, c in gcnt.most_common(3))
        log(f'{w:<14} {len(os_):>3} {r["concept_en"][:18]:<18} '
            f'{r["domain"][:9]:<9} {gs:<24} {num_rate:>4.0%} {dep}')
        keep.append((w, r, gcnt, num_rate, len(eids)))
    log()
    n_multi = sum(1 for *_, ne in keep if ne >= 2)
    log(f'кандидатов с n≥2 вхождений: {len(keep)}; из них в ≥2 документах: '
        f'{n_multi} (арбитр зависимости документов — LA-оговорка)')
    log('чтение: интересен кандидат, чей жанровый/NUM-профиль СОГЛАСУЕТСЯ '
        'с доменом концепта (напр. measure/time — при NUM выше базлайна; '
        'vessel — в mi-записях) И который живёт в ≥2 документах. '
        'Слой разведочный: p не заявляем (мал n), список — для ручной '
        'проверки и для симметричного прогона на стороне LA.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
