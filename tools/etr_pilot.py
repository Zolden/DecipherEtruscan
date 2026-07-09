# -*- coding: utf-8 -*-
"""Этап 14 (§BM): этрусский пилот — применимость нашего метода.

Данные: Larth-Etruscan-NLP (Vico & Niculae 2023, ACL ALP) — компиляция
ETP/CIE-производных: надпись, город, датировка, английский перевод (где есть).

Пилотные замеры (сопоставимо с LA-показателями):
1. Масштаб: надписей, токенов, типов, доля гапаксов; сравнение с LA.
2. Формульность: покрытие корпуса топ-шаблонами; топ-биграммы.
3. «Операторы» (известные служебные слова: clan 'сын', sec 'дочь',
   avils 'лет', ril 'в возрасте', lupu 'умер', zilath/zilc 'магистрат',
   turuce 'посвятил', mi 'я', cver 'дар'): частоты, позиции — прямой аналог
   нашего реестра §BC.
4. Морфология: цепочки генитивов -s/-sa/-al (паттерн «X-al Y-s clan»);
   доля слов с известными суффиксами — аналог калибровки §L.
5. Валидационный потенциал: сколько надписей имеют перевод (train/test для
   дистрибутивной семантики — то, чего нет у LA).
"""
import sys, csv, re
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

rows = []
with open('data/etruscan_larth.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        et = (r.get('Etruscan') or '').strip().lower()
        if not et: continue
        et = re.sub(r'[:\.\,;·]+', ' ', et)
        toks = [t for t in et.split() if re.fullmatch(r"[a-zθχφśšḫ'’vzunm-]+", t)]
        if toks:
            rows.append({'id': r.get('ID', '').strip(), 'city': r.get('City', ''),
                         'toks': toks, 'tr': (r.get('Translation') or '').strip()})

print(f'=== 1. масштаб ===')
n_tok = sum(len(r['toks']) for r in rows)
vocab = Counter(t for r in rows for t in r['toks'])
hapax = sum(1 for w, c in vocab.items() if c == 1)
print(f'надписей: {len(rows)}; токенов: {n_tok}; типов: {len(vocab)}; '
      f'гапаксов: {hapax} ({hapax / len(vocab):.0%})')
print(f'LA для сравнения: 1721 надпись, 900 лексиконных токенов, 618 типов, '
      f'гапаксов 82%')
with_tr = sum(1 for r in rows if len(r['tr']) > 3)
print(f'с переводом: {with_tr} ({with_tr / len(rows):.0%}) — валидационный '
      f'набор, которого у LA нет')

print(f'\n=== 2. формульность ===')
big = Counter()
for r in rows:
    for i in range(len(r['toks']) - 1):
        big[(r['toks'][i], r['toks'][i + 1])] += 1
print('топ-биграммы:', [(f'{a} {b}', c) for (a, b), c in big.most_common(8)])
# шаблон надписи = кортеж классов (known-word / X)
KNOWN = {'mi', 'clan', 'sec', 'sech', 'avils', 'ril', 'lupu', 'lupuce',
         'zilath', 'zilc', 'turuce', 'turce', 'cver', 'suthi', 'thui',
         'cehen', 'an', 'cn', 'ta', 'mini', 'muluvanice', 'mulveni', 'aisar',
         'ais', 'puia', 'ati', 'apa', 'atial', 'clens', 'clenar', 'etera',
         'spur', 'meθlum', 'am', 'ame', 'amce', 'svalce', 'svalas', 'tenθas',
         'tenu', 'zich', 'ziχu', 'acil', 'cerine', 'θui'}
tmpl = Counter()
for r in rows:
    t = tuple('K:' + w if w in KNOWN else 'X' for w in r['toks'][:6])
    tmpl[t] += 1
top10 = tmpl.most_common(10)
cover = sum(c for _, c in top10) / len(rows)
print(f'покрытие топ-10 шаблонов (первые 6 слов, K=известное слово): {cover:.0%}')

print(f'\n=== 3. операторы (аналог §BC) ===')
pos_stats = {}
for op in ['mi', 'clan', 'sec', 'avils', 'ril', 'lupu', 'lupuce', 'zilath',
           'turuce', 'turce', 'suthi', 'puia', 'zilc']:
    occ = ini = fin = 0
    for r in rows:
        for i, t in enumerate(r['toks']):
            if t == op:
                occ += 1
                ini += i == 0
                fin += i == len(r['toks']) - 1
    if occ:
        print(f'   {op:<8} ×{occ:>4}  нач {ini / occ:>4.0%}  фин {fin / occ:>4.0%}')

print(f'\n=== 4. морфология: суффиксы генитивов (калибровочный замер) ===')
for suf in ['s', 'al', 'sa', 'us', 'ns', 'ial', 'ns']:
    n = sum(c for w, c in vocab.items() if w.endswith(suf) and len(w) > len(suf) + 1)
    print(f'   -{suf:<4} типов с суффиксом: '
          f'{sum(1 for w in vocab if w.endswith(suf) and len(w) > len(suf) + 1):>5} '
          f'(токенов {n})')
# цепочка «X-al ... clan» (сын такой-то [матери])
chain = 0
for r in rows:
    for i in range(len(r['toks']) - 1):
        if r['toks'][i].endswith('al') and r['toks'][i + 1] in ('clan', 'sec'):
            chain += 1
print(f'паттерн «X-al + clan/sec» (генитив родителя + „сын/дочь“): ×{chain}')
