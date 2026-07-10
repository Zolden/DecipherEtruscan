# -*- coding: utf-8 -*-
"""§5: структурный (слотовый) разбор Liber Linteus по корпусной копии CIEW.

Данные: записи src='CIEW', eid='9001' заморозки v0.4 (130 строк, ~30%
полного текста LL — покрытие парсера, не памятника; честная оговорка).

Анализы:
  1. Формульный словарь LL: покрытие строк известными ритуальными
     операторами (vacl, fler, cletram, śrenχve, trin, θez-, nunθen-,
     etnam, ceia/hia/ciz/male, sacni-/cilθ-/spur-/meθlum-, eslem/zaθrum,
     теонимы tin/uni/neθuns/veive/crap-).
  2. Рефренные семейства: кластеризация строк по Жаккару токен-множеств
     (порог 0.5); нуль — перестановка токенов между строками с
     сохранением длин (R=1000): p для размера максимального семейства.
  3. Слот рефрена «(male) ceia hia etnam ciz vacl X»: инвентарь
     заполнителей X.
  4. Позиция vacl в строке: тест начальной позиции (нуль — равномерная
     позиция, R=10000, как §1).
  5. Календарные метки: строки с числительными/названиями сроков
     (zaθrum- «20», eslem zaθrum- «18+», celi, acale, tiur «месяц»).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_liber.py
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R_FAM = 1000
R_POS = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_liber.log')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


FORMULA = {
    'vacl': ('vacl',), 'fler': ('fler', 'flere', 'flerχva'),
    'cletram': ('cletram',), 'srenchve': ('srenchve',),
    'trin': ('trin', 'trinth'), 'thez-': ('thezi', 'thezine', 'thezeri'),
    'nunthen-': ('nunthen', 'nunthene', 'nuntheri', 'nunthenth'),
    'etnam': ('etnam',), 'ceia': ('ceia',), 'hia': ('hia',),
    'ciz': ('ciz',), 'male': ('male',),
    'sacni-': ('sacni', 'sacnicla', 'sacnicleri', 'sacnicstres',
               'sacnitn'), 'cilth-': ('cilth', 'cilthl', 'cilths',
                                      'cilthcval'),
    'spur-': ('spureri', 'spurestres', 'spurestresc'),
    'methlum-': ('methlum', 'methlumeri', 'methlumeric', 'methlumesc'),
    'календ.': ('zathrum', 'zathrums', 'zathrumsne', 'zathrumis',
                'eslem', 'celi', 'acale', 'tiur', 'tiurim'),
    'теонимы': ('tin', 'tins', 'tinsi', 'uni', 'unialti', 'nethuns',
                'nethunsl', 'veive', 'veives', 'crap', 'crapsti',
                'lusa', 'lusas', 'aiser', 'aiseras'),
}


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    ll = [r for r in corpus['records']
          if r['src'] == 'CIEW' and r['eid'] == '9001']
    lines = []
    for r in ll:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W'
              and '-' not in t['ascii'] and len(t['ascii']) >= 2]
        if len(ws) >= 2:
            lines.append((r['key'], ws))
    n_words_all = sum(1 for r in ll for t in r['toks'] if t['kind'] == 'W')
    log('=== §5: структурный разбор Liber Linteus (копия CIEW) ===')
    log(f'строк LL в корпусе: {len(ll)}; пригодных (≥2 чистых словоформ): '
        f'{len(lines)}; словоформ {n_words_all} ≈ '
        f'{n_words_all / 1330:.0%} опубликованного объёма (~1330 слов); '
        f'выравнивание строк частичное (флаг unaligned)')

    # --- 1. формульное покрытие --------------------------------------------
    log()
    log('--- 1. формульный словарь ---')
    n_cov = 0
    fam_counts = Counter()
    for key, ws in lines:
        hit = False
        for fam, forms in FORMULA.items():
            if any(w in forms for w in ws):
                fam_counts[fam] += 1
                hit = True
        n_cov += hit
    log(f'строк с ≥1 формульным элементом: {n_cov}/{len(lines)} '
        f'({n_cov / len(lines):.0%})')
    for fam, c in fam_counts.most_common():
        log(f'  {fam:<10} в {c} строках')

    # --- 2. рефренные семейства --------------------------------------------
    log()
    log('--- 2. рефренные семейства (Жаккар ≥0.5) ---')
    sets = [set(ws) for _, ws in lines]

    def families(sets_):
        n = len(sets_)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                a, b = sets_[i], sets_[j]
                if len(a & b) / len(a | b) >= 0.5:
                    parent[find(i)] = find(j)
        sizes = Counter(find(i) for i in range(n))
        return sorted(sizes.values(), reverse=True), sizes

    obs_sizes, sizes = families(sets)
    log(f'семейств размера ≥2: {sum(1 for s in obs_sizes if s >= 2)}; '
        f'крупнейшие: {obs_sizes[:6]}')
    rng = np.random.default_rng(SEED)
    all_tokens = [w for _, ws in lines for w in ws]
    lens = [len(ws) for _, ws in lines]
    sim_max = np.zeros(R_FAM)
    for r_i in range(R_FAM):
        perm = rng.permutation(len(all_tokens))
        it = iter(perm)
        sets_p = []
        for L in lens:
            sets_p.append({all_tokens[next(it)] for _ in range(L)})
        sim_max[r_i] = families(sets_p)[0][0]
    p_fam = float(((sim_max >= obs_sizes[0]).sum() + 1) / (R_FAM + 1))
    log(f'нуль (перестановка токенов, R={R_FAM}): max семейство '
        f'{sim_max.mean():.1f}±, максимум {int(sim_max.max())}; '
        f'наблюдаемое {obs_sizes[0]}; p={p_fam:.4f}')
    # состав крупнейшего семейства
    roots = Counter()
    parent = list(range(len(sets)))

    def find2(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= 0.5:
                parent[find2(i)] = find2(j)
    groups = {}
    for i in range(len(sets)):
        groups.setdefault(find2(i), []).append(i)
    big = sorted(groups.values(), key=len, reverse=True)
    for g in big[:3]:
        if len(g) < 2:
            continue
        log(f'  семейство ×{len(g)}:')
        for i in g[:4]:
            log(f'    {lines[i][0]}: {" ".join(lines[i][1])[:80]}')

    # --- 3. слот рефрена ceia hia -------------------------------------------
    log()
    log('--- 3. слот «(male) ceia hia etnam ciz vacl X» ---')
    fillers = []
    for key, ws in lines:
        if 'ceia' in ws and 'hia' in ws:
            if 'vacl' in ws:
                i = ws.index('vacl')
                fillers.append(' '.join(ws[i + 1:i + 3]) or '∅')
            else:
                fillers.append('(без vacl): ' + ' '.join(ws[-2:]))
    log(f'строк рефрена ceia-hia: {len(fillers)}; заполнители слота: '
        f'{fillers}')

    # --- 4. позиция vacl -----------------------------------------------------
    log()
    log('--- 4. позиция vacl в строке ---')
    occ = []
    for _, ws in lines:
        for i, w in enumerate(ws):
            if w == 'vacl':
                occ.append((i, len(ws)))
    ini = sum(1 for i, L in occ if i == 0)
    rng2 = np.random.default_rng(SEED)
    Ls = np.array([L for _, L in occ])
    sim = (rng2.random((R_POS, len(occ))) < 1.0 / Ls).sum(axis=1)
    p_ini = float(((sim >= ini).sum() + 1) / (R_POS + 1))
    log(f'vacl: {len(occ)} вхождений, начальная позиция {ini} '
        f'({ini / max(len(occ), 1):.0%}); нуль равномерный: p={p_ini:.4f}')

    # --- 5. календарные строки ----------------------------------------------
    log()
    log('--- 5. календарные метки ---')
    cal = [(key, ws) for key, ws in lines
           if any(w.startswith(('zathrum', 'eslem', 'tiur', 'acal',
                                'celi')) for w in ws)]
    log(f'строк с календарными метками: {len(cal)}')
    for key, ws in cal[:8]:
        log(f'  {key}: {" ".join(ws)[:80]}')

    # --- 6. календарная сетка v3: сквозная нумерация строк F&W --------------
    # ключ = «страница.номер»; номера < 900 — сквозные по свитку (проверка
    # монотонности ниже), >=900 — ненумерованные (исключены из сетки)
    log()
    log('--- 6. календарная сетка v3 (только нумерованные строки) ---')
    numbered = []
    for key, ws in lines:
        pg, num = key.split('.')
        if int(num) < 900:
            numbered.append((int(num), int(pg), ws))
    numbered.sort()
    seq_pages = [pg for _, pg, _ in numbered]
    mono = all(seq_pages[i] <= seq_pages[i + 1]
               for i in range(len(seq_pages) - 1))
    log(f'нумерованных строк: {len(numbered)}; max номер: '
        f'{numbered[-1][0] if numbered else 0}; сквозная нумерация '
        f'(страницы монотонны по номерам): {"ДА" if mono else "НЕТ"}')
    MONTHS = {'acal': 'acale', 'celi': 'celi', 'zathrum': 'zathrum-20',
              'eslem': 'eslem', 'tiur': 'tiur'}
    grid = []
    for num, pg, ws in numbered:
        hits = sorted({lab for pre, lab in MONTHS.items()
                       if any(w.startswith(pre) for w in ws)})
        if hits:
            grid.append((num, hits, ' '.join(ws)[:58]))
    log('хронологическая последовательность календарных строк (по свитку):')
    for num, hits, txt in grid:
        log(f'  стр.{num:>3}  {"+".join(hits):<22} {txt}')
    ac = [n for n, h, _ in grid if 'acale' in h]
    ce = [n for n, h, _ in grid if 'celi' in h]
    if ac and ce:
        log(f'порядок месяцев по свитку: acale @ {ac}, celi @ {ce} — '
            + ('acale < celi (июнь<сентябрь, ожидаемо)'
               if max(ac) < min(ce) else 'смешанный'))
    fam_pos = {}
    for num, pg, ws in numbered:
        st = set(ws)
        if 'ceia' in st and 'hia' in st:
            fam_pos.setdefault('ceia-hia', []).append(num)
        if 'cisum' in st or ('tul' in st and 'avils' in st):
            fam_pos.setdefault('датировочная', []).append(num)
        if 'sacnicleri' in st or 'spurestres' in st or 'sacnicstres' in st:
            fam_pos.setdefault('община', []).append(num)
    for fam, ps in sorted(fam_pos.items()):
        log(f'  семейство {fam}: строки {sorted(ps)}')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
