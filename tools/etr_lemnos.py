# -*- coding: utf-8 -*-
"""§6: Лемносская стела — количественная проверка тирренской параллели.

Данные: полная чистая транслитерация lang='lemn' из supplement v0.6
(CIEP 15999 содержит частичные варианты того же памятника и исключён из
этого анализа во избежание двойного счёта); контроли — токены lang='lat'
и lang='umb' записей корпуса.

Тест «этрусковидности концовок»: статистика — доля словоформ-токенов
языка L, чья финальная биграмма входит в инвентарь финальных биграмм
этрусского словаря (типов ≥5). Нуль — перемешивание букв ВНУТРИ каждого
слова (сохраняются длина и состав), R=10000, seed=42 → p и z. Если
лемнийский значимо «этрусковиднее» своего нуля, а латынь/умбрский — нет
или слабее, параллель количественно поддержана. Оговорки: (а) мал объём
(~50 токенов), (б) OCR-шум CIEW, (в) финальные биграммы — грубый прибор
(фонотактика+морфология вместе).

Плюс разведочно: кандидаты-когнаты (лемнийское слово ↔ этрусские типы с
общим началом ≥4 букв или расстоянием ≤1) — БЕЗ статистических претензий.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_lemnos.py
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_lemnos.log')
LEMNOS_SRC = 'SUPP:lemnos_wikipedia.csv'
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def main():
    os.makedirs('logs', exist_ok=True)
    log('RETRACT 2026-07-10: прежняя положительная атрибуция имела target '
        'leakage. Даже после очистки forced closed-set score не доказывает '
        'родство; см. §8 и audit type-disjoint result.')
    log()
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    recs = corpus['records']

    def toks_of(lang, src=None):
        out = []
        for r in recs:
            if r['lang'] == lang and r['kind'] == 'text' \
                    and 'forgery?' not in r['flags'] \
                    and r.get('variant_of') is None \
                    and (src is None or r['src'] == src):
                for t in r['toks']:
                    if t['kind'] == 'W' and '-' not in t['ascii'] \
                            and len(t['ascii']) >= 3:
                        out.append(t['ascii'])
        return out

    etr_view = [r for r in recs if r['lang'] == 'etr' and r['kind'] == 'text'
                and 'forgery?' not in r['flags']
                and r.get('variant_of') is None]
    vocab = {w for w in Counter(t['ascii'] for r in etr_view
                                for t in r['toks'] if t['kind'] == 'W')
             if '-' not in w and len(w) >= 3}
    end2 = Counter(w[-2:] for w in vocab)
    END2 = {e for e, c in end2.items() if c >= 5}
    log('=== §6: Лемносская стела — тирренская параллель ===')
    log(f'этрусский словарь: {len(vocab)} типов; инвентарь финальных '
        f'биграмм (≥5 типов): {len(END2)}')

    # CIEP 15999 contains two partial/variant readings of the same stele.
    # Use only the declared clean, complete supplement to avoid double count.
    lemn = toks_of('lemn', LEMNOS_SRC)
    log(f'лемнийских токенов: {len(lemn)}: {lemn}')

    rng = np.random.default_rng(SEED)

    def test_lang(tokens, name):
        toks = [w for w in tokens if len(w) >= 3]
        if not toks:
            return
        obs = np.mean([w[-2:] in END2 for w in toks])
        sims = np.zeros(R)
        arrs = [list(w) for w in toks]
        for r_i in range(R):
            s = 0
            for a in arrs:
                p = rng.permutation(len(a))
                s += (a[p[-2]] + a[p[-1]]) in END2
            sims[r_i] = s / len(arrs)
        p = float(((sims >= obs).sum() + 1) / (R + 1))
        z = float((obs - sims.mean()) / max(sims.std(), 1e-9))
        log(f'  {name:<22} токенов {len(toks):>4}: доля этрусских '
            f'финальных биграмм {obs:.0%} (нуль {sims.mean():.0%}±'
            f'{sims.std():.0%}), z={z:+.1f}, p={p:.4f}')
        return obs, z, p

    log()
    log('--- тест этрусковидности концовок (нуль: скрэмбл букв в слове) ---')
    test_lang(lemn, 'лемнийский (supplement)')
    test_lang(toks_of('lat'), 'латынь (контроль)')
    test_lang(toks_of('umb'), 'умбрский (контроль)')
    test_lang(toks_of('cel'), 'кельтский (контроль)')

    # --- дискриминативный маркёр: финальное -l (тирренский генитив) --------
    log()
    log('--- финальное -l (генитив II -l/-al — тирренский маркёр) ---')
    etr_l = np.mean([w[-1] == 'l' for w in vocab])
    log(f'  этрусский словарь: {etr_l:.1%} типов на -l')
    from scipy.stats import hypergeom as hg
    base = {}
    for name, toks in [('лемнийский', lemn), ('латынь', toks_of('lat')),
                       ('умбрский', toks_of('umb'))]:
        toks = [w for w in toks if len(w) >= 3]
        nl = sum(1 for w in toks if w[-1] == 'l')
        base[name] = (nl, len(toks))
        log(f'  {name:<12} {nl}/{len(toks)} ({nl / max(len(toks), 1):.1%})')
    # лемнийский vs латынь: гипергеометрический (обе выборки объединены)
    (a, na), (b, nb) = base['лемнийский'], base['латынь']
    p_lat = float(hg.sf(a - 1, na + nb, a + b, na))
    (c, nc) = base['умбрский']
    p_umb = float(hg.sf(a - 1, na + nc, a + c, na))
    log(f'  лемнийский > латынь по -l: p={p_lat:.4f}; > умбрский: '
        f'p={p_umb:.4f} (гипергеометрический, односторонний)')

    # --- кандидаты-когнаты (разведочно) -------------------------------------
    log()
    log('--- кандидаты-когнаты (разведочно, без p) ---')
    seen = set()
    for w in sorted(set(lemn)):
        cands = []
        for v in vocab:
            if v == w:
                cands.append(v + ' (точное)')
            elif len(w) >= 4 and len(v) >= 4 and v[:4] == w[:4]:
                cands.append(v)
        cands = sorted(set(cands))[:6]
        if cands and w not in seen:
            seen.add(w)
            log(f'  {w:<14} ~ {", ".join(cands)}')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
