# -*- coding: utf-8 -*-
"""§6.1: классификатор языка на суффиксных распределениях — куда ложится
лемнийский НАБОР токенов?

Модель: наивный Байес по признакам формы (финальные 1–3-граммы,
начальные 1–2) на ТОКЕНАХ трёх языков корпуса v0.6: этрусский (выборка),
латынь, умбрский. Атрибуция НАБОРА: сумма лог-правдоподобий по токенам.
Калибровка: бутстрэп — наборы размера n_lemn из отложенных токенов
каждого языка (R=2000, seed=42) → матрица «истина × атрибуция»; затем
атрибутируется лемнийский набор (33 токена) и сообщается маржа
лог-правдоподобий. Лемнийский набор — 33 токена чистого supplement;
частичные варианты CIEP 15999 исключены во избежание двойного счёта.
Оговорки: этрусская выборка доминирует корпус — берём сбалансированные
обучающие пулы.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_lemnos2.py
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
R_BOOT = 2000
OUT_LOG = os.path.join('logs', 'etr_lemnos2.log')
LEMNOS_SRC = 'SUPP:lemnos_wikipedia.csv'
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def feats(w):
    f = set()
    for k in (1, 2, 3):
        if len(w) > k:
            f.add(f's{k}:{w[-k:]}')
    for k in (1, 2):
        if len(w) > k:
            f.add(f'p{k}:{w[:k]}')
    return f


def main():
    os.makedirs('logs', exist_ok=True)
    log('RETRACT 2026-07-10: closed-set NB вынужденно относит отдельный OOD '
        'язык к etr/lat/umb и не атрибутирует Лемнос. См. §8 отчёта.')
    log()
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.8'
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

    pools = {'etr': toks_of('etr'), 'lat': toks_of('lat'),
             'umb': toks_of('umb')}
    # CIEP 15999 is a partial/variant reading of this same monument.
    lemn = toks_of('lemn', LEMNOS_SRC)
    rng = np.random.default_rng(SEED)
    log('=== §6.1: языковая атрибуция лемнийского набора ===')
    log(f'пулы токенов: '
        + ', '.join(f'{k}:{len(v)}' for k, v in pools.items())
        + f'; лемнийских: {len(lemn)}')

    # сбалансированные train/holdout пулы
    n_tr = 120
    langs = sorted(pools)
    train, hold = {}, {}
    for lg in langs:
        p = list(pools[lg])
        idx = rng.permutation(len(p))
        train[lg] = [p[i] for i in idx[:n_tr]]
        hold[lg] = [p[i] for i in idx[n_tr:]]
    log(f'train {n_tr} токенов/язык; holdout: '
        + ', '.join(f'{lg}:{len(hold[lg])}' for lg in langs))

    fi = {}
    for lg in langs:
        for w in train[lg]:
            for f_ in feats(w):
                fi.setdefault(f_, len(fi))

    def loglik(bag):
        # log P(bag | lang) при NB с add-1
        out = {}
        for lg in langs:
            cnt = Counter(f_ for w in train[lg] for f_ in feats(w))
            n = len(train[lg])
            s = 0.0
            for w in bag:
                for f_ in feats(w):
                    if f_ in fi:
                        s += np.log((cnt.get(f_, 0) + 1) / (n + 2))
            out[lg] = s / max(len(bag), 1)
        return out

    # бутстрэп-калибровка на holdout
    n_bag = len(lemn)
    conf = {lg: Counter() for lg in langs}
    for lg in langs:
        pool = hold[lg]
        for _ in range(R_BOOT // len(langs)):
            bag = [pool[i] for i in rng.integers(0, len(pool), n_bag)]
            ll = loglik(bag)
            conf[lg][max(ll, key=ll.get)] += 1
    log()
    log('--- бутстрэп-калибровка (наборы размера '
        f'{n_bag} из holdout) ---')
    log('истина \\ атрибуция: ' + '  '.join(langs))
    ok = 0
    tot = 0
    for lg in langs:
        row = [conf[lg].get(l2, 0) for l2 in langs]
        ok += conf[lg].get(lg, 0)
        tot += sum(row)
        log(f'  {lg}: ' + '  '.join(f'{x:>4}' for x in row))
    log(f'точность атрибуции наборов: {ok / tot:.0%}')

    ll = loglik(lemn)
    best = sorted(ll.items(), key=lambda x: -x[1])
    log()
    log('--- лемнийский набор ---')
    for lg, v in best:
        log(f'  logP/токен ({lg}): {v:.3f}')
    log(f'атрибуция: {best[0][0]} (маржа к следующему: '
        f'{best[0][1] - best[1][1]:.3f} на токен)')
    # значимость маржи: бутстрэп лемнийского набора (ресемпл с возвратом)
    margins = np.zeros(500)
    for i in range(500):
        bag = [lemn[j] for j in rng.integers(0, len(lemn), n_bag)]
        l2 = loglik(bag)
        b2 = sorted(l2.items(), key=lambda x: -x[1])
        margins[i] = (b2[0][1] - b2[1][1]) if b2[0][0] == best[0][0] \
            else -(b2[0][1] - b2[1][1])
    log(f'бутстрэп лемнийского набора (500): атрибуция «{best[0][0]}» '
        f'устойчива в {np.mean(margins > 0):.0%} ресемплов')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
