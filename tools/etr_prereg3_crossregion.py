# -*- coding: utf-8 -*-
"""Предрегистрация №3: межрегиональная парадигматика формулы.

Правила заморожены в validation/prereg3_crossregion_pairs.md (коммит ДО
прогона; прогон ОДИН; публикация как вышло). H1: одноклассовость
скелетных замещений (§13) на парах записей РАЗНЫХ регионов выше нуля
глобальной перестановки операторов (R=1000, seed=42), p<=0.05.
Реестр OPS импортируется из etr_minimal_pairs (§13) — идентичность
классов по построению.

Запуск (единственный):
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_prereg3_crossregion.py
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
OUT_LOG = os.path.join('logs', 'etr_prereg3_crossregion.log')
LOG = []

# реестр §13 (копия значений из tools/etr_minimal_pairs.py, часть 2)
OPS = {'clan': 'KIN', 'sec': 'KIN', 'sech': 'KIN', 'puia': 'KIN',
       'ruva': 'KIN', 'ati': 'KIN', 'apa': 'KIN', 'papa': 'KIN',
       'teta': 'KIN', 'nefts': 'KIN', 'tusurthir': 'KIN',
       'turce': 'VERB', 'turuce': 'VERB', 'muluvanice': 'VERB',
       'lupu': 'VERB', 'lupuce': 'VERB', 'svalce': 'VERB',
       'amce': 'VERB', 'ame': 'VERB', 'zilachnu': 'VERB',
       'zilachnce': 'VERB', 'tence': 'VERB', 'cerichunce': 'VERB',
       'zilath': 'TIT', 'zilch': 'TIT', 'zilc': 'TIT',
       'camthi': 'TIT', 'maru': 'TIT', 'purth': 'TIT',
       'marunuch': 'TIT',
       'mi': 'DEIX', 'mini': 'DEIX', 'mine': 'DEIX', 'ta': 'DEIX',
       'ca': 'DEIX', 'cn': 'DEIX', 'itun': 'DEIX', 'eca': 'DEIX',
       'avils': 'MEAS', 'avil': 'MEAS', 'ril': 'MEAS',
       'suthi': 'OBJ', 'suthina': 'OBJ', 'mutna': 'OBJ',
       'mlach': 'OBJ', 'cana': 'OBJ', 'flere': 'OBJ', 'fler': 'OBJ'}


def log(m=''):
    print(m)
    LOG.append(m)


def count_events(buckets):
    tot = same = 0
    for key, items in buckets.items():
        seen = sorted(set(items))
        for x in range(len(seen)):
            for y in range(x + 1, len(seen)):
                (a1, r1, w1), (a2, r2, w2) = seen[x], seen[y]
                if a1 != a2 and w1 != w2 and r1 != r2:
                    tot += 1
                    same += OPS[w1] == OPS[w2]
    return tot, same


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== Предрегистрация №3: межрегиональная парадигматика ===')
    log('правила: validation/prereg3_crossregion_pairs.md (до прогона)')
    recs = []
    for r in view:
        ws = tuple(t['ascii'] for t in r['toks']
                   if t['kind'] == 'W' and '-' not in t['ascii']
                   and len(t['ascii']) >= 2)
        if 3 <= len(ws) <= 8 and r.get('region'):
            recs.append((r['artifact_id'], r['region'], ws))
    log(f'записей 3–8 слов с регионом: {len(recs)}')

    buckets = defaultdict(list)
    for art, reg, ws in recs:
        sk = tuple(w if w in OPS else '*' for w in ws)
        if not any(w in OPS for w in ws):
            continue
        for i, w in enumerate(sk):
            if w == '*':
                continue
            buckets[(len(sk), i, sk[:i] + sk[i + 1:])].append((art, reg, w))
    obs_tot, obs_same = count_events(buckets)
    obs = obs_same / max(obs_tot, 1)
    log(f'межрегиональных событий: {obs_tot}; одноклассовых: {obs_same} '
        f'({100 * obs:.1f}%)')

    incid = []
    for key in sorted(buckets, key=str):
        for a, rg, w in sorted(set(buckets[key])):
            incid.append((key, a, rg, w))
    ops_arr = [w for *_, w in incid]
    rng = np.random.default_rng(SEED)
    sims = np.zeros(R)
    for r_i in range(R):
        perm = rng.permutation(len(ops_arr))
        byb = defaultdict(list)
        for (key, a, rg, _), pi in zip(incid, perm):
            byb[key].append((a, rg, ops_arr[pi]))
        tot, same = count_events(byb)
        sims[r_i] = same / max(tot, 1)
    p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
    log(f'нуль (глобальная перестановка операторов): '
        f'{100 * sims.mean():.1f}%±{100 * sims.std():.1f}%; p={p:.4f}')
    log(f'H1 {"ПОДТВЕРЖДЕНА" if p <= 0.05 else "НЕ ПОДТВЕРЖДЕНА"} '
        f'(критерий p<=0.05, единственный прогон)')
    log('\nоговорка: регион известен у ~54% записей (покрытие Бурман) — '
        'ограничение внешней валидности, не нуля.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
