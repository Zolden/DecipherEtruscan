# -*- coding: utf-8 -*-
"""§35 (серия 4, цикл 8): объект × надпись (Vol. IV) + инвентарь USEP.

1. Адрия (Vol. IV.1.1): тип объекта = первое латинское существительное
   титула (patera/pes/fragmentum/…); связь «тип × наличие надписи»
   (перестановочный нуль меток надписи, R=10000, seed=42) — несут ли
   определённые типы посуды надписи чаще (лайт-версия плана Sol №7).
2. USEP (data/external/usep_tyrsenian, Brown, CC BY-NC-SA): инвентарь
   35 ett-записей — материал/дата/музей; точечное обогащение
   artifact-графа (дескриптив).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_object_genre.py
"""
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
OUT_LOG = os.path.join('logs', 'etr_object_genre.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    log('=== §35: объект × надпись (Адрия) + USEP ===')
    import fitz
    doc = fitz.open(os.path.join('data', 'external', 'cie_online',
                                 'CIE-IV.1.1_Atria_20001-20422.pdf'))
    full = '\n'.join(doc[i].get_text() for i in range(len(doc)))
    parts = re.split(r'\n(20[0-4]\d\d)\s', full)
    tituli = {}
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        if 20001 <= num <= 20422 and num not in tituli:
            tituli[num] = parts[i + 1][:400]
    OBJ_WORDS = ['patera', 'pes', 'fundus', 'fragmentum', 'catillus',
                 'catilli', 'poculum', 'olla', 'amphora', 'tegula',
                 'calix', 'scyphus', 'urceus', 'lucerna', 'dolium',
                 'vas', 'ansa', 'orlo', 'oris']
    rows = []
    for num in sorted(tituli):
        t = tituli[num].lower()
        obj = next((w for w in OBJ_WORDS if re.search(
            r'\b' + w[:6], t[:120])), 'прочее')
        has = bool(re.search(r'\binscriptio\b|\blitterae?\b',
                             tituli[num]))
        rows.append((obj, has))
    grp = defaultdict(lambda: [0, 0])
    for obj, has in rows:
        grp[obj][has] += 1
    log(f'титулов: {len(rows)}; с надписью: {sum(h for _, h in rows)}')
    log(f'{"тип":<12} {"всего":>5} {"с надп.":>7} {"доля":>6}')
    for obj, (no, yes) in sorted(grp.items(), key=lambda kv: -sum(kv[1])):
        if no + yes >= 10:
            log(f'{obj:<12} {no + yes:>5} {yes:>7} {yes / (no + yes):>6.0%}')
    labs = np.array([h for _, h in rows])
    objs = [o for o, _ in rows]
    keys = sorted({o for o in objs if objs.count(o) >= 10})
    oi = {o: i for i, o in enumerate(keys)}
    oarr = np.array([oi.get(o, -1) for o in objs])

    def spread(lb):
        vals = [lb[oarr == i].mean() for i in range(len(keys))
                if (oarr == i).any()]
        return max(vals) - min(vals)

    obs = spread(labs)
    rng = np.random.default_rng(SEED)
    sims = np.zeros(R)
    for i in range(R):
        sims[i] = spread(labs[rng.permutation(len(labs))])
    p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
    log(f'разброс доли надписей между типами (n>=10): {obs:.3f}; '
        f'нуль {sims.mean():.3f}±{sims.std():.3f}; p={p:.4f}')

    # --- USEP ---------------------------------------------------------------
    log('\n--- USEP (Brown, CC BY-NC-SA): инвентарь ---')
    files = sorted(glob.glob(os.path.join('data', 'external',
                                          'usep_tyrsenian', '**', '*.xml'),
                             recursive=True))
    n_date = n_mat = 0
    mats = Counter()
    for p_ in files:
        s = open(p_, encoding='utf-8', errors='replace').read()
        m = re.search(r'<material[^>]*>([^<]+)<', s)
        if m:
            n_mat += 1
            mats[m.group(1).strip().lower()] += 1
        if re.search(r'notBefore|notAfter|origDate', s):
            n_date += 1
    log(f'файлов: {len(files)}; с материалом: {n_mat} '
        f'({dict(mats.most_common(5))}); с датировкой: {n_date}')
    log('чтение: USEP — точечное обогащение (музейный provenance для '
        'артефакт-графа); слияние с корпусом только вручную (§8.3 Sol: '
        'совпали 2 нормализованные editions).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
