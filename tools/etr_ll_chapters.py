# -*- coding: utf-8 -*-
"""§10 (=§5 v4): главная грамматика Liber Linteus — секции по датам,
валидация рукописными красными линиями, классовая структура секций.

Предпосылки: карта свитка (results/ll_scroll_map.csv, §5.8) и рукописные
границы глав Herbig (data/supplements/herbig_ll_structure.md): красные
линии-параграфы между стихами VI 8/9 и XI 13/14 ≈ границы сквозных
позиций 111/112 и 230/231 (±1–2 по точности выравнивания §9.1).

ПРЕ-ДЕКЛАРИРОВАННОЕ правило секций (до всех тестов): секция начинается
на каждой строке с классом CAL (календарная дата, разметка §5); строки
до первой CAL-строки — секция 0. Только точно позиционированные строки
(121); интервальные исключены — заявленная потеря мощности.

Тесты:
T1 (валидация рукописью): сумма расстояний от двух красных границ до
   ближайшего начала CAL-строки; нуль A — k CAL-позиций разыгрываются
   без возвращения по 121 позиции; нуль B (консервативнее к слипанию) —
   разыгрываются начала CAL-серий (рангов). R=10000, seed=42.
T2 (классовые пары в секциях): число секций, содержащих оба класса
   пары; совместный нуль — ОДИН общий циклический сдвиг классового
   вектора по последовательности строк на replicate (секционные границы
   фиксированы), min-p по семейству пар (корректный W–Y §8.8). R=10000.
T3 (порядок внутри секций): средняя нормированная позиция класса в
   секции (0=начало, 1=конец); тот же совместный нуль сдвигов, семейный
   min-p двусторонний.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ll_chapters.py
"""
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
BOUNDS = (111.5, 230.5)  # красные линии Herbig в сквозных позициях
PAIRS = [('VACL', 'THEO'), ('VACL', 'OFFER'), ('THEO', 'OFFER'),
         ('VERB-R', 'THEO'), ('VERB-R', 'VACL'), ('VERB-R', 'OFFER')]
CLASSES = ['CAL', 'VACL', 'THEO', 'OFFER', 'VERB-R', 'CONJ']
OUT_LOG = os.path.join('logs', 'etr_ll_chapters.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    rows = []
    with open(os.path.join('results', 'll_scroll_map.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'exact':
                cls = set(r['classes'].split()) - {'—'}
                rows.append((int(r['position']), cls))
    rows.sort()
    pos = np.array([p for p, _ in rows])
    cls_sets = [c for _, c in rows]
    n = len(rows)
    log('=== §10: секции LL по датам + рукописная валидация ===')
    log(f'точных строк: {n}; классы: '
        + ', '.join(f'{c}:{sum(1 for s in cls_sets if c in s)}'
                    for c in CLASSES))

    rng = np.random.default_rng(SEED)
    cal_idx = np.array([i for i, s in enumerate(cls_sets) if 'CAL' in s])
    k = len(cal_idx)
    # CAL-серии (слипшиеся подряд идущие CAL-строки => одно начало)
    runs = [i for j, i in enumerate(cal_idx)
            if j == 0 or cal_idx[j - 1] != i - 1]
    log(f'CAL-строк: {k}; CAL-серий (начал секций): {len(runs)}')

    # --- T1: красные линии vs начала CAL-строк ------------------------------
    cal_pos = pos[cal_idx]
    run_pos = pos[np.array(runs)]

    def sumdist(marks):
        return sum(min(abs(m - b) for m in marks) for b in BOUNDS)

    obs_a = sumdist(cal_pos)
    obs_b = sumdist(run_pos)
    log(f'\n--- T1: расстояние красных линий до ближайшей CAL-позиции ---')
    for b in BOUNDS:
        da = min(abs(m - b) for m in cal_pos)
        log(f'  граница {b}: ближайшая CAL-строка на расстоянии {da}')
    sims_a = np.zeros(R)
    sims_b = np.zeros(R)
    for i in range(R):
        sims_a[i] = sumdist(pos[rng.choice(n, size=k, replace=False)])
        sims_b[i] = sumdist(pos[rng.choice(n, size=len(runs),
                                           replace=False)])
    p_a = float(((sims_a <= obs_a).sum() + 1) / (R + 1))
    p_b = float(((sims_b <= obs_b).sum() + 1) / (R + 1))
    log(f'T1 нуль A (k={k} позиций): obs={obs_a:.1f}, нуль '
        f'{sims_a.mean():.1f}±{sims_a.std():.1f}, p={p_a:.4f}')
    log(f'T1 нуль B (начала серий, k={len(runs)}): obs={obs_b:.1f}, нуль '
        f'{sims_b.mean():.1f}±{sims_b.std():.1f}, p={p_b:.4f}')

    # --- секции по пре-декларированному правилу ------------------------------
    sec_id = np.zeros(n, int)
    cur = 0
    for i in range(n):
        if i in set(runs) and i > 0:
            cur += 1
        sec_id[i] = cur
    n_sec = cur + 1
    sizes = Counter(sec_id.tolist())
    log(f'\nсекций: {n_sec}; размеры: медиана '
        f'{int(np.median(list(sizes.values())))}, '
        f'мин {min(sizes.values())}, макс {max(sizes.values())}')

    # --- T2/T3 на совместном нуле циклических сдвигов -----------------------
    def pair_stats(cs):
        out = []
        for a, b in PAIRS:
            secs_a = {sec_id[i] for i, s in enumerate(cs) if a in s}
            secs_b = {sec_id[i] for i, s in enumerate(cs) if b in s}
            out.append(len(secs_a & secs_b))
        return np.array(out, float)

    def order_stats(cs):
        # средняя нормированная позиция класса внутри секций (размер>=3)
        relpos = np.zeros(n)
        for s in range(n_sec):
            idx = np.where(sec_id == s)[0]
            if len(idx) >= 2:
                relpos[idx] = np.linspace(0, 1, len(idx))
            else:
                relpos[idx] = 0.5
        out = []
        for c in CLASSES[1:]:  # CAL определяет секции — исключён
            xs = [relpos[i] for i, st in enumerate(cs) if c in st
                  and sizes[sec_id[i]] >= 3]
            out.append(np.mean(xs) if xs else np.nan)
        return np.array(out, float)

    obs_pairs = pair_stats(cls_sets)
    obs_ord = order_stats(cls_sets)
    ge_pairs = np.zeros(len(PAIRS))
    null_pair_max = np.zeros((R, len(PAIRS)))
    lo_ord = np.zeros(len(obs_ord))
    hi_ord = np.zeros(len(obs_ord))
    null_ord = np.zeros((R, len(obs_ord)))
    for i in range(R):
        sh = int(rng.integers(1, n))  # один общий сдвиг на replicate
        cs = cls_sets[sh:] + cls_sets[:sh]
        ps = pair_stats(cs)
        null_pair_max[i] = ps
        ge_pairs += ps >= obs_pairs
        os_ = order_stats(cs)
        null_ord[i] = os_
        lo_ord += np.where(np.isnan(os_) | np.isnan(obs_ord), 0,
                           os_ <= obs_ord)
        hi_ord += np.where(np.isnan(os_) | np.isnan(obs_ord), 0,
                           os_ >= obs_ord)

    log('\n--- T2: классовые пары в одной секции (совместный нуль сдвигов) ---')
    # семейный min-p: ранги наблюдений против max-статистики семейства
    raw_p = (ge_pairs + 1) / (R + 1)
    # p̃ по семейству: доля replicate, где ХОТЬ ОДНА пара достигла своего obs
    fam = ((null_pair_max >= obs_pairs[None, :]).any(axis=1).sum() + 1) / (R + 1)
    for (a, b), o, rp in zip(PAIRS, obs_pairs, raw_p):
        mu = null_pair_max[:, PAIRS.index((a, b))].mean()
        log(f'  {a}+{b:<8} секций вместе: {int(o):>2} (нуль {mu:.1f}); '
            f'raw p={rp:.4f}')
    log(f'  семейная поправка (min-p, общий сдвиг): худший случай '
        f'p̃<= {fam:.4f} для лучшей пары; консервативно Bonferroni x{len(PAIRS)}')

    log('\n--- T3: средняя позиция класса в секции (0=начало, 1=конец) ---')
    p_lo = (lo_ord + 1) / (R + 1)
    p_hi = (hi_ord + 1) / (R + 1)
    for c, o, pl, ph in zip(CLASSES[1:], obs_ord, p_lo, p_hi):
        if np.isnan(o):
            log(f'  {c:<7} — нет вхождений в секциях >=3 строк')
        else:
            log(f'  {c:<7} средн. позиция {o:.3f}; p(раньше)={pl:.4f}, '
                f'p(позже)={ph:.4f} (raw, двусторонне x2, семья x5)')

    log('\nчтение: T1 — совпадают ли рукописные границы глав с началом '
        'датированных секций; T2/T3 — устойчивая ли внутренняя структура '
        'секций. Ограничения: 121 точная строка из ~240 (интервальные вне '
        'выборки); границы красных линий с точностью ±1–2 позиции (§9.1).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
