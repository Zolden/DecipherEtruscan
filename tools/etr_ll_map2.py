# -*- coding: utf-8 -*-
"""§16 (серия 2, цикл 2): карта свитка v2 — продвижение интервальных
строк LL к точным позициям через Herbig-якоря + перезапуск T1–T3.

Механизм: (1) по выровненным точным парам (§9.1) для каждой колонки
Herbig оценивается оффсет (позиция − строка; медиана; колонки с ≥3
парами); (2) интервальная строка F&W с кандидатом (col, line) J≥0.25
получает предсказанную позицию offset_col + line и ПРОДВИГАЕТСЯ, только
если предсказание попадает в её страничный интервал (двойная проверка
независимыми источниками: Herbig-координата × F&W-страница) и слот не
занят точной строкой; (3) results/ll_scroll_map_v2.csv = точные +
продвинутые (kind='herbig'); (4) T1–T3 §10 повторяются на v2 БЕЗ
изменения правил (тот же CAL-критерий, те же нули, seed=42).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ll_map2.py
"""
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 10000
BOUNDS = (111.5, 230.5)
PAIRS = [('VACL', 'THEO'), ('VACL', 'OFFER'), ('THEO', 'OFFER'),
         ('VERB-R', 'THEO'), ('VERB-R', 'VACL'), ('VERB-R', 'OFFER')]
CLASSES = ['CAL', 'VACL', 'THEO', 'OFFER', 'VERB-R', 'CONJ']
OUT_LOG = os.path.join('logs', 'etr_ll_map2.log')
OUT_CSV = os.path.join('results', 'll_scroll_map_v2.csv')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log('=== §16: карта свитка v2 (Herbig-якоря) + T1–T3 v2 ===')
    exact, interval = [], []
    with open(os.path.join('results', 'll_scroll_map.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            cls = set(r['classes'].split()) - {'—'}
            if r['kind'] == 'exact':
                exact.append((int(r['position']), r['key'], cls, r['text']))
            else:
                a, b = r['position'].split('–')
                interval.append((int(a), int(b), r['key'], cls, r['text']))
    aln_exact, aln_int = {}, {}
    with open(os.path.join('results', 'herbig_fw_alignment.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'exact':
                aln_exact[int(r['fw_position'])] = (
                    r['herbig_col'], int(r['herbig_line']),
                    float(r['jaccard']))
            else:
                aln_int[r['fw_key']] = (r['herbig_col'],
                                        int(r['herbig_line']),
                                        float(r['jaccard']))
    log(f'точных строк: {len(exact)}; интервальных: {len(interval)}; '
        f'якорей: exact {len(aln_exact)}, interval {len(aln_int)}')

    # --- оффсеты колонок -----------------------------------------------------
    col_off = defaultdict(list)
    for pos, (col, line, j) in aln_exact.items():
        col_off[col].append(pos - line)
    offsets = {c: int(np.median(v)) for c, v in col_off.items()}
    log('оффсеты колонок (медиана; в скобках — число пар, 1-парные '
        'страхуются интервальной проверкой): '
        + ', '.join(f'{c}:{o}({len(col_off[c])})' for c, o in sorted(
            offsets.items(), key=lambda kv: kv[1])))
    # геометрическая интерполяция отсутствующих оффсетов: цепочка от
    # соседней якорной колонки по максимальному номеру строки Herbig
    # (неопределённость ±3-4 позиции; строки получат метку 'geo' и
    # обязаны попадать в свой F&W-интервал)
    col_order = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
                 'IX', 'X', 'XI', 'XII']
    maxline = defaultdict(int)
    with open(os.path.join('data', 'external', 'cie_online',
                           'herbig_ll_index.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['col'] in col_order and 'gamma' not in r['flags']:
                maxline[r['col']] = max(maxline[r['col']], int(r['line']))
    geo = {}
    for i, c in enumerate(col_order):
        if c in offsets:
            continue
        if i > 0 and col_order[i - 1] in offsets:
            prev = col_order[i - 1]
            geo[c] = offsets[prev] + maxline[prev]
        elif i + 1 < len(col_order) and col_order[i + 1] in offsets:
            geo[c] = offsets[col_order[i + 1]] - maxline[c]
    # второй проход (цепочка через geo)
    for i, c in enumerate(col_order):
        if c in offsets or c in geo:
            continue
        if i > 0 and col_order[i - 1] in geo:
            prev = col_order[i - 1]
            geo[c] = geo[prev] + maxline[prev]
    log('geo-оффсеты (интерполяция, ±3-4): '
        + ', '.join(f'{c}:{o}' for c, o in sorted(
            geo.items(), key=lambda kv: kv[1])) if geo else 'geo: нет')
    geo_cols = set(geo)
    offsets.update(geo)

    # --- продвижение интервальных строк --------------------------------------
    taken = {p for p, *_ in exact}
    promoted = []
    n_out = n_nocol = n_taken = 0
    for a, b, key, cls, text in interval:
        cand = aln_int.get(key)
        if not cand:
            continue
        col, line, j = cand
        if col not in offsets:
            n_nocol += 1
            continue
        pred = offsets[col] + line
        if not (a <= pred <= b):
            n_out += 1
            continue
        if pred in taken:
            n_taken += 1
            continue
        taken.add(pred)
        promoted.append((pred, key, cls, text,
                         'herbig-geo' if col in geo_cols else 'herbig'))
    n_geo = sum(1 for *_, kd in promoted if kd == 'herbig-geo')
    log(f'кандидатов с якорем: {len(aln_int)}; продвинуто: '
        f'{len(promoted)} (из них geo: {n_geo}; вне интервала: {n_out}, '
        f'нет оффсета колонки: {n_nocol}, слот занят: {n_taken})')

    rows2 = sorted([(p, 'exact', k, c, t) for p, k, c, t in exact]
                   + [(p, kd, k, c, t) for p, k, c, t, kd in promoted])
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['position', 'kind', 'key', 'classes', 'text'])
        for p, kd, k, c, t in rows2:
            wr.writerow([p, kd, k, ' '.join(sorted(c)) or '—', t])
    log(f'карта v2 записана: {OUT_CSV} ({len(rows2)} строк)')

    # --- T1–T3 v2 (правила §10 без изменений) --------------------------------
    pos = np.array([p for p, *_ in rows2])
    cls_sets = [c for _, _, _, c, _ in rows2]
    n = len(rows2)
    log(f'\nстрок в v2: {n}; классы: '
        + ', '.join(f'{c}:{sum(1 for s in cls_sets if c in s)}'
                    for c in CLASSES))
    rng = np.random.default_rng(SEED)
    cal_idx = np.array([i for i, s in enumerate(cls_sets) if 'CAL' in s])
    runs = [i for j, i in enumerate(cal_idx)
            if j == 0 or cal_idx[j - 1] != i - 1]
    log(f'CAL-строк: {len(cal_idx)}; CAL-серий: {len(runs)}')
    cal_pos = pos[cal_idx]
    run_pos = pos[np.array(runs)]

    def sumdist(marks):
        return sum(min(abs(m - b) for m in marks) for b in BOUNDS)

    obs_a, obs_b = sumdist(cal_pos), sumdist(run_pos)
    log(f'\n--- T1 v2 ---')
    for b in BOUNDS:
        log(f'  граница {b}: ближайшая CAL на {min(abs(m - b) for m in cal_pos)}')
    sims_a = np.zeros(R)
    sims_b = np.zeros(R)
    for i in range(R):
        sims_a[i] = sumdist(pos[rng.choice(n, size=len(cal_idx),
                                           replace=False)])
        sims_b[i] = sumdist(pos[rng.choice(n, size=len(runs),
                                           replace=False)])
    log(f'T1 нуль A: obs={obs_a:.1f}, нуль {sims_a.mean():.1f}'
        f'±{sims_a.std():.1f}, p={((sims_a <= obs_a).sum() + 1) / (R + 1):.4f}')
    log(f'T1 нуль B: obs={obs_b:.1f}, нуль {sims_b.mean():.1f}'
        f'±{sims_b.std():.1f}, p={((sims_b <= obs_b).sum() + 1) / (R + 1):.4f}')

    sec_id = np.zeros(n, int)
    cur = 0
    run_set = set(runs)
    for i in range(n):
        if i in run_set and i > 0:
            cur += 1
        sec_id[i] = cur
    n_sec = cur + 1
    sizes = Counter(sec_id.tolist())
    log(f'секций: {n_sec}; медиана {int(np.median(list(sizes.values())))}, '
        f'макс {max(sizes.values())}')

    def pair_stats(cs):
        out = []
        for a, b in PAIRS:
            sa = {sec_id[i] for i, s in enumerate(cs) if a in s}
            sb = {sec_id[i] for i, s in enumerate(cs) if b in s}
            out.append(len(sa & sb))
        return np.array(out, float)

    def order_stats(cs):
        relpos = np.zeros(n)
        for s in range(n_sec):
            idx = np.where(sec_id == s)[0]
            relpos[idx] = (np.linspace(0, 1, len(idx))
                           if len(idx) >= 2 else 0.5)
        out = []
        for c in CLASSES[1:]:
            xs = [relpos[i] for i, st in enumerate(cs) if c in st
                  and sizes[sec_id[i]] >= 3]
            out.append(np.mean(xs) if xs else np.nan)
        return np.array(out, float)

    obs_p = pair_stats(cls_sets)
    obs_o = order_stats(cls_sets)
    ge_p = np.zeros(len(PAIRS))
    any_hit = 0
    lo_o = np.zeros(len(obs_o))
    hi_o = np.zeros(len(obs_o))
    for i in range(R):
        sh = int(rng.integers(1, n))
        cs = cls_sets[sh:] + cls_sets[:sh]
        ps = pair_stats(cs)
        ge_p += ps >= obs_p
        any_hit += (ps >= obs_p).any()
        os_ = order_stats(cs)
        lo_o += np.where(np.isnan(os_) | np.isnan(obs_o), 0, os_ <= obs_o)
        hi_o += np.where(np.isnan(os_) | np.isnan(obs_o), 0, os_ >= obs_o)
    log('\n--- T2 v2 (совместный нуль сдвигов) ---')
    for (a, b), o, g in zip(PAIRS, obs_p, ge_p):
        log(f'  {a}+{b:<8} вместе: {int(o):>2}; raw p={(g + 1) / (R + 1):.4f}')
    log(f'  семейный min-p: p̃<={(any_hit + 1) / (R + 1):.4f}')
    log('\n--- T3 v2 ---')
    for c, o, l_, h_ in zip(CLASSES[1:], obs_o, lo_o, hi_o):
        if np.isnan(o):
            log(f'  {c:<7} — нет вхождений')
        else:
            log(f'  {c:<7} позиция {o:.3f}; p(раньше)={(l_ + 1) / (R + 1):.4f}, '
                f'p(позже)={(h_ + 1) / (R + 1):.4f} (raw)')
    log('\nчтение: правила §10 без изменений; выигрыш мощности — от '
        'продвинутых строк. Продвижение валидировано двойным источником '
        '(Herbig-координата × страничный интервал F&W).')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
