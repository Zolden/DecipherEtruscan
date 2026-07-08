# -*- coding: utf-8 -*-
"""Этап 2 (§2): калибровка морфем — суффиксы с нуль-моделями и репликацией.

Вход: data/etr_corpus.pkl (v0.3). Вид: lang=etr, kind=text, без forgery?,
без variant_of. Словарь типов — ascii-формы словоформ (ярус A, весь
корпус; сибилянты слиты проекцией) и богатые формы ETP-части (ярус B,
различия σ/ś/š сохранены).

Тесты:
  1. СКРИНИНГ суффиксов длины 1–3 (все концовки с n_типов ≥ 15).
     Статистика «поддержка парадигмы»: число типов w с суффиксом s, чья
     основа w[:-k] сама есть в словаре. Нуль — позиционная перестановка
     хвостов длины k среди всех типов длины ≥ k+2; при фиксированных
     основах это даёт ТОЧНЫЙ гипергеометрический нуль:
     support ~ Hypergeom(N=типов, K=типов с основой-в-словаре, n=типов
     с данным хвостом). p односторонние; семейная поправка — Бонферрони
     по числу протестированных концовок (консервативно).
  2. РЕПЛИКАЦИЯ топ-суффиксов по стратам: части ETP/CIEP, топ-регионы,
     века (датированная ETP-часть; границы -550 и -300).
  3. Ярус B: раздельные сибилянтные генитивы -s vs -σ/-ś (ETP-часть).
  4. ПАРАДИГМЫ: основы, засвидетельствованные голыми и с ≥2 разными
     значимыми суффиксами (разведочная карта, без теста).
  5. СЕМАНТИКА МОРФЕМ по глоссам (одно-словные переведённые записи):
     (а) -al/-s ↔ генитивная семантика глоссы («of-…», «…s’»);
     (б) -ce ↔ глагольная глосса (англ. прошедшее);
     (в) пол имени: mr-/mrs-глоссы Хилла ↔ концовки слова.
     Нуль — перестановка глосс между записями, R=10000, seed=42,
     семейный контроль Westfall–Young внутри каждого блока.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_morphemes.py
"""
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
from scipy.stats import hypergeom

sys.stdout.reconfigure(encoding='utf-8')

R = 10000
SEED = 42
MIN_TYPES = 15
OUT_LOG = os.path.join('logs', 'etr_morphemes.log')
OUT_CSV = os.path.join('results', 'morphemes_v1.csv')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def view_of(corpus):
    return [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]


def clean_types(counter):
    """Типы без повреждений и коротышек."""
    return {w for w in counter if '-' not in w and len(w) >= 2}


def suffix_test(vocab, k, min_types=MIN_TYPES):
    """Все концовки длины k: (suffix, n_s, support, E, p_hypergeom)."""
    elig = [w for w in vocab if len(w) >= k + 2]
    N = len(elig)
    stem_in = np.array([w[:-k] in vocab for w in elig])
    K = int(stem_in.sum())
    ends = Counter(w[-k:] for w in elig)
    out = []
    for s, n_s in ends.items():
        if n_s < min_types:
            continue
        idx = [i for i, w in enumerate(elig) if w.endswith(s)]
        supp = int(stem_in[idx].sum())
        E = n_s * K / N
        p = float(hypergeom.sf(supp - 1, N, K, n_s))
        out.append((s, n_s, supp, E, p))
    return out, N, K


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    view = view_of(corpus)
    log('=== Калибровка морфем (этап 2) ===')
    log(f'вид: {len(view)} записей; R={R}, seed={SEED}; '
        f'порог скрининга: n_типов ≥ {MIN_TYPES}')

    # словари
    cntA = Counter(t['ascii'] for r in view for t in r['toks']
                   if t['kind'] == 'W')
    vocabA = clean_types(cntA)
    etp_recs = [r for r in view if r['src'] == 'ETP']
    cntB = Counter(t['form'] for r in etp_recs for t in r['toks']
                   if t['kind'] == 'W')
    vocabB = clean_types(cntB)
    log(f'ярус A (весь корпус, ascii): {len(vocabA)} типов; '
        f'ярус B (ETP, богатые формы): {len(vocabB)} типов')

    # --- 1. скрининг суффиксов -------------------------------------------
    log()
    log('--- 1. скрининг концовок длины 1–3 (ярус A) ---')
    all_tests = []
    for k in (1, 2, 3):
        res, N, K = suffix_test(vocabA, k)
        log(f'k={k}: подходящих типов N={N}, с основой-в-словаре K={K} '
            f'({K / N:.0%}); протестировано концовок: {len(res)}')
        all_tests += [(k,) + t for t in res]
    m = len(all_tests)
    log(f'семейство: m={m} тестов; поправка Бонферрони')
    all_tests.sort(key=lambda t: (t[5], -t[3]))
    log(f'{"суфф.":<6} {"k":>2} {"типов":>6} {"поддержка":>9} {"E[под.]":>8} '
        f'{"обог.":>6} {"p":>10} {"p_adj":>10}')
    rows_csv = []
    sig = []
    for k, s, n_s, supp, E, p in all_tests[:28]:
        padj = min(1.0, p * m)
        mark = ' *' if padj < 0.05 else ''
        log(f'-{s:<5} {k:>2} {n_s:>6} {supp:>9} {E:>8.1f} '
            f'{supp / max(E, 1e-9):>6.1f} {p:>10.2e} {padj:>10.2e}{mark}')
        rows_csv.append({'suffix': s, 'k': k, 'n_types': n_s,
                         'support': supp, 'expected': f'{E:.2f}',
                         'p': f'{p:.3e}', 'p_adj': f'{padj:.3e}'})
        if padj < 0.05:
            sig.append(s)
    log('* — значимо после Бонферрони (α=0.05)')

    # --- 2. репликация по стратам ----------------------------------------
    log()
    log('--- 2. репликация топ-суффиксов по стратам ---')
    TOP = [s for s in ['s', 'l', 'al', 'ial', 'us', 'sa', 'ce', 'thi',
                       'isa', 'la', 'na', 'ei'] ]
    strata = {'ETP': etp_recs,
              'CIEP': [r for r in view if r['src'] == 'CIEP']}
    for reg, _ in Counter(r['region'] for r in view
                          if r['region']).most_common(5):
        strata[f'рег.{reg}'] = [r for r in view if r['region'] == reg]
    # века: датированная ETP-часть
    dated = [r for r in etp_recs if r['y_from'] is not None]
    strata['до -550'] = [r for r in dated if r['y_from'] >= 550]
    strata['-550…-300'] = [r for r in dated if 550 > r['y_from'] >= 300]
    strata['после -300'] = [r for r in dated if r['y_from'] < 300]
    log(f'(даты есть у {len(dated)} записей ETP-части)')
    hdr = f'{"суфф.":<6}' + ''.join(f'{k:>12}' for k in strata)
    log(hdr)
    for s in TOP:
        k = len(s)
        cells = []
        for recs in strata.values():
            v = clean_types(Counter(t['ascii'] for r in recs
                                    for t in r['toks'] if t['kind'] == 'W'))
            elig = [w for w in v if len(w) >= k + 2]
            N = len(elig)
            if N < 50:
                cells.append(f'{"–":>12}')
                continue
            Kk = sum(1 for w in elig if w[:-k] in v)
            n_s = sum(1 for w in elig if w.endswith(s))
            supp = sum(1 for w in elig if w.endswith(s) and w[:-k] in v)
            if n_s < 5:
                cells.append(f'{"–":>12}')
                continue
            p = float(hypergeom.sf(supp - 1, N, Kk, n_s))
            star = '*' if p < 0.05 else ' '
            cells.append(f'{supp:>4}/{n_s:<4}{star:>1}{p:>5.0e}'[:12].rjust(12))
        log(f'-{s:<5}' + ''.join(cells))
    log('в ячейке: поддержка/типов, * — p<0.05 (без семейной поправки — '
        'репликация знака, не открытие)')

    # --- 3. ярус B: сибилянтные генитивы ----------------------------------
    log()
    log('--- 3. ярус B (ETP, богатые формы): -s vs -σ/-ś/-š ---')
    for s_chars in [('s',), ('σ', 'ś', 'š', 'ς')]:
        n_s = supp = 0
        elig = [w for w in vocabB if len(w) >= 3]
        N = len(elig)
        Kk = sum(1 for w in elig if w[:-1] in vocabB)
        for w in elig:
            if w[-1] in s_chars:
                n_s += 1
                if w[:-1] in vocabB:
                    supp += 1
        if n_s:
            p = float(hypergeom.sf(supp - 1, N, Kk, n_s))
            log(f'  конечные {"/".join(s_chars):<8}: типов {n_s:>4}, '
                f'поддержка {supp:>3} (E={n_s * Kk / N:.1f}), p={p:.2e}')

    # --- 4a. парный тест парадигм ------------------------------------------
    # Правильный тест падежных суффиксов: одна ОСНОВА с двумя разными
    # суффиксами (velthina-s / velthina-l). Для пары (s1, s2):
    # A = основы, засвидетельствованные с s1; B = с s2; универсум U —
    # основы, извлекаемые снятием любого кандидатного суффикса.
    # Нуль: A и B — случайные подмножества U → |A∩B| ~ Hypergeom.
    log()
    log('--- 4a. парный тест парадигм (основа с двумя суффиксами) ---')
    CAND = ['s', 'l', 'al', 'ial', 'us', 'sa', 'isa', 'la', 'na', 'ce',
            'thi', 'c', 'm', 'ei']
    stems_of = {}
    for s in CAND:
        stems_of[s] = {w[:-len(s)] for w in vocabA
                       if w.endswith(s) and len(w) >= len(s) + 3}
    U = set()
    for s in CAND:
        U |= stems_of[s]
    NU = len(U)
    pair_res = []
    for i in range(len(CAND)):
        for j in range(i + 1, len(CAND)):
            A, B = stems_of[CAND[i]], stems_of[CAND[j]]
            ov = len(A & B)
            if not A or not B:
                continue
            E = len(A) * len(B) / NU
            p_hi = float(hypergeom.sf(ov - 1, NU, len(A), len(B)))
            p_lo = float(hypergeom.cdf(ov, NU, len(A), len(B)))
            pair_res.append((CAND[i], CAND[j], len(A), len(B), ov, E,
                             p_hi, p_lo))
    m_pairs = len(pair_res)
    log(f'универсум основ |U|={NU}; пар протестировано: {m_pairs} '
        f'(двусторонне; Бонферрони ×{2 * m_pairs})')
    log('ОБОГАЩЕНИЕ (общая основа с двумя суффиксами = падежная парадигма):')
    log(f'{"пара":<10} {"|A|":>5} {"|B|":>5} {"∩":>4} {"E[∩]":>7} '
        f'{"p":>10} {"p_adj":>10}')
    sig_pairs = []
    for s1, s2, a, b, ov, E, p_hi, p_lo in sorted(pair_res,
                                                  key=lambda t: t[6])[:8]:
        padj = min(1.0, p_hi * 2 * m_pairs)
        mark = ' *' if padj < 0.05 else ''
        log(f'-{s1}/-{s2:<6} {a:>5} {b:>5} {ov:>4} {E:>7.1f} {p_hi:>10.2e} '
            f'{padj:>10.2e}{mark}')
        if padj < 0.05:
            sig_pairs.append((s1, s2))
    log('ДЕПРЕССИЯ (комплементарное распределение — основа берёт только '
        'один из двух):')
    log(f'{"пара":<10} {"|A|":>5} {"|B|":>5} {"∩":>4} {"E[∩]":>7} '
        f'{"p":>10} {"p_adj":>10}')
    depl_pairs = []
    for s1, s2, a, b, ov, E, p_hi, p_lo in sorted(pair_res,
                                                  key=lambda t: t[7])[:10]:
        padj = min(1.0, p_lo * 2 * m_pairs)
        mark = ' *' if padj < 0.05 else ''
        log(f'-{s1}/-{s2:<6} {a:>5} {b:>5} {ov:>4} {E:>7.1f} {p_lo:>10.2e} '
            f'{padj:>10.2e}{mark}')
        if padj < 0.05:
            depl_pairs.append((s1, s2))
    sig_sufs = sorted({s for pr in sig_pairs for s in pr})
    log(f'значимых пар-парадигм: {len(sig_pairs)}; значимо комплементарных: '
        f'{len(depl_pairs)} → {depl_pairs}')

    # --- 4b. карта парадигм на значимых суффиксах --------------------------
    log()
    log('--- 4b. парадигмы: основа + ≥2 суффикса из значимых пар ---')
    stems = {}
    for w in vocabA:
        for s in sig_sufs:
            if w.endswith(s) and len(w) >= len(s) + 3:
                stems.setdefault(w[:-len(s)], set()).add(s)
    para = {st: sufs for st, sufs in stems.items() if len(sufs) >= 2}
    log(f'основ с ≥2 суффиксами: {len(para)}; с ≥3: '
        f'{sum(1 for v in para.values() if len(v) >= 3)}; с голой формой '
        f'вдобавок: {sum(1 for st in para if st in vocabA)}')
    top_para = sorted(para.items(), key=lambda x: (-len(x[1]), x[0]))[:14]
    for st, sufs in top_para:
        bare = ' (+голая)' if st in vocabA else ''
        log(f'  {st:<12} + {sorted(sufs)}{bare}')

    # --- 5. семантика морфем по глоссам ------------------------------------
    log()
    log('--- 5. семантика морфем по глоссам (одно-словные переведённые '
        'записи) ---')
    singles = [(r, r['toks'][0]['ascii'], ' '.join(r['trs']).lower())
               for r in view
               if r['trs'] and sum(t['kind'] == 'W' for t in r['toks']) == 1
               and '-' not in r['toks'][0]['ascii']]
    # берём первый W-токен как слово записи
    singles = [(w, g) for r, w, g in singles if len(w) >= 3]
    log(f'одно-словных переведённых записей: {len(singles)}')
    rng = np.random.default_rng(SEED)
    glosses = [g for _, g in singles]
    perm = np.argsort(rng.random((R, len(singles))), axis=1)

    def assoc(word_pred, gloss_pat, label):
        has_w = np.array([word_pred(w) for w, _ in singles])
        pat = re.compile(gloss_pat)
        has_g = np.array([bool(pat.search(g)) for g in glosses])
        n_w, n_g = int(has_w.sum()), int(has_g.sum())
        both = int((has_w & has_g).sum())
        gi = has_g.astype(np.int8)
        wi = np.where(has_w)[0]
        sim = gi[perm[:, wi]].sum(axis=1)
        p = ((sim >= both).sum() + 1) / (R + 1)
        base = n_g / len(singles)
        log(f'  {label:<28} n_w={n_w:>4} оба={both:>4} '
            f'точн.={both / max(n_w, 1):>4.0%} база={base:>4.0%} p={p:.4f}')
        return p

    GEN_G = r'^of-|\bof\b|’s|\'s'
    ps = []
    ps.append(assoc(lambda w: w.endswith('al'), GEN_G, '-al ↔ генитив глоссы'))
    ps.append(assoc(lambda w: w.endswith('s') and not w.endswith('es'),
                    GEN_G, '-s ↔ генитив глоссы'))
    ps.append(assoc(lambda w: w.endswith('sa') or w.endswith('isa'),
                    r'wife|husband|\bof\b|^of-', '-(i)sa ↔ посессив/супруж.'))
    ps.append(assoc(lambda w: w.endswith('ce'),
                    r'\w+ed\b|\bgave\b|\bbuilt\b|\bwrote\b|\bmade\b',
                    '-ce ↔ глагольная глосса'))
    ps.append(assoc(lambda w: w.endswith('c') and not w.endswith('ce'),
                    r'\band\b', '-c ↔ «and» (энклитика)'))
    log('  (семейная поправка Бонферрони ×5: значимо при p<0.01)')

    # пол имени по Хиллу: mr- vs mrs-/ms-
    log()
    log('--- пол имени (глоссы Хилла mr-/mrs-) ↔ концовки слова ---')
    male = [(w, g) for w, g in singles if g.startswith('mr-')]
    fem = [(w, g) for w, g in singles
           if g.startswith('mrs-') or g.startswith('ms-')]
    log(f'мужских глосс: {len(male)}; женских: {len(fem)}')
    # какие концовки различают пол: скрининг k=1,2 по |разности долей|
    gender_tests = []
    for k in (1, 2):
        ends = Counter(w[-k:] for w, _ in male + fem)
        for e, n in ends.items():
            if n < 15:
                continue
            pm = sum(1 for w, _ in male if w.endswith(e)) / max(len(male), 1)
            pf = sum(1 for w, _ in fem if w.endswith(e)) / max(len(fem), 1)
            gender_tests.append((e, k, pm, pf, abs(pm - pf)))
    gender_tests.sort(key=lambda t: -t[4])
    labels = np.array([0] * len(male) + [1] * len(fem))
    words_mf = [w for w, _ in male + fem]
    rngg = np.random.default_rng(SEED)
    permg = np.argsort(rngg.random((R, len(labels))), axis=1)
    log(f'{"конц.":<6} {"муж%":>6} {"жен%":>6} {"|Δ|":>6} {"p":>8} '
        f'{"p̃_сем":>8}')
    # Westfall–Young по семейству гендерных тестов
    obs_stats = []
    end_masks = []
    for e, k, pm, pf, d in gender_tests[:10]:
        mask = np.array([w.endswith(e) for w in words_mf])
        end_masks.append(mask)
        obs_stats.append(d)
    sim_d = np.zeros((R, len(end_masks)))
    for j, mask in enumerate(end_masks):
        lm = labels[permg]  # R × n перестановки меток
        n_m = (labels == 0).sum()
        n_f = (labels == 1).sum()
        pm_s = ((lm == 0) & mask[None, :]).sum(axis=1) / n_m
        pf_s = ((lm == 1) & mask[None, :]).sum(axis=1) / n_f
        sim_d[:, j] = np.abs(pm_s - pf_s)
    p_raw_g = ((sim_d >= np.array(obs_stats)[None, :]).sum(axis=0) + 1) / (R + 1)
    # min-p по семейству
    p_sim_g = np.zeros_like(sim_d)
    for j in range(sim_d.shape[1]):
        col = sim_d[:, j]
        order = np.sort(col)
        idx = np.searchsorted(order, col, side='left')
        p_sim_g[:, j] = (R - idx + 1) / (R + 1)
    minp_g = p_sim_g.min(axis=1)
    for j, (e, k, pm, pf, d) in enumerate(gender_tests[:10]):
        padj = ((minp_g <= p_raw_g[j]).sum() + 1) / (R + 1)
        log(f'-{e:<5} {pm:>6.0%} {pf:>6.0%} {d:>6.2f} {p_raw_g[j]:>8.4f} '
            f'{padj:>8.4f}')
    log('нуль: перестановка пола между словами; W–Y по 10 тестам')

    # --- сохранение ---------------------------------------------------------
    import csv as _csv
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['suffix', 'k', 'n_types',
                                           'support', 'expected', 'p',
                                           'p_adj'])
        w.writeheader()
        for row in rows_csv:
            w.writerow(row)
    log()
    log(f'таблица суффиксов записана: {OUT_CSV}')
    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
