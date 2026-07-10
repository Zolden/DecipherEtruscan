# -*- coding: utf-8 -*-
"""§7: концепт-лексикон эпохи → фонетический зонд заимствований (Wanderwörter).

Идея (владельца проекта): понятия, заведомо существовавшие в эпоху
бронзы/железа, собраны с их формами в РАСШИФРОВАННЫХ языках региона
(data/concepts/concept_lexicon.csv — переносимая структура, будет
использована и в Linear A). Понятия, чьи формы КОНВЕРГИРУЮТ между
языками (бродячие культурные слова), — зонд: их звуко-классовые скелеты
ищутся в этрусском словаре. Результат — «туман вероятного смысла»
(results/concept_fog_v1.csv) с калибровкой на известных словах.

Дисциплина против ловушки массового сравнения:
  1. Скелеты звуко-классов (P T K S M N L R W J H; гласные вон) —
     грубые, зато честные к межъязыковой записи.
  2. Конвергенция понятия = число пар языков со сходством скелетов
     ≥0.7; нуль — перестановка форм внутри каждого языка между
     понятиями (R=200) → p по пулу.
  3. ВАЛИДАЦИЯ НА ИЗВЕСТНОМ: слова с глоссами (Хилл/ETP_POS) — метрика
     hit@1/hit@5 «верхний концепт совпал с известным значением» против
     перестановки глосс (R=1000). Позитивные контроли: известные
     грецизмы vinum/qutum/aska…; негативные: исконные clan/avil/śuθi.
  4. Туман публикуется с оценкой точности из валидации, не сверх неё.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_concepts.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R_CONV = 200
R_VAL = 1000
SIM_PAIR = 0.7
SIM_ETR = 0.75
MIN_SK = 3  # минимальная длина скелета с обеих сторон (против ложняков)
OUT_LOG = os.path.join('logs', 'etr_concepts.log')
OUT_CSV = os.path.join('results', 'concept_fog_v1.csv')
LEX = os.path.join('data', 'concepts', 'concept_lexicon.csv')
LANGS = ['grc', 'lat', 'hit', 'akk', 'heb', 'egy', 'sum']
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


DIGRAPH = {'sh': 'S', 'th': 'T', 'kh': 'K', 'ch': 'K', 'ph': 'P',
           'ts': 'S', 'dj': 'S', 'tj': 'S', 'ng': 'N'}
# Примечание (N21): мягкий назальный класс m=n ПРОВЕРЕН И ОТКЛОНЁН —
# чинит qutum<kothon, но ломает негативный контроль (clan~caelum sim=1.0)
# и ухудшает hit@5 (p 0.007->0.033); классы остаются v1.
SINGLE = {'p': 'P', 'b': 'P', 'f': 'P', 'v': 'W', 'w': 'W',
          't': 'T', 'd': 'T', 'k': 'K', 'g': 'K', 'q': 'K', 'c': 'K',
          'x': 'K', 's': 'S', 'z': 'S', 'm': 'M', 'n': 'N', 'l': 'L',
          'r': 'R', 'j': 'J', 'y': 'J', 'h': 'H'}


def skeleton(form):
    f = form.lower().strip().strip('?')
    f = re.sub(r'[^a-z]', '', f)
    out = []
    i = 0
    while i < len(f):
        if f[i:i + 2] in DIGRAPH:
            out.append(DIGRAPH[f[i:i + 2]])
            i += 2
            continue
        c = SINGLE.get(f[i])
        if c:
            out.append(c)
        i += 1
    return ''.join(out)


def lev(a, b):
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a, b):
    if not a or not b:
        return 0.0
    return 1.0 - lev(a, b) / max(len(a), len(b))


CASE_ENDS = ('isa', 'ial', 'al', 'us', 'sa', 's', 'l', 'm', 'c')


def stems_of(w):
    out = {w}
    for e in CASE_ENDS:
        if w.endswith(e) and len(w) - len(e) >= 3:
            out.add(w[:-len(e)])
    return out


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = list(csv.DictReader(open(LEX, encoding='utf-8')))
    log('=== §7: концепт-лексикон и фонетический зонд ===')
    n_forms = 0
    concepts = []
    for r in rows:
        forms = {}
        for lg in LANGS:
            cell = (r.get(lg) or '').strip()
            fs = []
            for v in cell.split('/'):
                v = v.strip()
                if not v:
                    continue
                sk = skeleton(v)
                if len(sk) >= 2:
                    fs.append((v, sk, v.endswith('?')))
                    n_forms += 1
            if fs:
                forms[lg] = fs
        concepts.append({'id': r['id'], 'en': r['gloss_en'],
                         'ru': r['gloss_ru'], 'dom': r['domain'],
                         'forms': forms})
    log(f'понятий: {len(concepts)}; форм со скелетами ≥2: {n_forms}; '
        f'языков: {len(LANGS)}')

    # --- 1. конвергенция понятий -------------------------------------------
    def conv_score(forms):
        s = 0
        lgs = sorted(forms)
        for i in range(len(lgs)):
            for j in range(i + 1, len(lgs)):
                best = max((sim(a[1], b[1]) for a in forms[lgs[i]]
                            for b in forms[lgs[j]]), default=0)
                s += best >= SIM_PAIR
        return s

    obs = [conv_score(c['forms']) for c in concepts]
    # нуль: перестановка форм каждого языка между понятиями
    cols = {lg: [c['forms'].get(lg) for c in concepts] for lg in LANGS}
    pool = []
    for _ in range(R_CONV):
        sh = {lg: [cols[lg][i] for i in rng.permutation(len(concepts))]
              for lg in LANGS}
        for ci in range(len(concepts)):
            f = {lg: sh[lg][ci] for lg in LANGS if sh[lg][ci]}
            pool.append(conv_score(f))
    pool = np.array(pool)
    p_conv = [(float(((pool >= o).sum() + 1) / (len(pool) + 1)), o)
              for o in obs]
    conv_set = [c for c, (p, o) in zip(concepts, p_conv)
                if o >= 2 and p < 0.05]
    log(f'нуль конвергенции: пул {len(pool)} значений, средний счёт '
        f'{pool.mean():.2f}; конвергентных понятий (счёт≥2, p<0.05): '
        f'{len(conv_set)}')
    top = sorted(zip(concepts, p_conv), key=lambda x: (-x[1][1], x[1][0]))
    log('топ конвергентных (Wanderwort-кандидаты):')
    for c, (p, o) in top[:15]:
        log(f'  {c["en"]:<16} пар≥{SIM_PAIR}: {o}, p={p:.4f}')

    # --- 2. зонд на этрусском ----------------------------------------------
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.7'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    freq = Counter(t['ascii'] for r in view for t in r['toks']
                   if t['kind'] == 'W')
    vocab = sorted(w for w in freq if '-' not in w and len(w) >= 3)
    # глоссы известных слов
    gloss = {}
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        if r['trs'] and len(ws) == 1 and ws[0] in freq:
            gloss.setdefault(ws[0], set()).add(' '.join(r['trs']).lower())
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = re.sub(r'[^a-zθχφσςśšê\']', '',
                       (row.get('Etruscan') or '').lower())
            w = ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's',
                         'ς': 's', 'ś': 's', 'š': 's', 'ê': 'e',
                         "'": ''}.get(c, c) for c in w)
            trs = re.findall(r"'([^']+)'", row.get('Translations') or '')
            if w in freq and trs:
                gloss.setdefault(w, set()).update(t.lower() for t in trs)
    log(f'этрусских типов: {len(vocab)}; с известными глоссами: '
        f'{len(gloss)}')

    sk_cache = {w: [skeleton(s) for s in stems_of(w)] for w in vocab}
    # ярус A: конвергентные понятия (все языки); ярус B: контактный канал —
    # греческие/латинские формы ВСЕХ понятий (исторически вероятные доноры
    # этрусского: qutum<kothon, culichna<kylix и т.п.)
    conv_forms = []
    seen_f = set()
    for c in conv_set:
        for lg, fs in c['forms'].items():
            for v, sk, unc in fs:
                key = (c['id'], lg, v)
                if key not in seen_f:
                    seen_f.add(key)
                    conv_forms.append((c, lg, v, sk, 'A'))
    for c in concepts:
        for lg in ('grc', 'lat'):
            for v, sk, unc in c['forms'].get(lg, []):
                key = (c['id'], lg, v)
                if key not in seen_f:
                    seen_f.add(key)
                    conv_forms.append((c, lg, v, sk, 'B'))
    nA = sum(1 for x in conv_forms if x[4] == 'A')
    log(f'форм в зонде: {len(conv_forms)} (ярус A конвергентный: {nA}; '
        f'ярус B контактный grc/lat: {len(conv_forms) - nA})')

    def best_for(w):
        best = (0.0, None, None, None, '')
        for sk_w in sk_cache[w]:
            if len(sk_w) < MIN_SK:
                continue
            for c, lg, v, sk, tier in conv_forms:
                if len(sk) < MIN_SK:
                    continue
                s = sim(sk_w, sk)
                if s > best[0]:
                    best = (s, c, lg, v, tier)
        return best

    # --- 3. валидация на известных словах -----------------------------------
    log()
    log('--- валидация: известные слова (hit@1/hit@5 против перестановки) ---')
    known = sorted(w for w in gloss if w in sk_cache)
    top5 = {}
    for w in known:
        scored = []
        for sk_w in sk_cache[w]:
            if len(sk_w) < MIN_SK:
                continue
            for c, lg, v, sk, tier in conv_forms:
                if len(sk) < MIN_SK:
                    continue
                s = sim(sk_w, sk)
                if s >= SIM_ETR:
                    scored.append((s, c['en'], c['ru']))
        scored.sort(reverse=True)
        seen = []
        for s, en, ru in scored:
            if en not in [e for _, e, _ in seen]:
                seen.append((s, en, ru))
            if len(seen) == 5:
                break
        top5[w] = seen

    def gloss_hit(cen, cru, gset):
        keys = set(re.findall(r'[a-z]{3,}', cen.lower()))
        return any(k in g for g in gset for k in keys)

    def eval_hits(gmap):
        h1 = h5 = n = 0
        for w in known:
            if not top5[w]:
                continue
            n += 1
            hits = [gloss_hit(en, ru, gmap[w]) for _, en, ru in top5[w]]
            h1 += hits[0]
            h5 += any(hits)
        return h1, h5, n

    h1, h5, n_eval = eval_hits(gloss)
    sims_h1 = np.zeros(R_VAL)
    sims_h5 = np.zeros(R_VAL)
    gvals = [gloss[w] for w in known]
    for r_i in range(R_VAL):
        perm = rng.permutation(len(known))
        gmap = {w: gvals[perm[i]] for i, w in enumerate(known)}
        a, b, _ = eval_hits(gmap)
        sims_h1[r_i] = a
        sims_h5[r_i] = b
    p1 = float(((sims_h1 >= h1).sum() + 1) / (R_VAL + 1))
    p5 = float(((sims_h5 >= h5).sum() + 1) / (R_VAL + 1))
    log(f'известных слов с кандидатами (sim≥{SIM_ETR}): {n_eval}')
    log(f'hit@1: {h1} (нуль {sims_h1.mean():.1f}±{sims_h1.std():.1f}), '
        f'p={p1:.4f}')
    log(f'hit@5: {h5} (нуль {sims_h5.mean():.1f}±{sims_h5.std():.1f}), '
        f'p={p5:.4f}')
    log('контроли:')
    for w in ['vinum', 'qutum', 'aska', 'culichna', 'pruchum',
              'clan', 'avil', 'suthi', 'puia', 'tular']:
        if w in top5 and top5[w]:
            s, en, ru = top5[w][0]
            g = sorted(gloss.get(w, ['—']))[0][:30]
            log(f'  {w:<10} глосса={g!r:<32} топ-концепт: {en} '
                f'({ru}) sim={s:.2f}')
        elif w in freq:
            log(f'  {w:<10} кандидатов ≥{SIM_ETR} нет')

    # --- 4. туман для непереведённых ----------------------------------------
    n_fog = 0
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wcsv = csv.writer(f, lineterminator='\n')
        wcsv.writerow(['word', 'freq', 'known_gloss', 'concept_en',
                       'concept_ru', 'domain', 'lang', 'form', 'sim',
                       'tier'])
        for w in vocab:
            s, c, lg, v, tier = best_for(w)
            if s >= SIM_ETR and c is not None:
                g = sorted(gloss[w])[0][:40] if w in gloss else ''
                wcsv.writerow([w, freq[w], g, c['en'], c['ru'], c['dom'],
                               lg, v, f'{s:.3f}', tier])
                n_fog += 1
    log()
    log(f'туман записан: {OUT_CSV} — {n_fog} слов с sim≥{SIM_ETR}; '
        f'точность слоя оценивать по hit@1 валидации выше, не выше её')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
