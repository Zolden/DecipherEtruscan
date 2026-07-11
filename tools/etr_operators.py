# -*- coding: utf-8 -*-
"""Этап 1 (§1): реестр операторов v1 + слот-шаблоны + валидация по переводам.

Вход: data/etr_corpus.pkl (заморозка v0.2), data/ETP_POS.csv (глоссарий
Larth с переводами — для независимого подтверждения глосс реестра).

Аналитический вид: lang=='etr' and kind=='text' and 'forgery?' not in flags.
Сопоставление форм — по ascii-проекции токенов (θ→th и т.д.): она
унифицирует конвенции ETP- и CIEP-частей; варианты написания одного
оператора склеены в группы явным списком OPERATORS.

Статистика:
  1. Позиционные профили операторов (начальная/финальная позиция среди
     словоформ-токенов записи, только записи с ≥2 словоформами).
     Нуль-модель: позиция каждого вхождения равновероятна внутри своей
     записи (перестановка позиций). R=10000, seed=42. p — разведочные
     ОДНОСТОРОННИЕ; семейный контроль — Westfall–Young min-p по всему
     семейству тестов (операторы × {нач, фин}).
  2. Репликация знака позиционного смещения: ETP-часть vs CIEP-часть и
     по топ-регионам (порог n≥5 вхождений в страте).
  3. Слот-шаблоны: токены классифицируются (OP:группа / GEN? по суффиксам
     -s/-l-генитивов / NUM / GAP / X), считается покрытие топ-шаблонами.
  4. Валидация по переводам: точность/полнота ассоциации «оператор в
     тексте ↔ ключевое слово в переводе» на переведённых записях вида;
     нуль — перестановка переводов между записями (R=10000, seed=42),
     семейный контроль Westfall–Young. Плюс мультиметочная
     классификация записей (эпитафия/посвящение/говорящий предмет/
     границы) правилами ПО ТЕКСТУ против золота ПО ПЕРЕВОДУ.
     Примечание к плану: правила фиксированы априори, обучаемых
     параметров нет, поэтому LOO вырождается в прямую оценку на всех
     переведённых записях; строгость обеспечивает перестановочный нуль.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_operators.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R = 10000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_operators.log')
OUT_CSV = os.path.join('results', 'operators_v1.csv')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


# --- Реестр операторов v1 ---------------------------------------------------
# (группа, ascii-формы, глосса, класс, regex ключевых слов перевода | None)
# Глоссы — общепринятые (справочники: Bonfante & Bonfante 2002, The
# Etruscan Language; Wallace 2008, Zikh Rasna); подтверждение по ETP_POS
# считается ниже автоматически. Формы — ascii-проекция (унификация частей).
OPERATORS = [
    ('mi',     ('mi',), 'я (говорящий предмет)', 'дейксис',
     r'\bi\b|\bme\b'),
    ('mini',   ('mini', 'mine'), 'меня', 'дейксис', r'\bme\b'),
    ('clan',   ('clan',), 'сын (ном.)', 'родство', r'\bson\b'),
    ('clens',  ('clens', 'clensi'), 'сына (ген./дат.)', 'родство', r'\bson'),
    ('sec',    ('sec', 'sech'), 'дочь', 'родство', r'\bdaughter'),
    ('puia',   ('puia',), 'жена', 'родство', r'\bwife\b'),
    ('ati',    ('ati',), 'мать', 'родство', r'\bmother\b'),
    ('apa',    ('apa',), 'отец', 'родство', r'\bfather\b'),
    ('lupu',   ('lupu', 'lupuce'), 'умер(ший)', 'глагол-витал',
     r'\bdie[ds]?\b|\bdead\b|\bdeceas'),
    ('svalce', ('svalce', 'svalthas', 'svalas'), 'жил / при жизни',
     'глагол-витал', r'\blive[ds]\b|\bliving\b'),
    ('avils',  ('avils', 'avilxs'), 'лет (ген. возраста)', 'мера-возраст',
     r'\byears?\b|\bage\b|\baged\b'),
    ('avil',   ('avil',), 'год', 'мера-возраст', r'\byears?\b'),
    ('ril',    ('ril',), 'в возрасте', 'мера-возраст',
     r'\bage[d]?\b|\byears?\b'),
    ('turce',  ('turce', 'turuce', 'turice', 'turke'), 'посвятил/дал',
     'глагол-посвящ', r'\bdedicat|\bgave\b|\bgives?\b|\boffer'),
    ('muluvanice', ('muluvanice', 'muluvanike', 'mulvanice', 'mulvenice',
                    'mulvannice', 'muluvenice'), 'подарил/посвятил',
     'глагол-посвящ', r'\bgave\b|\bdedicat|\bdonat|\bpresent'),
    ('zinace', ('zinace', 'zinake'), 'сделал/изготовил', 'глагол-изгот',
     r'\bmade\b|\bmake[rs]?\b|\bfashion'),
    ('zilath', ('zilath', 'zilach', 'zilth', 'zilc', 'zilci', 'zilachnce',
                'zilachnuce'), 'магистрат(-ство); zilaθ', 'титул',
     r'\bzila|\bmagistra|\bpraetor|\bgovernor'),
    ('suthi',  ('suthi', 'suti', 'suthith', 'suthiu'), 'гробница',
     'объект-погреб', r'\btomb\b|\bgrave\b|\bsepulch|\bburial'),
    ('thui',   ('thui',), 'здесь', 'дейксис', r'\bhere\b'),
    ('cver',   ('cver', 'cvera'), 'дар/вотив', 'объект-дар',
     r'\bgift\b|\bvotive'),
    ('tular',  ('tular',), 'границы', 'объект-граница', r'\bboundar'),
    ('spur',   ('spur', 'spurana', 'spureni'), 'город/община (основа)',
     'община', r'\bcity\b|\bcommunit|\bpeople\b|\btown\b'),
    ('ais',    ('ais', 'eis', 'aisar', 'eiser'), 'бог(и)', 'сакрал',
     r'\bgod'),
    ('ame',    ('ame', 'amce'), 'быть (есть/был)', 'связка',
     r'\bwas\b|\bis\b|\bare\b|\bbe\b'),
    ('naper',  ('naper',), 'мера (площади)', 'мера', r'\bnaper|\bmeasur'),
    ('tiur',   ('tiur', 'tiurs'), 'месяц/луна', 'календарь',
     r'\bmonth\b|\bmoon\b'),
    ('itun',   ('itun', 'ita', 'ica', 'eca', 'ca', 'cn', 'cen', 'cehen'),
     'этот/эту (дейктики)', 'дейксис', r'\bthis\b|\bthese\b'),
]
# кандидаты генитивных суффиксов из пилота (для слот-классов, НЕ реестр)
GEN_SUFFIXES = ('al', 'als', 'ials', 'ial', 'us', 'la', 'sa', 'isa', 's')


def view_of(corpus):
    return [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags']
            and r.get('variant_of') is None]


def words_ascii(rec):
    return [t['ascii'] for t in rec['toks'] if t['kind'] == 'W']


def load_etp_pos():
    """ETP_POS.csv: ascii-форма → набор англ. глосс (для подтверждения)."""
    out = {}
    path = os.path.join('data', 'ETP_POS.csv')
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            w = (r.get('Etruscan') or '').strip().lower()
            w = re.sub(r'[^a-zθχφσςśšê\']', '', w)
            w = ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's',
                         'ς': 's', 'ś': 's', 'š': 's', 'ê': 'e',
                         "'": ''}.get(c, c) for c in w)
            trs = re.findall(r"'([^']+)'", r.get('Translations') or '')
            if w and trs:
                out.setdefault(w, set()).update(t.lower() for t in trs)
    return out


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log('FIXED 2026-07-10: совместный нуль — одна общая перестановка порядка слов в записи на replicate (R=5000); прежние p̃ отозваны в §8.')
    log()
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = view_of(corpus)
    log(f'=== Реестр операторов v1 (этап 1) ===')
    log(f'вид: {len(view)} записей (lang=etr, text, без forgery?); '
        f'R={R}, seed={SEED}')

    # --- 0. подтверждение глосс по ETP_POS --------------------------------
    etp_pos = load_etp_pos()
    log()
    log('--- подтверждение глосс реестра по ETP_POS (независимый глоссарий '
        'Larth) ---')
    for name, forms, gloss, cls, kw in OPERATORS:
        hits = {f: sorted(etp_pos[f])[:4] for f in forms if f in etp_pos}
        if hits:
            log(f'  {name:<12} ETP_POS: {hits}')
    log('  (формы без строки выше в ETP_POS отсутствуют — глосса только по '
        'справочникам)')

    # --- 1. позиционные профили с нуль-моделью ----------------------------
    rng = np.random.default_rng(SEED)
    multi = [r for r in view if len(words_ascii(r)) >= 2]
    log()
    log(f'--- позиционные профили (записи с ≥2 словоформами: {len(multi)}) ---')
    occ = {}  # name -> список длин записей и позиций
    for rec in multi:
        ws = words_ascii(rec)
        for name, forms, *_ in OPERATORS:
            fs = set(forms)
            for i, w in enumerate(ws):
                if w in fs:
                    occ.setdefault(name, []).append((i, len(ws)))
    tests = []  # (name, side, obs, n, o)
    op_index = {}
    for name, forms, gloss, cls, kw in OPERATORS:
        o = occ.get(name, [])
        if len(o) < 5:
            continue
        n = len(o)
        ini = sum(1 for i, L in o if i == 0)
        fin = sum(1 for i, L in o if i == L - 1)
        op_index[name] = len(op_index)
        tests.append((name, 'нач', ini, n, o))
        tests.append((name, 'фин', fin, n, o))
    # СОВМЕСТНЫЙ нуль (исправление 2026-07-10, см. §8/§8.7): одна общая
    # перестановка порядка слов внутри каждой записи на replicate; все
    # статистики семейства считаются из одного разыгранного мира.
    R_WY = 5000
    rec_labels = []
    tested_ops = set(op_index)
    for rec in multi:
        ws = words_ascii(rec)
        lab = np.full(len(ws), -1, dtype=np.int16)
        hit = False
        for name in tested_ops:
            fs = set(dict((n_, f_) for n_, f_, *_ in OPERATORS)[name])
            for i, w in enumerate(ws):
                if w in fs:
                    lab[i] = op_index[name]
                    hit = True
        if hit:
            rec_labels.append(lab)
    nOps = len(op_index)
    sims = np.zeros((R_WY, len(tests)), dtype=np.int32)
    col_ini = {op_index[nm]: j for j, (nm, sd, *_ ) in enumerate(tests)
               if sd == 'нач'}
    col_fin = {op_index[nm]: j for j, (nm, sd, *_ ) in enumerate(tests)
               if sd == 'фин'}
    for r_i in range(R_WY):
        for lab in rec_labels:
            perm = rng.permutation(len(lab))
            first = lab[perm[0]]
            last = lab[perm[-1]]
            if first >= 0:
                sims[r_i, col_ini[first]] += 1
            if last >= 0:
                sims[r_i, col_fin[last]] += 1
    R_eff = R_WY
    obs_arr = np.array([t[2] for t in tests])
    p_raw = ((sims >= obs_arr).sum(axis=0) + 1) / (R_eff + 1)
    # Westfall–Young min-p: p-значение каждого симулированного значения в
    # рамках его теста, затем распределение минимума по семейству
    p_sim = np.zeros_like(sims, dtype=np.float64)
    for j in range(len(tests)):
        col = sims[:, j]
        order = np.sort(col)
        idx = np.searchsorted(order, col, side='left')
        p_sim[:, j] = (R_eff - idx + 1) / (R_eff + 1)
    minp = p_sim.min(axis=1)
    p_adj = np.array([((minp <= p).sum() + 1) / (R_eff + 1)
                      for p in p_raw])
    log(f'{"оператор":<12} {"глосса":<22} {"n":>4} {"нач%":>5} {"фин%":>5} '
        f'{"p(нач)":>8} {"p(фин)":>8} {"p̃_сем(нач)":>10} {"p̃_сем(фин)":>10}')
    rows_csv = []
    by_name = {}
    for j, (name, side, obs, n, o) in enumerate(tests):
        d = by_name.setdefault(name, {})
        d[side] = (obs, n, p_raw[j], p_adj[j])
    for name, forms, gloss, cls, kw in OPERATORS:
        if name not in by_name:
            continue
        d = by_name[name]
        (ini, n, p_i, pa_i) = d['нач']
        (fin, _, p_f, pa_f) = d['фин']
        log(f'{name:<12} {gloss[:22]:<22} {n:>4} {ini / n:>5.0%} '
            f'{fin / n:>5.0%} {p_i:>8.4f} {p_f:>8.4f} {pa_i:>10.4f} '
            f'{pa_f:>10.4f}')
        rows_csv.append({'operator': name, 'forms': ' '.join(forms),
                         'gloss': gloss, 'class': cls, 'n': n,
                         'init': ini, 'fin': fin,
                         'p_init': f'{p_i:.5f}', 'p_fin': f'{p_f:.5f}',
                         'p_adj_init': f'{pa_i:.5f}',
                         'p_adj_fin': f'{pa_f:.5f}'})
    log('нуль-модель: позиция вхождения равновероятна внутри записи; '
        'p̃_сем — Westfall–Young по всему семейству тестов')

    # --- 2. репликация знака смещения -------------------------------------
    log()
    log('--- репликация: доля начальных позиций по стратам (n≥5) ---')
    def strat_stats(recs):
        st = {}
        for rec in recs:
            ws = words_ascii(rec)
            if len(ws) < 2:
                continue
            for name, forms, *_ in OPERATORS:
                fs = set(forms)
                for i, w in enumerate(ws):
                    if w in fs:
                        a, b = st.get(name, (0, 0))
                        st[name] = (a + (i == 0), b + 1)
        return st
    parts = {'ETP': [r for r in view if r['src'] == 'ETP'],
             'CIEP': [r for r in view if r['src'] == 'CIEP']}
    top_regions = [reg for reg, _ in Counter(
        r['region'] for r in view if r['region']).most_common(5)]
    for reg in top_regions:
        parts[f'рег.{reg}'] = [r for r in view if r['region'] == reg]
    st_all = {k: strat_stats(v) for k, v in parts.items()}
    hdr = 'оператор     ' + ' '.join(f'{k:>10}' for k in parts)
    log(hdr)
    for name, forms, gloss, cls, kw in OPERATORS:
        if name not in by_name or by_name[name]['нач'][1] < 8:
            continue
        cells = []
        for k in parts:
            a_b = st_all[k].get(name)
            cells.append(f'{a_b[0]/a_b[1]:>7.0%}/{a_b[1]:<2}' if a_b and a_b[1] >= 5
                         else f'{"–":>10}')
        log(f'{name:<12} ' + ' '.join(cells))

    # --- 3. слот-шаблоны ---------------------------------------------------
    log()
    log('--- слот-шаблоны (классы: OP:имя / GEN? / NUM / GAP / X) ---')
    op_of = {}
    for name, forms, *_ in OPERATORS:
        for f in forms:
            op_of[f] = name
    def tok_class(t):
        if t['kind'] == 'N':
            return 'NUM'
        if t['kind'] == 'G':
            return 'GAP'
        a = t['ascii']
        if a in op_of:
            return 'OP:' + op_of[a]
        for suf in GEN_SUFFIXES:
            if a.endswith(suf) and len(a) > len(suf) + 2:
                return 'GEN?'
        return 'X'
    tmpl = Counter()
    for rec in multi:
        tmpl[tuple(tok_class(t) for t in rec['toks'])[:6]] += 1
    top = tmpl.most_common(12)
    cov10 = sum(c for _, c in top[:10]) / len(multi)
    log(f'мультисловных записей: {len(multi)}; уникальных шаблонов (первые '
        f'6 слотов): {len(tmpl)}; покрытие топ-10: {cov10:.0%}')
    for t, c in top:
        log(f'  ×{c:>4}  {" ".join(t)}')

    # --- 4. валидация по переводам ----------------------------------------
    log()
    log('--- ассоциация «оператор в тексте ↔ ключевое слово в переводе» ---')
    tr_view = [r for r in view if r['trs']]
    trs = [' '.join(r['trs']).lower() for r in tr_view]
    log(f'переведённых записей вида: {len(tr_view)}')
    rng2 = np.random.default_rng(SEED)
    assoc_tests = []
    for name, forms, gloss, cls, kw in OPERATORS:
        if not kw:
            continue
        fs = set(forms)
        has_op = np.array([any(w in fs for w in words_ascii(r))
                           for r in tr_view])
        n_op = int(has_op.sum())
        if n_op < 5:
            continue
        pat = re.compile(kw)
        has_kw = np.array([bool(pat.search(t)) for t in trs])
        both = int((has_op & has_kw).sum())
        assoc_tests.append((name, gloss, n_op, int(has_kw.sum()), both,
                            has_op, has_kw))
    sims2 = np.zeros((R, len(assoc_tests)), dtype=np.int32)
    perm_idx = np.argsort(rng2.random((R, len(tr_view))), axis=1)
    for j, (*_, has_op, has_kw) in enumerate(assoc_tests):
        kw_int = has_kw.astype(np.int8)
        op_idx = np.where(has_op)[0]
        sims2[:, j] = kw_int[perm_idx[:, op_idx]].sum(axis=1)
    obs2 = np.array([t[4] for t in assoc_tests])
    p2 = ((sims2 >= obs2).sum(axis=0) + 1) / (R + 1)
    p_sim2 = np.zeros_like(sims2, dtype=np.float64)
    for j in range(len(assoc_tests)):
        col = sims2[:, j]
        order = np.sort(col)
        idx = np.searchsorted(order, col, side='left')
        p_sim2[:, j] = (R - idx + 1) / (R + 1)
    minp2 = p_sim2.min(axis=1)
    p2_adj = np.array([((minp2 <= p).sum() + 1) / (R + 1) for p in p2])
    log(f'{"оператор":<12} {"n_op":>5} {"n_kw":>5} {"оба":>4} {"точн.":>6} '
        f'{"база":>6} {"p":>8} {"p̃_сем":>8}')
    for j, (name, gloss, n_op, n_kw, both, has_op, has_kw) in enumerate(
            assoc_tests):
        prec = both / n_op
        base = n_kw / len(tr_view)
        log(f'{name:<12} {n_op:>5} {n_kw:>5} {both:>4} {prec:>6.0%} '
            f'{base:>6.0%} {p2[j]:>8.4f} {p2_adj[j]:>8.4f}')
        for row in rows_csv:
            if row['operator'] == name:
                row.update({'n_translated': n_op, 'kw_hits': both,
                            'precision': f'{prec:.3f}',
                            'base_rate': f'{base:.3f}',
                            'p_assoc': f'{p2[j]:.5f}',
                            'p_assoc_adj': f'{p2_adj[j]:.5f}'})
    log('нуль: перестановка переводов между переведёнными записями')

    # --- 5. мультиметочная классификация по правилам ----------------------
    log()
    log('--- классификация записей правилами ПО ТЕКСТУ против золота ПО '
        'ПЕРЕВОДУ ---')
    RULES = {
        'эпитафия': ({'clan', 'clens', 'sec', 'puia', 'ati', 'apa', 'lupu',
                      'svalce', 'avils', 'ril', 'suthi'},
                     r'\bson\b|\bdaughter|\bwife\b|\bmother\b|\bfather\b|'
                     r'\bdie[ds]?\b|\bdead\b|\blived\b|\byears?\b|\btomb\b|'
                     r'\bgrave\b'),
        'посвящение': ({'turce', 'muluvanice', 'zinace', 'cver'},
                       r'\bdedicat|\bgave\b|\boffer|\bmade\b|\bgift\b|'
                       r'\bvotive'),
        'говорящий': ({'mi', 'mini'}, r'\bi\b|\bme\b'),
        'границы': ({'tular'}, r'\bboundar'),
    }
    op_names_of_rec = {}
    for rec in tr_view:
        names = set()
        ws = set(words_ascii(rec))
        for name, forms, *_ in OPERATORS:
            if ws & set(forms):
                names.add(name)
        op_names_of_rec[rec['rid']] = names
    rng3 = np.random.default_rng(SEED)
    perm3 = np.argsort(rng3.random((R, len(tr_view))), axis=1)
    for label, (ops, kw) in RULES.items():
        pat = re.compile(kw)
        pred = np.array([bool(op_names_of_rec[r['rid']] & ops)
                         for r in tr_view])
        gold = np.array([bool(pat.search(t)) for t in trs])
        tp = int((pred & gold).sum())
        prec = tp / max(pred.sum(), 1)
        rec_ = tp / max(gold.sum(), 1)
        f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
        # нуль: перестановка золота
        gold_int = gold.astype(np.int8)
        pi = np.where(pred)[0]
        sim_tp = gold_int[perm3[:, pi]].sum(axis=1)
        p = ((sim_tp >= tp).sum() + 1) / (R + 1)
        log(f'  {label:<12} pred={int(pred.sum()):>4} gold={int(gold.sum()):>4} '
            f'TP={tp:>4} точн.={prec:>4.0%} полн.={rec_:>4.0%} '
            f'F1={f1:.2f} p={p:.4f}')

    # --- реестр в CSV -------------------------------------------------------
    fields = ['operator', 'forms', 'gloss', 'class', 'n', 'init', 'fin',
              'p_init', 'p_fin', 'p_adj_init', 'p_adj_fin', 'n_translated',
              'kw_hits', 'precision', 'base_rate', 'p_assoc', 'p_assoc_adj']
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for row in rows_csv:
            w.writerow({k: row.get(k, '') for k in fields})
    log()
    log(f'реестр записан: {OUT_CSV}')
    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
