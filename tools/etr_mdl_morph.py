# -*- coding: utf-8 -*-
r"""Аудит Sol №2: held-out продуктивная морфология через MDL.

Вопрос: реальна ли этрусская суффиксальная морфология как ПРОДУКТИВНАЯ
система (а не артефакт частотных хвостов)? Метод: двухчастный код (MDL).
Сплит 80/20 ПО ПАМЯТНИКАМ (artifact_id, seed=42). На train жадно растится
лексикон суффиксов S (всегда содержит ∅); каждый тип анализируется как
основа(>=3 симв.)+суффикс из S (или ∅):

  L(model) = Σ_{s∈S}(len(s)+1)·H0 + Σ_{основы}(len(st)+1)·H0,
             H0 = log2(|алфавит train|+1)
  L(data)  = Σ_токенов [−log2 p_stem(основа) − log2 p_suf(суффикс)],
             add-1 по счётам выбранных анализов (категории основ —
             наблюдённые; категории суффиксов — весь S)

Выбор анализа типа — argmin ПОЛНОЙ поштучной дельты DL при замороженных
p: данные (f·[−log2 p_stem − log2 p_suf]) плюс дельта лексикона основ
(новая основа платит (len+1)·H0, освобождённая — экономит); проход по
типам последовательный в сортированном порядке, после прохода —
пересчёт p; 3 раунда. ВАЖНОЕ отклонение от чернового плана аудита:
назначение по чистой data-стоимости токена (без лексиконной дельты)
эмпирически даёт вырожденный минимум S=∅ — все переключения
откатываются, т.к. ~2/3 DL лежит в лексиконе основ, которого поштучное
data-назначение «не видит»; это зафиксировано в логе как негатив
литеральной процедуры (стандартное лекарство Morfessor Baseline —
локальный шаг по полной DL).

Кандидаты — словоконечные подстроки длины 1..4 train-типов
(основа-остаток >=3, типов >= 15); на каждом шаге приближённый выигрыш
DL кандидата при замороженных вероятностях: переключаются только типы,
оканчивающиеся на кандидата, и только если их полная поштучная дельта
DL отрицательна; лучший положительный кандидат добавляется, затем
точный пересчёт (3 раунда) и точный DL. Если точный пересчёт даёт
неположительный реальный выигрыш, кандидат откатывается и исключается
(логируется). Стоп: нет положительного выигрыша или |S\{∅}| >= 60.

Оценка на отложенных памятниках (не пересекаются с train):
  a) биты/токен финальной модели (для новой основы — посимвольная
     unigram-модель train, add-1, с символом конца) против базлайнов:
     B0 = unigram по train-типам с посимвольным backoff для OOV;
     B1 = посимвольная 3-граммная LM (train, add-0.5);
  b) ПРОДУКТИВНОСТЬ (ключевая метрика): доля новых test-типов,
     разложимых как train-основа (из финальных анализов) + непустой
     суффикс из S; нуль — R=200 случайных наборов той же мощности и
     того же мультимножества длин из пула кандидатов (без выбранного
     набора как целого); p = (1+#{null>=obs})/(R+1);
  c) точность против справочного набора REF (падежные/глагольные
     форманты по стандартным грамматикам, Wallace 2008) + тот же нуль;
  d) парадигмы: train-основы с >=2 разными непустыми суффиксами.

Артефакты: logs/etr_mdl_morph.log; results/mdl_suffixes_v1.csv
(suffix, dl_gain_bits = точный выигрыш DL на шаге добавления,
n_types_train = типов с этим суффиксом в финальных анализах, in_ref).
p — разведочные.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_mdl_morph.py
"""
import csv
import math
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
R_NULL = 200
MAX_SUF = 60      # стоп жадного роста
MIN_TYPES = 15    # порог кандидата: типов train с этим концом
MAX_SUF_LEN = 4
MIN_STEM = 3
OUT_LOG = os.path.join('logs', 'etr_mdl_morph.log')
OUT_CSV = os.path.join('results', 'mdl_suffixes_v1.csv')
LOG = []

# справочные форманты (Wallace 2008: генитивы -s/-l/-al/-ial, артикуляции
# -isa/-sa/-si/-ale, локатив -i/-thi/-e, претерит -ce/-ke, -u, аблатив/
# прочие -a/-na/-ri, энклитика -m)
REF = {'s', 'l', 'al', 'ial', 'isa', 'sa', 'si', 'ale', 'i', 'thi',
       'e', 'ce', 'ke', 'u', 'a', 'na', 'ri', 'm'}


def log(msg=''):
    print(msg)
    LOG.append(msg)


def analyses_of(w, S):
    """Допустимые анализы типа w: [(основа, суффикс)]; порядок фиксирован
    (∅, затем суффиксы длины 1..4) — при равной стоимости берётся первый."""
    out = [(w, '')]
    for k in range(1, MAX_SUF_LEN + 1):
        if len(w) - k >= MIN_STEM:
            s = w[-k:]
            if s in S:
                out.append((w[:-k], s))
    return out


def counts_of(assign, freq):
    """Счёты токенов по основам/суффиксам и число типов на основу."""
    cs, cu, users = Counter(), Counter(), Counter()
    for w, (st, su) in assign.items():
        f = freq[w]
        cs[st] += f
        cu[su] += f
        users[st] += 1
    return cs, cu, users


def assign_round(types_sorted, freq, S, assign, cs, cu, users, N, H0):
    """Один раунд последовательного переназначения анализов: argmin полной
    поштучной дельты DL при замороженных p (данные + дельта лексикона
    основ, отслеживаемая инкрементально по числу типов на основу)."""
    Vs, Vu = len(cs), len(S)
    live = Counter(users)
    new = dict(assign)
    for w in types_sorted:
        st_c, su_c = new[w]
        f = freq[w]
        best_sc, best_a = None, None
        for st, su in analyses_of(w, S):
            sc = f * (-math.log2((cs.get(st, 0) + 1) / (N + Vs))
                      - math.log2((cu.get(su, 0) + 1) / (N + Vu)))
            if st != st_c:
                if live.get(st, 0) == 0:
                    sc += (len(st) + 1) * H0
                if live.get(st_c, 0) == 1:
                    sc -= (len(st_c) + 1) * H0
            if best_sc is None or sc < best_sc - 1e-12:
                best_sc, best_a = sc, (st, su)
        if best_a != (st_c, su_c):
            live[st_c] -= 1
            live[best_a[0]] += 1
            new[w] = best_a
    return new


def total_dl(assign, freq, S, N, H0):
    """Точная двухчастная длина описания (биты)."""
    cs, cu, _ = counts_of(assign, freq)
    Vs, Vu = len(cs), len(S)
    lm = sum((len(s) + 1) * H0 for s in sorted(S))
    lm += sum((len(st) + 1) * H0 for st in sorted(cs))
    ld = 0.0
    for w in sorted(assign):
        st, su = assign[w]
        ld += freq[w] * (-math.log2((cs[st] + 1) / (N + Vs))
                         - math.log2((cu[su] + 1) / (N + Vu)))
    return lm + ld, lm, ld


def approx_gain(c, Tc, assign, freq, cs, cu, users, S, N, H0):
    """Приближённый выигрыш DL от добавления суффикса c при замороженных
    вероятностях: переключаются только типы из Tc, которым это выгодно
    (2 итерации уточнения массы c); суффиксная часть L(data) пересчитана
    точно при фиксированных анализах, основная — по затронутым основам
    с поправкой знаменателя. Возвращает (gain_bits, [переключаемые типы])."""
    Vs, Vu = len(cs), len(S)
    lc = len(c)
    F_est = sum(freq[w] for w in Tc)
    sw = []
    for _ in range(2):
        p_c_bits = -math.log2((F_est + 1) / (N + Vu + 1))
        sw, F_sw = [], 0
        added_stems, removed_users = Counter(), Counter()
        for w in Tc:
            st_o, su_o = assign[w]
            f = freq[w]
            old = (-math.log2((cs[st_o] + 1) / (N + Vs))
                   - math.log2((cu.get(su_o, 0) + 1) / (N + Vu)))
            st_n = w[:-lc]
            new = (-math.log2((cs.get(st_n, 0) + 1) / (N + Vs)) + p_c_bits)
            d = f * (new - old)
            # поштучный вклад в лексикон основ
            if users.get(st_n, 0) + added_stems.get(st_n, 0) == 0:
                d += (len(st_n) + 1) * H0
            if users.get(st_o, 0) - removed_users.get(st_o, 0) == 1:
                d -= (len(st_o) + 1) * H0
            if d < -1e-12:
                sw.append(w)
                F_sw += f
                added_stems[st_n] += 1
                removed_users[st_o] += 1
        F_est = F_sw
    if not sw:
        return 0.0, []
    F_sw = sum(freq[w] for w in sw)
    d_su, d_st, users_d = Counter(), Counter(), Counter()
    for w in sw:
        st_o, su_o = assign[w]
        f = freq[w]
        d_su[su_o] -= f
        d_st[st_o] -= f
        d_st[w[:-lc]] += f
        users_d[st_o] -= 1
        users_d[w[:-lc]] += 1
    # суффиксная часть L(data): до/после при фиксированных анализах
    old_su = sum(cu[s] * -math.log2((cu[s] + 1) / (N + Vu))
                 for s in sorted(S) if cu[s] > 0)
    new_su = F_sw * -math.log2((F_sw + 1) / (N + Vu + 1))
    for s in sorted(S):
        cnt = cu[s] + d_su.get(s, 0)
        if cnt > 0:
            new_su += cnt * -math.log2((cnt + 1) / (N + Vu + 1))
    # основная часть L(data): затронутые основы + смена знаменателя Vs
    n_new = sum(1 for st, d in users_d.items()
                if users.get(st, 0) == 0 and d > 0)
    n_free = sum(1 for st, d in users_d.items()
                 if users.get(st, 0) > 0 and users.get(st, 0) + d == 0)
    Vs2 = Vs + n_new - n_free
    aff = sorted(d_st)
    old_st = sum(cs.get(st, 0) * -math.log2((cs.get(st, 0) + 1) / (N + Vs))
                 for st in aff)
    new_st = 0.0
    for st in aff:
        cnt = cs.get(st, 0) + d_st[st]
        if cnt > 0:
            new_st += cnt * -math.log2((cnt + 1) / (N + Vs2))
    mass_aff = sum(cs.get(st, 0) for st in aff)
    denom_corr = (N - mass_aff) * math.log2((N + Vs2) / (N + Vs))
    d_data = (new_su - old_su) + (new_st - old_st) + denom_corr
    # L(model): суффикс c + новые основы − освобождённые основы
    d_model = (lc + 1) * H0
    for st, d in users_d.items():
        u0 = users.get(st, 0)
        if u0 == 0 and d > 0:
            d_model += (len(st) + 1) * H0
        elif u0 > 0 and u0 + d == 0:
            d_model -= (len(st) + 1) * H0
    return -(d_data + d_model), sw


def main():
    t0 = time.time()
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.10'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== Аудит Sol №2: held-out продуктивная морфология (MDL) ===')
    log(f'корпус v{corpus["meta"]["freeze_version"]}; канонический вид: {len(view)} записей')

    # --- сплит по памятникам ---
    arts = sorted({r['artifact_id'] for r in view})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(arts))
    n_tr = int(0.8 * len(arts))
    train_arts = {arts[i] for i in perm[:n_tr]}
    tr_freq, te_freq = Counter(), Counter()
    n_rec_tr = n_rec_te = 0
    for r in view:
        is_tr = r['artifact_id'] in train_arts
        if is_tr:
            n_rec_tr += 1
        else:
            n_rec_te += 1
        tgt = tr_freq if is_tr else te_freq
        for t in r['toks']:
            if t['kind'] == 'W' and '-' not in t['ascii'] \
                    and len(t['ascii']) >= 3:
                tgt[t['ascii']] += 1
    N = sum(tr_freq.values())
    N_te = sum(te_freq.values())
    types_sorted = sorted(tr_freq)
    log(f'памятников: {len(arts)}; сплит по памятникам (seed={SEED}): '
        f'train {n_tr} / test {len(arts) - n_tr}')
    log(f'train: {n_rec_tr} записей, {N} токенов, {len(types_sorted)} типов; '
        f'test: {n_rec_te} записей, {N_te} токенов, {len(te_freq)} типов')

    alphabet = sorted({ch for w in types_sorted for ch in w})
    A = len(alphabet)
    H0 = math.log2(A + 1)
    log(f'алфавит train: {A} символов; H0 = log2({A}+1) = {H0:.4f} бит/симв')

    # --- пул кандидатов ---
    cand = {}
    for w in types_sorted:
        for k in range(1, MAX_SUF_LEN + 1):
            if len(w) - k >= MIN_STEM:
                cand.setdefault(w[-k:], []).append(w)
    pool = {c: ws for c, ws in cand.items() if len(ws) >= MIN_TYPES}
    by_len = Counter(len(c) for c in pool)
    log(f'кандидатов (конечные подстроки 1..{MAX_SUF_LEN}, основа>={MIN_STEM}, '
        f'типов>={MIN_TYPES}): {len(pool)} '
        f'(по длинам: {dict(sorted(by_len.items()))})')

    # --- жадный рост S ---
    log()
    log('--- жадный рост лексикона суффиксов (MDL, train) ---')
    log('замечание (негатив литеральной процедуры): назначение анализов по '
        'чистой data-стоимости токена даёт вырожденный S=∅ — все кандидаты '
        'откатываются точным пересчётом, т.к. ~2/3 DL лежит в лексиконе '
        'основ; здесь назначение = argmin полной поштучной дельты DL '
        '(данные + лексикон основ), как в Morfessor Baseline')
    S = {''}
    assign = {w: (w, '') for w in types_sorted}
    cs, cu, users = counts_of(assign, tr_freq)
    dl_cur, lm, ld = total_dl(assign, tr_freq, S, N, H0)
    dl_start = dl_cur
    log(f'DL старт (S={{∅}}): {dl_cur:.1f} бит = модель {lm:.1f} '
        f'+ данные {ld:.1f}')
    added = []  # (суффикс, прибл. выигрыш, точный выигрыш)
    banned = set()
    while True:
        best_g, best_c, best_sw = 0.0, None, None
        for c in sorted(pool):
            if c in S or c in banned:
                continue
            g, sw = approx_gain(c, pool[c], assign, tr_freq,
                                cs, cu, users, S, N, H0)
            if g > best_g + 1e-9:
                best_g, best_c, best_sw = g, c, sw
        if best_c is None:
            log('стоп: нет кандидата с положительным выигрышем DL')
            break
        snap = dict(assign)
        S.add(best_c)
        for w in best_sw:
            assign[w] = (w[:-len(best_c)], best_c)
        cs, cu, users = counts_of(assign, tr_freq)
        for _ in range(3):
            assign = assign_round(types_sorted, tr_freq, S, assign,
                                  cs, cu, users, N, H0)
            cs, cu, users = counts_of(assign, tr_freq)
        dl_new, lm, ld = total_dl(assign, tr_freq, S, N, H0)
        realized = dl_cur - dl_new
        if realized <= 1e-6:
            S.discard(best_c)
            banned.add(best_c)
            assign = snap
            cs, cu, users = counts_of(assign, tr_freq)
            log(f"- '-{best_c}' отвергнут: прибл. {best_g:.1f} бит, но "
                f"точный DL {dl_cur:.1f} → {dl_new:.1f} "
                f"({realized:+.1f}); откат и исключение")
            continue
        n_asgn = sum(1 for w in assign if assign[w][1] == best_c)
        log(f"+ '-{best_c}': прибл. {best_g:.1f} бит; точный DL {dl_cur:.1f}"
            f" → {dl_new:.1f} (выигрыш {realized:.1f}); "
            f"переключено {len(best_sw)}, в финале раунда назначено {n_asgn} типов")
        added.append((best_c, best_g, realized))
        dl_cur = dl_new
        if len(S) - 1 >= MAX_SUF:
            log(f'стоп: |S \\ {{∅}}| достиг {MAX_SUF}')
            break
    S_non = sorted(s for s in S if s)
    n_assigned = Counter(su for _, su in assign.values())
    log(f'итог: |S \\ {{∅}}| = {len(S_non)}; DL {dl_cur:.1f} бит '
        f'(сжатие против старта: {100 * (1 - dl_cur / dl_start):.2f}%)')
    log('суффиксы (порядок добавления): '
        + ' '.join(f'-{c}' for c, _, _ in added))

    # --- финальные вероятности train ---
    Vs, Vu = len(cs), len(S)
    stems_train = set(cs)

    def p_stem_bits(st):
        return -math.log2((cs.get(st, 0) + 1) / (N + Vs))

    def p_suf_bits(su):
        return -math.log2((cu.get(su, 0) + 1) / (N + Vu))

    # посимвольная unigram train (add-1, категории = алфавит + END)
    char_cnt = Counter()
    for w in types_sorted:
        f = tr_freq[w]
        for ch in w:
            char_cnt[ch] += f
    char_denom = sum(char_cnt.values()) + N + (A + 1)

    def char_cost(sq):
        b = sum(-math.log2((char_cnt.get(ch, 0) + 1) / char_denom)
                for ch in sq)
        return b + -math.log2((N + 1) / char_denom)

    def model_cost(w):
        best = None
        for st, su in analyses_of(w, S):
            sb = p_stem_bits(st) if st in stems_train else char_cost(st)
            c = sb + p_suf_bits(su)
            if best is None or c < best - 1e-12:
                best = c
        return best

    # B0: unigram train-типов (add-1, +OOV-категория) с посимв. backoff
    Vt = len(tr_freq)
    b0_denom = N + Vt + 1

    def b0_cost(w):
        cw = tr_freq.get(w, 0)
        if cw > 0:
            return -math.log2((cw + 1) / b0_denom)
        return -math.log2(1 / b0_denom) + char_cost(w)

    # B1: посимвольная 3-грамма (add-0.5), BOS='^', EOS='$'
    tri, ctx = Counter(), Counter()
    for w in types_sorted:
        f = tr_freq[w]
        seq = '^^' + w + '$'
        for i in range(2, len(seq)):
            tri[(seq[i - 2:i], seq[i])] += f
            ctx[seq[i - 2:i]] += f
    V1 = A + 1

    def b1_cost(w):
        seq = '^^' + w + '$'
        return sum(-math.log2((tri.get((seq[i - 2:i], seq[i]), 0) + 0.5)
                              / (ctx.get(seq[i - 2:i], 0) + 0.5 * V1))
                   for i in range(2, len(seq)))

    # --- (a) биты/токен на test ---
    log()
    log('--- (a) сжатие test-потока (памятники не пересекаются с train) ---')
    te_types = sorted(te_freq)
    oov_types = [w for w in te_types if w not in tr_freq]
    oov_tok = sum(te_freq[w] for w in oov_types)
    log(f'test-токенов: {N_te}; новых типов (OOV): {len(oov_types)} из '
        f'{len(te_types)} ({100 * oov_tok / N_te:.1f}% токенов)')
    bits_m = sum(te_freq[w] * model_cost(w) for w in te_types) / N_te
    bits_b0 = sum(te_freq[w] * b0_cost(w) for w in te_types) / N_te
    bits_b1 = sum(te_freq[w] * b1_cost(w) for w in te_types) / N_te
    log(f'биты/токен: MDL-модель {bits_m:.3f} | B0 (unigram типов + '
        f'char-backoff) {bits_b0:.3f} | B1 (char-3gram add-0.5) {bits_b1:.3f}')
    log(f'модель vs B0: {bits_b0 - bits_m:+.3f} бит/токен; '
        f'модель vs B1: {bits_b1 - bits_m:+.3f} бит/токен '
        f'(положительное = модель лучше)')

    # --- (b) продуктивность на новых типах ---
    log()
    log('--- (b) продуктивность: новые test-типы = train-основа + суффикс ---')
    suf_by_len = {k: {s for s in S_non if len(s) == k}
                  for k in range(1, MAX_SUF_LEN + 1)}

    def decomposable(w, sets_by_len):
        for k in range(1, MAX_SUF_LEN + 1):
            if len(w) - k >= MIN_STEM and w[-k:] in sets_by_len[k] \
                    and w[:-k] in stems_train:
                return True
        return False

    if not S_non or not oov_types:
        log('НЕГАТИВ: пустой лексикон суффиксов или нет новых test-типов — '
            'продуктивность не оценивается')
        obs_frac = float('nan')
        p_prod = p_prec = float('nan')
        prec = float('nan')
        null_prod = null_prec_arr = np.array([])
    else:
        obs_dec = sum(1 for w in oov_types if decomposable(w, suf_by_len))
        obs_frac = obs_dec / len(oov_types)
        log(f'разложимы: {obs_dec} из {len(oov_types)} новых типов '
            f'({100 * obs_frac:.1f}%)')
        # нуль: случайные наборы той же мощности и мультимножества длин
        pool_by_len = {k: sorted(c for c in pool if len(c) == k)
                       for k in range(1, MAX_SUF_LEN + 1)}
        need = Counter(len(s) for s in S_non)
        chosen = set(S_non)
        null_sets = []
        guard = 0
        while len(null_sets) < R_NULL and guard < 100 * R_NULL:
            guard += 1
            cur = set()
            for k in sorted(need):
                idx = rng.choice(len(pool_by_len[k]), size=need[k],
                                 replace=False)
                cur.update(pool_by_len[k][i] for i in idx)
            if cur == chosen:
                continue  # без выбранного набора как целого
            null_sets.append(cur)
        null_prod, null_prec = [], []
        for ns in null_sets:
            nbl = {k: {s for s in ns if len(s) == k}
                   for k in range(1, MAX_SUF_LEN + 1)}
            nd = sum(1 for w in oov_types if decomposable(w, nbl))
            null_prod.append(nd / len(oov_types))
            null_prec.append(len(ns & REF) / len(ns))
        null_prod = np.array(null_prod)
        null_prec_arr = np.array(null_prec)
        p_prod = float((1 + (null_prod >= obs_frac - 1e-12).sum())
                       / (len(null_sets) + 1))
        log(f'нуль (R={len(null_sets)}, случайные наборы из пула, тот же '
            f'мультинабор длин): среднее {100 * null_prod.mean():.1f}%, '
            f'max {100 * null_prod.max():.1f}%; p = {p_prod:.4f}')

        # --- (c) сверка с REF ---
        log()
        log('--- (c) сверка с известными формантами (REF, Wallace 2008) ---')
        inter = sorted(set(S_non) & REF)
        prec = len(inter) / len(S_non)
        p_prec = float((1 + (null_prec_arr >= prec - 1e-12).sum())
                       / (len(null_sets) + 1))
        log(f'|S∩REF| = {len(inter)} из |S\\{{∅}}| = {len(S_non)} → '
            f'точность {100 * prec:.1f}%; нуль: среднее '
            f'{100 * null_prec_arr.mean():.1f}%, max '
            f'{100 * null_prec_arr.max():.1f}%; p = {p_prec:.4f}')
        log(f'S∩REF: {" ".join("-" + s for s in inter) or "—"}')
        log(f'S\\REF: '
            f'{" ".join("-" + s for s in S_non if s not in REF) or "—"}')

    # --- (d) парадигмы ---
    log()
    log('--- (d) парадигмы (train, финальные анализы) ---')
    stem_sufs = {}
    for w in sorted(assign):
        st, su = assign[w]
        stem_sufs.setdefault(st, Counter())[su] += tr_freq[w]
    parad = {st: c for st, c in stem_sufs.items()
             if sum(1 for s in c if s) >= 2}
    log(f'основ с >=2 разными непустыми суффиксами: {len(parad)} '
        f'из {len(stem_sufs)}')
    log('топ-15 по токенам основы:')
    for st in sorted(parad, key=lambda x: (-cs[x], x))[:15]:
        parts = ' '.join(
            f"{'∅' if not s else '-' + s}({n})"
            for s, n in sorted(parad[st].items(), key=lambda kv: (-kv[1], kv[0])))
        log(f'  {st:<10} n={cs[st]:>3}  {parts}')

    # --- CSV ---
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wtr = csv.writer(f, lineterminator='\n')
        wtr.writerow(['suffix', 'dl_gain_bits', 'n_types_train', 'in_ref'])
        for c, _, g_real in added:
            wtr.writerow([c, f'{g_real:.2f}', n_assigned.get(c, 0),
                          int(c in REF)])
    log()
    log(f'csv записан: {OUT_CSV} ({len(added)} суффиксов)')
    log(f'время: {time.time() - t0:.1f} с')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
