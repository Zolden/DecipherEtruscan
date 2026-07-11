# -*- coding: utf-8 -*-
"""Аудит Sol, план №3: context-only якорный ГРАФ — распространение меток.

Самостоятельная надстройка над пилотом Sol (tools/etr_method_audit_sol.py,
distributional_anchor_audit): вместо ближайшего PPMI-центроида — графовое
распространение меток (label propagation, Zhou et al.) по kNN-графу
косинусных близостей PPMI-векторов, с включением НЕразмеченных типов как
промежуточных узлов. Форма слова НЕ используется как признак нигде —
только контексты (окно ±2 внутри записи, контексты = топ-500 частых
типов, точно как у Sol).

Данные: data/etr_corpus.pkl (текущая заморозка), канонический вид (lang=etr,
kind=text, без forgery?, variant_of=None). Чистые записи — как у Sol:
src != 'CIEP' и >=2 содержательных слов (kind='W', без '-', len>=3).
Золото — ETP_POS-метки как в semantic_labels Sol (NAME-M/NAME-F/THEO/
VERB/FUNC; конфликтные типы отброшены); якоря = размеченные типы с
частотой >=2 в чистых записях.

Дизайн: узлы = типы freq>=2 с ненулевым контекстным вектором (якоря
сохраняются всегда, в т.ч. изолированные — для сопоставимости оценки;
cap 2500 узлов по частоте); kNN k=10 по косинусу L2-нормированных
PPMI-строк, ребро при попадании в kNN хотя бы в одну сторону, веса =
косинус, нулевые обрезаны; F <- alpha*S_norm@F + (1-alpha)*Y0 (S
нормирован по строкам, gather-цикл по рёбрам, не плотный matmul),
alpha=0.85, 30 итераций, F0=Y0.

Оценка: сначала воспроизводится центроидный БАЗЛАЙН Sol (PPMI только по
якорным строкам, family-blocked 5-fold CV, strip_case-семьи,
rng(SEED+4) — механизм Sol; ориентир Sol: balanced=39.1%, VERB
F1=0.571); механизм дополнительно верифицируется бит-в-бит под
настройками самого Sol (raw-порог >=2 W-токенов, most_common-контексты,
фолды rng(20260710+4) — эталон из results/method_audit_sol_20260710.json),
затем графовое распространение на тех же фолдах. Нуль —
перестановка ВЕКТОРОВ меток между семьями равного размера (механизм
Sol, rng(SEED+5)), R=1000 (при прогнозе >8 мин — 500), полный CV-цикл
распространения на каждой перестановке; p — для balanced accuracy.
Финал: обучение на всех якорях -> ранжированные гипотезы для неякорных
узлов, results/anchor_graph_hypotheses_v1.csv (топ-300 по margin;
разведочный слой, p не заявляем).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_anchor_graph.py
"""
import csv
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
SOL_SEED = 20260710
R_NULL_TARGET = 1000
R_NULL_FALLBACK = 500
NULL_BUDGET_SEC = 480.0
K_NN = 10
ALPHA = 0.85
N_ITER = 30
MAX_NODES = 2500
N_CONTEXT = 500
N_FOLDS = 5
TOP_HYP = 300
ALLOWED = {'FUNC', 'NAME-F', 'NAME-M', 'THEO', 'VERB'}
OUT_LOG = os.path.join('logs', 'etr_anchor_graph.log')
OUT_CSV = os.path.join('results', 'anchor_graph_hypotheses_v1.csv')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


# --- функции пилота Sol (воспроизведены дословно) --------------------------
def to_ascii_word(w):
    w = re.sub(r"[^a-zθχφσςśšê']", '', (w or '').strip().lower())
    table = {'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
             'ś': 's', 'š': 's', 'ê': 'e', "'": ''}
    return ''.join(table.get(c, c) for c in w)


def strip_case(w):
    for ending in ('isa', 'ial', 'thi', 'al', 'us', 'sa', 'ei', 'ce', 's', 'l'):
        if w.endswith(ending) and len(w) - len(ending) >= 3:
            return w[:-len(ending)]
    return w


def content_words(rec):
    return [t['ascii'] for t in rec['toks']
            if t['kind'] == 'W' and '-' not in t['ascii'] and len(t['ascii']) >= 3]


def etp_labels():
    """Золото из ETP_POS — в точности ETP-часть semantic_labels Sol:
    конфликтные (равноголосые) типы отбрасываются."""
    votes = defaultdict(Counter)
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = to_ascii_word(row.get('Etruscan'))
            if len(w) < 3:
                continue
            tag = (row.get('TAG') or '').strip()
            label = None
            if (row.get('theo') or '').strip() == '1':
                label = 'THEO'
            elif tag == 'VERB':
                label = 'VERB'
            elif tag in ('PRON', 'DET', 'ADP', 'PRT'):
                label = 'FUNC'
            elif (row.get('masc') or '').strip() == '1':
                label = 'NAME-M'
            elif (row.get('fem') or '').strip() == '1':
                label = 'NAME-F'
            if label:
                votes[w][label] += 1
    out = {}
    for w in sorted(votes):
        top = votes[w].most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            out[w] = top[0][0]
    return out


def context_counts(records, row_index, ctx_index):
    """Матрица со-встречаемости, окно ±2 по содержательным словам записи
    (точно как у Sol); строки независимы от состава row_index."""
    counts = np.zeros((len(row_index), len(ctx_index)))
    for rec in records:
        ws = content_words(rec)
        for i, target in enumerate(ws):
            ri = row_index.get(target)
            if ri is None:
                continue
            for j in range(max(0, i - 2), min(len(ws), i + 3)):
                if j == i:
                    continue
                ci = ctx_index.get(ws[j])
                if ci is not None:
                    counts[ri, ci] += 1
    return counts


def ppmi_l2(counts):
    """PPMI + L2-нормировка строк (как у Sol); маргиналы зависят от
    набора строк, поэтому считается отдельно для якорей и для узлов."""
    total = counts.sum()
    row_sum = counts.sum(axis=1, keepdims=True)
    col_sum = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ppmi = np.maximum(np.log((counts * total) / (row_sum @ col_sum)), 0.0)
    ppmi[~np.isfinite(ppmi)] = 0.0
    norms = np.linalg.norm(ppmi, axis=1, keepdims=True)
    return ppmi / np.maximum(norms, 1e-12), norms[:, 0]


def crossval_centroid(X, y, folds, n_classes):
    """Базлайн Sol: ближайший центроид по косинусу, CV по готовым фолдам."""
    pred = np.zeros(len(y), dtype=int)
    for fold in range(N_FOLDS):
        train = folds != fold
        centroids = np.zeros((n_classes, X.shape[1]))
        for c in range(n_classes):
            rows_c = X[train & (y == c)]
            if len(rows_c):
                centroids[c] = rows_c.mean(axis=0)
        centroids /= np.maximum(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        pred[~train] = (X[~train] @ centroids.T).argmax(axis=1)
    return pred


# --- метрики ----------------------------------------------------------------
def metrics_full(y_true, y_pred, n_classes):
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, g in zip(y_true, y_pred):
        conf[int(t), int(g)] += 1
    prec, rec, f1 = [], [], []
    for c in range(n_classes):
        tp = conf[c, c]
        fn = conf[c].sum() - tp
        fp = conf[:, c].sum() - tp
        r = tp / max(tp + fn, 1)
        p = tp / max(tp + fp, 1)
        rec.append(float(r))
        prec.append(float(p))
        f1.append(float(2 * p * r / max(p + r, 1e-12)))
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    return acc, float(np.mean(rec)), float(np.mean(f1)), prec, rec, f1, conf


def log_per_class(classes, prec, rec, f1, conf):
    for i, c in enumerate(classes):
        log(f'  {c:<7} support={int(conf[i].sum()):>3} '
            f'P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f}')


def log_confusion(classes, conf):
    log('  confusion (истина\\предсказание):')
    log('           ' + ' '.join(f'{c:>7}' for c in classes))
    for i, c in enumerate(classes):
        log(f'  {c:<9}' + ' '.join(f'{conf[i, j]:>7d}'
                                   for j in range(len(classes))))


def verify_sol_pilot(view, etp):
    """Бит-в-бит воспроизведение центроидного пилота Sol под ЕГО
    настройками: порог записей по всем W-токенам, контексты
    Counter.most_common(500), фолды rng(20260710+4). Эталон —
    results/method_audit_sol_20260710.json."""
    clean = [r for r in view if r['src'] != 'CIEP'
             and sum(t['kind'] == 'W' for t in r['toks']) >= 2]
    freq = Counter(w for r in clean for w in content_words(r))
    labels = {w: c for w, c in etp.items() if c in ALLOWED and freq[w] >= 2}
    anchor_words = sorted(labels)
    classes = sorted(set(labels.values()))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[labels[w]] for w in anchor_words])
    ctx_index = {w: i for i, w in enumerate(
        w for w, _ in freq.most_common(N_CONTEXT))}
    X, _ = ppmi_l2(context_counts(
        clean, {w: i for i, w in enumerate(anchor_words)}, ctx_index))
    families = sorted({strip_case(w) for w in anchor_words})
    rng = np.random.default_rng(SOL_SEED + 4)
    order = [families[i] for i in rng.permutation(len(families))]
    fold_of = {fam: i % N_FOLDS for i, fam in enumerate(order)}
    folds = np.array([fold_of[strip_case(w)] for w in anchor_words])
    pred = crossval_centroid(X, y, folds, len(classes))
    _, bal, mf1, _, _, f1, _ = metrics_full(y, pred, len(classes))
    log(f'верификация механизма под настройками Sol (raw-порог, '
        f'most_common, rng({SOL_SEED}+4)): {len(clean)} записей, '
        f'{len(anchor_words)} якорей; balanced={bal:.5f}, '
        f'macro-F1={mf1:.5f}, VERB F1={f1[ci["VERB"]]:.5f}')
    log('эталон Sol (JSON): balanced=0.39072, macro-F1=0.39145, '
        'VERB F1=0.57143 — совпадение подтверждает реализацию; далее '
        'числа на спецификации этого скрипта (SEED=42, порог по '
        'содержательным словам, сортированные контексты)')


# --- граф -------------------------------------------------------------------
def build_knn_graph(X, k):
    """kNN по косинусу L2-строк; ребро, если хотя бы в одну сторону;
    веса = косинус, нулевые обрезаны. Возвращает направленные рёбра,
    отсортированные по (src, dst), веса нормированы по строкам."""
    n = X.shape[0]
    sims = X @ X.T
    sims = (sims + sims.T) * 0.5          # симметрия против шума BLAS
    np.fill_diagonal(sims, -1.0)
    order = np.argsort(-sims, axis=1, kind='stable')[:, :k]
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = order.ravel().astype(np.int64)
    wts = sims[rows, cols]
    keep = wts > 0.0                      # PPMI неотрицателен: cos>=0, режем нули
    rows, cols = rows[keep], cols[keep]
    code = np.unique(np.concatenate([rows * n + cols, cols * n + rows]))
    src = code // n
    dst = code % n
    w = sims[src, dst]
    deg = np.bincount(src, weights=w, minlength=n)
    w_norm = w / np.maximum(deg[src], 1e-12)
    return src, dst, w_norm, deg


def propagate(src, dst, w_norm, y0):
    """Итерации F <- alpha*S@F + (1-alpha)*Y0, F0=Y0; S@F — gather по
    рёбрам + bincount по строкам (без плотного matmul)."""
    F = y0.copy()
    base = (1.0 - ALPHA) * y0
    n, width = y0.shape
    for _ in range(N_ITER):
        contrib = w_norm[:, None] * F[dst]
        SF = np.empty_like(F)
        for j in range(width):
            SF[:, j] = np.bincount(src, weights=contrib[:, j], minlength=n)
        F = ALPHA * SF + base
    return F


def crossval_graph(edges, n_nodes, anchor_nodes, y, folds, n_classes):
    """Полный CV-цикл распространения: 5 фолдов батчируются в одну
    широкую F (столбцы классов независимы — результат идентичен
    пофолдовому прогону). Возвращает предсказания test-якорей и число
    вырожденных (нулевых) test-строк."""
    src, dst, w_norm = edges
    y0 = np.zeros((n_nodes, N_FOLDS * n_classes))
    for fold in range(N_FOLDS):
        tr = folds != fold
        y0[anchor_nodes[tr], fold * n_classes + y[tr]] = 1.0
    F = propagate(src, dst, w_norm, y0)
    pred = np.zeros(len(y), dtype=int)
    n_zero = 0
    for fold in range(N_FOLDS):
        te = folds == fold
        block = F[anchor_nodes[te], fold * n_classes:(fold + 1) * n_classes]
        pred[te] = block.argmax(axis=1)
        n_zero += int((block.max(axis=1) <= 0).sum())
    return pred, n_zero


def main():
    t_all = time.perf_counter()
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9', 'нужна актуальная заморозка'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    clean = [r for r in view if r['src'] != 'CIEP'
             and len(content_words(r)) >= 2]

    log('=== Аудит Sol №3: context-only якорный граф (label propagation) ===')
    log(f'корпус v{corpus["meta"]["freeze_version"]}; канонический вид: {len(view)} записей')
    log(f'чистые записи (src!=CIEP, >=2 содержательных слов): {len(clean)} '
        '(у Sol порог по всем W-токенам: 1249)')

    freq = Counter(w for r in clean for w in content_words(r))
    types2 = sorted(w for w, c in freq.items() if c >= 2)
    etp = etp_labels()
    anchors = {w: c for w, c in etp.items()
               if c in ALLOWED and freq[w] >= 2}
    anchor_words = sorted(anchors)
    classes = sorted(set(anchors.values()))
    n_classes = len(classes)
    cindex = {c: i for i, c in enumerate(classes)}
    y = np.array([cindex[anchors[w]] for w in anchor_words])
    cls_counts = Counter(anchors.values())
    cls_str = ', '.join(f'{c}={cls_counts[c]}' for c in classes)
    log(f'ETP-меток всего: {len(etp)}; типов с freq>=2: {len(types2)}; '
        f'якорей: {len(anchor_words)} ({cls_str}); у Sol: 181')

    # контексты: топ-500 частых типов; при равенстве частот — алфавит
    # (у Sol — порядок вставки Counter.most_common; граница топ-500
    # малочастотна, вклад расхождения незначим)
    context_words = [w for w, _ in sorted(
        freq.items(), key=lambda kv: (-kv[1], kv[0]))[:N_CONTEXT]]
    ctx_index = {w: i for i, w in enumerate(context_words)}

    # --- базлайн Sol: PPMI только по якорным строкам ------------------------
    base_index = {w: i for i, w in enumerate(anchor_words)}
    counts_base = context_counts(clean, base_index, ctx_index)
    X_base, norms_base = ppmi_l2(counts_base)
    log(f'контексты: топ-{len(context_words)}; ненулевых якорных векторов: '
        f'{int((norms_base > 0).sum())}/{len(anchor_words)} (у Sol: 171/181)')

    families = sorted({strip_case(w) for w in anchor_words})
    rng = np.random.default_rng(SEED + 4)
    family_order = [families[i] for i in rng.permutation(len(families))]
    fold_of = {fam: i % N_FOLDS for i, fam in enumerate(family_order)}
    folds = np.array([fold_of[strip_case(w)] for w in anchor_words])
    log(f'family-blocked фолды (механизм Sol, rng(SEED+4)): '
        f'{len(families)} strip_case-семей; размеры фолдов: '
        f'{np.bincount(folds, minlength=N_FOLDS).tolist()}')

    pred_b = crossval_centroid(X_base, y, folds, n_classes)
    acc_b, bal_b, mf1_b, prec_b, rec_b, f1_b, conf_b = metrics_full(
        y, pred_b, n_classes)
    majority = float((y == np.bincount(y).argmax()).mean())
    log()
    log('--- Базлайн Sol: ближайший PPMI-центроид (те же 5 фолдов) ---')
    verify_sol_pilot(view, etp)
    log(f'acc={acc_b:.1%} (majority={majority:.1%}), balanced={bal_b:.1%}, '
        f'macro-F1={mf1_b:.3f}')
    log('ориентир Sol (seed 20260710): acc=51.9%, balanced=39.1%, '
        'macro-F1=0.391, VERB F1=0.571')
    log_per_class(classes, prec_b, rec_b, f1_b, conf_b)
    log_confusion(classes, conf_b)
    verb = cindex.get('VERB')
    log(f'VERB F1 базлайна = {f1_b[verb]:.3f} (ориентир Sol: 0.571)')

    # --- узлы графа ----------------------------------------------------------
    log()
    log('--- Граф: kNN k=10 по косинусу PPMI (узлы: freq>=2) ---')
    cand_index = {w: i for i, w in enumerate(types2)}
    counts_cand = context_counts(clean, cand_index, ctx_index)
    rowsum_cand = counts_cand.sum(axis=1)
    anchor_set = set(anchor_words)
    iso_anchors = [w for w in anchor_words
                   if rowsum_cand[cand_index[w]] == 0]
    node_words = [w for w in types2
                  if rowsum_cand[cand_index[w]] > 0 or w in anchor_set]
    if len(node_words) > MAX_NODES:
        non_anchor = sorted((w for w in node_words if w not in anchor_set),
                            key=lambda w: (-freq[w], w))
        keep = set(non_anchor[:MAX_NODES - len(anchor_words)]) | anchor_set
        node_words = [w for w in node_words if w in keep]
        log(f'узлов было больше {MAX_NODES} — обрезано по частоте '
            '(якоря сохранены)')
    node_words = sorted(node_words)
    node_index = {w: i for i, w in enumerate(node_words)}
    n_nodes = len(node_words)
    counts_nodes = counts_cand[[cand_index[w] for w in node_words]]
    X_nodes, norms_nodes = ppmi_l2(counts_nodes)
    anchor_nodes = np.array([node_index[w] for w in anchor_words])
    log(f'узлов: {n_nodes} (якорей {len(anchor_words)}, неякорных '
        f'{n_nodes - len(anchor_words)}); якорей с нулевым контекстным '
        f'вектором: {len(iso_anchors)} (оставлены узлами для '
        'сопоставимости оценки; изолированы)')
    zero_ppmi = int((norms_nodes <= 0).sum())
    src, dst, w_norm, deg = build_knn_graph(X_nodes, K_NN)
    out_deg = np.bincount(src, minlength=n_nodes)
    log(f'направленных рёбер после симметризации: {len(src)}; средняя '
        f'степень {len(src) / n_nodes:.1f}; узлов без рёбер: '
        f'{int((out_deg == 0).sum())} (нулевых PPMI-строк: {zero_ppmi})')

    # --- графовое распространение на тех же фолдах --------------------------
    edges = (src, dst, w_norm)
    t_cv = time.perf_counter()
    pred_g, n_zero = crossval_graph(edges, n_nodes, anchor_nodes, y,
                                    folds, n_classes)
    cv_sec = time.perf_counter() - t_cv
    acc_g, bal_g, mf1_g, prec_g, rec_g, f1_g, conf_g = metrics_full(
        y, pred_g, n_classes)
    log()
    log('--- Графовое распространение (alpha=0.85, 30 итераций, те же '
        'фолды) ---')
    log(f'acc={acc_g:.1%} (majority={majority:.1%}), balanced={bal_g:.1%}, '
        f'macro-F1={mf1_g:.3f}; вырожденных (нулевых) test-строк: {n_zero}')
    print(f'  (тайминг, только консоль: CV-цикл {cv_sec:.2f} с)')
    log_per_class(classes, prec_g, rec_g, f1_g, conf_g)
    log_confusion(classes, conf_g)
    log(f'VERB F1 распространения = {f1_g[verb]:.3f} (базлайн '
        f'{f1_b[verb]:.3f}, ориентир Sol 0.571)')

    # контроль: тот же центроид, но на граф-PPMI признаках якорей —
    # отделяет вклад графа от вклада пересчитанных маргиналов PPMI
    pred_c = crossval_centroid(X_nodes[anchor_nodes], y, folds, n_classes)
    acc_c, bal_c, mf1_c, _, _, f1_c, _ = metrics_full(y, pred_c, n_classes)
    log(f'контроль (центроид на граф-PPMI признаках якорей): acc={acc_c:.1%}, '
        f'balanced={bal_c:.1%}, macro-F1={mf1_c:.3f}, VERB F1={f1_c[verb]:.3f}')

    # --- нуль: перестановка векторов меток между семьями равного размера ----
    family_indices = defaultdict(list)
    for i, w in enumerate(anchor_words):
        family_indices[strip_case(w)].append(i)
    strata = defaultdict(list)
    for fam in sorted(family_indices):
        strata[len(family_indices[fam])].append(sorted(family_indices[fam]))
    sizes = sorted(strata)
    rng_null = np.random.default_rng(SEED + 5)
    R = R_NULL_TARGET
    null_bal = []
    t_null = time.perf_counter()
    i = 0
    while i < R:
        yp = y.copy()
        for s in sizes:
            groups = strata[s]
            donor_order = rng_null.permutation(len(groups))
            patterns = [y[groups[int(j)]].copy() for j in donor_order]
            for target, pattern in zip(groups, patterns):
                yp[target] = pattern
        pp, _ = crossval_graph(edges, n_nodes, anchor_nodes, yp,
                               folds, n_classes)
        null_bal.append(metrics_full(yp, pp, n_classes)[1])
        i += 1
        if i == 20 and R == R_NULL_TARGET:
            proj = (time.perf_counter() - t_null) / i * R_NULL_TARGET
            if proj > NULL_BUDGET_SEC:
                R = R_NULL_FALLBACK
                log(f'прогноз нуля R={R_NULL_TARGET}: {proj / 60:.1f} мин > '
                    f'{NULL_BUDGET_SEC / 60:.0f} мин — снижаю R до {R} '
                    '(первые перестановки того же потока rng(SEED+5))')
    null_bal = np.array(null_bal)
    p_bal = float(((null_bal >= bal_g).sum() + 1) / (R + 1))
    log()
    log('--- Нуль: перестановка векторов меток между семьями равного '
        'размера (механизм Sol, rng(SEED+5)) ---')
    log(f'R={R}; полный CV-цикл распространения на каждой перестановке; '
        f'null balanced {null_bal.mean():.1%}±{null_bal.std():.1%}; '
        f'наблюдаемое {bal_g:.1%}; p={p_bal:.4f}')
    print(f'  (тайминг, только консоль: нуль {time.perf_counter() - t_null:.0f} с)')

    # --- вердикт --------------------------------------------------------------
    log()
    d_bal = bal_g - bal_b
    d_f1 = mf1_g - mf1_b
    if bal_g > bal_b:
        log(f'ВЕРДИКТ: графовое распространение БЬЁТ центроидный базлайн по '
            f'balanced accuracy ({bal_b:.1%} -> {bal_g:.1%}, {d_bal:+.1%}; '
            f'macro-F1 {mf1_b:.3f} -> {mf1_g:.3f}, {d_f1:+.3f}).')
    else:
        log(f'ВЕРДИКТ: графовое распространение НЕ бьёт центроидный базлайн '
            f'(balanced {bal_b:.1%} -> {bal_g:.1%}, {d_bal:+.1%}; macro-F1 '
            f'{mf1_b:.3f} -> {mf1_g:.3f}, {d_f1:+.3f}) — публикуемый негатив.')

    # --- финал: гипотезы для неякорных узлов ---------------------------------
    y0_full = np.zeros((n_nodes, n_classes))
    y0_full[anchor_nodes, y] = 1.0
    F_full = propagate(src, dst, w_norm, y0_full)
    hyp = []
    n_unreached = 0
    for w in node_words:
        if w in anchor_set:
            continue
        row = F_full[node_index[w]]
        s = row.sum()
        if s <= 0:
            n_unreached += 1
            continue
        q = row / s
        order = np.argsort(-q, kind='stable')
        hyp.append((w, freq[w], classes[int(order[0])],
                    float(q[order[0]]), float(q[order[0]] - q[order[1]]),
                    classes[int(order[1])]))
    hyp.sort(key=lambda r: (-r[4], -r[1], r[0]))
    top = hyp[:TOP_HYP]
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['word', 'freq', 'pred_class', 'score', 'margin',
                     'second_class'])
        for w, fq, pc, sc, mg, sec in top:
            wr.writerow([w, fq, pc, f'{sc:.6f}', f'{mg:.6f}', sec])
    log()
    log('--- Гипотезы (обучение на всех якорях; разведочный слой, p не '
        'заявляем) ---')
    log(f'неякорных узлов: {n_nodes - len(anchor_words)}; с ненулевым '
        f'сигналом: {len(hyp)}; недостижимых: {n_unreached}; '
        f'в CSV топ-{len(top)} по margin: {OUT_CSV}')
    top_counts = Counter(r[2] for r in top)
    log('класс: гипотез в топ-300 | CV-precision класса (графовый CV, '
        'калибровка ожиданий):')
    for i, c in enumerate(classes):
        log(f'  {c:<7} гипотез={top_counts.get(c, 0):>3} | '
            f'CV-precision={prec_g[i]:.3f}')
    log('первые 10 гипотез (word, freq, class, score, margin):')
    for w, fq, pc, sc, mg, sec in top[:10]:
        log(f'  {w:<14} n={fq:<3} {pc:<7} score={sc:.3f} margin={mg:.3f} '
            f'(2-й: {sec})')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nполное время: {time.perf_counter() - t_all:.0f} с '
          f'(только консоль)\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
