# -*- coding: utf-8 -*-
"""Этап 3 (§3): дистрибутивная семантика с учителем — классы слов.

Задача: предсказывать СЕМАНТИЧЕСКИЙ КЛАСС слова (не перевод) с
калиброванной вероятностью; обучение и метрика — на словах с глоссами,
применение — ранжированные гипотезы для слов без перевода.

Данные: одно-словные переведённые записи вида v0.3 (тип слова = ascii-
форма; глоссы агрегируются по типу). Классы по фиксированным априорным
правилам на глоссах (порядок = приоритет):
  NAME-M  глосса начинается «mr-»
  NAME-F  «mrs-» / «ms-»
  THEO    god|goddess|deity|divine
  KIN     son|daughter|wife|husband|mother|father
  VERB    англ. прошедшее (\\w+ed) | gave|built|made|wrote|is|was
  OTHER   остальное
Классы с n<15 вливаются в OTHER (правило объявлено заранее).

Признаки БЕЗ УТЕЧКИ переводов: конечные n-граммы 1–3 ($-якорь), начальные
1–2 (^-якорь), внутренние биграммы, длина (бакет), частота (бакет),
регион большинства, соседи-операторы и позиция (для слов, встречающихся
и в мультисловных записях).

Модель: биномиальный наивный Байес (add-1), реализация numpy.
Оценка: train/test 80/20 по ТИПАМ слов (детерминированная стратификация,
seed=42), точность против (а) мажоритарного базлайна, (б) правила
гендерных суффиксов из §2; перестановочный нуль (R=1000 перетренировок
на перемешанных метках). Калибровка: надёжность по децилям уверенности.
Выход: results/semantic_hypotheses_v1.csv — ранжированные гипотезы
классов для непереведённых слов.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_semantics.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

R_PERM = 1000
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_semantics.log')
OUT_CSV = os.path.join('results', 'semantic_hypotheses_v1.csv')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def view_of(corpus):
    return [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]


def gloss_class(g):
    """Класс глоссы. Приоритет: маркер лица (mrs-/ms-/mr-) по самой ранней
    позиции — покрывает и генитивы «of-mr-…»; затем THEO/KIN/VERB/OTHER."""
    # позиции маркеров; 'mr-' не должен срабатывать внутри 'mrs-'
    g2 = g.replace('mrs-', '###-').replace('ms-', '###-')
    i_f = min([i for i in (g.find('mrs-'), g.find('ms-')) if i != -1],
              default=-1)
    i_m = g2.find('mr-')
    if i_f != -1 and (i_m == -1 or i_f < i_m):
        return 'NAME-F'
    if i_m != -1:
        return 'NAME-M'
    if re.search(r'\bgod|goddess|deit|divine', g):
        return 'THEO'
    if re.search(r'\bson\b|\bdaughter|\bwife\b|\bhusband|\bmother\b'
                 r'|\bfather\b', g):
        return 'KIN'
    if re.search(r'\w+ed\b|\bgave\b|\bbuilt\b|\bmade\b|\bwrote\b|\bis\b'
                 r'|\bwas\b', g):
        return 'VERB'
    return 'OTHER'

OP_FORMS = {  # операторные группы §1 (ascii) для признаков-соседей
    'mi': {'mi', 'mini', 'mine'}, 'kin': {'clan', 'clens', 'clensi', 'sec',
    'sech', 'puia', 'ati', 'apa'}, 'vital': {'lupu', 'lupuce', 'svalce',
    'avils', 'avil', 'ril'}, 'dedic': {'turce', 'turuce', 'muluvanice',
    'mulvanice', 'zinace', 'cver'}, 'deix': {'itun', 'ita', 'ica', 'eca',
    'ca', 'cn', 'cen', 'thui'}, 'tomb': {'suthi', 'suti'},
}


def features_of(w, freq, region, nbr_ops, mean_rel_pos):
    f = set()
    for k in (1, 2, 3):
        if len(w) > k:
            f.add(f'suf{k}:{w[-k:]}')
    for k in (1, 2):
        if len(w) > k:
            f.add(f'pre{k}:{w[:k]}')
    for i in range(len(w) - 1):
        f.add(f'bg:{w[i:i + 2]}')
    f.add(f'len:{min(len(w), 10)}')
    f.add(f'freq:{min(freq, 5)}')
    if region:
        f.add(f'reg:{region}')
    for op in nbr_ops:
        f.add(f'nbr:{op}')
    if mean_rel_pos is not None:
        f.add(f'pos:{int(mean_rel_pos * 3)}')  # трети записи
    return f


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = view_of(corpus)
    log('=== Семантика с учителем: классы слов (этап 3) ===')
    log(f'вид: {len(view)} записей; R_perm={R_PERM}, seed={SEED}')

    # --- сбор словных данных ----------------------------------------------
    word_glosses = {}
    word_freq = Counter()
    word_regions = {}
    word_nbrs = {}
    word_pos = {}
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        for i, w in enumerate(ws):
            if '-' in w or len(w) < 3:
                continue
            word_freq[w] += 1
            if r['region']:
                word_regions.setdefault(w, Counter())[r['region']] += 1
            if len(ws) >= 2:
                for gname, forms in OP_FORMS.items():
                    if any(x in forms for x in ws if x != w):
                        word_nbrs.setdefault(w, set()).add(gname)
                word_pos.setdefault(w, []).append(i / (len(ws) - 1))
        if r['trs'] and len(ws) == 1 and '-' not in ws[0] and len(ws[0]) >= 3:
            word_glosses.setdefault(ws[0], []).append(
                ' '.join(r['trs']).lower())

    def label_of(glosses):
        votes = Counter(gloss_class(g) for g in glosses)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None  # ничья — тип отбрасывается (лог ниже)
        return top[0][0]

    labeled = {}
    ties = 0
    for w, gs in word_glosses.items():
        lab = label_of(gs)
        if lab is None:
            ties += 1
        else:
            labeled[w] = lab
    cnt = Counter(labeled.values())
    log(f'размеченных типов: {len(labeled)} (ничьих отброшено: {ties}); '
        f'классы: {dict(cnt.most_common())}')
    small = {c for c, n in cnt.items() if n < 15}
    if small:
        log(f'классы с n<15 влиты в OTHER: {sorted(small)}')
        labeled = {w: ('OTHER' if c in small else c)
                   for w, c in labeled.items()}
        cnt = Counter(labeled.values())
    classes = sorted(cnt)
    log(f'итоговые классы: {dict(Counter(labeled.values()).most_common())}')

    # --- матрица признаков --------------------------------------------------
    def word_feats(w):
        reg = None
        if w in word_regions:
            reg = word_regions[w].most_common(1)[0][0]
        mp = None
        if w in word_pos:
            mp = float(np.mean(word_pos[w]))
        return features_of(w, word_freq[w], reg,
                           word_nbrs.get(w, ()), mp)

    words_lab = sorted(labeled)  # детерминизм
    feats_lab = [word_feats(w) for w in words_lab]
    feat_index = {}
    for fs in feats_lab:
        for f in fs:
            if f not in feat_index:
                feat_index[f] = len(feat_index)
    nF = len(feat_index)
    log(f'признаков: {nF}')
    X = np.zeros((len(words_lab), nF), dtype=np.int8)
    for i, fs in enumerate(feats_lab):
        for f in fs:
            X[i, feat_index[f]] = 1
    y = np.array([classes.index(labeled[w]) for w in words_lab])
    nC = len(classes)

    # --- train/test (стратифицированно, seed) -------------------------------
    rng = np.random.default_rng(SEED)
    test_idx = []
    for c in range(nC):
        idx = np.where(y == c)[0]
        idx = idx[rng.permutation(len(idx))]
        test_idx += list(idx[:max(1, len(idx) // 5)])
    test_mask = np.zeros(len(y), bool)
    test_mask[test_idx] = True
    tr, te = ~test_mask, test_mask
    log(f'train: {int(tr.sum())}, test: {int(te.sum())} (стратиф. 80/20)')

    def nb_fit_predict(Xtr, ytr, Xte):
        prior = np.log(np.bincount(ytr, minlength=nC) + 1)
        like1 = np.zeros((nC, nF))
        like0 = np.zeros((nC, nF))
        for c in range(nC):
            Xc = Xtr[ytr == c]
            n1 = Xc.sum(axis=0)
            like1[c] = np.log((n1 + 1) / (len(Xc) + 2))
            like0[c] = np.log((len(Xc) - n1 + 1) / (len(Xc) + 2))
        lp = prior[None, :] + Xte @ (like1 - like0).T + like0.sum(axis=1)[None, :]
        lp -= lp.max(axis=1, keepdims=True)
        P = np.exp(lp)
        P /= P.sum(axis=1, keepdims=True)
        return P

    P_te = nb_fit_predict(X[tr], y[tr], X[te])
    pred = P_te.argmax(axis=1)
    acc = float((pred == y[te]).mean())
    maj = int(np.bincount(y[tr]).argmax())
    acc_maj = float((y[te] == maj).mean())

    # базлайн гендерных суффиксов §2
    def gender_rule(w):
        if w.endswith(('i', 'ei')) and not w.endswith('vi'):
            return classes.index('NAME-F') if 'NAME-F' in classes else maj
        if w.endswith(('s', 'as', 'es', 'us')):
            return classes.index('NAME-M') if 'NAME-M' in classes else maj
        return maj
    te_words = [w for w, m in zip(words_lab, te) if m]
    acc_rule = float(np.mean([gender_rule(w) == y[te][i]
                              for i, w in enumerate(te_words)]))

    log()
    log(f'--- качество на отложенных типах ---')
    log(f'точность NB: {acc:.1%}; мажоритарный: {acc_maj:.1%}; '
        f'правило суффиксов §2: {acc_rule:.1%}')
    log(f'{"класс":<8} {"n_te":>5} {"точн.":>6} {"полн.":>6} {"F1":>6}')
    for c in range(nC):
        tp = int(((pred == c) & (y[te] == c)).sum())
        fp = int(((pred == c) & (y[te] != c)).sum())
        fn = int(((pred != c) & (y[te] == c)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        log(f'{classes[c]:<8} {int((y[te] == c).sum()):>5} {pr:>6.0%} '
            f'{rc:>6.0%} {f1:>6.2f}')
    conf = np.zeros((nC, nC), int)
    for a, b in zip(y[te], pred):
        conf[a, b] += 1
    log('матрица ошибок (строки — истина, столбцы — предсказание):')
    log('      ' + ' '.join(f'{c:>7}' for c in classes))
    for c in range(nC):
        log(f'{classes[c]:<6}' + ' '.join(f'{conf[c, d]:>7}'
                                          for d in range(nC)))

    # --- перестановочный нуль ----------------------------------------------
    rng2 = np.random.default_rng(SEED)
    accs = np.zeros(R_PERM)
    ytr = y[tr].copy()
    for r_i in range(R_PERM):
        yp = ytr[rng2.permutation(len(ytr))]
        Pp = nb_fit_predict(X[tr], yp, X[te])
        accs[r_i] = (Pp.argmax(axis=1) == y[te]).mean()
    p_acc = float(((accs >= acc).sum() + 1) / (R_PERM + 1))
    log(f'перестановочный нуль (R={R_PERM}): средняя случайная точность '
        f'{accs.mean():.1%}, максимум {accs.max():.1%}; p={p_acc:.4f}')

    # --- калибровка ---------------------------------------------------------
    # сырые вероятности NB самоуверенны; строим отображение «уверенность →
    # эмпирическая точность» по 5-fold CV на train и применяем к тесту и
    # гипотезам (столбец prob_cal)
    BINS = (0.0, 0.5, 0.7, 0.9, 1.01)
    rng_cv = np.random.default_rng(SEED)
    tr_idx = np.where(tr)[0]
    fold = rng_cv.permutation(len(tr_idx)) % 5
    cv_conf, cv_ok = [], []
    for f_i in range(5):
        va = tr_idx[fold == f_i]
        fit = tr_idx[fold != f_i]
        Pv = nb_fit_predict(X[fit], y[fit], X[va])
        cv_conf.append(Pv.max(axis=1))
        cv_ok.append(Pv.argmax(axis=1) == y[va])
    cv_conf = np.concatenate(cv_conf)
    cv_ok = np.concatenate(cv_ok)
    bin_acc = {}
    for b in range(len(BINS) - 1):
        m = (cv_conf >= BINS[b]) & (cv_conf < BINS[b + 1])
        bin_acc[b] = float(cv_ok[m].mean()) if m.sum() >= 5 else None

    def calibrate(conf):
        b = np.searchsorted(BINS, conf, side='right') - 1
        return np.array([
            float(conf[i]) if bin_acc.get(int(x)) is None else bin_acc[int(x)]
            for i, x in enumerate(b)
        ])

    log()
    log('--- калибровка (CV-отображение бинов; проверка на тесте) ---')
    confdc = P_te.max(axis=1)
    cal_te = calibrate(confdc)
    log(f'{"бин сырой ув.":<14} {"n":>4} {"эмп. точн.":>10} '
        f'{"калибр. оценка":>15}')
    for b in range(len(BINS) - 1):
        m = (confdc >= BINS[b]) & (confdc < BINS[b + 1])
        if m.sum():
            log(f'[{BINS[b]:.1f}, {min(BINS[b + 1], 1.0):.1f})    '
                f'{int(m.sum()):>4} '
                f'{float((pred[m] == y[te][m]).mean()):>10.1%} '
                f'{float(cal_te[m].mean()):>15.1%}')

    # --- гипотезы для непереведённых слов -----------------------------------
    log()
    log('--- гипотезы для непереведённых типов ---')
    unlabeled = sorted(w for w in word_freq
                       if w not in word_glosses and len(w) >= 3
                       and '-' not in w)
    Xu = np.zeros((len(unlabeled), nF), dtype=np.int8)
    for i, w in enumerate(unlabeled):
        for f in word_feats(w):
            j = feat_index.get(f)
            if j is not None:
                Xu[i, j] = 1
    Pu = nb_fit_predict(X, y, Xu)  # финальная модель на всех размеченных
    pu = Pu.argmax(axis=1)
    log(f'непереведённых типов: {len(unlabeled)}; '
        f'распределение предсказаний: '
        f'{dict(Counter(classes[c] for c in pu).most_common())}')
    conf_u = Pu.max(axis=1)
    cal_u = calibrate(conf_u)
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wcsv = csv.writer(f, lineterminator='\n')
        wcsv.writerow(['word', 'freq', 'pred_class', 'prob_raw',
                       'prob_cal'] + [f'p_{c}' for c in classes])
        order = np.argsort(-cal_u, kind='stable')
        for i in order:
            wcsv.writerow([unlabeled[i], word_freq[unlabeled[i]],
                           classes[pu[i]], f'{conf_u[i]:.3f}',
                           f'{cal_u[i]:.3f}'] +
                          [f'{Pu[i, c]:.3f}' for c in range(nC)])
    log(f'гипотезы записаны: {OUT_CSV} (сортировка по калиброванной '
        f'уверенности)')
    log(f'типов с калиброванной уверенностью ≥0.7: '
        f'{int((cal_u >= 0.7).sum())}; ≥0.6: {int((cal_u >= 0.6).sum())}')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
