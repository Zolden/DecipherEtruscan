# -*- coding: utf-8 -*-
"""§3v2: семантика с учителем — контекст записи + структурная разметка ETP_POS.

Отличия от v1 (tools/etr_semantics.py):
  МЕТКИ: к глоссам Хилла добавлены структурные столбцы ETP_POS
    (theo=1 → THEO; TAG=VERB → VERB; TAG∈{PRON,DET,ADP,PRT} → FUNC;
    masc/fem=1 → NAME-M/F; внутри ETP_POS приоритет THEO>VERB>FUNC>NAME).
    При конфликте источников приоритет у ETP_POS (структурная разметка
    научной традиции); конфликты логируются.
  ПРИЗНАКИ: + контекст записей — соседние операторные группы (v1),
    со-встречающиеся частотные слова (топ-200), суффиксный тип соседей
    (-s/-l/-al/...), доля одно-словных вхождений.
  Цель: открыть классы THEO/VERB/FUNC, не видимые по форме слова (N14).

Модель/оценка как в v1: биномиальный NB (numpy), train/test 80/20 по
типам (seed=42), перестановочный нуль R=1000, калибровка 5-fold CV.
Выход: results/semantic_hypotheses_v2.csv.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_semantics2.py
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
OUT_LOG = os.path.join('logs', 'etr_semantics2.log')
OUT_CSV = os.path.join('results', 'semantic_hypotheses_v2.csv')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def view_of(corpus):
    return [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]


def gloss_class(g):
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


def to_ascii_word(w):
    w = re.sub(r'[^a-zθχφσςśšê\']', '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def etp_pos_labels():
    out = {}
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            w = to_ascii_word(r.get('Etruscan'))
            if not w or len(w) < 3:
                continue
            tag = (r.get('TAG') or '').strip()
            lab = None
            if (r.get('theo') or '').strip() == '1':
                lab = 'THEO'
            elif tag == 'VERB':
                lab = 'VERB'
            elif tag in ('PRON', 'DET', 'ADP', 'PRT'):
                lab = 'FUNC'
            elif (r.get('masc') or '').strip() == '1':
                lab = 'NAME-M'
            elif (r.get('fem') or '').strip() == '1':
                lab = 'NAME-F'
            if lab:
                out.setdefault(w, Counter())[lab] += 1
    final = {}
    for w, votes in out.items():
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            order = {'THEO': 0, 'VERB': 1, 'FUNC': 2, 'NAME-M': 3,
                     'NAME-F': 3}
            top.sort(key=lambda t: order.get(t[0], 9))
        final[w] = top[0][0]
    return final


OP_FORMS = {
    'mi': {'mi', 'mini', 'mine'}, 'kin': {'clan', 'clens', 'clensi', 'sec',
    'sech', 'puia', 'ati', 'apa'}, 'vital': {'lupu', 'lupuce', 'svalce',
    'avils', 'avil', 'ril'}, 'dedic': {'turce', 'turuce', 'muluvanice',
    'mulvanice', 'zinace', 'cver'}, 'deix': {'itun', 'ita', 'ica', 'eca',
    'ca', 'cn', 'cen', 'thui'}, 'tomb': {'suthi', 'suti'},
}
NBR_SUF = ('s', 'l', 'al', 'ial', 'sa', 'isa', 'ce')


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.6'
    view = view_of(corpus)
    log('=== Семантика v2: контекст + ETP_POS (этап 3v2) ===')
    log(f'вид: {len(view)} записей; R_perm={R_PERM}, seed={SEED}')

    # --- словные статистики и контексты ------------------------------------
    word_freq = Counter()
    word_glosses = {}
    word_regions = {}
    word_nbrops = {}
    word_pos = {}
    word_ctx = {}
    word_nbrsuf = {}
    word_single = Counter()
    for r in view:
        ws = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
        ok = [w for w in ws if '-' not in w and len(w) >= 3]
        for i, w in enumerate(ws):
            if '-' in w or len(w) < 3:
                continue
            word_freq[w] += 1
            if len(ws) == 1:
                word_single[w] += 1
            if r['region']:
                word_regions.setdefault(w, Counter())[r['region']] += 1
            if len(ws) >= 2:
                for gname, forms in OP_FORMS.items():
                    if any(x in forms for x in ws if x != w):
                        word_nbrops.setdefault(w, set()).add(gname)
                word_pos.setdefault(w, []).append(i / (len(ws) - 1))
                for x in ok:
                    if x != w:
                        word_ctx.setdefault(w, Counter())[x] += 1
                for j in (i - 1, i + 1):
                    if 0 <= j < len(ws):
                        for suf in NBR_SUF:
                            if ws[j].endswith(suf):
                                word_nbrsuf.setdefault(w, set()).add(suf)
                                break
        if r['trs'] and len(ws) == 1 and '-' not in ws[0] and len(ws[0]) >= 3:
            word_glosses.setdefault(ws[0], []).append(
                ' '.join(r['trs']).lower())

    ctx_vocab = [w for w, _ in word_freq.most_common(200)]
    ctx_set = set(ctx_vocab)

    # --- метки ---------------------------------------------------------------
    hill = {}
    for w, gs in word_glosses.items():
        votes = Counter(gloss_class(g) for g in gs)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue
        hill[w] = top[0][0]
    etp = {w: c for w, c in etp_pos_labels().items() if w in word_freq}
    conflicts = [(w, etp[w], hill[w]) for w in set(etp) & set(hill)
                 if etp[w] != hill[w]]
    log(f'метки: Хилл {len(hill)} типов; ETP_POS {len(etp)} типов '
        f'(в корпусе); конфликтов {len(conflicts)} — приоритет ETP_POS')
    for w, e, h in sorted(conflicts)[:8]:
        log(f'   конфликт: {w}: ETP_POS={e} vs Хилл={h}')
    labeled = dict(hill)
    labeled.update(etp)
    cnt = Counter(labeled.values())
    log(f'размеченных типов: {len(labeled)}; классы: '
        f'{dict(cnt.most_common())}')
    small = {c for c, n in cnt.items() if n < 15}
    if small:
        log(f'классы с n<15 влиты в OTHER: {sorted(small)}')
        labeled = {w: ('OTHER' if c in small else c)
                   for w, c in labeled.items()}
    classes = sorted(set(labeled.values()))
    log(f'итоговые классы: {dict(Counter(labeled.values()).most_common())}')

    # --- признаки ------------------------------------------------------------
    def word_feats(w):
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
        f.add(f'freq:{min(word_freq[w], 5)}')
        if w in word_regions:
            f.add(f'reg:{word_regions[w].most_common(1)[0][0]}')
        for op in word_nbrops.get(w, ()):
            f.add(f'nbr:{op}')
        if w in word_pos:
            f.add(f'pos:{int(float(np.mean(word_pos[w])) * 3)}')
        for x, c in word_ctx.get(w, Counter()).most_common(20):
            if x in ctx_set:
                f.add(f'ctx:{x}')
        for suf in word_nbrsuf.get(w, ()):
            f.add(f'nbrsuf:{suf}')
        sf = word_single[w] / word_freq[w]
        f.add(f'single:{int(sf * 3)}')
        return f

    words_lab = sorted(labeled)
    feat_index = {}
    rows_feats = []
    for w in words_lab:
        fs = word_feats(w)
        rows_feats.append(fs)
        for f in fs:
            if f not in feat_index:
                feat_index[f] = len(feat_index)
    nF = len(feat_index)
    nC = len(classes)
    log(f'признаков: {nF}')
    X = np.zeros((len(words_lab), nF), dtype=np.int8)
    for i, fs in enumerate(rows_feats):
        for f in fs:
            X[i, feat_index[f]] = 1
    y = np.array([classes.index(labeled[w]) for w in words_lab])

    rng = np.random.default_rng(SEED)
    test_idx = []
    for c in range(nC):
        idx = np.where(y == c)[0]
        idx = idx[rng.permutation(len(idx))]
        test_idx += list(idx[:max(1, len(idx) // 5)])
    test_mask = np.zeros(len(y), bool)
    test_mask[test_idx] = True
    tr, te = ~test_mask, test_mask
    log(f'train: {int(tr.sum())}, test: {int(te.sum())}')

    def nb_fit_predict(Xtr, ytr, Xte):
        prior = np.log(np.bincount(ytr, minlength=nC) + 1)
        like1 = np.zeros((nC, nF))
        like0 = np.zeros((nC, nF))
        for c in range(nC):
            Xc = Xtr[ytr == c]
            n1 = Xc.sum(axis=0)
            like1[c] = np.log((n1 + 1) / (len(Xc) + 2))
            like0[c] = np.log((len(Xc) - n1 + 1) / (len(Xc) + 2))
        lp = (prior[None, :] + Xte @ (like1 - like0).T
              + like0.sum(axis=1)[None, :])
        lp -= lp.max(axis=1, keepdims=True)
        P = np.exp(lp)
        P /= P.sum(axis=1, keepdims=True)
        return P

    P_te = nb_fit_predict(X[tr], y[tr], X[te])
    pred = P_te.argmax(axis=1)
    acc = float((pred == y[te]).mean())
    maj = int(np.bincount(y[tr]).argmax())
    acc_maj = float((y[te] == maj).mean())
    log()
    log(f'--- качество (отложенные {int(te.sum())} типов) ---')
    log(f'точность NB v2: {acc:.1%}; мажоритарный: {acc_maj:.1%} '
        f'(v1 было: 60.0% при 5 классах — прямое сравнение см. ниже)')
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
    log('матрица ошибок (строки — истина):')
    log('        ' + ' '.join(f'{c:>7}' for c in classes))
    for c in range(nC):
        log(f'{classes[c]:<8}' + ' '.join(f'{conf[c, d]:>7}'
                                          for d in range(nC)))

    # абляция: без контекстных признаков (= v1-набор) на тех же метках
    ctx_pref = ('ctx:', 'nbr:', 'nbrsuf:', 'pos:', 'single:')
    keep = np.array([not any(f.startswith(p) for p in ctx_pref)
                     for f, _ in sorted(feat_index.items(),
                                        key=lambda kv: kv[1])])
    Xa = X[:, keep]

    def nb_fp(Xtr, ytr, Xte, nf):
        prior = np.log(np.bincount(ytr, minlength=nC) + 1)
        like1 = np.zeros((nC, nf))
        like0 = np.zeros((nC, nf))
        for c in range(nC):
            Xc = Xtr[ytr == c]
            n1 = Xc.sum(axis=0)
            like1[c] = np.log((n1 + 1) / (len(Xc) + 2))
            like0[c] = np.log((len(Xc) - n1 + 1) / (len(Xc) + 2))
        lp = (prior[None, :] + Xte @ (like1 - like0).T
              + like0.sum(axis=1)[None, :])
        return lp.argmax(axis=1)

    acc_abl = float((nb_fp(Xa[tr], y[tr], Xa[te], int(keep.sum()))
                     == y[te]).mean())
    log(f'абляция (только формальные признаки, те же метки): {acc_abl:.1%} '
        f'→ вклад контекста: {acc - acc_abl:+.1%}')

    # перестановочный нуль
    rng2 = np.random.default_rng(SEED)
    accs = np.zeros(R_PERM)
    ytr_c = y[tr].copy()
    for r_i in range(R_PERM):
        yp = ytr_c[rng2.permutation(len(ytr_c))]
        Pp = nb_fit_predict(X[tr], yp, X[te])
        accs[r_i] = (Pp.argmax(axis=1) == y[te]).mean()
    p_acc = float(((accs >= acc).sum() + 1) / (R_PERM + 1))
    log(f'перестановочный нуль: среднее {accs.mean():.1%}, '
        f'max {accs.max():.1%}, p={p_acc:.4f}')

    # калибровка (CV-бины)
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
        mm = (cv_conf >= BINS[b]) & (cv_conf < BINS[b + 1])
        bin_acc[b] = float(cv_ok[mm].mean()) if mm.sum() >= 5 else None

    def calibrate(conf):
        b = np.searchsorted(BINS, conf, side='right') - 1
        return np.array([bin_acc.get(int(x)) or float(conf[i])
                         for i, x in enumerate(b)])

    confdc = P_te.max(axis=1)
    cal_te = calibrate(confdc)
    log()
    log('--- калибровка на тесте ---')
    for b in range(len(BINS) - 1):
        mm = (confdc >= BINS[b]) & (confdc < BINS[b + 1])
        if mm.sum():
            log(f'[{BINS[b]:.1f},{min(BINS[b + 1], 1.0):.1f}) n={int(mm.sum()):>4} '
                f'эмп={float((pred[mm] == y[te][mm]).mean()):.1%} '
                f'калибр={float(cal_te[mm].mean()):.1%}')

    # --- гипотезы v2 ----------------------------------------------------------
    REGISTRY = set('mi mini mine clan clens clensi sec sech puia ati apa '
                   'lupu lupuce svalce svalthas svalas avils avil ril turce '
                   'turuce turice turke muluvanice muluvanike mulvanice '
                   'mulvenice mulvannice muluvenice zinace zinake zilath '
                   'zilach zilth zilc zilci zilachnce zilachnuce suthi suti '
                   'suthith suthiu thui cver cvera tular spur spurana '
                   'spureni ais eis aisar eiser ame amce naper tiur tiurs '
                   'itun ita ica eca ca cn cen cehen etnam vacl fler'.split())
    unlabeled = sorted(w for w in word_freq
                       if w not in labeled and len(w) >= 3 and '-' not in w
                       and w not in REGISTRY)
    log(f'(v5: из гипотез исключены {len(REGISTRY)} форм реестра §1 — '
        f'функциональные слова, см. N15/§3.9)')
    Xu = np.zeros((len(unlabeled), nF), dtype=np.int8)
    for i, w in enumerate(unlabeled):
        for f in word_feats(w):
            j = feat_index.get(f)
            if j is not None:
                Xu[i, j] = 1
    Pu = nb_fit_predict(X, y, Xu)
    pu = Pu.argmax(axis=1)
    conf_u = Pu.max(axis=1)
    cal_u = calibrate(conf_u)
    log()
    log(f'--- гипотезы v2: {len(unlabeled)} непереведённых типов ---')
    log(f'распределение: '
        f'{dict(Counter(classes[c] for c in pu).most_common())}')
    log(f'калиброванная уверенность ≥0.7: {int((cal_u >= 0.7).sum())}; '
        f'≥0.6: {int((cal_u >= 0.6).sum())}')
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['word', 'freq', 'pred_class', 'prob_raw', 'prob_cal']
                      + [f'p_{c}' for c in classes])
        order = np.argsort(-cal_u, kind='stable')
        for i in order:
            wcsv.writerow([unlabeled[i], word_freq[unlabeled[i]],
                           classes[pu[i]], f'{conf_u[i]:.3f}',
                           f'{cal_u[i]:.3f}'] +
                          [f'{Pu[i, c]:.3f}' for c in range(nC)])
    log(f'гипотезы записаны: {OUT_CSV}')

    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
