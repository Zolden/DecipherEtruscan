# -*- coding: utf-8 -*-
"""§3.7 (v3): различима ли морфологически структура имени —
преномен (личное имя) vs гентилиций (родовое)?

Метки: столбцы prae/nomen ETP_POS (только типы, встречающиеся в корпусе
v0.4; типы с обеими метками отброшены с логом). Признаки — только ФОРМА
слова (конечные 1–3-граммы, начальные 1–2, биграммы, длина). NB, 80/20
(seed=42), перестановочный нуль R=1000; топ различающих признаков.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_semantics3.py
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
R_PERM = 1000
OUT_LOG = os.path.join('logs', 'etr_semantics3.log')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def to_ascii_word(w):
    w = re.sub(r'[^a-zθχφσςśšê\']', '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def feats(w):
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
    return f


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    in_corpus = {t['ascii'] for r in corpus['records'] for t in r['toks']
                 if t['kind'] == 'W'}
    votes = {}
    with open(os.path.join('data', 'ETP_POS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = to_ascii_word(row.get('Etruscan'))
            if len(w) < 3 or w not in in_corpus:
                continue
            p = (row.get('prae') or '').strip() == '1'
            n = (row.get('nomen') or '').strip() == '1'
            if p:
                votes.setdefault(w, set()).add(0)
            if n:
                votes.setdefault(w, set()).add(1)
    both_words = sorted(w for w, labs in votes.items() if len(labs) > 1)
    lab = {w: next(iter(labs)) for w, labs in votes.items()
           if len(labs) == 1}
    words = sorted(lab)
    y = np.array([lab[w] for w in words])
    log('=== §3.7 (v3): преномен vs гентилиций по форме слова ===')
    log(f'типов: {len(words)} (prae {int((y == 0).sum())}, '
        f'nomen {int((y == 1).sum())}; с обеими метками отброшено '
        f'{len(both_words)}: {both_words})')

    fi = {}
    rows = []
    for w in words:
        fs = feats(w)
        rows.append(fs)
        for f_ in fs:
            fi.setdefault(f_, len(fi))
    X = np.zeros((len(words), len(fi)), np.int8)
    for i, fs in enumerate(rows):
        for f_ in fs:
            X[i, fi[f_]] = 1
    rng = np.random.default_rng(SEED)
    te = np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        te[idx[rng.permutation(len(idx))][:len(idx) // 5]] = True
    tr = ~te

    def nb(Xtr, ytr, Xte):
        pri = np.log(np.bincount(ytr, minlength=2) + 1)
        l1 = np.zeros((2, X.shape[1]))
        l0 = np.zeros((2, X.shape[1]))
        for c in (0, 1):
            Xc = Xtr[ytr == c]
            n1 = Xc.sum(axis=0)
            l1[c] = np.log((n1 + 1) / (len(Xc) + 2))
            l0[c] = np.log((len(Xc) - n1 + 1) / (len(Xc) + 2))
        lp = pri[None, :] + Xte @ (l1 - l0).T + l0.sum(axis=1)[None, :]
        return lp.argmax(axis=1), l1 - l0

    pred, w_mat = nb(X[tr], y[tr], X[te])
    acc = float((pred == y[te]).mean())
    maj = float((y[te] == np.bincount(y[tr]).argmax()).mean())
    accs = np.zeros(R_PERM)
    ytr_ = y[tr].copy()
    for r_i in range(R_PERM):
        accs[r_i] = (nb(X[tr], ytr_[rng.permutation(len(ytr_))],
                        X[te])[0] == y[te]).mean()
    p = float(((accs >= acc).sum() + 1) / (R_PERM + 1))
    log(f'точность: {acc:.1%} (мажоритарный {maj:.1%}; нуль '
        f'{accs.mean():.1%}, max {accs.max():.1%}; p={p:.4f})')
    _, w_full = nb(X, y, X[:1])
    diff = w_full[1] - w_full[0]  # >0 → за гентилиций
    names = sorted(fi, key=fi.get)
    order = np.argsort(diff)
    log('топ признаков ЗА гентилиций: '
        + ', '.join(f'{names[i]}({diff[i]:+.1f})' for i in order[-8:][::-1]
                    if names[i].startswith(('suf', 'pre'))))
    log('топ признаков ЗА преномен: '
        + ', '.join(f'{names[i]}({diff[i]:+.1f})' for i in order[:8]
                    if names[i].startswith(('suf', 'pre'))))
    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
