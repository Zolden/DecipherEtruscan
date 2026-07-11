# -*- coding: utf-8 -*-
"""§22 (серия 3, цикл 3): парадигмная специфика Etruscan↔Raetic —
сверх общей фонотактики (§19).

Вопрос: совпадают ли ПАРАДИГМЫ (пары суффиксов, живущие на одной
основе), а не только инвентарь исходов? Конструкция:
- Суффиксы: выученный S (results/mdl_suffixes_v1.csv, 60).
- Этрусские парадигмные пары E: пары (s1,s2), обе формы stem+s1 и
  stem+s2 аттестованы в каноническом виде, основа >=3; порог >=5 основ.
- Ретийская сторона: словоформы TIR (Raetic, чистые); жадное (длиннейшее)
  разложение stem+suffix по S; основы с >=2 формами дают ретийские пары.
- Статистика: доля ретийских пар, входящих в E. Нуль — перестановка
  суффиксов между ретийскими словами (основы и маргиналы суффиксов
  сохранены; R=1000, seed=42): проверяется именно СОСТЫКОВКА пар, не
  инвентарь. Мощность заведомо мала (ретийских основ с >=2 формами
  единицы) — негатив публикуем.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_tir_paradigms.py
"""
import csv
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
MIN_STEM = 3
E_MIN_STEMS = 5
OUT_LOG = os.path.join('logs', 'etr_tir_paradigms.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


TIR_MAP = {'þ': 'th', 'χ': 'ch', 'φ': 'ph', 'ś': 's', 'š': 's',
           'θ': 'th', 'ϑ': 'th'}


def norm_tir(w):
    w = (w or '').lower()
    w = ''.join(TIR_MAP.get(c, c) for c in unicodedata.normalize('NFD', w)
                if not unicodedata.combining(c))
    return re.sub(r'[^a-z]', '', w)


def main():
    os.makedirs('logs', exist_ok=True)
    log('=== §22: парадигмная сверка Etruscan↔Raetic ===')
    S = []
    with open(os.path.join('results', 'mdl_suffixes_v1.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            S.append(r['suffix'])
    S_set = set(S)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    vocab = {t['ascii'] for r in view for t in r['toks']
             if t['kind'] == 'W' and '-' not in t['ascii']
             and len(t['ascii']) >= 3}

    # --- этрусские парадигмные пары E ---------------------------------------
    stem_sufs = defaultdict(set)
    for w in sorted(vocab):
        for k in (1, 2, 3, 4):
            if len(w) - k >= MIN_STEM and w[-k:] in S_set:
                stem_sufs[w[:-k]].add(w[-k:])
    pair_stems = Counter()
    for st, sufs in stem_sufs.items():
        ss = sorted(sufs)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                pair_stems[(ss[i], ss[j])] += 1
    E = {p for p, n in pair_stems.items() if n >= E_MIN_STEMS}
    log(f'этрусских парадигмных пар (>= {E_MIN_STEMS} основ): {len(E)} '
        f'из {len(pair_stems)} наблюдённых')
    top_e = sorted(pair_stems.items(), key=lambda kv: -kv[1])[:8]
    log('топ E: ' + ', '.join(f'(-{a},-{b}):{n}' for (a, b), n in top_e))

    # --- ретийское разложение -------------------------------------------------
    raet = set()
    with open(os.path.join('data', 'external', 'tir', 'tir_words.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'Raetic' in (r['language'] or ''):
                w = norm_tir(r['title'])
                if len(w) >= MIN_STEM + 1:
                    raet.add(w)
    raet = sorted(raet)

    def decomp(w):
        for k in (4, 3, 2, 1):
            if len(w) - k >= MIN_STEM and w[-k:] in S_set:
                return w[:-k], w[-k:]
        return None

    dec = [d for d in (decomp(w) for w in raet) if d]
    r_stems = defaultdict(set)
    for st, su in dec:
        r_stems[st].add(su)
    multi = {st: sorted(ss) for st, ss in r_stems.items() if len(ss) >= 2}
    r_pairs = []
    for st, ss in sorted(multi.items()):
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                r_pairs.append((st, (ss[i], ss[j])))
    log(f'ретийских словоформ: {len(raet)}; разложимо по S: {len(dec)}; '
        f'основ с >=2 формами: {len(multi)}; ретийских пар: {len(r_pairs)}')
    for st, ss in sorted(multi.items()):
        att = [f'-{s}' + ('*' if tuple(sorted((ss[0], s))) else '')
               for s in ss]
        log(f'  основа {st:<10} формы: ' + ' '.join('-' + s for s in ss))
    if not r_pairs:
        log('НЕГАТИВ: ретийских парадигмных пар нет — тест не выполним '
            'на данных TIR; публикуем как границу.')
    else:
        obs = sum(1 for _, p in r_pairs if p in E) / len(r_pairs)
        log(f'\nдоля ретийских пар в E: {obs:.1%} '
            f'({sum(1 for _, p in r_pairs if p in E)}/{len(r_pairs)})')
        # нуль: перестановка суффиксов между ретийскими словами
        stems_seq = [st for st, _ in dec]
        sufs_seq = [su for _, su in dec]
        rng = np.random.default_rng(SEED)
        sims = np.zeros(R)
        for r_i in range(R):
            perm = rng.permutation(len(sufs_seq))
            rs = defaultdict(set)
            for st, pi in zip(stems_seq, perm):
                rs[st].add(sufs_seq[pi])
            tot = hit = 0
            for st, ss in rs.items():
                ss = sorted(ss)
                for i in range(len(ss)):
                    for j in range(i + 1, len(ss)):
                        tot += 1
                        hit += (ss[i], ss[j]) in E
            sims[r_i] = hit / max(tot, 1)
        p = float((1 + (sims >= obs - 1e-12).sum()) / (R + 1))
        log(f'нуль (перестановка суффиксов, основы сохранены): '
            f'{sims.mean():.1%}±{sims.std():.1%}; p={p:.4f}')
        log('чтение: малый n — результат разведочный при любом исходе; '
            'совпадение пар сверх нуля значило бы общность ПАРАДИГМ, '
            'не только исходов.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
