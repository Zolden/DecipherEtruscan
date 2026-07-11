# -*- coding: utf-8 -*-
"""§13 (план Sol №5): минимальные пары формул — взаимозаменяемость слов
в идентичных контекстах.

Идея: пары записей С РАЗНЫХ ПАМЯТНИКОВ, идентичные пословно, кроме
ровно ОДНОЙ позиции, дают контролируемый контраст: слова, замещающие
друг друга в одинаковой формуле, — кандидаты одного парадигматического/
семантического класса. Это «минимальные пары» уровня формулы.

Метод: записи канонического вида, 3..8 слов; для каждой записи и каждой
позиции i ключ (n, i, кортеж без i) → корзина; пары в корзине различаются
только в i. Пары одного памятника и тождественные слова исключены.
Валидация: на рёбрах замещения, где ОБА слова имеют ETP_POS-метку
(NAME-M/F/THEO/VERB/FUNC, механизм §8.2), — доля одноклассовых против
нуля перестановки меток по размеченным узлам графа (R=1000, seed=42,
структура рёбер фиксирована).

Выход: results/minimal_pairs_v1.csv (word_a, word_b, n_contexts,
позиции, классы) + лог. Кластеры замещения — разведочный слой.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_minimal_pairs.py
"""
import csv
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
R = 1000
OUT_LOG = os.path.join('logs', 'etr_minimal_pairs.log')
OUT_CSV = os.path.join('results', 'minimal_pairs_v1.csv')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def to_ascii_word(w):
    w = re.sub(r"[^a-zθχφσςśšê']", '', (w or '').strip().lower())
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def etp_labels():
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
    for w, cnt in votes.items():
        top = cnt.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            out[w] = top[0][0]
    return out


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== §13: минимальные пары формул (план Sol №5) ===')
    recs = []
    for r in view:
        ws = tuple(t['ascii'] for t in r['toks']
                   if t['kind'] == 'W' and '-' not in t['ascii']
                   and len(t['ascii']) >= 2)
        if 3 <= len(ws) <= 8:
            recs.append((r['artifact_id'], ws))
    log(f'записей 3–8 слов: {len(recs)}')

    buckets = defaultdict(list)
    for art, ws in recs:
        n = len(ws)
        for i in range(n):
            key = (n, i, ws[:i] + ws[i + 1:])
            buckets[key].append((art, ws[i]))
    pair_ctx = defaultdict(set)   # (a,b) -> set(context keys)
    pair_pos = defaultdict(Counter)
    n_ctx = 0
    for key, items in buckets.items():
        if len(items) < 2:
            continue
        seen = sorted({(a, w) for a, w in items})
        for x in range(len(seen)):
            for y in range(x + 1, len(seen)):
                (a1, w1), (a2, w2) = seen[x], seen[y]
                if a1 == a2 or w1 == w2:
                    continue
                pr = tuple(sorted((w1, w2)))
                pair_ctx[pr].add(key)
                pair_pos[pr][key[1]] += 1
                n_ctx += 1
    pairs = {pr: len(ctxs) for pr, ctxs in pair_ctx.items()}
    log(f'корзин-контекстов с >=2 записями: '
        f'{sum(1 for its in buckets.values() if len(its) >= 2)}; '
        f'пар замещения (разные памятники): {len(pairs)}')

    lab = etp_labels()
    lab_pairs = {pr: (lab.get(pr[0]), lab.get(pr[1]))
                 for pr in pairs if pr[0] in lab and pr[1] in lab}
    log(f'пар, где оба слова размечены ETP_POS: {len(lab_pairs)}')
    if lab_pairs:
        obs_same = sum(a == b for a, b in lab_pairs.values()) / len(lab_pairs)
        # нуль: перестановка меток по размеченным узлам графа замещений
        nodes = sorted({w for pr in lab_pairs for w in pr})
        node_lab = np.array([lab[w] for w in nodes])
        ni = {w: i for i, w in enumerate(nodes)}
        edges = [(ni[a], ni[b]) for a, b in lab_pairs]
        rng = np.random.default_rng(SEED)
        sims = np.zeros(R)
        for r_i in range(R):
            pl = node_lab[rng.permutation(len(nodes))]
            sims[r_i] = sum(pl[i] == pl[j] for i, j in edges) / len(edges)
        p = float((1 + (sims >= obs_same - 1e-12).sum()) / (R + 1))
        log(f'одноклассовых рёбер: {100 * obs_same:.1f}% против нуля '
            f'{100 * sims.mean():.1f}%±{100 * sims.std():.1f}%; p={p:.4f}')
        cls_cnt = Counter(tuple(sorted(v)) for v in lab_pairs.values())
        log('классовые сочетания рёбер: '
            + ', '.join(f'{a}+{b}:{n}' for (a, b), n in
                        cls_cnt.most_common(8)))

    log('\nтоп-20 пар замещения по числу контекстов:')
    top = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    for (a, b), n in top:
        la, lb = lab.get(a, '·'), lab.get(b, '·')
        log(f'  {a:<12} ~ {b:<12} контекстов {n}  [{la}/{lb}]')

    # интересное меньшинство: замещения вне ономастики
    non_name = [(pr, n) for pr, n in pairs.items()
                if lab.get(pr[0]) not in (None, 'NAME-M', 'NAME-F')
                and lab.get(pr[1]) not in (None, 'NAME-M', 'NAME-F')]
    log(f'\nзамещения, где оба слова размечены и НЕ имена: {len(non_name)}')
    for (a, b), n in sorted(non_name, key=lambda kv: -kv[1])[:12]:
        log(f'  {a:<12} ~ {b:<12} контекстов {n}  '
            f'[{lab[a]}/{lab[b]}]')

    # --- часть 2: скелетные пары (имена → '*', слоты операторов) -----------
    OPS = {'clan': 'KIN', 'sec': 'KIN', 'sech': 'KIN', 'puia': 'KIN',
           'ruva': 'KIN', 'ati': 'KIN', 'apa': 'KIN', 'papa': 'KIN',
           'teta': 'KIN', 'nefts': 'KIN', 'tusurthir': 'KIN',
           'turce': 'VERB', 'turuce': 'VERB', 'muluvanice': 'VERB',
           'lupu': 'VERB', 'lupuce': 'VERB', 'svalce': 'VERB',
           'amce': 'VERB', 'ame': 'VERB', 'zilachnu': 'VERB',
           'zilachnce': 'VERB', 'tence': 'VERB', 'cerichunce': 'VERB',
           'zilath': 'TIT', 'zilch': 'TIT', 'zilc': 'TIT',
           'camthi': 'TIT', 'maru': 'TIT', 'purth': 'TIT',
           'marunuch': 'TIT',
           'mi': 'DEIX', 'mini': 'DEIX', 'mine': 'DEIX', 'ta': 'DEIX',
           'ca': 'DEIX', 'cn': 'DEIX', 'itun': 'DEIX', 'eca': 'DEIX',
           'avils': 'MEAS', 'avil': 'MEAS', 'ril': 'MEAS',
           'suthi': 'OBJ', 'suthina': 'OBJ', 'mutna': 'OBJ',
           'mlach': 'OBJ', 'cana': 'OBJ', 'flere': 'OBJ', 'fler': 'OBJ'}
    log('\n--- часть 2: скелетные минимальные пары (слоты операторов) ---')
    log(f'реестр операторов: {len(OPS)} форм, классы '
        f'{sorted(set(OPS.values()))}')
    sk_buckets = defaultdict(list)
    for art, ws in recs:
        sk = tuple(w if w in OPS else '*' for w in ws)
        if not any(w in OPS for w in ws):
            continue
        n = len(sk)
        for i in range(n):
            if sk[i] == '*':
                continue
            sk_buckets[(n, i, sk[:i] + sk[i + 1:])].append((art, sk[i]))
    events = []  # (context_key, op_a, op_b) по разным памятникам
    for key, items in sk_buckets.items():
        seen = sorted({(a, w) for a, w in items})
        for x in range(len(seen)):
            for y in range(x + 1, len(seen)):
                (a1, w1), (a2, w2) = seen[x], seen[y]
                if a1 != a2 and w1 != w2:
                    events.append((key, w1, w2))
    log(f'событий замещения оператор↔оператор: {len(events)}')
    if events:
        obs_same2 = sum(OPS[a] == OPS[b] for _, a, b in events) / len(events)
        # нуль: глобальная перестановка операторов по инцидентностям
        incid = []
        for key, items in sk_buckets.items():
            for a, w in sorted(set(items)):
                incid.append((key, a, w))
        ops_arr = [w for _, _, w in incid]
        rng2 = np.random.default_rng(SEED + 1)
        sims2 = np.zeros(R)
        for r_i in range(R):
            perm = rng2.permutation(len(ops_arr))
            byb = defaultdict(list)
            for (key, art, _), pi in zip(incid, perm):
                byb[key].append((art, ops_arr[pi]))
            tot = same = 0
            for key, items in byb.items():
                seen = sorted(set(items))
                for x in range(len(seen)):
                    for y in range(x + 1, len(seen)):
                        (a1, w1), (a2, w2) = seen[x], seen[y]
                        if a1 != a2 and w1 != w2:
                            tot += 1
                            same += OPS[w1] == OPS[w2]
            sims2[r_i] = same / max(tot, 1)
        p2 = float((1 + (sims2 >= obs_same2 - 1e-12).sum()) / (R + 1))
        log(f'одноклассовых замещений: {100 * obs_same2:.1f}% против нуля '
            f'{100 * sims2.mean():.1f}%±{100 * sims2.std():.1f}%; p={p2:.4f}')
        ev_cnt = Counter(tuple(sorted((a, b))) for _, a, b in events)
        log('топ замещений операторов:')
        for (a, b), n in ev_cnt.most_common(12):
            log(f'  {a:<10} ~ {b:<10} {n:>3}  [{OPS[a]}/{OPS[b]}]')

    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['word_a', 'word_b', 'n_contexts', 'positions',
                     'label_a', 'label_b'])
        for (a, b), n in sorted(pairs.items(),
                                key=lambda kv: (-kv[1], kv[0])):
            wr.writerow([a, b, n,
                         ' '.join(str(p_) for p_ in
                                  sorted(pair_pos[(a, b)])),
                         lab.get(a, ''), lab.get(b, '')])
    log(f'\ncsv записан: {OUT_CSV} ({len(pairs)} пар)')
    log('чтение: замещаемость в идентичной формуле — слабый парадигматический '
        'сигнал; классовый тест несёт p, списки — разведочные.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
