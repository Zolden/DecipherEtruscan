# -*- coding: utf-8 -*-
"""§14.1 (план Sol №6): held-out restoration benchmark — предсказуемость
формул как фундамент для предложений по лакунам.

Протокол: сплит 80/20 ПО ПАМЯТНИКАМ (artifact_id, seed=42). В test-записях
(3..8 слов) маскируется КАЖДАЯ позиция по очереди; предсказание из train:
  M (модель): точный контекст (n, i, остальные слова) → голосование
  train-слов; бэкофф на скелетный контекст (имена → '*'; реестр
  операторов §13); бэкофф на слот-приор (init/mid/fin); бэкофф на
  глобальную частоту. B (базлайн): слот-приор + частота.
Метрики: top-1/top-5 accuracy, стратифицированно: все цели / цели из
реестра операторов (закрытый класс) / прочие (открытый: имена и т.д.);
покрытие точным/скелетным контекстом. Никакой статистики значимости —
это бенчмарк-линейка; предложения по реальным лакунам — отдельный слой
с этой калибровкой.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_restoration.py
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42
OUT_LOG = os.path.join('logs', 'etr_restoration.log')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


OPS = {'clan', 'sec', 'sech', 'puia', 'ruva', 'ati', 'apa', 'papa',
       'teta', 'nefts', 'tusurthir', 'turce', 'turuce', 'muluvanice',
       'lupu', 'lupuce', 'svalce', 'amce', 'ame', 'zilachnu',
       'zilachnce', 'tence', 'cerichunce', 'zilath', 'zilch', 'zilc',
       'camthi', 'maru', 'purth', 'marunuch', 'mi', 'mini', 'mine',
       'ta', 'ca', 'cn', 'itun', 'eca', 'avils', 'avil', 'ril',
       'suthi', 'suthina', 'mutna', 'mlach', 'cana', 'flere', 'fler'}


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    assert corpus['meta'].get('freeze_version') == '0.9'
    view = [r for r in corpus['records']
            if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r.get('variant_of') is None]
    log('=== §14.1: restoration benchmark (план Sol №6) ===')
    arts = sorted({r['artifact_id'] for r in view})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(arts))
    train_arts = {arts[i] for i in perm[:int(0.8 * len(arts))]}
    tr_recs, te_recs = [], []
    for r in view:
        ws = tuple(t['ascii'] for t in r['toks']
                   if t['kind'] == 'W' and '-' not in t['ascii']
                   and len(t['ascii']) >= 2)
        if 3 <= len(ws) <= 8:
            (tr_recs if r['artifact_id'] in train_arts
             else te_recs).append(ws)
    log(f'записей 3–8 слов: train {len(tr_recs)} / test {len(te_recs)} '
        f'(сплит по памятникам, seed={SEED})')

    # --- train-структуры -----------------------------------------------------
    exact = defaultdict(Counter)
    skel = defaultdict(Counter)
    slot_prior = {'init': Counter(), 'mid': Counter(), 'fin': Counter()}
    global_freq = Counter()

    def slot_of(i, n):
        return 'init' if i == 0 else 'fin' if i == n - 1 else 'mid'

    for ws in tr_recs:
        n = len(ws)
        sk = tuple(w if w in OPS else '*' for w in ws)
        for i in range(n):
            exact[(n, i, ws[:i] + ws[i + 1:])][ws[i]] += 1
            skel[(n, i, sk[:i] + sk[i + 1:])][ws[i]] += 1
            slot_prior[slot_of(i, n)][ws[i]] += 1
            global_freq[ws[i]] += 1

    def ranked(counter, k=5):
        return [w for w, _ in counter.most_common(k)]

    glob_top = ranked(global_freq)
    prior_top = {s: ranked(c) for s, c in slot_prior.items()}

    def predict_model(ws, sk, i):
        n = len(ws)
        out = []
        used = ('none',)
        ex = exact.get((n, i, ws[:i] + ws[i + 1:]))
        if ex:
            out += ranked(ex)
            used = ('exact',)
        skc = skel.get((n, i, sk[:i] + sk[i + 1:]))
        if skc:
            if used == ('none',):
                used = ('skel',)
            out += [w for w in ranked(skc) if w not in out]
        for w in prior_top[slot_of(i, n)] + glob_top:
            if w not in out:
                out.append(w)
        return out[:5], used[0]

    def predict_base(ws, i):
        n = len(ws)
        out = list(prior_top[slot_of(i, n)])
        out += [w for w in glob_top if w not in out]
        return out[:5]

    # --- оценка ---------------------------------------------------------------
    strata = {'все': lambda w: True, 'операторы': lambda w: w in OPS,
              'открытый класс': lambda w: w not in OPS}
    hit = {k: Counter() for k in strata}
    nev = Counter()
    cover = Counter()
    for ws in te_recs:
        sk = tuple(w if w in OPS else '*' for w in ws)
        for i, tgt in enumerate(ws):
            preds, used = predict_model(ws, sk, i)
            bpreds = predict_base(ws, i)
            cover[used] += 1
            for name, cond in strata.items():
                if not cond(tgt):
                    continue
                nev[name] += 1
                hit[name]['m1'] += preds[:1] == [tgt]
                hit[name]['m5'] += tgt in preds
                hit[name]['b1'] += bpreds[:1] == [tgt]
                hit[name]['b5'] += tgt in bpreds
    tot = sum(cover.values())
    log(f'событий маскирования: {tot}; покрытие: точный контекст '
        f'{cover["exact"] / tot:.1%}, скелетный {cover["skel"] / tot:.1%}, '
        f'только приоры {cover["none"] / tot:.1%}')
    log()
    log(f'{"страта":<15} {"n":>5} {"M top1":>7} {"M top5":>7} '
        f'{"B top1":>7} {"B top5":>7}')
    for name in strata:
        n = nev[name]
        if n == 0:
            continue
        h = hit[name]
        log(f'{name:<15} {n:>5} {h["m1"] / n:>7.1%} {h["m5"] / n:>7.1%} '
            f'{h["b1"] / n:>7.1%} {h["b5"] / n:>7.1%}')
    log()
    log('чтение: линейка предсказуемости формул held-out по памятникам. '
        'Открытый класс (имена) принципиально малопредсказуем — '
        'восстановление лакун честно только для операторных слотов; '
        'к реальным лакунам применять с этой калибровкой.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
