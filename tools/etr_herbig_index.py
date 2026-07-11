# -*- coding: utf-8 -*-
"""§9.1: Herbig 1919-21 (CIE Suppl. I, Liber Linteus) — извлечение Index
verborum и выравнивание с F&W-слоем корпуса.

Источник: data/external/cie_online/Supplementum_I.pdf (studietruschi.org,
общественное достояние по возрасту). Index verborum (PDF-стр. 35-40)
даёт каждое слово LL со ссылками (колонка I-XII, строка) — независимый
второй свидетель текста с АВТОРИТЕТНЫМИ координатами издания.

Метод: (1) координатный разбор двухколоночного индекса (заголовочное
слово при малом x, продолжения с отступом); (2) нормализация OCR
(ϑ/&→th, χ→ch, ç→c, ś→s, греческие Χ/Π/Ι в римских числах, γ-строки
как '/'|'y'); (3) карта (col,line)→множество слов; (4) выравнивание с
нумерованными строками F&W (results/ll_scroll_map.csv, позиции 1-242):
монотонное DP (Нидлман-Вунш), score=Жаккар множеств слов, gap=-0.15;
(5) интервальные строки — привязка к оставшимся строкам Herbig между
выровненными соседями по лучшему Жаккару.

Выходы: data/supplements/herbig_ll_index.csv (word_raw, word_norm, col,
line, flags — коммитится, PD), results/herbig_fw_alignment.csv,
logs/etr_herbig_index.log. Разведочный слой: покрытие индекса неполно
(только читаемые Хербигом слова), Жаккар занижен по построению.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_herbig_index.py
"""
import csv
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
PDF = os.path.join('data', 'external', 'cie_online', 'Supplementum_I.pdf')
OUT_IDX = os.path.join('data', 'external', 'cie_online',
                       'herbig_ll_index.csv')  # НЕ в supplements: freeze
# всасывает supplements/*.csv как записи, а это словарный слой
OUT_ALN = os.path.join('results', 'herbig_fw_alignment.csv')
OUT_LOG = os.path.join('logs', 'etr_herbig_index.log')
PAGES = range(35, 41)  # Index verborum
X_SPLIT = 290.0        # граница печатных колонок
ENTRY_X = {'L': 84.0, 'R': 313.0}   # порог x заголовочного слова
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


# --- нормализация ----------------------------------------------------------
WORD_MAP = {'ϑ': 'th', '&': 'th', 'θ': 'th', 'χ': 'ch', 'φ': 'ph',
            'ç': 'c', 'ś': 's', 'š': 's', 'σ': 's', 'ς': 's',
            'α': 'a', 'ν': 'v', 'µ': 'm', 'μ': 'm', 'é': 'e', 'è': 'e'}


def norm_word(w):
    w = (w or '').strip().lower()
    w = ''.join(WORD_MAP.get(c, c) for c in w)
    w = re.sub(r'[^a-z]', '', w)
    return w


ROMAN_TR = {'Χ': 'X', 'Π': 'II', 'Ι': 'I', 'У': 'V', 'U': 'II', 'H': 'II',
            'l': 'I', 'i': 'I', '1': 'I', 'В': 'B'}
ROMAN_RE = re.compile(r'^[IVXΧΠΙУUHli1]+$')
VALID_COLS = {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
              'X', 'XI', 'XII'}


def fix_roman(tok):
    t = ''.join(ROMAN_TR.get(c, c) for c in tok)
    return t if t in VALID_COLS else None


# --- 1. разбор страниц индекса ---------------------------------------------
def read_entries():
    import fitz
    doc = fitz.open(PDF)
    entries = []  # list of str (полный текст статьи)
    for pno in PAGES:
        words = doc[pno].get_text('words')
        if not words:
            continue
        for zone, (xlo, xhi) in (('L', (0, X_SPLIT)), ('R', (X_SPLIT, 1e9))):
            zw = [w for w in words if xlo <= w[0] < xhi]
            # шапка страницы: y < 85 (LIBER..., номер, INDEX VERBORUM)
            zw = [w for w in zw if w[1] > 85]
            zw.sort(key=lambda w: (w[1], w[0]))
            # кластеризация по y с порогом 3.5pt
            bands = []
            for w in zw:
                if bands and abs(w[1] - bands[-1][0]) <= 3.5:
                    bands[-1][1].append(w)
                else:
                    bands.append([w[1], [w]])
            for _, bw in bands:
                bw.sort(key=lambda w: w[0])
                x0 = bw[0][0]
                text = ' '.join(w[4] for w in bw)
                if re.fullmatch(r'[A-ZΖΘ]\.?|Index verborum,?', text.strip()):
                    continue  # буквы-рубрики
                first = text.split()[0] if text.split() else ''
                # заголовок должен начинаться с буквы (не цифра/скобка/пункт.)
                is_entry = (x0 <= ENTRY_X[zone]
                            and not re.fullmatch(r'[\d(),.;:•■\-\[\]]+',
                                                 first))
                if is_entry:
                    entries.append(text)
                elif entries:
                    entries[-1] += ' ' + text
    return entries


# --- 2. разбор статьи: слово + ссылки --------------------------------------
def parse_entry(entry):
    """Возвращает (words_norm:set, refs:[(col,line,flags)], ok:bool)."""
    e = entry.strip()
    if not e:
        return set(), [], False
    # кросс-ссылки 'X v. Y' без римских — пропуск
    toks = e.split()
    # заголовочные формы: всё до первого римского числа / 'Fragm.'
    head, i = [], 0
    while i < len(toks):
        t = toks[i].strip('.,;')
        if fix_roman(t) or t.startswith('Fragm'):
            break
        head.append(toks[i])
        i += 1
    if i == 0 or i == len(toks):
        return set(), [], False
    # формы слова: элементы head, похожие на слова (в т.ч. 'X an Y')
    NOISE = {'an', 'v', 'vix', 'adn', 'cf', 'sim', 'in', 't', 'ad',
             'bis', 'extr', 'sq', 'sqq', 'coll', 'ib', 'ibid', 'ubi',
             'et', 'krall', 'torp', 'herbig', 'bugge', 'fragm', 'nov',
             'de', 'vel'}
    words = set()
    for h in head:
        h2 = re.sub(r'[\[\](){}?.,;:]', '', h)
        n = norm_word(h2)
        if len(n) >= 2 and n not in NOISE:
            words.add(n)
    if not words:
        return set(), [], False
    # ссылки: скобки вырезаем (пометив '?'), затем col→числа
    rest = ' '.join(toks[i:])
    unc_global = False
    def _paren(m):
        nonlocal unc_global
        if '?' in m.group(0):
            unc_global = True
        return ' (bis) ' if 'bis' in m.group(0) else ' '
    rest = re.sub(r'\([^)]*\)', _paren, rest)
    refs = []
    col = None
    gamma = False
    for raw in re.split(r'\s+', rest):
        t = raw.strip('.,;')
        if not t:
            continue
        unc = '?' in raw or unc_global
        if t.startswith('Fragm') or t in ('nov', 'a', 'b'):
            col = 'FN'
            gamma = False
            continue
        # γ-маркер: '/', 'y', 'γ', либо прилеплен к римскому 'VIII/'
        if t in ('/', 'y', 'γ'):
            gamma = True
            continue
        had_gamma = t.endswith('/')
        rc = fix_roman(t.rstrip('/'))
        if rc:
            col = rc
            gamma = had_gamma
            continue
        m = re.fullmatch(r'(\d{1,2})(/)?', t)
        if m and col:
            line = int(m.group(1))
            if 1 <= line <= 30:
                fl = []
                if unc:
                    fl.append('unc')
                if gamma:
                    fl.append('gamma')
                if col == 'FN':
                    fl.append('fragm_nov')
                refs.append((col, line, '+'.join(fl)))
            if m.group(2):
                gamma = True
        elif t == '(bis)' and refs:
            refs.append(refs[-1])
    return words, refs, bool(refs)


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('data', 'supplements'), exist_ok=True)
    log('=== §9.1: Herbig Index verborum → карта (col,line) и выравнивание с F&W ===')
    entries = read_entries()
    log(f'статей индекса (заголовочный x): {len(entries)}')
    rows = []          # (word_raw_first, word_norm, col, line, flags)
    line_words = {}    # (col,line)->set(words)
    n_bad = 0
    for e in entries:
        words, refs, ok = parse_entry(e)
        if not ok:
            n_bad += 1
            continue
        for w in sorted(words):
            for col, line, fl in refs:
                rows.append((e.split()[0], w, col, line, fl))
                if col != 'FN' and 'gamma' not in fl:
                    line_words.setdefault((col, line), set()).add(w)
    log(f'разобрано статей: {len(entries) - n_bad}; не разобрано: {n_bad}; '
        f'ссылок (слово,кол,строка): {len(rows)}')
    col_order = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
                 'IX', 'X', 'XI', 'XII']
    log(f'строк Herbig с ≥1 словом (без γ/FN): {len(line_words)}; '
        f'по колонкам: ' + ', '.join(
            f'{c}:{sum(1 for (cc, _) in line_words if cc == c)}'
            for c in col_order))

    with open(OUT_IDX, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['entry_head', 'word_norm', 'col', 'line', 'flags'])
        for r in sorted(set(rows)):
            wr.writerow(r)
    log(f'индекс записан: {OUT_IDX}')

    # --- 3. выравнивание с F&W (ll_scroll_map) -----------------------------
    fw = []
    with open(os.path.join('results', 'll_scroll_map.csv'),
              encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ws = set(row['text'].split())
            fw.append((row['position'], row['kind'], row['key'], ws))
    fw_exact = [(int(p), k, ws) for p, kd, k, ws in fw if kd == 'exact'
                for _ in [0]]
    fw_exact.sort()
    # Herbig-строки в порядке свитка
    hb = sorted(line_words.items(),
                key=lambda kv: (col_order.index(kv[0][0]), kv[0][1]))
    log(f'\nвыравнивание: {len(fw_exact)} точных строк F&W × {len(hb)} строк Herbig')

    SKEL = str.maketrans('', '', 'aeiou')

    def _match(a, b):
        # мягкое равенство слов: точно / общий префикс>=4 / скелет согласных
        if a == b:
            return True
        if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
            return True
        sa, sb = a.translate(SKEL), b.translate(SKEL)
        return len(sa) >= 3 and sa == sb

    def jac(a, b):
        # мягкий Жаккар (жадное паросочетание)
        if not a or not b:
            return 0.0
        bb = set(b)
        hit = 0
        for w in a:
            for v in bb:
                if _match(w, v):
                    bb.discard(v)
                    hit += 1
                    break
        return hit / (len(a) + len(b) - hit)

    GAP = -0.15
    n, m = len(fw_exact), len(hb)
    D = np.full((n + 1, m + 1), -1e9)
    D[0, :] = np.arange(m + 1) * GAP
    D[:, 0] = np.arange(n + 1) * GAP
    B = np.zeros((n + 1, m + 1), np.int8)
    S = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            S[i, j] = jac(fw_exact[i][2], hb[j][1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = (D[i - 1, j - 1] + S[i - 1, j - 1], 0)
            if D[i - 1, j] + GAP > best[0]:
                best = (D[i - 1, j] + GAP, 1)
            if D[i, j - 1] + GAP > best[0]:
                best = (D[i, j - 1] + GAP, 2)
            D[i, j], B[i, j] = best
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if B[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif B[i, j] == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    matched = [(pi, pj) for pi, pj in pairs if S[pi, pj] >= 0.2]
    js = [S[pi, pj] for pi, pj in matched]
    log(f'DP-пар: {len(pairs)}; с Жаккаром ≥0.2: {len(matched)} '
        f'(средний J={np.mean(js):.2f})' if matched else 'нет пар')
    # границы колонок в сквозной нумерации F&W
    col_pos = {}
    for pi, pj in matched:
        col_pos.setdefault(hb[pj][0][0], []).append(fw_exact[pi][0])
    log('колонки Herbig → диапазон позиций F&W (по выровненным):')
    for c in col_order:
        if c in col_pos:
            log(f'  Col. {c:<5} позиции {min(col_pos[c])}–{max(col_pos[c])} '
                f'(n={len(col_pos[c])})')

    # --- 4. интервальные строки F&W → свободные строки Herbig --------------
    used_h = {pj for _, pj in matched}
    free_h = [j for j in range(m) if j not in used_h]
    fw_int = [(p, k, ws) for p, kd, k, ws in fw if kd == 'interval']
    n_int_hit = 0
    int_rows = []
    for p, k, ws in fw_int:
        best_j, best_s = None, 0.0
        for j in free_h:
            s = jac(ws, hb[j][1])
            if s > best_s:
                best_j, best_s = j, s
        if best_j is not None and best_s >= 0.25:
            n_int_hit += 1
            int_rows.append((p, k, hb[best_j][0], best_s))
    log(f'\nинтервальных строк F&W: {len(fw_int)}; получили кандидата '
        f'(col,line) с J≥0.25: {n_int_hit}')

    with open(OUT_ALN, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['fw_position', 'fw_key', 'herbig_col', 'herbig_line',
                     'jaccard', 'kind'])
        for pi, pj in matched:
            wr.writerow([fw_exact[pi][0], fw_exact[pi][1],
                         hb[pj][0][0], hb[pj][0][1], f'{S[pi, pj]:.3f}',
                         'exact'])
        for p, k, (c, ln), s in int_rows:
            wr.writerow([p, k, c, ln, f'{s:.3f}', 'interval-candidate'])
    log(f'выравнивание записано: {OUT_ALN}')
    log('\nчтение: разведочный слой; Жаккар занижен (индекс содержит только '
        'уверенные чтения Herbig, F&W-строки — только уцелевшие слова). '
        'Структурные якоря издания (красные линии VI 8/9, XI 13/14 и др.) — '
        'в data/supplements/herbig_ll_structure.md.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
