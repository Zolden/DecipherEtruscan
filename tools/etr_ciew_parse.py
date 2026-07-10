# -*- coding: utf-8 -*-
"""Парсер CIEW (Fowler & Wolfe 1965) из OCR-слоя → большие тексты + валидация.

Вход: data/external/fowler_wolfe/fowler_wolfe_vol1_djvu.txt (см. README
там же: происхождение, нумерация, легенда символов).

Выходы:
  data/external/fowler_wolfe/ciew_bigtexts.csv — построчно большие
    тексты: 7002 (Tabula Capuana, TLE 2), 9001 (Liber Linteus, версия
    Runes), 9002 (Лемносская стела), 9003 (печень Пьяченцы), 9011–9016
    (Харсекин); 9021 (Пирги) парсится, но в корпус НЕ идёт (дубль
    supplement-а; хранится для сверки).
  logs/etr_ciew_parse.log — статистика + пословная валидация CIE-записей
    CIEW против CIEP-части корпуса (совпадающие CIE-номера!).

Декодирование (черновик в README, финальная таблица здесь):
  однозначно: 6→θ, 9→θ (в буквенном контексте), X→χ, B→8?? нет: B — не
    буква F&W; «~»→[---] (утраченные группы), ромбы/мусор → '-' (по
    букве), '+' (разлом) → '-', {латиница} → удаляется (лог),
    '/' в начале продолжения — склейка строки.
  неоднозначно (OCR смешивает глифы) — решает ЛЕКСИКОННЫЙ
    ДИЗАМБИГУАТОР: для токена с символами из AMBIG генерируются варианты
    (8→f|θ, 0→o|θ, 1→i|l, 5→s, O→o|ś(редко)), выбирается вариант,
    засвидетельствованный в ascii-словаре замороженного корпуса
    (v0.3) или в списке слов опорных текстов; единственное совпадение →
    'ocr-fixed', иначе дефолт (8→f, 0→o, 1→i, 5→s, O→o) + флаг 'ocr?'.

Колонки Liber Linteus: номера строк в OCR идут последовательностями с
рестартами (новая колонка при падении номера) — колонка отслеживается
счётчиком, ключ строки = (колонка, номер).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_ciew_parse.py
"""
import csv
import difflib
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

SRC = os.path.join('data', 'external', 'fowler_wolfe',
                   'fowler_wolfe_vol1_djvu.txt')
OUT_CSV = os.path.join('data', 'external', 'fowler_wolfe',
                       'ciew_bigtexts.csv')
OUT_LOG = os.path.join('logs', 'etr_ciew_parse.log')
LOG = []

BIG = ['7002', '9001', '9002', '9003', '9011', '9012', '9013', '9014',
       '9015', '9016', '9021']
LANG = {'9002': 'lemn'}
TITLE = {'7002': 'Tabula Capuana (TLE 2)', '9001': 'Liber Linteus (Runes)',
         '9002': 'Лемносская стела', '9003': 'печень Пьяченцы',
         '9011': 'Харсекин 1', '9012': 'Харсекин 2', '9013': 'Харсекин 3',
         '9014': 'Харсекин 4', '9015': 'Харсекин 5', '9016': 'Харсекин 6',
         '9021': 'Пирги (Паллоттино)'}


def log(msg=''):
    print(msg)
    LOG.append(msg)


# --- декодирование -----------------------------------------------------------
JUNK = set('«»©®�■□♦◊¤*^%$#=?!;,`\'"“”_|\\')
KEEP = set('ABCDEFGHIKLMNOPQRSTUVXZ0123456789 .•[]<>{}/-+~()')
AMBIG = {'8': ('f', 'θ'), '0': ('o', 'θ'), '1': ('i', 'l'),
         '5': ('s',), 'O': ('o', 'ś')}
BASE = {'6': 'θ', '9': 'θ', 'X': 'χ', '8': 'f', '0': 'o', '1': 'i',
        '5': 's'}


def to_ascii(w):
    return ''.join({'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
                    'ś': 's', 'š': 's', 'ê': 'e', "'": ''}.get(c, c)
                   for c in w)


def pre_clean(s):
    """OCR-строка → нормализованный текст в наших конвенциях разметки."""
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'\{[^}]*\}', ' ', s)  # латиница в {} — вон (считается)
    s = s.replace('~', ' [---] ')
    out = []
    for ch in s:
        if ch in KEEP:
            out.append(ch)
        elif ch in JUNK or ord(ch) > 0x2000:
            out.append('-')
        elif ch.isalpha():
            out.append(ch.upper())
        else:
            out.append('-')
    s = ''.join(out)
    s = s.replace('+', '-')
    s = re.sub(r'\(\s*T', 'ś', s)  # глиф ś у F&W OCR-ится как «(T»
    s = re.sub(r'[()]', '', s)
    s = re.sub(r'-{1,}', lambda m: '-' * len(m.group(0)), s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def decode_token(tok, lexicon, stats):
    """Токен после pre_clean → (декодированный, флаг)."""
    core = tok
    # однозначные замены (в буквенном контексте всё равно буквы)
    def base_map(t):
        return ''.join(BASE.get(c, c) for c in t)
    letters = [c for c in core if c.isalnum()]
    if not letters:
        return tok, ''
    if all(c.isdigit() for c in letters):
        return tok, 'num'  # арабское числительное F&W
    amb = [c for c in set(core) if c in AMBIG]
    lowered = base_map(core).lower()
    if not amb or all(c not in AMBIG or len(AMBIG[c]) == 1 for c in core):
        return lowered, ''
    # варианты
    def variants(t):
        outs = ['']
        for c in t:
            opts = AMBIG.get(c, None)
            if opts is None:
                opts = (BASE.get(c, c).lower() if c.isalnum() else c,)
            outs = [o + v for o in outs for v in opts][:16]
        return outs
    vs = variants(core)
    marks = re.compile(r'[\[\]<>•.\-/ ]')
    hits = [v for v in vs
            if to_ascii(marks.sub('', v)) in lexicon and marks.sub('', v)]
    if len(set(hits)) == 1:
        stats['ocr_fixed'] += 1
        return hits[0], 'ocr-fixed'
    stats['ocr_ambig'] += 1
    return lowered, 'ocr?'


def decode_line(line, lexicon, stats):
    toks = []
    flags = set()
    for tok in line.split():
        d, fl = decode_token(tok, lexicon, stats)
        toks.append(d)
        if fl and fl != 'num':
            flags.add(fl)
    return ' '.join(toks), sorted(flags)


# --- постраничный сбор строк больших записей ---------------------------------
PAGE_RE = re.compile(r'Corpus Inscriptionum Etruscarum Wisconsinense')


def collect_big(body):
    """Возвращает {entry: [((page, line), text), …]} для записей BIG.

    Атрибуция: строка с явным префиксом записи → этой записи; «голая»
    строка «номер текст» → большой записи, если на странице встречаются
    префиксные строки ровно одной большой записи (страницы LL). При
    повторе (page, line) остаётся более ДЛИННЫЙ текст (колонko-расщеплённые
    страницы дублируют строки огрызками); '/'-продолжения доклеиваются.
    """
    ent_res = {e: re.compile(r'^\s*' + r'\s?'.join(e)
                             + r'\s+(\d{1,3})\s+(\S.*)$') for e in BIG}
    bare_re = re.compile(r'^\s*(\d{1,3})\s+(\S.{2,})$')
    pages = re.split(PAGE_RE, body)
    out = {e: {} for e in BIG}
    order = {e: [] for e in BIG}
    for p_i, page in enumerate(pages):
        page_rows = []  # (entry|None, num, text)
        for raw in page.split('\n'):
            hit = None
            for e, rx in ent_res.items():
                m = rx.match(raw)
                if m:
                    hit = (e, int(m.group(1)), m.group(2).strip())
                    break
            if hit:
                page_rows.append(hit)
                continue
            m = bare_re.match(raw)
            if m and int(m.group(1)) <= 150 and not re.match(
                    r'^\d+\s+\d', raw.strip()):
                page_rows.append((None, int(m.group(1)), m.group(2).strip()))
        ents_here = {e for e, _, _ in page_rows if e}
        big_here = ents_here.pop() if len(ents_here) == 1 else None
        for e, num, txt in page_rows:
            ent = e or big_here
            if ent is None:
                continue
            cont = txt.startswith('/')
            if cont:
                txt = txt[1:].strip()
            key = (p_i, num)
            d = out[ent]
            if key not in d:
                d[key] = txt
                order[ent].append(key)
            elif cont or d[key].endswith('/'):
                d[key] = d[key].rstrip('/') + ' ' + txt
            elif len(txt) > len(d[key]):
                d[key] = txt
        # v2: на одно-записных страницах подбираем НЕнумерованные текстовые
        # строки (колонко-расщеплённый OCR потерял их номера); ключ
        # (page, 900+idx), флаг у потребителя — по отсутствию номера ≤300
        if big_here is not None:
            u_idx = 0
            for raw in page.split('\n'):
                raw = raw.strip()
                if not raw or bare_re.match(raw) or any(
                        rx.match(raw) for rx in ent_res.values()):
                    continue
                if re.fullmatch(r'[\d\s]+', raw):
                    continue
                letters = len(re.findall(r'[A-Za-z]', raw))
                if letters >= 8 and letters >= 0.5 * len(raw.replace(' ', '')):
                    key = (p_i, 900 + u_idx)
                    if key not in out[big_here]:
                        out[big_here][key] = raw
                        order[big_here].append(key)
                        u_idx += 1
    # v3: нумерованные ОГРЫЗКИ колонко-расщеплённых страниц наследуют текст
    # ненумерованной строки той же страницы, начинающейся теми же буквами
    # (выравнивание + дедупликация); использованные ненумерованные удаляются
    for e in BIG:
        d = out[e]
        by_page = {}
        for (p, num) in list(d):
            by_page.setdefault(p, []).append(num)
        for p, nums in by_page.items():
            tiny = [n for n in sorted(nums) if n < 900 and len(
                re.findall(r'[A-Za-z]', d[(p, n)])) <= 4]
            unal = [n for n in sorted(nums) if n >= 900]
            used = set()
            for n in tiny:
                pref = re.sub(r'[^A-Za-z]', '', d[(p, n)]).upper()[:3]
                if len(pref) < 2:
                    continue
                for u in unal:
                    if u in used:
                        continue
                    ut = re.sub(r'[^A-Za-z]', '', d[(p, u)]).upper()
                    if ut.startswith(pref):
                        d[(p, n)] = d[(p, u)]
                        used.add(u)
                        break
            for u in used:
                del d[(p, u)]
                order[e].remove((p, u))
    return {e: [(k, out[e][k]) for k in order[e]] for e in BIG}


def main():
    os.makedirs('logs', exist_ok=True)
    t = open(SRC, encoding='utf-8').read()
    # защита от битой загрузки (archive.org может отдать заглушку/redirect)
    assert len(t) > 400_000 and 'Corpus Inscriptionum' in t, (
        f'вход {SRC} подозрительно мал ({len(t)} байт) или не тот файл — '
        'перескачайте: см. data/external/fowler_wolfe/README.md')
    t = t.replace('\r\n', '\n').replace('\r', '\n')  # нормализация eol
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    # лексикон — ТОЛЬКО из не-CIEW источников: иначе петля обратной связи
    # (наши же декодированные слова меняли бы дизамбигуацию при перепрогоне)
    view = [r for r in corpus['records']
            if r['lang'] in ('etr', 'lemn') and r['kind'] == 'text'
            and not r['src'].startswith('CIEW')]
    lexicon = {t2['ascii'] for r in view for t2 in r['toks']
               if t2['kind'] == 'W'}
    # опорные слова известных больших текстов (изданий) — для дизамбигуации
    lexicon |= {to_ascii(w) for w in (
        'fler etnam tesim celucn cletram śrenχve trin θezine χim tarc mutin '
        'anancveś nac cal vacl ścanince saucsaθ persin ara nunθene śaθaś '
        'naχve heχz male ceia hia zaθrumsne luśaś fulinuśnes θuvas munistas '
        'tameresca ilacve tulerase χurvar teśiameitale alśase atranes '
        'zilacal seleitala acnaśvers itanim heramve pulumχva śelace vacal '
        'tmial avilχval amuce snuiaφ hulaieś naφuθ śiaśi maraś mav '
        'sialχveiś aviś evisθo śeronaiθ zivai śeronai morinail aker '
        'tavarśio vanalasial').split()}
    log('=== Парсер CIEW ===')
    log(f'лексикон дизамбигуации: {len(lexicon)} ascii-типов')

    ia = t.find('Alphabet-Ordered Index', 6000)
    body = t[:ia]
    stats = Counter()
    out_rows = []
    big_rows = collect_big(body)
    for e in BIG:
        rows = big_rows[e]
        n_words = 0
        for (page, num), raw in rows:
            cleaned = pre_clean(raw)
            if not re.search(r'[A-Za-z]', cleaned):
                continue
            dec, fl = decode_line(cleaned, lexicon, stats)
            n_words += len([x for x in dec.split()
                            if re.search(r'[a-zθχśê]', x)])
            if num >= 900:
                fl = sorted(set(fl) | {'unaligned'})
            out_rows.append({'entry': e, 'col': page, 'line': num,
                             'ocr': raw[:120], 'text': dec,
                             'flags': ' '.join(fl),
                             'title': TITLE[e]})
        n_un = sum(1 for (_, n_), _ in rows if n_ >= 900)
        log(f'  {e} ({TITLE[e]}): строк {len(rows)} '
            f'(из них ненумерованных v2: {n_un}), словоформ ~{n_words}')
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['entry', 'col', 'line', 'text',
                                          'flags', 'ocr', 'title'], lineterminator='\n')
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    log(f'декод-статистика: {dict(stats)}')
    log(f'большие тексты записаны: {OUT_CSV} ({len(out_rows)} строк)')

    # --- валидация CIE-извлечений против CIEP-части ------------------------
    log()
    log('--- валидация: CIE-записи CIEW ↔ CIEP-часть (совпадающие номера) ---')
    cie_rows = {}
    row_re = re.compile(r'^\s*(\d{1,4})\s+(\d{1,2})\s+(\S.*)$')
    for raw in body.split('\n'):
        m = row_re.match(raw)
        if not m:
            continue
        num = int(m.group(1))
        if 1 <= num <= 5606 or 8001 <= num <= 8464:
            cie_rows.setdefault(str(num), []).append(
                (int(m.group(2)), m.group(3).strip()))
    log(f'CIE-записей извлечено (паттерн «номер строка текст»): '
        f'{len(cie_rows)}')
    ciep_by_eid = {}
    for r in corpus['records']:
        if r['src'] == 'CIEP':
            ciep_by_eid.setdefault(r['eid'], []).append(r)
    both = sorted(set(cie_rows) & set(ciep_by_eid), key=int)
    log(f'номеров, есть и в CIEW-извлечении, и в CIEP-части: {len(both)}')

    def squash_ciew(num):
        s = ' '.join(txt for _, txt in sorted(cie_rows[num]))
        s = pre_clean(s)
        dec, _ = decode_line(s, lexicon, stats)
        return to_ascii(re.sub(r'[^a-zθχφσςśšê]', '',
                               dec.replace(' ', '')))

    def squash_ciep(recs):
        return ''.join(t2['ascii'] for r in recs for t2 in r['toks']
                       if t2['kind'] == 'W')

    ratios = []
    for num in both:
        a = squash_ciew(num)
        b = squash_ciep(ciep_by_eid[num])
        if len(a) >= 6 and len(b) >= 6 and re.search(r'[a-hk-z]', b):
            sm = difflib.SequenceMatcher(None, a, b)
            contain = sum(bl.size for bl in sm.get_matching_blocks()) \
                / min(len(a), len(b))
            ratios.append((contain, sm.ratio(), num, a[:40], b[:40]))
    ratios.sort(reverse=True)
    import numpy as np
    rr = np.array([x[1] for x in ratios])
    cc = np.array([x[0] for x in ratios])
    log(f'сопоставимых пар (≥6 букв, CIEP не чисто-числовая): {len(rr)}')
    if len(rr):
        log(f'симметричное сходство: медиана {np.median(rr):.2f}; '
            f'≥0.8: {(rr >= 0.8).mean():.0%}')
        log(f'вложенность (совпавшее/короткая сторона — устойчиво к разнице '
            f'покрытия памятника): медиана {np.median(cc):.2f}; '
            f'≥0.8: {(cc >= 0.8).mean():.0%}; ≥0.6: {(cc >= 0.6).mean():.0%}')
        log('примеры лучших (по вложенности):')
        for c_, r_, num, a, b in ratios[:5]:
            log(f'  CIE {num}: влож {c_:.2f} симм {r_:.2f}  CIEW={a!r} '
                f'CIEP={b!r}')
        log('примеры худших:')
        for c_, r_, num, a, b in ratios[-5:]:
            log(f'  CIE {num}: влож {c_:.2f} симм {r_:.2f}  CIEW={a!r} '
                f'CIEP={b!r}')

    # --- CIE-записи CIEW как кандидат-источник v0.5 -------------------------
    shared = set(both)
    cont_of = {num: c_ for c_, _, num, _, _ in ratios}
    out2 = os.path.join('data', 'external', 'fowler_wolfe',
                        'ciew_cie_entries.csv')
    n_excl = 0
    with open(out2, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['entry', 'text', 'flags',
                                          'shared_with_ciep', 'containment'], lineterminator='\n')
        w.writeheader()
        for num in sorted(cie_rows, key=int):
            s = ' '.join(txt for _, txt in sorted(cie_rows[num]))
            dec, fl = decode_line(pre_clean(s), lexicon, stats)
            if len(re.sub(r'[^a-zθχśê]', '', dec)) < 3:
                continue
            sh = num in shared
            n_excl += not sh
            w.writerow({'entry': num, 'text': dec, 'flags': ' '.join(fl),
                        'shared_with_ciep': int(sh),
                        'containment': f'{cont_of.get(num, ""):.2f}'
                        if num in cont_of else ''})
    log()
    log(f'CIE-записи CIEW: {out2}; CIEW-эксклюзивных (нет в CIEP, '
        f'кандидаты в корпус v0.5): {n_excl}')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
