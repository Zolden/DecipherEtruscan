# -*- coding: utf-8 -*-
"""Этап 0 (§0): заморозка этрусского корпуса → data/etr_corpus.pkl

Источники (data/):
  etruscan_larth.csv — компиляция Larth-Etruscan-NLP: 561 строка ETP-части
      (без key; дословно совпадает с ETP_fix.csv — проверяется здесь же) +
      6578 строк CIEP-части (key непуст; каждая строка — проекция колонки T
      ИЛИ C файла CIEP_pymupdf.csv, проверяется здесь же).
  ETP_fix.csv — ETP-данные (контрольный источник для сверки).
  CIEP_pymupdf.csv — конкорданс CIEP (pymupdf-извлечение): key, T
      (дипломатическое чтение), C (исправленное), A (глоссы через дефис),
      CIEP (номер записи = ID в larth).

Единица корпуса:
  ETP-часть: уникальная пара (ID, текст); варианты перевода сливаются.
  CIEP-часть: ОДНА строка CIEP_pymupdf.csv (T и C — два чтения одной
      записи, а не две записи; larth хранит их отдельными строками, что
      двоило бы статистику). Предпочтённое чтение: C, иначе T.

Нормализация текста (NORM_VERSION, все правила см. parse_segment):
  - html-сущности (&gt;) → символ; NFC; ’/ʼ → '; Σ → σ.
  - | и отдельно стоящие / // — границы строк надписи; • : ; и отдельно
    стоящие точки — разделители слов.
  - Лейденская разметка: [абв] реставрация (флаг restored; [---]/[…] —
    токен-лакуна G), <абв> эмендация (emended), (абв) раскрытие
    аббревиатуры (expanded), {абв} лишние буквы писца — исключаются из
    формы (scribal_extra), точки при буквах и подстрочные точки U+0323 —
    неуверенное чтение (uncertain), прогоны дефисов -- — повреждение
    (damaged), / внутри слова — перенос строки (line_split).
  - «[8-10 letters]» → лакуна; «[vacat]» → пусто (не лакуна).
  - Токены только из IVXLC с ≥1 заглавной → числительные (kind=N).
  - В CIEP-части одиночный дефис между буквами — разделитель слов
    (конвенция транскрипции конкорданса); в ETP-части дефис-разделитель
    не используется (проверено профилированием).
  - ascii-проекция формы (θ→th, χ→ch, φ→ph, σ/ς/ś/š→s, ê→e, '→∅) — для
    сопоставимости частей корпуса (CIEP транскрибирован ASCII); проекция
    с потерями и используется ТОЛЬКО для кросс-сверок, не для морфологии.

Язык записей (lang): по умолчанию 'etr'; документированные исключения —
  LANG_BY_CIE / LANG_BY_ETP (умбрский, цизальпинско-кельтский, латинский).
  ВНИМАНИЕ: конкорданс CIEP включает фалискские/латинские записи, не все
  из которых можно выявить автоматически, — ключевые результаты следующих
  этапов реплицируются на чистой ETP-части (см. §0 отчёта).

Детерминизм: порядок файлов фиксирован, множества сериализуются
  сортированными, timestamp-ов нет; повторный прогон обязан давать
  пустой git diff (pkl побайтово, sha256 в data/etr_corpus.sha256).

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_freeze.py
"""
import csv
import glob
import hashlib
import html
import os
import pickle
import random
import re
import sys
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

NORM_VERSION = '0.1'
DATA = 'data'
OUT_PKL = os.path.join(DATA, 'etr_corpus.pkl')
OUT_SHA = os.path.join(DATA, 'etr_corpus.sha256')
OUT_LOG = os.path.join('logs', 'etr_freeze.log')
OUT_SAMPLE = os.path.join('validation', 'sample50.md')

LINE_SEP = '§'  # внутренний маркер границы строки надписи
LETTERS = set("abcdefghijklmnopqrstuvwxyzθχφσςśšê'")
ASCII_MAP = {'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
             'ś': 's', 'š': 's', 'ê': 'e', "'": ''}

# Языковые назначения ID-группам CIEP-части (номер CIEP → (lang, обоснование)).
LANG_BY_CIE = {
    '15896': ('umb', 'умбрский: Игувинские таблицы — ikuvina, esunu, fertu, '
                     'prinuvatus; глоссы "make-sacrifice", "onbehalfofthepeople"'),
    '14651': ('cel', 'цизальпинско-кельтский: камень Брионы — kuitos lekatos, '
                     'tanotaliknoi, anokopokios, setupokios'),
    # найдены ручным просмотром выборки sample50 (первый проход):
    '17299': ('lat', 'ранняя латынь: fovrio c(aii) = M. Furius, формула '
                     '«de praidad» (из добычи)'),
    '17181': ('lat', 'ранняя латынь: romanom, n(vmmvs) — легенды '
                     'республиканских монет'),
    '17286': ('lat', 'ранняя латынь: {dv}<b>onor{o}<vm> (duonorum→bonorum), '
                     'h{e}<i>c (hic) — орфография эпитафий Сципионов'),
}
# То же для ETP-части (ID → (lang, обоснование)).
LANG_BY_ETP = {
    'Pe 1.211': ('lat', 'латинская: L SCARPIUS SCARPIAE L POPA'),
    'Ar 1.3': ('lat', 'латинская: CN LABERIUS A F POM'),
    'Um 1.7': ('lat', 'латинская половина билингвы гаруспика: L CAFATIUS L F '
                      'STE HARUSPEX FULGURIATOR'),
    'ETP 240': ('etr-lat', 'этрусский текст латинским письмом: VEL BER COMSN '
                           'A VELOSA I ~ «Vel Percomsna, сын Vela»'),
}

LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_csv(name):
    with open(os.path.join(DATA, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def clean_tr(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()


def to_ascii(form):
    return ''.join(ASCII_MAP.get(c, c) for c in form)


def pre_string(s):
    """Строковая предобработка до токенизации."""
    s = html.unescape(s)
    s = unicodedata.normalize('NFC', s)
    s = s.replace('’', "'").replace('ʼ', "'").replace('Σ', 'σ')
    s = re.sub(r'\[\s*vacat\s*\]', ' ', s)  # пустое место на камне — не лакуна
    s = re.sub(r'\[\d+\s*[-–]\s*\d+\s*letters?\]', ' [---] ', s)  # лакуна n–m букв
    s = s.replace('|', f' {LINE_SEP} ')
    s = re.sub(r'[•:;·]+', ' ', s)
    return s


def parse_segment(seg, stats):
    """Один сегмент → токен-словарь {'form','kind','flags'} или None."""
    flags = set()
    if '{' in seg or '}' in seg:  # Лейден {..}: лишние буквы писца — исключить;
        # НО если после вычёркивания букв не остаётся (в CIEP встречаются
        # записи целиком в скобках — семантика конкорданса неясна),
        # содержимое сохраняется с флагом braced, чтобы не терять текст.
        seg2 = re.sub(r'\{[^}]*\}', '', seg).replace('{', '').replace('}', '')
        if any(c.isalpha() for c in seg2):
            if seg2 != seg:
                flags.add('scribal_extra')
            seg = seg2
        else:
            flags.add('braced')
            seg = seg.replace('{', '').replace('}', '')
    if '<' in seg or '>' in seg:  # эмендация: пропущенное писцом, остаётся
        flags.add('emended')
        seg = seg.replace('<', '').replace('>', '')
    if '[' in seg or ']' in seg:  # реставрация в лакуне: остаётся с флагом
        flags.add('restored')
        seg = seg.replace('[', '').replace(']', '')
    if '(' in seg or ')' in seg:  # раскрытие аббревиатуры
        flags.add('expanded')
        seg = seg.replace('(', '').replace(')', '')
    if '/' in seg:  # перенос строки внутри слова
        flags.add('line_split')
        seg = seg.replace('/', '')
    if '.' in seg:  # точки при буквах: неуверенное чтение (конвенция ETP)
        flags.add('uncertain')
        seg = seg.replace('.', '')
    d = unicodedata.normalize('NFD', seg)
    if '̣' in d:  # подстрочная точка — неуверенная буква
        flags.add('uncertain')
        seg = unicodedata.normalize('NFC', d.replace('̣', ''))
    if not seg:
        return None
    if not seg.replace('-', ''):  # только дефисы → лакуна
        return {'form': '-', 'kind': 'G', 'flags': tuple(sorted(flags))}
    if re.fullmatch(r'[ivxlcIVXLC]+', seg) and re.search(r'[IVXLC]', seg):
        return {'form': seg.upper(), 'kind': 'N', 'flags': tuple(sorted(flags))}
    if re.search(r'[A-Z]', seg):
        flags.add('caps')
    seg = seg.lower()
    bad = [c for c in seg if c not in LETTERS and c != '-']
    if bad:
        flags.add('stripped_chars')
        stats['stripped_chars'].update(bad)
        seg = ''.join(c for c in seg if c in LETTERS or c == '-')
    if not seg.replace('-', ''):
        stats['dropped_tokens'] += 1
        return None
    if '-' in seg:
        flags.add('damaged')
    tok = {'form': seg, 'kind': 'W', 'flags': tuple(sorted(flags))}
    return tok


def parse_text(raw, dash_split, stats):
    """Текст надписи → (токены, число строк надписи)."""
    s = pre_string(raw)
    toks = []
    n_lines = 1
    for t in s.split():
        if t == LINE_SEP or (t and set(t) <= {'/'}):
            n_lines += 1
            continue
        if set(t) <= {'.'}:
            continue
        if dash_split:
            parts = re.split(r'(?<=[^-])-(?=[^-])', t)
            if len(parts) > 1:
                stats['dash_splits'] += 1
        else:
            parts = [t]
        for seg in parts:
            tok = parse_segment(seg, stats)
            if tok is not None:
                toks.append(tok)
    for tok in toks:
        tok['ascii'] = to_ascii(tok['form']) if tok['kind'] == 'W' else tok['form']
    return toks, n_lines


def classify_kind(toks):
    words = [t for t in toks if t['kind'] == 'W']
    single = sum(1 for t in words if len(t['form']) == 1)
    if len(words) >= 8 and single >= 0.8 * len(words):
        return 'abecedarium?'  # алфавитная надпись, не связный текст
    return 'text'


def seg_class(toks):
    words = [t for t in toks if t['kind'] == 'W']
    if not words:
        return 'empty'
    if len(words) > 1:
        return 'multi'
    return 'long-unseg' if len(words[0]['form']) >= 12 else 'single'


def parse_year(s):
    s = (s or '').strip()
    return float(s) if s else None


def build_records():
    larth = load_csv('etruscan_larth.csv')
    etp_fix = load_csv('ETP_fix.csv')
    ciep = load_csv('CIEP_pymupdf.csv')
    stats = {'dash_splits': 0, 'dropped_tokens': 0,
             'stripped_chars': Counter(), 'caps_records': []}
    records = []

    # --- ETP-часть -------------------------------------------------------
    etp_rows = [r for r in larth if not (r.get('key') or '').strip()]
    ciep_rows_larth = [r for r in larth if (r.get('key') or '').strip()]
    log(f'larth: {len(larth)} строк = ETP-часть {len(etp_rows)} '
        f'+ CIEP-часть {len(ciep_rows_larth)}')

    # сверка с контрольным ETP_fix.csv
    key_set = {((r['ID'] or '').strip(), (r['Etruscan'] or '').strip())
               for r in etp_rows}
    missing = [r for r in etp_fix
               if ((r['ID'] or '').strip(), (r['Etruscan'] or '').strip())
               not in key_set]
    log(f'сверка ETP_fix ⊆ larth(ETP-часть): не найдено {len(missing)} из '
        f'{len(etp_fix)} (ожидание: 0)')
    assert not missing, 'ETP_fix содержит строки, отсутствующие в larth'

    merged_tr = dup_rows = 0
    by_key = {}
    id_seq = Counter()
    for r in etp_rows:
        eid = (r['ID'] or '').strip()
        raw = (r['Etruscan'] or '').strip()
        tr = clean_tr(r['Translation'])
        k = (eid, raw)
        if k in by_key:
            dup_rows += 1
            rec = by_key[k]
            if tr and tr not in rec['trs']:
                rec['trs'].append(tr)
                merged_tr += 1
            continue
        id_seq[eid] += 1
        toks, n_lines = parse_text(raw, dash_split=False, stats=stats)
        lang, why = LANG_BY_ETP.get(eid, ('etr', ''))
        rec = {
            'rid': f'ETP:{eid}' + (f'#{id_seq[eid]}' if id_seq[eid] > 1 else ''),
            'src': 'ETP', 'eid': eid, 'key': '',
            'city': (r['City'] or '').strip() or None,
            'y_from': parse_year(r['Year - From']),
            'y_to': parse_year(r['Year - To']),
            'lang': lang, 'kind': classify_kind(toks),
            'raw': raw, 'raw_T': None, 'raw_C': None, 'reading': '',
            'trs': [tr] if tr else [],
            'toks': toks, 'n_lines': n_lines, 'seg': seg_class(toks),
        }
        if any('caps' in t['flags'] for t in toks) and lang == 'etr':
            stats['caps_records'].append(rec['rid'])
        by_key[k] = rec
        records.append(rec)
    log(f'ETP-часть: {len(by_key)} записей после слияния {dup_rows} строк-дублей '
        f'(влитых вариантов перевода: {merged_tr})')

    # --- CIEP-часть ------------------------------------------------------
    # контрольная проекция: каждая larth-строка CIEP-части обязана дословно
    # (по буквам) совпадать с T или C своей записи (CIEP-номер, key)
    def squash(s):
        s = pre_string(s or '').lower()
        s = unicodedata.normalize('NFD', s).replace('̣', '')
        return ''.join(c for c in s if c in LETTERS)

    ciep_by = {}
    for r in ciep:
        ciep_by.setdefault(((r['CIEP'] or '').strip(), (r['key'] or '').strip()),
                           []).append(r)
    n_t = n_c = n_bad = 0
    for r in ciep_rows_larth:
        k = ((r['ID'] or '').strip(), (r['key'] or '').strip())
        et = squash(r['Etruscan'])
        ts = [squash(x['T']) for x in ciep_by.get(k, [])]
        cs = [squash(x['C']) for x in ciep_by.get(k, [])]
        if et in ts:
            n_t += 1
        elif et in cs:
            n_c += 1
        else:
            n_bad += 1
    log(f'сверка larth(CIEP-часть) ↔ CIEP_pymupdf: текст=T {n_t}, текст=C {n_c}, '
        f'не сопоставлено {n_bad} (ожидание: 0)')
    assert n_bad == 0, 'larth CIEP-часть не является проекцией T/C'

    ik_seq = Counter()
    n_tc = {'C': 0, 'T': 0}
    for r in ciep:
        t_raw = (r['T'] or '').strip()
        c_raw = (r['C'] or '').strip()
        a = clean_tr(r['A'])
        num = (r['CIEP'] or '').strip()
        key = (r['key'] or '').strip()
        raw = c_raw or t_raw
        reading = 'C' if c_raw else 'T'
        n_tc[reading] += 1
        ik = (num, key)
        ik_seq[ik] += 1
        toks, n_lines = parse_text(raw, dash_split=True, stats=stats)
        lang, why = LANG_BY_CIE.get(num, ('etr', ''))
        rec = {
            'rid': f'CIEP:{num}:{key}'
                   + (f'#{ik_seq[ik]}' if ik_seq[ik] > 1 else ''),
            'src': 'CIEP', 'eid': num, 'key': key,
            'city': None, 'y_from': None, 'y_to': None,
            'lang': lang, 'kind': classify_kind(toks),
            'raw': raw, 'raw_T': t_raw or None, 'raw_C': c_raw or None,
            'reading': reading,
            'trs': [a] if a else [],
            'toks': toks, 'n_lines': n_lines, 'seg': seg_class(toks),
        }
        if any('caps' in tok['flags'] for tok in toks) and lang == 'etr':
            stats['caps_records'].append(rec['rid'])
        records.append(rec)
    log(f'CIEP-часть: {len(ciep)} записей (чтение C: {n_tc["C"]}, '
        f'только T: {n_tc["T"]}); двойных чтений T&C: '
        f'{sum(1 for r in ciep if (r["T"] or "").strip() and (r["C"] or "").strip())}')

    # --- supplement-механизм ---------------------------------------------
    supp_files = sorted(glob.glob(os.path.join(DATA, 'supplements', '*.csv')))
    n_supp = 0
    for path in supp_files:
        fname = os.path.basename(path)
        with open(path, encoding='utf-8') as f:
            for i, r in enumerate(csv.DictReader(f)):
                raw = (r.get('text') or '').strip()
                if not raw:
                    continue
                toks, n_lines = parse_text(raw, dash_split=False, stats=stats)
                tr = clean_tr(r.get('translation'))
                records.append({
                    'rid': f'SUPP:{fname}:{i}',
                    'src': f'SUPP:{fname}',
                    'eid': (r.get('source_id') or '').strip(), 'key': '',
                    'city': (r.get('city') or '').strip() or None,
                    'y_from': parse_year(r.get('date_from')),
                    'y_to': parse_year(r.get('date_to')),
                    'lang': (r.get('language') or 'etr').strip(),
                    'kind': classify_kind(toks),
                    'raw': raw, 'raw_T': None, 'raw_C': None, 'reading': '',
                    'trs': [tr] if tr else [],
                    'toks': toks, 'n_lines': n_lines, 'seg': seg_class(toks),
                })
                n_supp += 1
    log(f'supplements: файлов {len(supp_files)}, записей {n_supp}')
    return records, stats, larth, etp_fix, ciep


def losslessness_check(records):
    """Буквенный состав raw (минус {..}-вычеркнутое) == буквенный состав форм."""
    def sig_raw(raw):
        s = pre_string(raw)
        out = []
        for t in s.split():  # {..} вычёркивается токен-условно, как в parse_segment
            t2 = re.sub(r'\{[^}]*\}', '', t).replace('{', '').replace('}', '')
            if not any(c.isalpha() for c in t2):
                t2 = t.replace('{', '').replace('}', '')
            out.append(t2)
        s = ' '.join(out)
        s = unicodedata.normalize('NFD', s).replace('̣', '')
        s = unicodedata.normalize('NFC', s).lower()
        return sorted(c for c in s if c in LETTERS)

    def sig_toks(toks):
        out = []
        for t in toks:
            if t['kind'] in ('W', 'N'):
                out.extend(c for c in t['form'].lower() if c in LETTERS)
        return sorted(out)

    bad = []
    for rec in records:
        if sig_raw(rec['raw']) != sig_toks(rec['toks']):
            bad.append(rec['rid'])
    return bad


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('validation', exist_ok=True)
    os.makedirs(os.path.join(DATA, 'supplements'), exist_ok=True)

    log('=== Заморозка этрусского корпуса (этап 0), '
        f'нормализация v{NORM_VERSION} ===')
    in_sha = {}
    for f in ['etruscan_larth.csv', 'ETP_fix.csv', 'CIEP_pymupdf.csv']:
        in_sha[f] = sha256_file(os.path.join(DATA, f))
        log(f'вход: {f}  sha256={in_sha[f][:16]}…')
    log()

    records, stats, larth, etp_fix, ciep = build_records()

    # --- внутренние проверки ---------------------------------------------
    log()
    log('=== внутренние проверки ===')
    rids = Counter(r['rid'] for r in records)
    dup_rids = [k for k, v in rids.items() if v > 1]
    log(f'уникальность rid: дублей {len(dup_rids)} (ожидание: 0)')
    assert not dup_rids

    bad = losslessness_check(records)
    log(f'потерь букв (raw ↔ токены, по буквенному составу): '
        f'{len(bad)}/{len(records)} записей '
        f'({1 - len(bad) / len(records):.2%} прошло)')
    for rid in bad[:20]:
        log(f'   несовпадение: {rid}')

    y_bad = [r['rid'] for r in records
             if r['y_from'] is not None and r['y_to'] is not None
             and r['y_from'] < r['y_to']]
    log(f'датировки с y_from < y_to (годы до н.э., ожидание 0): {len(y_bad)}')

    log(f'ETP-часть: dash-split не применялся (конвенция); '
        f'CIEP-часть: токенов, разрезанных по одиночному дефису: '
        f'{stats["dash_splits"]}')
    log(f'выброшенные при чистке токены (без букв): {stats["dropped_tokens"]}')
    log(f'вычищенные символы: '
        f'{dict(sorted(stats["stripped_chars"].items()))}')
    log(f'записи lang=etr с caps-токенами (кандидаты в латинские, '
        f'проверить глазами): {stats["caps_records"]}')

    # --- языки, виды, сегментация ------------------------------------------
    log()
    log('=== состав корпуса ===')
    for field in ['src', 'lang', 'kind', 'seg', 'reading']:
        c = Counter((r[field].split(":")[0] if field == "src" else r[field])
                    for r in records)
        log(f'{field}: {dict(sorted(c.items()))}')

    # --- канонические числа (основной вид: lang=etr, kind=text) -----------
    log()
    log('=== канонические числа (lang=etr, kind=text) ===')
    view = [r for r in records if r['lang'] == 'etr' and r['kind'] == 'text']
    words = [t['form'] for r in view for t in r['toks'] if t['kind'] == 'W']
    vocab = Counter(words)
    hapax = sum(1 for c in vocab.values() if c == 1)
    n_num = sum(1 for r in view for t in r['toks'] if t['kind'] == 'N')
    n_gap = sum(1 for r in view for t in r['toks'] if t['kind'] == 'G')
    with_tr = sum(1 for r in view if r['trs'])
    log(f'записей: {len(view)}; словоформ-токенов: {len(words)}; '
        f'типов: {len(vocab)}; гапаксов: {hapax} ({hapax / len(vocab):.0%})')
    log(f'числительных-токенов: {n_num}; лакун-токенов: {n_gap}')
    log(f'записей с переводом: {with_tr} ({with_tr / len(view):.0%})')
    for part in ['ETP', 'CIEP']:
        pv = [r for r in view if r['src'] == part]
        pw = [t['form'] for r in pv for t in r['toks'] if t['kind'] == 'W']
        ptr = sum(1 for r in pv if r['trs'])
        log(f'  {part}: записей {len(pv)}, токенов {len(pw)}, '
            f'типов {len(set(pw))}, с переводом {ptr}')
    fl = Counter(f for r in view for t in r['toks'] for f in t['flags'])
    log(f'флаги токенов: {dict(sorted(fl.items()))}')

    # пересечение словарей частей через ascii-проекцию
    etp_asc = {t['ascii'] for r in view if r['src'] == 'ETP'
               for t in r['toks'] if t['kind'] == 'W'}
    ciep_asc = {t['ascii'] for r in view if r['src'] == 'CIEP'
                for t in r['toks'] if t['kind'] == 'W'}
    log(f'ascii-типов: ETP {len(etp_asc)}, CIEP {len(ciep_asc)}, '
        f'общих {len(etp_asc & ciep_asc)} — сопоставимость частей')

    # --- сборка и сериализация --------------------------------------------
    corpus = {
        'meta': {
            'norm_version': NORM_VERSION,
            'builder': 'tools/etr_freeze.py',
            'inputs_sha256': in_sha,
            'lang_rules': {'by_cie': LANG_BY_CIE, 'by_etp': LANG_BY_ETP,
                           'default': 'etr'},
            'ascii_map': ASCII_MAP,
            'note': 'Замороженный корпус: НЕ редактировать; добавления — '
                    'data/supplements/*.csv. Основной аналитический вид: '
                    "lang=='etr' and kind=='text'.",
        },
        'records': records,
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(corpus, f, protocol=4)
    sha = sha256_file(OUT_PKL)
    with open(OUT_SHA, 'w', encoding='utf-8') as f:
        f.write(f'{sha}  {os.path.basename(OUT_PKL)}  norm_v{NORM_VERSION}\n')
    log()
    log(f'записан {OUT_PKL}: {len(records)} записей, sha256={sha}')

    # --- выборка для ручной сверки (seed=42) -------------------------------
    rng = random.Random(42)
    sample = rng.sample(view, 50)
    with open(OUT_SAMPLE, 'w', encoding='utf-8') as f:
        f.write('# Ручная валидационная выборка (50 записей, seed=42)\n\n'
                'Сверить: чтение, разбор на токены, флаги, перевод. '
                'Отметки в колонке «OK?» (да/нет/примечание).\n\n'
                '| # | rid | raw | токены (флаги) | перевод | OK? |\n'
                '|---|-----|-----|----------------|---------|-----|\n')
        for i, r in enumerate(sample, 1):
            toks = ' '.join(
                t['form'] + ('⟨' + ','.join(t['flags']) + '⟩' if t['flags'] else '')
                for t in r['toks'])
            tr = (r['trs'][0][:60] if r['trs'] else '—').replace('|', '¦')
            raw = r['raw'][:70].replace('|', '¦')
            f.write(f'| {i} | {r["rid"]} | {raw} | {toks[:90]} | {tr} |  |\n')
    log(f'записана выборка для ручной сверки: {OUT_SAMPLE}')

    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
