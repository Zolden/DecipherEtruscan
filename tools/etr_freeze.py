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

v0.5 — CIEW-эксклюзивные CIE-записи (ciew_cie_entries.csv,
  shared_with_ciep=0): надписи CIE, которых НЕТ в CIEP-части (у Хилла
  пропущены) — src='CIEW-CIE', метаданные (регион/язык/подделки) через
  тот же join Бурман по CIE-номеру; порог качества — глобальная
  валидация CIEW↔CIEP (вложенность медиана 0.91), плюс флаг 'ocr?' на
  записях с неразрешённой дизамбигуацией. Общие с CIEP номера НЕ
  грузятся (двойной счёт).

v0.4 — источник CIEW (Fowler & Wolfe 1965, парсинг tools/etr_ciew_parse.py
  → data/external/fowler_wolfe/ciew_bigtexts.csv): большие тексты
  построчно — 7002 Tabula Capuana, 9001 Liber Linteus (Runes), 9002
  Лемносская стела (lang='lemn'), 9003 печень Пьяченцы, 9011–9016
  Харсекин; 9021 (Пирги) НЕ грузится (дубль supplement-а). Флаги OCR
  ('ocr?', 'ocr-fixed') — на уровне записи. Токены из одних цифр —
  числительные (F&W конвертировал римские в арабские).

v0.3 — слияние ВАРИАНТОВ ЧТЕНИЯ одного памятника (негатив N4 отчёта):
  записи с одинаковым (src, eid, key), одинаковым числом словоформ и
  совпадением ≥80% ascii-токенов считаются вариантами чтения; канон —
  первая запись группы (порядок файла), остальные получают variant_of
  (исключаются из аналитического вида), их переводы вливаются в канон.
  Пример: ETP:LL 6 (luśao/luśaσ/luśaσ' — одна строка LL в трёх чтениях).

v0.2 — метаданные из кросс-таблицы Бурман (data/external/burman/,
  Zenodo 10.5281/zenodo.17209666, CC0; версия 1.0.3): join по CIE-номеру
  даёт CIEP-записям (а) регион по сигле ET Рикса/Майзера, (б) уточнение
  языка по составу ссылок (только Bakkum → фалискский 'fal'; только
  CIL → 'lat'), (в) флаг 'forgery?' по Notes «Considered …» (подделки/
  подозрительные — исключаются из канонического вида), (г) xref
  (Trismegistos/ET1/ET2/TLE). eid ≥ ~14000 в кросс-таблице отсутствуют —
  это до-нумерация Хилла («CIE in progressu»), не настоящие CIE-номера.
  ETP-части регион ставится по сигле её собственного eid (Cr 2.20 → Cr).

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
FREEZE_VERSION = '0.7'
CIEW_CIE_CSV = os.path.join('data', 'external', 'fowler_wolfe',
                            'ciew_cie_entries.csv')
CIEW_CSV = os.path.join('data', 'external', 'fowler_wolfe',
                        'ciew_bigtexts.csv')
CIEW_SKIP = {'9021', '9002'}  # Пирги и Лемнос уже в supplements (двойной счёт)
CIEW_META = {'7002': ('Cm', 'Capua')}  # регион/город, где известны
CIEW_LANG = {'9002': 'lemn'}
DATA = 'data'
BURMAN_CSV = os.path.join('data', 'external', 'burman',
                          'burman_concordance_1.0.3.csv')

# Сиглы регионов ET (Rix/Meiser); имена — только уверенно отождествляемые,
# прочие сиглы хранятся как есть.
REGION_SIGLA = {'Cr', 'Ta', 'AT', 'Vc', 'AV', 'Vs', 'Ve', 'Cl', 'Pe', 'Ar',
                'Co', 'Vt', 'Fs', 'Ru', 'Vn', 'Po', 'Cm', 'Fa', 'La', 'AS',
                'AH', 'Ad', 'OA', 'OI', 'Um', 'Sp', 'Li', 'Na', 'Af'}
REGION_NAMES = {'Cr': 'Caere', 'Ta': 'Tarquinii', 'AT': 'ager Tarquiniensis',
                'Vc': 'Vulci', 'AV': 'ager Vulcentanus', 'Vs': 'Volsinii',
                'Ve': 'Veii', 'Cl': 'Clusium', 'Pe': 'Perusia',
                'Ar': 'Arretium', 'Co': 'Cortona', 'Vt': 'Volaterrae',
                'Fs': 'Faesulae', 'Ru': 'Rusellae', 'Vn': 'Vetulonia',
                'Po': 'Populonia', 'Cm': 'Campania', 'Fa': 'Falerii',
                'La': 'Latium', 'AS': 'ager Saenensis', 'AH': 'ager Hortanus',
                'Ad': 'Atria', 'Um': 'Umbria'}
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
    '15999': ('lemn', 'лемнийский: фрагменты стелы из Каминьи — sivai, '
                      'fokiasiale, evistho; памятник также представлен '
                      'полным supplement-изданием'),
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


def load_burman():
    """Кросс-таблица Бурман: CIE-номер → метаданные (регион/язык/подделка/xref).

    Ключ — CIE-номер без ведущих нулей (+суффикс, если есть), как в eid
    CIEP-части. При нескольких строках на номер: ET берётся первый
    непустой, ссылки объединяются по ИЛИ, Notes конкатенируются.
    """
    out = {}
    if not os.path.exists(BURMAN_CSV):
        return out
    with open(BURMAN_CSV, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f, delimiter=';'))
    hdr = rows[0]
    for row in rows[1:]:
        r = dict(zip(hdr, row))
        m = re.match(r'CIE (\d+)(.*)$', (r.get('CIE') or '').strip())
        if not m:
            continue
        key = str(int(m.group(1))) + m.group(2).strip()
        et = (r.get('Rix. ET1') or '').strip() or (r.get('Meiser. ET2') or '').strip()
        d = out.setdefault(key, {'et': '', 'bakkum': False, 'cil': False,
                                 'notes': [], 'tm': '', 'tle': ''})
        if et and not d['et']:
            d['et'] = et
        d['bakkum'] = d['bakkum'] or bool((r.get('Bakkum') or '').strip())
        d['cil'] = d['cil'] or any((r.get(c) or '').strip() for c in
                                   ['CIL I', 'CIL I(2)', 'CIL III',
                                    'CIL VI', 'CIL XI'])
        nt = (r.get('Notes') or '').strip()
        if nt:
            d['notes'].append(nt)
        if not d['tm']:
            d['tm'] = (r.get('Trismegistos') or '').strip()
        if not d['tle']:
            d['tle'] = (r.get('TLE') or '').strip()
    return out


def et_region(et_siglum):
    """'ET1 Cl 1.0234' → 'Cl'; None, если сиглы нет/не распознана."""
    m = re.match(r'ET[12] ([A-Za-z]+)[ .]', et_siglum + ' ')
    if m and m.group(1) in REGION_SIGLA:
        return m.group(1)
    return None


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
    if re.fullmatch(r'\d+', seg):  # арабские числительные (источник CIEW)
        return {'form': seg, 'kind': 'N', 'flags': tuple(sorted(flags))}
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
        pref = eid.split()[0] if ' ' in eid else ''
        region = pref if pref in REGION_SIGLA else None
        rec = {
            'rid': f'ETP:{eid}' + (f'#{id_seq[eid]}' if id_seq[eid] > 1 else ''),
            'src': 'ETP', 'eid': eid, 'key': '',
            'city': (r['City'] or '').strip() or None,
            'y_from': parse_year(r['Year - From']),
            'y_to': parse_year(r['Year - To']),
            'region': region, 'region_name': REGION_NAMES.get(region),
            'xref': None, 'flags': (),
            'provenance': None, 'note': None,
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

    burman = load_burman()
    bj = Counter()  # статистика join
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
        # join с кросс-таблицей Бурман (регион/язык/подделки/xref)
        region = None
        xref = None
        rflags = set()
        b = burman.get(num)
        if b is not None:
            bj['eid найден'] += 1
            region = et_region(b['et']) if b['et'] else None
            xref = {'tm': b['tm'], 'et': b['et'], 'tle': b['tle']}
            if num not in LANG_BY_CIE:
                if b['et']:
                    pass  # есть в Etruskische Texte → этрусская, lang как есть
                elif b['bakkum']:
                    lang = 'fal'
                    bj['язык → fal (Bakkum)'] += 1
                elif b['cil']:
                    lang = 'lat'
                    bj['язык → lat (только CIL)'] += 1
            if any(n.startswith('Considered') for n in b['notes']):
                rflags.add('forgery?')
                bj['флаг forgery?'] += 1
            if region:
                bj['есть регион'] += 1
        rec = {
            'rid': f'CIEP:{num}:{key}'
                   + (f'#{ik_seq[ik]}' if ik_seq[ik] > 1 else ''),
            'src': 'CIEP', 'eid': num, 'key': key,
            'city': None, 'y_from': None, 'y_to': None,
            'region': region, 'region_name': REGION_NAMES.get(region),
            'xref': xref, 'flags': tuple(sorted(rflags)),
            'provenance': None, 'note': None,
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
    log(f'join Бурман (по записям): {dict(sorted(bj.items()))}'
        if bj else 'join Бурман: кросс-таблица не найдена, метаданные не добавлены')

    # --- supplement-механизм ---------------------------------------------
    supp_files = sorted(glob.glob(os.path.join(DATA, 'supplements', '*.csv')))
    n_supp = 0
    for path in supp_files:
        fname = os.path.basename(path)
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            required = {'text', 'provenance'}
            missing_cols = required - set(reader.fieldnames or ())
            assert not missing_cols, (
                f'{path}: отсутствуют обязательные колонки supplement: '
                f'{sorted(missing_cols)}')
            for i, r in enumerate(reader):
                raw = (r.get('text') or '').strip()
                if not raw:
                    continue
                provenance = (r.get('provenance') or '').strip()
                assert provenance, (
                    f'{path}, строка CSV {i + 2}: для непустого text '
                    'обязателен provenance')
                toks, n_lines = parse_text(raw, dash_split=False, stats=stats)
                tr = clean_tr(r.get('translation'))
                records.append({
                    'rid': f'SUPP:{fname}:{i}',
                    'src': f'SUPP:{fname}',
                    'eid': (r.get('source_id') or '').strip(), 'key': '',
                    'city': (r.get('city') or '').strip() or None,
                    'y_from': parse_year(r.get('date_from')),
                    'y_to': parse_year(r.get('date_to')),
                    'region': None, 'region_name': None,
                    'xref': None, 'flags': (),
                    'provenance': provenance,
                    'note': (r.get('note') or '').strip() or None,
                    'lang': (r.get('language') or 'etr').strip(),
                    'kind': classify_kind(toks),
                    'raw': raw, 'raw_T': None, 'raw_C': None, 'reading': '',
                    'trs': [tr] if tr else [],
                    'toks': toks, 'n_lines': n_lines, 'seg': seg_class(toks),
                })
                n_supp += 1
    log(f'supplements: файлов {len(supp_files)}, записей {n_supp}')

    # --- источник CIEW (большие тексты Fowler & Wolfe) ---------------------
    n_ciew = 0
    if os.path.exists(CIEW_CSV):
        with open(CIEW_CSV, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                e = row['entry'].strip()
                if e in CIEW_SKIP:
                    continue
                raw = (row['text'] or '').strip()
                if not raw:
                    continue
                toks, n_lines = parse_text(raw, dash_split=False, stats=stats)
                region, city = CIEW_META.get(e, (None, None))
                rflags = tuple(sorted(set((row['flags'] or '').split())))
                records.append({
                    'rid': f'CIEW:{e}:{row["col"]}.{row["line"]}',
                    'src': 'CIEW', 'eid': e,
                    'key': f'{row["col"]}.{row["line"]}',
                    'city': city, 'y_from': None, 'y_to': None,
                    'region': region,
                    'region_name': REGION_NAMES.get(region),
                    'xref': None, 'flags': rflags,
                    'provenance': None, 'note': None,
                    'lang': CIEW_LANG.get(e, 'etr'),
                    'kind': classify_kind(toks),
                    'raw': raw, 'raw_T': None, 'raw_C': None, 'reading': '',
                    'trs': [],
                    'toks': toks, 'n_lines': n_lines,
                    'seg': seg_class(toks),
                })
                n_ciew += 1
        log(f'CIEW: записей {n_ciew} (из {CIEW_CSV}; 9021 пропущен — '
            f'дубль supplement-а Пирги)')
    else:
        log('CIEW: файл не найден, источник не подключён')

    # --- v0.5: CIEW-эксклюзивные CIE-записи --------------------------------
    n_cie = 0
    if os.path.exists(CIEW_CIE_CSV):
        cie_lang = Counter()
        with open(CIEW_CIE_CSV, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['shared_with_ciep'] == '1':
                    continue
                num = row['entry'].strip()
                raw = (row['text'] or '').strip()
                toks, n_lines = parse_text(raw, dash_split=False, stats=stats)
                lang = 'etr'
                region = None
                xref = None
                rflags = set((row['flags'] or '').split())
                b = burman.get(num)
                if b is not None:
                    region = et_region(b['et']) if b['et'] else None
                    xref = {'tm': b['tm'], 'et': b['et'], 'tle': b['tle']}
                    if b['et']:
                        pass
                    elif b['bakkum']:
                        lang = 'fal'
                    elif b['cil']:
                        lang = 'lat'
                    if any(n_.startswith('Considered') for n_ in b['notes']):
                        rflags.add('forgery?')
                cie_lang[lang] += 1
                records.append({
                    'rid': f'CIEW-CIE:{num}',
                    'src': 'CIEW-CIE', 'eid': num, 'key': '',
                    'city': None, 'y_from': None, 'y_to': None,
                    'region': region,
                    'region_name': REGION_NAMES.get(region),
                    'xref': xref, 'flags': tuple(sorted(rflags)),
                    'provenance': None, 'note': None,
                    'lang': lang, 'kind': classify_kind(toks),
                    'raw': raw, 'raw_T': None, 'raw_C': None, 'reading': '',
                    'trs': [],
                    'toks': toks, 'n_lines': n_lines,
                    'seg': seg_class(toks),
                })
                n_cie += 1
        log(f'CIEW-CIE (v0.5): записей {n_cie}, языки {dict(cie_lang)}')
    return records, stats, larth, etp_fix, ciep


BIG_ART = {'9001': 'ART:LL', '7002': 'ART:TCap', '9002': 'ART:Lemnos',
           '9003': 'ART:Liver'}
CIE_ART = {'15910': 'ART:LL', '15911': 'ART:TCo', '8682': 'ART:TCap',
           '15999': 'ART:Lemnos'}


def artifact_of(rec):
    """artifact_id (v0.7): один физический памятник для всех его чтений
    из разных источников (ETP/CIEP/CIEW/supplement). Единица независимости
    для confirmatory-перестановок (аудит Sol, §8.1)."""
    e, src = rec['eid'], rec['src']
    if src in ('CIEP', 'CIEW-CIE'):
        return CIE_ART.get(e, f'CIE:{e}')
    if src == 'CIEW':
        return BIG_ART.get(e, f'CIEW:{e}')
    if src == 'ETP':
        if e.startswith('LL'):
            return 'ART:LL'
        if e == 'ETP 74':
            return 'ART:TCo'
        return f'ETP:{e}'
    if src.startswith('SUPP'):
        low = src.lower()
        if 'pyrgi' in low:
            return 'ART:Pyrgi'
        if 'lemnos' in low:
            return 'ART:Lemnos'
    return f'{src}:{e}'


def merge_variants(records):
    """Помечает варианты чтения одного памятника (см. докстринг, v0.3).

    Возвращает число записей, помеченных как варианты. Детерминизм:
    группы и каноны — в порядке следования записей.
    """
    groups = {}
    for r in records:
        r['variant_of'] = None
        groups.setdefault((r['src'], r['eid'], r['key']), []).append(r)
    n_var = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        canon = []  # [(запись, ascii-сигнатура словоформ)]
        for r in g:
            sig = [t['ascii'] for t in r['toks'] if t['kind'] == 'W']
            if not sig:
                continue  # пустые/только-лакуны не сливаем
            hit = None
            for c, csig in canon:
                if len(sig) == len(csig):
                    agree = sum(a == b for a, b in zip(sig, csig))
                    if agree >= 0.8 * len(sig):
                        hit = c
                        break
            if hit is None:
                canon.append((r, sig))
            else:
                r['variant_of'] = hit['rid']
                hit['flags'] = tuple(sorted(set(hit['flags'])
                                            | {'has_variants'}))
                for tr in r['trs']:
                    if tr not in hit['trs']:
                        hit['trs'].append(tr)
                n_var += 1
    return n_var


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
    for path in sorted(glob.glob(os.path.join(DATA, 'supplements', '*.csv'))):
        key = os.path.relpath(path, DATA).replace(os.sep, '/')
        in_sha[key] = sha256_file(path)
        log(f'вход: {key}  sha256={in_sha[key][:16]}…')
    log()

    records, stats, larth, etp_fix, ciep = build_records()
    for rec in records:
        rec['artifact_id'] = artifact_of(rec)
    n_art = len({rec['artifact_id'] for rec in records})
    log(f'artifact_id (v0.7): {n_art} памятников на {len(records)} записей')
    n_var = merge_variants(records)
    log(f'варианты чтения (v0.3): помечено variant_of {n_var} записей')

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
    assert not bad, 'потеря букв при нормализации raw ↔ токены'

    y_bad = [r['rid'] for r in records
             if r['y_from'] is not None and r['y_to'] is not None
             and r['y_from'] < r['y_to']]
    log(f'датировки с y_from < y_to (годы до н.э., ожидание 0): {len(y_bad)}')
    assert not y_bad, 'некорректный порядок границ датировки'

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
    creg = Counter(r['region'] or '∅' for r in records)
    log(f'region: {dict(sorted(creg.items(), key=lambda x: -x[1]))}')
    log(f'записей с флагом forgery?: '
        f'{sum(1 for r in records if "forgery?" in r["flags"])}')

    # --- канонические числа (вид: etr, text, не подделка, не вариант) -----
    log()
    log('=== канонические числа (lang=etr, kind=text, без forgery?, '
        'без variant_of) ===')
    view = [r for r in records if r['lang'] == 'etr' and r['kind'] == 'text'
            and 'forgery?' not in r['flags'] and r['variant_of'] is None]
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
    for part in ['ETP', 'CIEP', 'SUPP']:
        pv = [r for r in view if r['src'].split(':')[0] == part]
        pw = [t['form'] for r in pv for t in r['toks'] if t['kind'] == 'W']
        ptr = sum(1 for r in pv if r['trs'])
        log(f'  {part}: записей {len(pv)}, токенов {len(pw)}, '
            f'типов {len(set(pw))}, с переводом {ptr}')
    vreg = sum(1 for r in view if r['region'] or r['city'])
    log(f'записей вида с регионом или городом: {vreg} ({vreg / len(view):.0%})')
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
    if os.path.exists(BURMAN_CSV):
        in_sha['burman_concordance_1.0.3.csv'] = sha256_file(BURMAN_CSV)
    if os.path.exists(CIEW_CSV):
        in_sha['ciew_bigtexts.csv'] = sha256_file(CIEW_CSV)
    if os.path.exists(CIEW_CIE_CSV):
        in_sha['ciew_cie_entries.csv'] = sha256_file(CIEW_CIE_CSV)
    corpus = {
        'meta': {
            'norm_version': NORM_VERSION,
            'freeze_version': FREEZE_VERSION,
            'builder': 'tools/etr_freeze.py',
            'inputs_sha256': in_sha,
            'lang_rules': {'by_cie': LANG_BY_CIE, 'by_etp': LANG_BY_ETP,
                           'burman': 'Bakkum-only→fal; CIL-only→lat; '
                                     'Notes Considered…→flags forgery?',
                           'default': 'etr'},
            'ascii_map': ASCII_MAP,
            'region_names': REGION_NAMES,
            'supplement_schema': {
                'required': ('text', 'provenance'),
                'preserved': ('provenance', 'note'),
            },
            'note': 'Замороженный корпус: НЕ редактировать; добавления — '
                    'data/supplements/*.csv. Основной аналитический вид: '
                    "lang=='etr' and kind=='text' and 'forgery?' not in "
                    "flags and variant_of is None.",
        },
        'records': records,
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(corpus, f, protocol=4)
    sha = sha256_file(OUT_PKL)
    with open(OUT_SHA, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'{sha}  {os.path.basename(OUT_PKL)}  norm_v{NORM_VERSION}\n')
    log()
    log(f'записан {OUT_PKL}: {len(records)} записей, sha256={sha}')

    # --- выборка для ручной сверки (seed=42) -------------------------------
    rng = random.Random(42)
    sample = rng.sample(view, 50)
    # Keep human review persistent across deterministic freeze reruns.  The
    # sample may change when the corpus changes, so preserve annotations by
    # stable rid rather than by row number.
    existing_ok = {}
    existing_sample_rids = set()
    if os.path.exists(OUT_SAMPLE):
        with open(OUT_SAMPLE, encoding='utf-8') as old:
            for line in old:
                if not line.startswith('|'):
                    continue
                cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
                if len(cells) == 6 and cells[1] != 'rid' and cells[1] != '-----':
                    existing_sample_rids.add(cells[1])
                    if cells[5]:
                        existing_ok[cells[1]] = cells[5]
    new_sample_rids = {r['rid'] for r in sample}
    lost_annotations = sorted(set(existing_ok) - new_sample_rids)
    assert not lost_annotations, (
        'новая sample50 вытесняет вручную проверенные rid; сохранить старую '
        'выборку как versioned review перед freeze: ' + ', '.join(lost_annotations)
    )
    with open(OUT_SAMPLE, 'w', encoding='utf-8', newline='\n') as f:
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
            ok = existing_ok.get(r['rid'], '').replace('|', '¦')
            f.write(f'| {i} | {r["rid"]} | {raw} | {toks[:90]} | {tr} | {ok} |\n')
    preserved = sum(r['rid'] in existing_ok for r in sample)
    log(f'записана выборка для ручной сверки: {OUT_SAMPLE}; '
        f'предыдущих rid: {len(existing_sample_rids)}; '
        f'сохранено ручных отметок по rid: {preserved}')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
