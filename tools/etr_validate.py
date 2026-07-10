# -*- coding: utf-8 -*-
"""Этап 0 (§0): внешняя валидация замороженного корпуса.

ETP-онлайн (etp.classics.umass.edu) на момент заморозки НЕДОСТУПЕН
(ECONNREFUSED), поэтому ручная сверка 50 записей с ETP-БД невозможна;
замена: (а) детерминированная выборка validation/sample50.md для ручной
проверки по изданиям, (б) автоматическая сверка трёх больших текстов с
опубликованными чтениями (зашиты ниже с атрибуцией источника):

  1. Cippus Perusinus  ↔ записи CIEP:4538:* (пословная конкорданс-проекция);
  2. Tabula Cortonensis ↔ запись ETP:ETP 74#6 (полный текст);
  3. Liber Linteus col. III (образец) ↔ лексика LL-записей ETP-части —
     слабая проверка (в корпусе другие строки LL), только пересечение
     формульной лексики.

Метод сравнения: корпусная сторона — конкатенация ascii-форм ТОКЕНОВ
записи (валидируется весь конвейер raw→токены); опубликованная сторона —
ascii-«сквош» (θ→th, χ→ch, φ→ph, σ/ς/ś/š→s, ê/ɜ→e, всё не-буквенное
прочь, без границ слов). Сквош устраняет различия издательских конвенций
(σ vs ś vs s; ê vs ɜ) и СЛОВОДЕЛЕНИЯ (издания делят слова по-разному:
scɜvɜś ↔ σcê vês). Числительные (IVXLC) исключаются с обеих сторон:
зона числительных TCo повреждена и читается изданиями по-разному.

Расхождения, найденные при первом прогоне, разобраны поштучно; подлинные
издательские варианты занесены в EXPLAINED_VARIANTS (белый список с
обоснованием) — вердикт требует нуля НЕОБЪЯСНЁННЫХ расхождений.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_validate.py
"""
import os
import pickle
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')

OUT_LOG = os.path.join('logs', 'etr_validate.log')
LOG = []


def log(msg=''):
    print(msg)
    LOG.append(msg)


# --- опорные тексты (внешние публикации) -----------------------------------
# Cippus Perusinus, транскрипция по F. Roncalli (цит. по статье
# en.wikipedia.org/wiki/Cippus_Perusinus, снято 2026-07-08).
CP_TEXT = """
teurat tanna la rezu l ame vaχr lautn velθinaś eśtla afunaś sleleθ caru
tezan fuśleri tesnś teiś raśneś ipa ama hen naper XII velθinaθuraś araś
peraśc emulm lescul zuci enesci epl tularu aulesi velθinas arznal clensi
θii θil ścuna cenu eplc felic larθalś afuneś clen θunχulθe falaś χiem
fuśle velθina hinθa cape municlet masu naper śran czl θii falaśt velθina
hut naper penezś masu acnina clel afuna velθina mler zinia inte mamer cnl
velθina zia śatene tesne eca velθina θuraś θaura helu tesne raśne cei
tesnś teiś raśneś χimθ śpel θuta ścuna afuna mena hen naper ci cnl hare
utuśe velθina śatena zuci enesci ipa śpelaneθi fulumχva śpelθi reneθi
eśtac velθina acilune turune ścune zea zuci enesci aθumicś afunaś penθna
ama velθina afuna θuruni ein zeriuna cla θil θunχulθl iχ ca ceχa ziχuχe
"""

# Tabula Cortonensis, секция I лицевой стороны (транскрипция по статье
# en.wikipedia.org/wiki/Tabula_Cortonensis, снято 2026-07-08; ɜ —
# «перевёрнутый эпсилон», особый знак этой таблички = ê данных Larth).
TCO_SECTION1 = """
et pɜtruiś scɜvɜś ɜliuntś vinac restmc cenu tɜnθur śar cusuθuraś
larisalisvla pes spante tɜnθur sa śran śarc clθii tɜrsna θui spanθi
mlɜśieθic raśna inni pes pɜtruś pavac traulac tiur tɜnθurs tɜnθaś
zacinat priniserac zal
"""

# Liber Linteus, col. III 12–17 (van der Meer 2007: 78–82; цит. по статье
# en.wikipedia.org/wiki/Liber_Linteus, снято 2026-07-08).
LL_COL3 = """
fler etnam tesim etnam celucn cletram śrenχve trin θezine χim fler
tarc mutin um anancveś nac cal tarc θezi vacl an ścanince saucsaθ persin
cletram śrenχve iχ ścanince clt vacl ara nunθene śaθaś naχve heχz male
"""

ASCII_MAP = {'θ': 'th', 'χ': 'ch', 'φ': 'ph', 'σ': 's', 'ς': 's',
             'ś': 's', 'š': 's', 'ê': 'e', 'ɜ': 'e', "'": ''}
LETTERS = set('abcdefghijklmnopqrstuvwxyzθχφσςśšêɜ\'')

# Подлинные расхождения изданий (разобраны вручную при первом прогоне):
# ключ — сквош-форма, которой нет в другой стороне; значение — объяснение.
EXPLAINED_VARIANTS = {
    'rezus': 'CP, строка 1 повреждена; чтения расходятся: CIEP «rezus» vs '
             'Roncalli «(la) rezu (l)» / др. «larezul» — вариант издания',
    'finthamuniclet': 'CP: CIEP читает finθa, Roncalli hinθa (классическая '
                      'мена f/h) — вариант издания',
    'clthii': 'TCo: изд. clθii vs корпус clθil (мена i/l на повреждённом '
              'месте) — вариант издания',
    'tenthas': 'TCo: изд. tɜnθa[ś] с реставрированным ś; корпус даёт tênθa '
               'без реставрации — вариант реставрации',
}


def squash(s):
    """ascii-сквош без границ слов; числительные IVXLC выброшены."""
    s = unicodedata.normalize('NFC', s)
    words = re.split(r'\s+', s)
    words = [w for w in words if not re.fullmatch(r'[IVXLC]+', w.strip('.·'))]
    s = ''.join(words).lower()
    s = ''.join(c for c in s if c in LETTERS)
    return ''.join(ASCII_MAP.get(c, c) for c in s)


def words_of(s):
    out = []
    for w in re.split(r'\s+', s.strip()):
        if not w or re.fullmatch(r'[IVXLC]+', w):
            continue
        sq = squash(w)
        if sq:
            out.append(sq)
    return out


def main():
    os.makedirs('logs', exist_ok=True)
    corpus = pickle.load(open(os.path.join('data', 'etr_corpus.pkl'), 'rb'))
    recs = corpus['records']
    log('=== Внешняя валидация корпуса (этап 0) ===')
    log('ETP-онлайн недоступен (ECONNREFUSED) — сверка по опубликованным '
        'чтениям трёх больших текстов; см. докстринг скрипта.')
    log()

    def squash_rec(rec):
        """Корпусная сторона: конкатенация ascii-форм словоформ-токенов."""
        return ''.join(t['ascii'] for t in rec['toks'] if t['kind'] == 'W')

    unexplained = 0

    # --- 1. Cippus Perusinus ↔ CIEP:4538 -------------------------------
    log('--- 1. Cippus Perusinus (Roncalli) ↔ CIEP:4538:* ---')
    cp_sq = squash(CP_TEXT)
    cp_recs = [r for r in recs if r['src'] == 'CIEP' and r['eid'] == '4538']
    ok = expl = bad = 0
    for r in cp_recs:
        rs = squash_rec(r)
        if not rs:
            continue
        if rs in cp_sq:
            ok += 1
        elif rs in EXPLAINED_VARIANTS:
            expl += 1
            log(f'   вариант издания: {r["rid"]} {rs!r} — '
                f'{EXPLAINED_VARIANTS[rs]}')
        else:
            bad += 1
            log(f'   НЕОБЪЯСНЕНО: {r["rid"]} {r["raw"][:50]!r} → {rs[:40]!r}')
    unexplained += bad
    log(f'записей CIEP:4538: подстрока опубликованного текста {ok}, '
        f'объяснённые варианты изданий {expl}, необъяснённых {bad}')

    # --- 2. Tabula Cortonensis ↔ ETP:ETP 74#6 ---------------------------
    log()
    log('--- 2. Tabula Cortonensis (секция I) ↔ ETP:ETP 74#6 ---')
    tco = [r for r in recs if r['rid'] == 'ETP:ETP 74#6']
    assert len(tco) == 1, 'запись ETP:ETP 74#6 не найдена'
    rec_sq = squash_rec(tco[0])
    pub_words = words_of(TCO_SECTION1)
    hit = [w for w in pub_words if w in rec_sq]
    miss = [w for w in pub_words if w not in rec_sq]
    expl2 = [w for w in miss if w in EXPLAINED_VARIANTS]
    bad2 = [w for w in miss if w not in EXPLAINED_VARIANTS]
    unexplained += len(bad2)
    log(f'слов опубликованной секции I в записи: {len(hit)}/{len(pub_words)}; '
        f'объяснённые варианты: {len(expl2)}, необъяснённые: {len(bad2)}')
    for w in expl2:
        log(f'   вариант издания: {w!r} — {EXPLAINED_VARIANTS[w]}')
    if bad2:
        log(f'   НЕОБЪЯСНЕНЫ: {bad2}')

    # --- 3. Liber Linteus col. III ↔ лексика LL-записей (слабая) --------
    log()
    log('--- 3. Liber Linteus col. III (van der Meer) ↔ LL-записи (слабая) ---')
    ll_recs = [r for r in recs if r['src'] == 'ETP'
               and r['eid'].startswith('LL')]
    ll_vocab = {t['ascii'] for r in ll_recs for t in r['toks']
                if t['kind'] == 'W'}
    pub_vocab = set(words_of(LL_COL3))
    inter = sorted(ll_vocab & pub_vocab)
    log(f'LL-записей в корпусе: {len(ll_recs)} (столбцы 2–11, другие строки, '
        f'чем опубликованный образец col. III — проверка только лексики)')
    log(f'пересечение словарей: {len(inter)} типов: {inter}')

    # --- 4. графема ê = ɜ Кортонской таблички ---------------------------
    log()
    log('--- 4. графема ê (Larth) = ɜ (перевёрнутый эпсилон TCo) ---')
    e_eids = sorted({r['eid'] for r in recs
                     if any('ê' in t['form'] for t in r['toks']
                            if t['kind'] == 'W')})
    log(f'записи с ê встречаются только в ID: {e_eids} '
        f'(ожидание: только ETP 74 = Tabula Cortonensis)')

    # --- вердикт ---------------------------------------------------------
    log()
    verdicts = [unexplained == 0,
                len(inter) >= 3,
                e_eids == ['ETP 74']]
    log(f'вердикты (необъяснённых расхождений 0; LL-пересечение≥3; '
        f'ê только TCo): {verdicts}')
    log('ИТОГ: ' + ('корпус согласуется с опубликованными чтениями '
                    '(все расхождения — задокументированные варианты изданий)'
                    if all(verdicts) else
                    'ЕСТЬ НЕОБЪЯСНЁННЫЕ РАСХОЖДЕНИЯ — разобрать до этапа 2'))

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
