# INPUT FROM SOL — аудит и новые направления, 2026-07-10

Статус после полного перепрогона: freeze v0.6, SHA-256 корпуса
`f5cba35e…`; 6795 этрусских records, 12 450 W-токенов, 7384 типа.
Подробный аудит: `METHOD_AUDIT_SOL_20260710.md`; числа:
`results/method_audit_sol_20260710.json`.
После исправления всех оставшихся text-writers два последовательных прогона
22 скриптов дали побайтово одинаковые SHA-256 для 37/37 артефактов.

## Самое важное

1. **Лемносский положительный вывод отозван.** CIEP 15999 содержал два
   фрагмента самой Лемносской стелы (`sivai`, `evistho` и др.), ошибочно
   размеченные `etr`; они попадали в обучение и затем служили «точными
   когнатами» цели. После исправления и type-disjoint split разные корректные
   NB выбирают `lat` 67–70, `etr` 30–33 раз из 100. Это не латинское родство,
   а неспособность суффиксного классификатора атрибутировать отдельный малый
   язык. Допустимый вывод только отрицательный; конкретные заранее известные
   параллели (`aviz sialχviz`, -l) обсуждать отдельно.
2. **Ключевая морфологическая комплементарность частично устойчива.** Старый
   all-match относил `X-ial` сразу к `-ial/-al/-l`, `X-isa` к `-isa/-sa/-s`.
   Longest-match + полный контроль оставляет 5 из заявленных 10 пар. Главная
   `-s/-al` сохраняется на source forms: obs=5, E=21.42,
   `p_Bonf(×91)=1.67e-5`; `-s/-l`
   тоже сохраняется. Исчезают, в частности, вложенные `-l/-al` и `-l/-ial`.
   Sensitivity без `ocr?`-records и без токенов
   `damaged/uncertain/restored/emended` оставляет те же 5/10; для `-s/-al`
   obs=4, E=15.77, `p_Bonf(×91)=.00178`.
   Массовая депрессия 27/86 невложенных пар указывает также на misfit простого
   hypergeom-null/лексические классы: сама депрессия ещё не доказывает падеж.
3. **Заявленный Westfall–Young в operator/second-position/LL коде неверен:**
   тесты симулируются независимо и совместная зависимость не сохраняется.
   Основные очень сильные сигналы переживают консервативный Bonferroni, но
   опубликованные `p̃` не следует цитировать. `turce` имеет 11/22 вторых
   позиций, raw `p=.0006`, однако против остальных dedicatory verbs различия
   нет (11/22 против 7/21, Fisher `p=.358`). Результат совместим с гипотезой
   общего жанрового слота dedicatory verb, но не доказывает её; уникальный
   «оператор Ваккернагеля» не подтверждён.
4. **Зависимая единица — памятник/источник, не record.** 5363 CIEP-records
   соответствуют 4203 CIE и лишь 2130 TM; Liber Linteus представлен CIEP,
   ETP и CIEW. Record-level permutation часто псевдореплицирует. Нужен
   обязательный `artifact_id` и перестановки/bootstraps по artifact, затем
   внутри source/genre/region/time strata.
5. 20.7% токенов имеют epistemic-флаги damaged/uncertain/restored/emended/OCR;
   1470 типов встречаются только в таких чтениях. Все открытия обязаны иметь
   clean-reading replication и sensitivity по альтернативным чтениям.

## Исправленные грубые ошибки

- `tools/etr_freeze.py`: CIE 15999 теперь `lemn`; из etr-вида удалены 2
  records / 5 токенов. `etr_lemnos*.py` берут цель только из чистого
  supplement, без дублирования CIEP.
- `tools/etr_semantics3.py`: prae/nomen теперь агрегируются между CSV-строками;
  исключены 3 конфликтных типа. 354→351, accuracy 77.1%→75.4%, `p=.001`
  сохранилось.
- Supplement loader теперь действительно требует непустой `provenance`,
  сохраняет `provenance/note`, фиксирует schema и хэши supplement-ов.
- Два полных прогона 22 скриптов дали одинаковые SHA-256 для 37/37
  артефактов; `compileall` и 10 regression-тестов прошли. Ручная
  `validation/sample50.md` всё ещё имеет 0/50 заполненных `OK?` — это
  следующий P0.

Calibration bug в `etr_semantics.py` и `etr_semantics2.py` исправлен:
`bin_acc.get(...) or raw_conf` заменял empirical accuracy 0.0 сырой
уверенностью; теперь `None` проверяется явно. Это не реабилитирует старую
калибровку: fixed split многократно использован при разработке.

## Новые численные результаты

### A. Context-only semantics (без формы и без CIEP)

`tools/etr_method_audit_sol.py`: только 1249 multiword records, gold только
ETP_POS, 5-fold с case-stripped families целиком в одном fold, признаки только
PPMI-контекст. После исключения tied gold — 181 размеченный тип freq≥2,
5 классов:

- accuracy .519 против majority .497 (отдельного теста этой разницы нет);
- balanced accuracy **.391** против family-block permutation-null .198,
  **p=.0005** (R=2000; полные векторы меток переставляются между семействами
  одинакового размера, сохраняя marginals и внутрисемейную зависимость);
- macro-F1 .391; VERB precision .606 / recall .541 / F1 **.571**;
- NAME-M F1 .675; THEO .133, FUNC .157.

Глобальный тест показывает, что контекст несёт некоторый сигнал. VERB имеет
лучший **описательный** F1 среди слабых неименных классов, но отдельного
VERB-null и парного сравнения с form baseline не было. До вывода о глаголах
и генерации значений нужны per-class test, nested artifact-level CV и
bootstrap по памятникам.

### B. Внешний Raetic→Etruscan transfer

Добавлен официальный TIR (см. ниже) и `tools/etr_raetic_transfer.py`.
Интерпретируемый categorical NB с uniform class prior обучается **только** на 90 чистых Raetic
словарных формах TIR (финальные 1–4-граммы) и без refit проверяется на 506
attested non-suffix ETP_POS-типах с хотя бы одним token-clean вхождением вне
record-level `ocr?`.
Шесть типов, встречающихся только в помеченных чтениях, исключены. Модель явно
воздерживается, если ни одна обучающая 1–4-грамма не известна:

- покрытие **435/506 = 86.0%**; на покрытых типах accuracy **.676**,
  balanced accuracy **.756**, macro-F1 .613;
- точное пересечение нормализованных Raetic train и Etruscan test surfaces:
  **0**;
- conditional target-family bootstrap 95% CI balanced accuracy
  **[.714, .795]** (фиксирует малый TIR train);
- lemma-block permutation обучающих меток: `p_raw=.00070`,
  `p_Bonferroni×4=.00280`; marginal-preserving target-family block null:
  `p_raw=.00010`, `p_Bonferroni×4=.00040`;
- GEN precision/recall .878/.721; NOM .661/.589; PERT .274/.958;
- descriptive sensitivity с empirical TIR class prior усиливает результат до
  accuracy **.708**, balanced accuracy **.780**;
- строгая графемная нормализация: покрытие **356/480 = 74.2%**, balanced
  accuracy **.719**; sensitivity только на 66 TIR-строках с `checklevel=0`:
  покрытие 420/506 и balanced **.723**.

После просмотра основного результата отдельно выбрана **post-hoc** более
простая модель: один самый длинный известный суффикс длины 2–4, без сложения
вложенных n-грамм. Её покрытие 258/506 (51.0%), accuracy .888, balanced .896;
на subset, совпадающем с eligibility кандидатов по `freq_clean>=2`, `n=102`,
accuracy .931, balanced .939. Именно эта post-hoc модель калибрует 265 строк
`results/raetic_transfer_candidates.csv` (GEN 127, NOM 118, PERT 20). Для
GEN/NOM/PERT matched precision соответственно .938 (61/65), 1.000 (31/31),
.500 (3/6); Wilson CI [.852,.976], [.890,1.000], [.188,.812] обязательны.
Поле `already_present_in_etp_pos` показывает, что 23/265 форм уже встречаются
в других (исключённых из gold) строках ETP_POS и потому даже формально не
являются новыми словарными кандидатами.
Record-level `ocr?` теперь исключается из clean frequency, а Liber Linteus и
Tabula Capuana склеиваются между ETP/CIEP/CIEW как единые памятники. Поэтому
список сократился с 278 до 265. Главные единицы неопределённости вынесены в
CSV: 133/265 строк опираются на суффикс, встреченный в TIR train лишь один
раз, 59/265 имеют только один artifact-cluster; одновременно `TIR support>=2`
и `artifact clusters>=2` выполняются лишь для **91/265**.

Это измеримое согласие прежде всего **номинальной/ономастической** суффиксации,
а не независимое доказательство родства или всего падежного строя: 86/90
обучающих TIR-форм — proper nouns, а анализ TIR сам использует этрусские
параллели. CSV — только exploratory очередь ручной проверки;
`score_margin_nonprobabilistic`
там является разностью log-likelihood, не вероятностью. Индивидуальных
candidate p-values и FDR-контроля нет, поэтому строки нельзя читать как
установленные падежи или переводы.

### C. Устойчивые/ослабленные старые сигналы

- Просопографический MI остаётся при artifact/source-cluster permutation:
  **p=.0005**; индивидуально устойчивы 7, а не прежние 10 основ.
- Временная неоднородность сохраняется при record-null, но после условной
  перестановки внутри geography: **p=.0225**. Monte Carlo по интервалам дат:
  MI median .568, 95% [.540, .608]. География объясняет часть эффекта.
- Genre `p=.038` не переживает семейство 5 жанров (`p_Bonf=.190`): тренд,
  не подтверждённое открытие.
- Semantics v2: accuracy .596, но macro-F1 ≈.38; FUNC F1=0, THEO=.10,
  VERB=.31; контекст улучшает лишь 4/349, McNemar `p=.125`; ни одной
  гипотезы с `prob_cal≥.7`.
- Concept fog: исправленный глобальный count-null даёт `p=.0005`, но
  max-stat FWER оставляет лишь `shekel-weight` и `cumin`; лексикон частично
  model-generated и параметры выбраны на том же gold. Никаких индивидуальных
  значений из fog заявлять нельзя до внешней ревизии лексикона.

## Новые подходы, рекомендуемый порядок

### 1. Artifact graph + lattice чтений (P0)

Сделать graph `artifact → edition/reading → record → token`, где ETP/CIEP/CIEW/
supplement — не независимые тексты, а наблюдения одного памятника. Не выбирать
одну строку навсегда: хранить альтернативные чтения как lattice с весами
edition/checklevel. Любую статистику считать (a) по MAP-чтению, (b) Monte
Carlo по lattice, (c) cluster bootstrap по artifact. Критерий открытия:
знак и CI устойчивы во всех трёх режимах и минимум в двух источниках.

### 2. Held-out productive morphology: MDL/FST (P1)

Вместо перебора хвостов и hypergeom-universe обучать конечный трансдьюсер/
MDL-сегментацию на train-artifacts. Алломорфы (`-s/-ś`, `-l/-al`, `-ce/-ke`)
задаются как конкурирующие анализы; longest-match — обязательный baseline.
Оценка только на полностью held-out stem families/sites/centuries:
compression gain, held-out word likelihood, recovery ETP_POS case labels и
precision парадигм. Сравнить с char-n-gram LM и length/final-letter null.

### 3. Context-only anchor graph + latent ritual slots (P1)

Продолжить положительный PPMI-пилот: bipartite graph `word ↔ context/slot`,
anchors только из внешнего expert gold, без формы слова. Для LL/Capua —
semi-Markov/HMM на периодах, а не строках; latent states должны предсказывать
удержанные anchors (CAL/THEO/OFFER/VERB), а не получать названия post hoc.
Nested CV по artifact; permutation anchors внутри genre/source; publish
negatives. Главная цель — VERB, где описательная точка F1=.571 задаёт
ориентир, но отдельного классового null пока нет.

### 4. Cross-Tyrsenian multi-task transfer (P2)

TIR даёт 112 inscriptions, явно классифицированных `language=Raetic`, 139
Raetic lexical entries и 11
морфем. Обучать общий character-FST с раздельными language heads и явными
соответствиями, но проверять Etruscan→Raetic и Raetic→Etruscan симметрично,
leave-lemma-family-out. Использовать transfer только как prior; неизвестное
этрусское значение принимать лишь при независимом context/genre/translation
подтверждении. Текущий case-transfer — положительный feasibility test.

### 5. Formula minimal pairs (P2, высокий upside)

Искать пары/кластеры надписей с одинаковым artifact-clean шаблоном и одной
заменой токена. На переведённой ETP-части измерить, совпадает ли заменённый
этрусский токен с единственным изменившимся английским concept; split строго
по formula-family и artifact. Затем применить только правила с held-out
precision/CI и FDR. Это ближе к естественному «контролируемому эксперименту»
в формульном корпусе, чем document-level bag-of-words/IBM-1.

### 6. Held-out restoration как обязательный sanity benchmark (P1)

Искусственно маскировать буквы и токены только в надёжных чтениях; split по
физическому объекту, formula-family и near-duplicate cluster. Сравнивать
top-k/MRR с character n-gram и Markov baseline, отдельно по длине и жанру.
Это проверяет, выучил ли pipeline структуру языка, и помогает реставрациям,
но не является семантической дешифровкой. Подход переносится из Pythia и
вычислительных обзоров гораздо лучше, чем нейронный «перевод» без близкого
high-resource языка.

### 7. Onomastic/object graph и held-out morpheme alignment (P1–P2)

USEP группирует разрозненные CIEP labels на одном зеркале/объекте. Строить
граф `label ↔ изображённая фигура ↔ object/site/date`; нуль — перестановка
внутри `object-class × period × site`, holdout — целый объект и name-family.
Для TIR/Lemnian сравнивать не слова целиком, а заранее заданные морфемные
соответствия: leave-one-correspondence-out, MRR/top-k, matched random lexemes
с теми же length/frequency/genre, permutation p и cluster bootstrap CI.
Имена/заимствования анализировать отдельным stratum.

## Новые внешние материалы

- `data/external/tir/`: воспроизводимая API-выгрузка **TIR**, 389 inscriptions,
  293 objects, 154 word entries, 11 morphemes; exact revision IDs, URLs,
  checksums; CC BY-SA 3.0. Очистка внутренней разметки чтений автоматически
  совпала с официальным rendered plain text для 389/389 записей. Для языка
  использовать только `language=Raetic` (112 inscriptions), не все 389.
- `data/external/usep_tyrsenian/`: pinned Brown USEP EpiDoc commit
  `d34663b…`: 35 `ett` records (26 с транскрипцией, 9 metadata-only) и один
  `xrr` metadata-only; CC BY-NC-SA 4.0. Ценность — object grouping, музейный
  provenance, материал/дата/изображения/библиография, а не новый массовый
  текст. С frozen corpus полностью совпали лишь 2 нормализованные editions;
  сливать автоматически нельзя.
- `data/external/lexlep_celtic_etruscan/`: 11 revision-pinned записей
  Lexicon Leponticum, отобранных как Etruscan с возможным кельтским
  материалом; CC BY-NC-SA 4.0. Это контактная ономастика, не общий корпус;
  только 3 имеют уже уверенный локальный concordance, `SH·1` может быть
  нелингвистическим.
- `data/external/etruscan_reference/`, `raetic_reference/` и
  `computational_decipherment_reference/`: открытые peer-reviewed обзоры
  Belfiore 2020, Salomon 2020 (включая personal-name tables) и Braović et al.
  2024; README фиксируют DOI, лицензию, область применимости и SHA-256.
- Mnamon Lemnian, первичная публикация Hephaistia, IESP и Eichner найдены, но
  не скопированы: нет пригодной лицензии на перераспределение. Использовать
  только как ссылочные/ручные источники и не превращать их correspondences в
  независимый gold без проверки циркулярности. Ссылки:
  https://mnamon.sns.it/index.php?id=26&lang=en&page=Lingua ;
  https://www.persee.fr/doc/crai_0065-0536_2010_num_154_1_92847 ;
  https://iesp.tarchna.di.unimi.it/ ;
  https://jolr.ru/files/%28125%29jlr2013-10%281-42%29.pdf .
- Официальный Институт исследований этрусков открыл **весь CIE** онлайн,
  включая fascicles 2006/2017 и официальный Liber Linteus supplement:
  https://www.studietruschi.org/corpus-inscriptionum-etruscarum-online .
  Это приоритетный источник для artifact IDs, изображений, чтений, дат и
  библиографии; PDF не коммитить без проверки условий, сначала сделать
  manifest и targeted extraction новых CIE 20001–21071/6325–6723.
- Там же открыт исторический архив REE/Studi Etruschi (кроме последних
  томов): https://www.studietruschi.org/rivista-di-epigrafia-etrusca-online .
- Peer-reviewed open volume **Rhaeti&Co** (2024), DOI
  `10.60973/RHAETI2024BOOK`, CC BY-NC-ND 4.0; главы Hajnal и Marchesini дают
  явные сравнительные таблицы этрусского/ретийского/лемнийского. Ссылка:
  https://alteritas.it/wp-content/uploads/2024/11/RhaetiCo-light.pdf .
- Güell Pratdepàdua 2025: открытая магистерская работа с корпусом **66**
  votive `turce` inscriptions, включая San Casciano dei Bagni; полезна для
  внешней проверки жанрового слота, но лицензия CC BY-NC-ND — не извлекать и
  не публиковать производную таблицу без разрешения:
  https://hdl.handle.net/2445/223748 .
- OpenEtruscan пока не считать источником данных: заявленные JSON/CSV/RDF
  dumps возвращают 404, DVC remote в опубликованном repo помечен retired,
  live API в основном дублирует Larth и числа сайта/API расходятся.

## Оценка целей

**Сильная цель (полный перевод произвольного текста):** на текущих данных
маловероятна. 70.6% records однословны, 81% типов — hapax, переводы/ETP_POS
частично происходят из той же научной традиции, а длинные тексты не имеют
полного независимого билингвального gold. Нужен новый длинный билингв,
словарь/глоссы или качественно новая археологическая привязка; алгоритм не
создаст отсутствующую информацию.

**Слабая цель (новые проверяемые сведения):** реалистична и уже дала
результат: устойчивая комплементарность-кандидат `-s/-al`, artifact-robust
география, ослабленная, но реальная временная структура, глобальный
context-only сигнал и измеримый Raetic transfer. Следующий научно полезный результат вероятнее
будет структурным/морфологическим с CI, а не «переводом слова».
