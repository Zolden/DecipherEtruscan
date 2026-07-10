# CIE online (studietruschi.org) — манифест

Источник: https://www.studietruschi.org/corpus-inscriptionum-etruscarum-online
(официальный Istituto di Studi Etruschi; фасцикулы как PDF-сканы).
manifest.csv — перечень фасцикулов с диапазонами tituli (снято 2026-07-10).

Приоритеты извлечения (адресно, не массовый OCR):
1. Supplementum Herbig 1919–21 (Liber Lintei) — PD по возрасту. СДЕЛАНО
   2026-07-10: Index verborum извлечён (tools/etr_herbig_index.py →
   data/supplements/herbig_ll_index.csv), структурные якоря периодов —
   data/supplements/herbig_ll_structure.md; PDF скачан (Supplementum_I.pdf,
   51 МБ, OCR-слой института; PDF не коммитится — размер).
2. Vol. IV (Gaucci 2017): tit. 20001–21071. ПОПРАВКА 2026-07-10: прежняя
   пометка «нет в корпусе» НЕВЕРНА — 105 CIE-номеров диапазона уже в
   корпусе через CIEP (Хилл), из них 41 в части IV.1.1. Часть IV.1.1
   (Atria 20001–20422) скачана (born-digital, полный текстовый слой);
   перепись: 422 титула, 23 Rix ET Ad-конкорданса, чтения в основном
   факсимиле-рисунками («ex apographo»), в прозе — точечно (vipus,
   tiniaś, ialu…). Потенциал: ~950 новых коротких надписей (керамические
   метки Адрии) на весь том; 2017 — в копирайте, извлекать факты-чтения.
3. Vol. II.1.5 (2006) и II.2.2 (1996) — проверить покрытие против CIEW.
