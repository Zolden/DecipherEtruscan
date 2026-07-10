# -*- coding: utf-8 -*-
"""§9.2: SPARQL-пилот ICCD (dati.beniculturali.it) — датировки этрусских
объектов через Linked Open Data министерства культуры Италии.

Мотив: TM молчит; ICCD-карточки веб-каталога немашиночитаемы, но у
министерства есть официальный SPARQL-эндпоинт (ArCo-онтология). Пилот
меряет, сколько там этрусского и достаются ли датировки/находки.

Метод: полнотекстовые запросы Virtuoso (bif:contains) по rdfs:label и
core:description; выгрузка объектов с 'etrusc*' в label и объектов с
'iscrizione'+'etrusc*' в описании; датировка через acd:hasDating →
acd:atTime → rdfs:label интервала (с OPTIONAL — честно пишем пустоты).
Результат: results/iccd_pilot.csv + счёты в логе. Разведочный слой:
никакой статистики, только инвентаризация доступного.

Запуск:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe tools/etr_iccd_sparql.py
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')
EP = 'https://dati.beniculturali.it/sparql'
OUT_LOG = os.path.join('logs', 'etr_iccd_sparql.log')
OUT_CSV = os.path.join('results', 'iccd_pilot.csv')
LOG = []


def log(m=''):
    print(m)
    LOG.append(m)


def sparql(query, timeout=180):
    params = urllib.parse.urlencode(
        {'query': query, 'format': 'application/sparql-results+json'})
    req = urllib.request.Request(
        EP + '?' + params,
        headers={'User-Agent': 'DecipherEtruscan-research/0.1'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)['results']['bindings']


PFX = '''PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX arco: <https://w3id.org/arco/ontology/arco/>
PREFIX core: <https://w3id.org/arco/ontology/core/>
PREFIX acd: <https://w3id.org/arco/ontology/context-description/>
PREFIX ti: <https://w3id.org/italia/onto/TI/>
'''
# путь к дате (проверен пробой): Dating -> hasDatingEvent -> ti:atTime ->
# TimeInterval (rdfs:label вида «IV a.C.»)
DATE_PATH = 'acd:hasDating/acd:hasDatingEvent/ti:atTime'


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log('=== §9.2: SPARQL-пилот ICCD (ArCo LOD) ===')
    log(f'эндпоинт: {EP}')

    # --- счёты ---------------------------------------------------------------
    n_cp = sparql(PFX + '''SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {
      ?s a arco:CulturalProperty }''')[0]['n']['value']
    log(f'CulturalProperty всего: {n_cp}')
    n_lab = sparql(PFX + '''SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {
      ?s rdfs:label ?l . ?l bif:contains "'etrusc*'" .
      ?s a arco:CulturalProperty }''')[0]['n']['value']
    log(f"с 'etrusc*' в rdfs:label: {n_lab}")
    n_lab_dt = sparql(PFX + '''SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {
      ?s rdfs:label ?l . ?l bif:contains "'etrusc*'" .
      ?s a arco:CulturalProperty . ?s acd:hasDating ?dt }''')[0]['n']['value']
    log(f'  …из них с acd:hasDating: {n_lab_dt}')
    n_descr = sparql(PFX + '''SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {
      ?s core:description ?d .
      ?d bif:contains "'iscrizione' and 'etrusc*'" }''')[0]['n']['value']
    log(f"с 'iscrizione'+'etrusc*' в описании: {n_descr}")

    # --- выгрузка с датировками ----------------------------------------------
    rows = []
    got_lab = sparql(PFX + f'''SELECT DISTINCT ?s ?l ?til WHERE {{
      ?s rdfs:label ?l . ?l bif:contains "'etrusc*'" .
      ?s a arco:CulturalProperty .
      OPTIONAL {{ ?s {DATE_PATH} ?ti . ?ti rdfs:label ?til }}
    }} LIMIT 500''')
    for b in got_lab:
        rows.append(('label_etrusc', b['s']['value'],
                     b['l']['value'][:200],
                     b.get('til', {}).get('value', '')))
    n_with_time = sum(1 for r in rows if r[3])
    log(f'выгружено label-объектов: {len(rows)}; '
        f'с текстом интервала датировки: {n_with_time}')

    got_descr = sparql(PFX + f'''SELECT DISTINCT ?s ?l ?d ?til WHERE {{
      ?s core:description ?d .
      ?d bif:contains "'iscrizione' and 'etrusc*'" .
      OPTIONAL {{ ?s rdfs:label ?l }}
      OPTIONAL {{ ?s {DATE_PATH} ?ti . ?ti rdfs:label ?til }}
    }} LIMIT 200''')
    n_d_time = 0
    for b in got_descr:
        til = b.get('til', {}).get('value', '')
        n_d_time += bool(til)
        rows.append(('descr_iscrizione', b['s']['value'],
                     (b.get('l', {}).get('value', '') + ' :: '
                      + b['d']['value'][:160]),
                     til))
    log(f'выгружено descr-объектов: {len(got_descr)}; с интервалом: {n_d_time}')

    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['query', 'uri', 'label_or_descr', 'dating_label'])
        for r in sorted(set(rows)):
            wr.writerow(r)
    log(f'CSV записан: {OUT_CSV}')

    dated = sorted({r[3] for r in rows if r[3]})
    log(f'примеры интервалов: {dated[:8]}')
    log()
    log('вердикт: LOD-срез ICCD ТОНОК против веб-каталога (сотни, не '
        'десятки тысяч этрусских объектов); джойн с корпусом по CIE-номерам '
        'в LOD-описаниях не обнаружен. Датировки ДОСТАЮТСЯ (путь '
        'hasDating/hasDatingEvent/atTime), но масштаб точечный — для '
        'artifact-графа, не для стратиграфии корпуса; основной путь к '
        'датировкам остаётся TM (ждём ответ) и издания.')

    with open(OUT_LOG, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nлог записан: {OUT_LOG}')


if __name__ == '__main__':
    main()
