# Ручная валидационная выборка (50 записей, seed=42)

Сверить: чтение, разбор на токены, флаги, перевод. Отметки в колонке «OK?» (да/нет/примечание).

| # | rid | raw | токены (флаги) | перевод | OK? |
|---|-----|-----|----------------|---------|-----|
| 1 | CIEP:16150:25 | selieifcle | selieifcle | selieifcle | да |
| 2 | CIEP:923:2 | en-------- | en--------⟨damaged⟩ | — | да |
| 3 | ETP:La 2.3 | araz silqetenas spurianas | araz silqetenas spurianas | (I am) Araz Silqetenas Spurianas. | да |
| 4 | CIEW:9001:94.917 | reuχzina . caveθ • zuslevac • ma<e>ra <•> s<ur>/ | reuχzina caveθ zuslevac maera⟨emended⟩ sur⟨emended,line_split⟩ | — | нет: парсер CIEW — F&W 9001.156 продолжается θi; в корпус попал обломок sur |
| 5 | CIEP:4720:2 | ferene | ferene | — | да |
| 6 | CIEP:3991:1 | se(th) casni(al) | seth⟨expanded⟩ casnial⟨expanded⟩ | — | да |
| 7 | CIEP:3427:1 | serturi | serturi | ms-serturi | да |
| 8 | CIEP:1473:1B | laut(nitha) | lautnitha⟨expanded⟩ | — | да |
| 9 | CIEW:9001:94.188 | meθlumeric . enas • raχθ • suo • nunθenθ | meθlumeric enas raχθ suo⟨erratum⟩ nunθenθ | — | нет: парсер CIEW — F&W 9001.188 и Herbig: suθ, не suo |
| 10 | CIEP:742:2B | peciania(sa) | pecianiasa⟨expanded⟩ | mr-pecianias-sons-wife | да |
| 11 | CIEP:16936:1B | evrfia | evrfia | — | да |
| 12 | CIEW:9001:94.910 | • ic • <e>svit<n/ | ic esvitn⟨emended,line_split⟩ | — | нет: парсер CIEW — обломок F&W 9001.151; потеряны начало строки и enas |
| 13 | CIEP:14316:1 | fe | fe | — | да |
| 14 | CIEP:438:4 | turce | turce | donated | да |
| 15 | CIEP:15370:2 | egnatvleiae c(aii) l(ibertvs) tyce | egnatvleiae caii⟨expanded⟩ libertvs⟨expanded⟩ tyce | ms-egnatuleiasmrcaiussfreedmanmstyceturtellia | да |
| 16 | CIEP:9179:2 | t(it)e thuma mi | tite⟨expanded⟩ thuma mi | — | да |
| 17 | ETP:ETP 100 | larθia | larθia | — | да |
| 18 | ETP:ETP 84 | vel : flere [ :---] [---] veluske : [---] | vel flere -⟨restored⟩ -⟨restored⟩ veluske -⟨restored⟩ | — | да |
| 19 | CIEP:553:1 | ar(n)th achuni(al) | arnth⟨expanded⟩ achunial⟨expanded⟩ | — | да |
| 20 | CIEP:3230:2 | vl | vl | ms-fasimrs-larthismr-vels | да |
| 21 | CIEP:3676:2 | puia | puia | — | да |
| 22 | CIEP:12325:1A | arthes | arthes | — | да |
| 23 | CIEP:15638:1A | iaiu | iaiu | iaiu | да |
| 24 | ETP:ETP 56 | tutis | tutis | — | да |
| 25 | CIEP:14678:1 | ankvenesankariateveiiae | ankvenesankariateveiiae | — | да |
| 26 | CIEP:2757:1 | thania tutn(i)al | thania tutnial⟨expanded⟩ | — | да |
| 27 | CIEW:7002:83.15 | epnicei nunθ<c>uci iei tu<rza> i ri<e>[na]<l t>ae iti l<a> halχ aper t | epnicei nunθcuci⟨emended⟩ iei turza⟨emended⟩ i rienal⟨emended,restored⟩ tae⟨emended⟩ iti l | — | нет: парсер CIEW — θ/I/φ прочитаны e/l/4, переносы не склеены |
| 28 | CIEP:16402:1 | ------a--ithm------ | ------a--ithm------⟨damaged⟩ | mr-caearithma | да |
| 29 | CIEP:20032:1 | ka | ka | — | да |
| 30 | CIEP:14304:1 | th(an)i(al) | thanial⟨expanded⟩ | — | да |
| 31 | CIEP:9110:2A | m(arcia)l | marcial⟨expanded⟩ | ms-marcis | да |
| 32 | CIEP:3305:1 | arsuthin | arsuthin | — | да |
| 33 | CIEP:9996:1 | t(it)i | titi⟨expanded⟩ | — | да |
| 34 | CIEP:15342:1C | zi(lchus) | zilchus⟨expanded⟩ | — | да |
| 35 | CIEP:4800:1 | arnthlatin | arnthlatin | mr-arnthlatinie | да |
| 36 | CIEW-CIE:3776 | peonei • ceisis | peonei ceisis | — |  |
| 37 | ETP:Vs 1.178 | vel leinies : larθial : ruva : arnθialum ¦ clan : velusum : prumaθσ :  | vel leinies larθial ruva arnθialum clan velusum prumaθσ avils semφσ lupuce | Vel Leinies, brother of Larth (and) son o f Arnth (and) grea | да |
| 38 | CIEW-CIE:119 | [θ]<a>na mi<nia avles cai>n<al> | θana⟨emended,restored⟩ minia⟨emended⟩ avles cainal⟨emended⟩ | — |  |
| 39 | CIEW-CIE:3480 | lθ . cincuni [a] -<l>a • <rafl--> | lθ cincuni a⟨restored⟩ -la⟨damaged,emended⟩ rafl--⟨damaged,emended⟩ | — |  |
| 40 | CIEP:1944:1 | fastia crmartnei(al) rutmate(s)a | fastia crmartneial⟨expanded⟩ rutmatesa⟨expanded⟩ | — | да |
| 41 | CIEP:17334:2L | vesvnae | vesvnae | — | да |
| 42 | CIEP:9183:4 | dextrorsvms | dextrorsvms | — | да |
| 43 | CIEP:6421:1 | minimuluvanicemamarceapuniievenala | minimuluvanicemamarceapuniievenala | — | да |
| 44 | CIEP:4797:2 | purnei(al) | purneial⟨expanded⟩ | mrs-purneis | да |
| 45 | CIEP:1872:1 | fastiacainei----minasa | fastiacainei----minasa⟨damaged⟩ | — | да |
| 46 | CIEP:3160:2 | rav(a)e | ravae⟨expanded⟩ | — | да |
| 47 | CIEW-CIE:475 | zerapiu > lautni fraucnal | zerapiu lautni fraucnal | — |  |
| 48 | CIEP:6214:2 | muca | muca | — | да |
| 49 | CIEP:733:1 | artnei(al) l(ar)th(us) | artneial⟨expanded⟩ larthus⟨expanded⟩ | mr-thanamrs-artneismr-larths | да |
| 50 | CIEP:531:2 | etrual | etrual | — | да |
