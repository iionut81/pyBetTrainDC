# GOAL_BY_70 —9noiembrie2026

Actualizat 2026-11-09T09:05:59.094+02:00, Europe/Bucharest. [Review anterior](../Review_2026-11-08/REVIEW.md) și [lecții](../Review_2026-11-08/LESSONS_FOR_NEXT_SCAN.md) predate la2026-11-09T08:59:22.313+02:00; cercetare începută2026-11-09T08:59:38.380+02:00. Corecția editorială a numelor din preambulul lecțiilor este documentată; rezultatele și verificările cerute existau înaintea scanării. Fără modificări de modele, ponderi, calibrare, cote bookmaker sau xG inventat.

Inventar: **29 ligi,4 partide,4 modeleGol70 și3cornere**, 124 istorice unice anterioare00:00RO9noiembrie. Primele20asociații din [clasamentul UEFA încheiat în2026](https://www.uefa.com/nationalassociations/uefarankings/?year=2026), plusRomânia,Serbia și7divizii secunde urmărite. Lista celor29 de ligi, inclusiv toate cele26 fără partide identificate, este în [universe_coverage.csv](../Goal70_2026-11-09/universe_coverage.csv). Anglia,Italia,Spania,Germania,Franța,Portugalia,Olanda,Belgia,Turcia,Cehia,Grecia,Polonia,Danemarca,Norvegia,Cipru,Elveția,Austria,Scoția,Suedia,Croația sunt cele20; suplimenteRomânia/Serbia șiE1,D2,SP2,I2,F2,B2,TR2.

FotMob global/calendar sezon șiESPN au fost confruntate. Norvegia1,România1,Spania2 sunt ligile cu partide. ESPN nesuportat pentruPolonia,Croația,Serbia,Belgia2: absențele lor sunt acoperite de calendarul sezonului aceluiași furnizorFotMob, fără confirmare independentă exhaustivă. ESPN gol pentruRomânia contrazice clubulBotoșani șiFotMob. LasPalmas–Eldense a fost omis dinFotMob global; [clubul confirmă9noiembrie19:30Canare=21:30RO](https://www.udlaspalmas.es/noticias/el-partido-j13-laspalmaseldense-se-disputara-manana-lunes-9-de-noviembre-a-las-1930-horas-hora-canaria-en-el-estadio-gran-canaria), la felESPN; endpointul individualFotMob a confirmat ulterior noua oră și neînceperea. Inventarul inițial3partide și toate predicțiile inițiale sunt păstrate înRevisions/; cele3valori/ranguri/timestampuri nu au fost rescrise.

[Excluderi](../Goal70_2026-11-09/exclusions.csv):54partide din afara universului convenit, fără a le declara inexistente sau calculate. [Cupe](../Goal70_2026-11-09/cups_coverage.json): programul global consultat este inventariat separat; nu aparUCL/UEL/UECL eligibile și nu proiectăm formulele naționale automat peste cupe. Calendarul global nu este dovadă exhaustivă pentru toate cupele. [Statusuri individuale](../Goal70_2026-11-09/status_at_selection.csv):4/4 neîncepute, ultima verificare09:03RO;3partide au fost confirmate și prin surse oficiale distincte, iarLasPalmas prin comunicatul oficial reprogramării.

**Cercetare parțială**, cu calcul complet pentru eligibilele identificate: o observațieFirstHalf lipsește în istoriculLasPalmas/Eldense, deci nu există modelcornere pentru ea. ContextAPI pentru4partide,8echipe și80rânduri din ultimele10oficiale;9FirstHalf și1listă de goluri nereconciliate rămân lipsă în aceste serii. Titularii nu sunt confirmați; cronologia completă a schimbărilor de antrenor și recuperarea bolnavilorVålerenga nu sunt verificate. Confirmarea independentă exhaustivăFirstHalf lipsește. Nu numim cercetarea top global complet verificat.

Metodă: numai campionatul actual, minimum5observații sezon și3pe teren per echipă, fără observații cerute lipsă. λsezon=(GF_A/n_A+GA_B/n_B+GF_B/n_B+GA_A/n_A)/2; analog pe teren; λ=0,5λsezon+0,5λteren. Gol70:1−exp(−λ),stres0,8λ. Cornere:PoissonCDF(7,λ),stres1,2λ. GF70 numără TOATE golurile valide≤70, inclusiv45+; nu aplicăm70/90. FirstHalf≤7inclusiv45+; lipsa nu devinezero. Procente exploratorii necalibrate, frecvențele istorice separate și eșantioanele suprapuse;1/p este exclusiv cotă matematică.

## GOAL_BY_70: NO BET

**Lider statistic Kristiansund–Vålerenga,20:00RO:90,90%brut,85,30%stres,1,1001matematic.**47/49istoric și23/24teren; GF70propriu11/12acasă și12/12afară. Nu îl transform în recomandare: [NFF](https://www.fotball.no/turneringer/eliteserien/2026/kristiansund---valerenga-utsatt/) confirmă documente medicale pentru12jucători și mutarea partidei din7în9noiembrie. [Vålerenga](https://www.vif-fotball.no/nyheter/kristiansund-kampen-flyttes-til-mandag) confirmă motivul. Nu avem lista celor recuperați sau aprobarea medicală individuală de azi; istoricul de sezon nu descrie această situație. Nu deducem că apărarea slăbită garantează goluri: atacul, ritmul și selecția sunt și ele incerte. Este o judecată contextuală pentru această partidă, fără penalizare numerică sau filtru nou automat.

Etapa26,Nordmøre,gazonartificial. [ClubulKBK](https://www.kristiansundbk.no/lag/import/tournament/eliteserien-2026/season/fotballsesongen-2026/match/round-26-kristiansund-x-valerenga) indică gazdele pe15cu23puncte după25meciuri, deci puncte importante în lupta de jos; miza nu impune gol≤70. API:Skiri4-3-3/Moesgaard4-4-2, XIprobabil; Igor/Tufekčić șiDrammeh/Rashani în listaAPI, care omite focarul colectiv. Formația nu dovedește tactica. Verificările actuale de lot au prioritate față de o listăAPI aparent scurtă.

Alternative: Sabadell–RealSociedadB87,70%/81,30%,19/24și10/11;3meciuri0–0Sabadell în sezon,2deschideri tardiveSanse, probabilitate mai mică și risc de control defensiv. Botoșani–Oțelul85,72%/78,93%,24/28și11/12,GFpropriu8/6și4/6; contribuție ofensivă mai modestă a oaspeților și echipă în schimbare. LasPalmas–Eldense84,01%/76,93%,20/24; alertă meteo ridicată și reprogramare, context/loturi incomplet confirmate. Nu forțez înlocuirea liderului cu un meci mai slab documentat.

## Lecții aplicate din review

| Lecție | Verificare și dovadă nouă | Statut și efect |
| --- | --- | --- |
| Model/frecvență/AR20 | Toate4partidele calculate înaintea alegerii;Sabadell94,84/88,57versus24/24 | Aplicată; nu înlocuim modelul cu100%istoric |
| GF70propriu,tardive/0–0 |11/12și12/12la lider;Sabadell3zero-zero,Sanse2tardive în sezon; tabele exacte | Aplicată;NO BETGol70, fără procente modificate |
| Intern/Europa/cupă | Formula doar sezon intern;80rânduri oficiale cu competiție separată | Aplicată;9FirstHalf și1gol nereconciliate rămân explicite |
| Dispersie,exact7/≥8 |Sabadell2exact7/0≥8, varianțe teren1,70/1,87; alternative în tabele | Aplicată; susține preferința, eșantioane5/6 |
| Loturi și antrenori actuali |NFF12bolnaviVIF;Sanzsuspendat/Quadriincert contrazicXIvechi;Ansotegi confirmat primar | Aplicată parțial; focarul oprește recomandareaGol70, titulari neconfirmați |
| Program/status independent |LasPalmas omis în global identificat dinESPN+club;4/4neîncepute individual | Aplicată; revizia inventarului păstrată, fără predicție retrospectivă |
| Meteo/odihnă |PrognozeOpenMeteo șiAPI, ultimele date verificate; itinerare incomplete | Aplicată parțial; fără ajustări numerice |
| Cauze și calibrare |Brighton15înaintearoșului23; eșecurile nu au cauze tactice demonstrate | Neverificate exhaustiv; fărătuning sau promisiuni |

## Clasament complet al universului identificat

| Rang original | Meci /RO | Brut | AR20 | 1/p matematic | Istoric unic | Teren | Rol/lipsă |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kristiansund–Vålerenga 20:00 | 90.90% | 85.30% | 1.1001 | 47/49 | 23/24 | comparat |
| 2 | Sabadell–Real Sociedad B 21:30 | 87.70% | 81.30% | 1.1402 | 19/24 | 10/11 | comparat |
| 3 | Botoșani–Oțelul Galați 20:00 | 85.72% | 78.93% | 1.1666 | 24/28 | 11/12 | comparat |
| 4 | Las Palmas–Eldense 21:30 | 84.01% | 76.93% | 1.1903 | 20/24 | 9/12 | comparat |

## Kristiansund–Vålerenga — numărători și sensibilitate

| Eșantion | n / lipsă | GF / GA | reușite | tardiv /0–0 | medie /varianță | exact7 /≥8 |
| --- | --- | --- | --- | --- | --- | --- |
| home | 25 /0 | 21 /37 | 24 | 0 /1 | 2.32 /2.226666666666667 | — |
| away | 25 /0 | 31 /34 | 24 | 1 /0 | 2.6 /3.9166666666666674 | — |
| home_venue | 12 /0 | 11 /13 | 11 | 0 /1 | 2 /1.4545454545454546 | — |
| away_venue | 12 /0 | 12 /20 | 12 | 0 /0 | 2.6666666666666665 /4.606060606060607 | — |

λsezon=2.46,λteren=2.3333333333333335,λ50/50=2.3966666666666665. Numai-sezon91.46%,numai-teren90.30%,brut90.90%,stres85.30%. Fără recalibrare contextuală.

OpenMeteo la2026-11-09T20:00RO:7.8°C, ploaie10%,vânt27km/h. Prognoză, fără inspecție teren. APIpoate diferi; unitatea vântuluiAPI nu este explicită.

### Kristiansund — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-03-15 | Kristiansund–Brann | acasă | 3 | success | 10 | fail |
| 2026-04-06 | Kristiansund–Bodø/Glimt | acasă | 29 | success | 5 | success |
| 2026-04-11 | Tromsø–Kristiansund | afară | 60 | success | 5 | success |
| 2026-04-19 | Kristiansund–Fredrikstad | acasă | 48 | success | 4 | success |
| 2026-04-26 | Aalesund–Kristiansund | afară | 62 | success | 8 | fail |
| 2026-05-03 | Kristiansund–Hamarkameratene | acasă | 60 | success | 3 | success |
| 2026-05-10 | Sandefjord–Kristiansund | afară | 45+5 | success | 2 | success |
| 2026-05-16 | Molde–Kristiansund | afară | 42 | success | 4 | success |
| 2026-05-20 | Lillestrøm–Kristiansund | afară | 14 | success | 5 | success |
| 2026-05-24 | Kristiansund–Viking | acasă | 2 | success | 5 | success |
| 2026-05-29 | Vålerenga–Kristiansund | afară | 1 | success | 3 | success |
| 2026-07-12 | Rosenborg–Kristiansund | afară | 25 | success | 6 | success |
| 2026-07-18 | Kristiansund–Sarpsborg 08 | acasă | 0–0 | fail | 6 | success |
| 2026-07-25 | Kristiansund–Start | acasă | 15 | success | 5 | success |
| 2026-08-02 | KFUM–Kristiansund | afară | 16 | success | 4 | success |
| 2026-08-09 | Kristiansund–Molde | acasă | 25 | success | 7 | success |
| 2026-08-16 | Fredrikstad–Kristiansund | afară | 4 | success | 5 | success |
| 2026-08-30 | Hamarkameratene–Kristiansund | afară | 49 | success | 7 | success |
| 2026-09-06 | Kristiansund–Tromsø | acasă | 14 | success | 5 | success |
| 2026-09-13 | Viking–Kristiansund | afară | 4 | success | 5 | success |
| 2026-09-19 | Kristiansund–Rosenborg | acasă | 31 | success | 8 | fail |
| 2026-10-10 | Bodø/Glimt–Kristiansund | afară | 44 | success | 4 | success |
| 2026-10-17 | Kristiansund–Aalesund | acasă | 27 | success | 5 | success |
| 2026-10-25 | Kristiansund–KFUM | acasă | 11 | success | 0 | success |
| 2026-11-01 | Sarpsborg 08–Kristiansund | afară | 3 | success | 4 | success |

### Kristiansund — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-01 | Eliteserien | Sarpsborg 08–Kristiansund | 2-4 | 3 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105037) |
| 2026-10-28 | Cup | Kristiansund–Aalesund | 0-2 | 38 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6284091) |
| 2026-10-25 | Eliteserien | Kristiansund–KFUM | 2-2 | 11 | success | 0 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5104933) |
| 2026-10-17 | Eliteserien | Kristiansund–Aalesund | 1-4 | 27 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105028) |
| 2026-10-10 | Eliteserien | Bodø/Glimt–Kristiansund | 3-1 | 44 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105017) |
| 2026-09-19 | Eliteserien | Kristiansund–Rosenborg | 1-3 | 31 | success | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105013) |
| 2026-09-16 | Cup | Brattvåg–Kristiansund | 1-2 | 5 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6144857) |
| 2026-09-13 | Eliteserien | Viking–Kristiansund | 8-1 | 4 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105008) |
| 2026-09-06 | Eliteserien | Kristiansund–Tromsø | 2-0 | 14 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5104996) |
| 2026-08-30 | Eliteserien | HamKam–Kristiansund | 2-2 | 49 | success | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5104987) |

### Vålerenga — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-03-15 | Vålerenga–Sandefjord | acasă | 5 | success | 7 | success |
| 2026-03-22 | Rosenborg–Vålerenga | afară | 59 | success | 11 | fail |
| 2026-04-06 | Vålerenga–Viking | acasă | 45 | success | 7 | success |
| 2026-04-12 | Fredrikstad–Vålerenga | afară | 25 | success | 2 | success |
| 2026-04-19 | Vålerenga–Lillestrøm | acasă | 86 | fail | 7 | success |
| 2026-04-26 | Molde–Vålerenga | afară | 21 | success | 6 | success |
| 2026-05-03 | Vålerenga–KFUM | acasă | 2 | success | 6 | success |
| 2026-05-08 | Hamarkameratene–Vålerenga | afară | 7 | success | 3 | success |
| 2026-05-16 | Vålerenga–Sarpsborg 08 | acasă | 6 | success | 1 | success |
| 2026-05-25 | Start–Vålerenga | afară | 5 | success | 2 | success |
| 2026-05-29 | Vålerenga–Kristiansund | acasă | 1 | success | 3 | success |
| 2026-07-11 | Tromsø–Vålerenga | afară | 12 | success | 5 | success |
| 2026-07-16 | Vålerenga–Aalesund | acasă | 39 | success | 6 | success |
| 2026-07-26 | Brann–Vålerenga | afară | 24 | success | 7 | success |
| 2026-07-31 | Vålerenga–Hamarkameratene | acasă | 45+3 | success | 3 | success |
| 2026-08-08 | Vålerenga–Bodø/Glimt | acasă | 19 | success | 8 | fail |
| 2026-08-16 | Aalesund–Vålerenga | afară | 3 | success | 7 | success |
| 2026-08-30 | Vålerenga–Molde | acasă | 11 | success | 3 | success |
| 2026-09-06 | Sarpsborg 08–Vålerenga | afară | 28 | success | 6 | success |
| 2026-09-12 | Lillestrøm–Vålerenga | afară | 25 | success | 4 | success |
| 2026-09-20 | Vålerenga–Fredrikstad | acasă | 52 | success | 2 | success |
| 2026-10-12 | KFUM–Vålerenga | afară | 26 | success | 7 | success |
| 2026-10-18 | Vålerenga–Brann | acasă | 36 | success | 6 | success |
| 2026-10-24 | Bodø/Glimt–Vålerenga | afară | 65 | success | 5 | success |
| 2026-11-01 | Vålerenga–Tromsø | acasă | 9 | success | 2 | success |

### Vålerenga — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-01 | Eliteserien | Vålerenga–Tromsø | 4-0 | 9 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105038) |
| 2026-10-28 | Cup | Sarpsborg 08–Vålerenga | 3-2 | 12 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6284181) |
| 2026-10-24 | Eliteserien | Bodø/Glimt–Vålerenga | 0-1 | 65 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5104932) |
| 2026-10-18 | Eliteserien | Vålerenga–Brann | 3-1 | 36 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105027) |
| 2026-10-12 | Eliteserien | KFUM–Vålerenga | 2-1 | 26 | success | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105024) |
| 2026-09-20 | Eliteserien | Vålerenga–Fredrikstad | 1-1 | 52 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105010) |
| 2026-09-17 | Cup | Skeid–Vålerenga | 1-4 | 13 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6144868) |
| 2026-09-12 | Eliteserien | Lillestrøm–Vålerenga | 2-0 | 25 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105003) |
| 2026-09-06 | Eliteserien | Sarpsborg 08–Vålerenga | 2-2 | 28 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5104998) |
| 2026-08-30 | Eliteserien | Vålerenga–Molde | 3-4 | 11 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5104992) |

## Sabadell–Real Sociedad B — numărători și sensibilitate

| Eșantion | n / lipsă | GF / GA | reușite | tardiv /0–0 | medie /varianță | exact7 /≥8 |
| --- | --- | --- | --- | --- | --- | --- |
| home | 12 /0 | 8 /10 | 9 | 0 /3 | 1.5 /1.1818181818181819 | — |
| away | 12 /0 | 11 /16 | 10 | 2 /0 | 2.25 /2.75 | — |
| home_venue | 5 /0 | 6 /3 | 4 | 0 /1 | 1.8 /1.7 | — |
| away_venue | 6 /0 | 8 /9 | 6 | 0 /0 | 2.8333333333333335 /1.7666666666666664 | — |

λsezon=1.875,λteren=2.3166666666666664,λ50/50=2.095833333333333. Numai-sezon84.66%,numai-teren90.14%,brut87.70%,stres81.30%. Fără recalibrare contextuală.

OpenMeteo la2026-11-09T21:00RO:15.4°C, ploaie0%,vânt3.3km/h. Prognoză, fără inspecție teren. APIpoate diferi; unitatea vântuluiAPI nu este explicită.

### Sabadell — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-17 | Sporting Gijón–Sabadell | afară | 0–0 | fail | 4 | success |
| 2026-08-23 | Castellón–Sabadell | afară | 0–0 | fail | 5 | success |
| 2026-08-29 | Sabadell–Almería | acasă | 53 | success | 3 | success |
| 2026-09-07 | Sabadell–Córdoba | acasă | 7 | success | 2 | success |
| 2026-09-13 | Mallorca–Sabadell | afară | 10 | success | 2 | success |
| 2026-09-20 | Sabadell–Real Oviedo | acasă | 23 | success | 2 | success |
| 2026-09-26 | Celta Fortuna–Sabadell | afară | 6 | success | 6 | success |
| 2026-10-04 | Sabadell–FC Andorra | acasă | 0–0 | fail | 5 | success |
| 2026-10-10 | AD Ceuta FC–Sabadell | afară | 26 | success | 2 | success |
| 2026-10-18 | Burgos CF–Sabadell | afară | 63 | success | 6 | success |
| 2026-10-25 | Sabadell–Tenerife | acasă | 47 | success | 4 | success |
| 2026-10-31 | Albacete–Sabadell | afară | 10 | success | 7 | success |

### Sabadell — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-31 | LaLiga2 | Albacete–Sabadell | 2-1 | 10 | success | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868514) |
| 2026-10-25 | LaLiga2 | Sabadell–Tenerife | 1-1 | 47 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868513) |
| 2026-10-18 | LaLiga2 | Burgos CF–Sabadell | 1-0 | 63 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868492) |
| 2026-10-10 | LaLiga2 | AD Ceuta FC–Sabadell | 1-2 | 26 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868486) |
| 2026-10-04 | LaLiga2 | Sabadell–FC Andorra | 0-0 | 0–0 | fail | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868479) |
| 2026-09-26 | LaLiga2 | Celta Fortuna–Sabadell | 1-1 | 6 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868460) |
| 2026-09-20 | LaLiga2 | Sabadell–Real Oviedo | 3-1 | 23 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868458) |
| 2026-09-13 | LaLiga2 | Mallorca–Sabadell | 2-0 | 10 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868444) |
| 2026-09-07 | LaLiga2 | Sabadell–Córdoba | 3-2 | 7 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868434) |
| 2026-08-29 | LaLiga2 | Sabadell–Almería | 1-0 | 53 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868424) |

### Real Sociedad B — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-14 | Real Sociedad B–Castellón | acasă | 7 | success | 7 | success |
| 2026-08-22 | Albacete–Real Sociedad B | afară | 6 | success | 5 | success |
| 2026-08-31 | Burgos CF–Real Sociedad B | afară | 13 | success | 5 | success |
| 2026-09-05 | Real Sociedad B–Tenerife | acasă | 60 | success | 3 | success |
| 2026-09-12 | FC Andorra–Real Sociedad B | afară | 11 | success | 5 | success |
| 2026-09-19 | Real Sociedad B–Mallorca | acasă | 81 | fail | 6 | success |
| 2026-09-26 | AD Ceuta FC–Real Sociedad B | afară | 48 | success | 5 | success |
| 2026-10-04 | Real Sociedad B–Granada | acasă | 28 | success | 2 | success |
| 2026-10-10 | Celta Fortuna–Real Sociedad B | afară | 12 | success | 2 | success |
| 2026-10-17 | Real Sociedad B–Almería | acasă | 23 | success | 3 | success |
| 2026-10-24 | Las Palmas–Real Sociedad B | afară | 8 | success | 6 | success |
| 2026-10-31 | Real Sociedad B–Girona | acasă | 86 | fail | 1 | success |

### Real Sociedad B — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-31 | LaLiga2 | Real Sociedad B–Girona | 1-0 | 86 | fail | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868522) |
| 2026-10-24 | LaLiga2 | Las Palmas–Real Sociedad B | 1-0 | 8 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868512) |
| 2026-10-17 | LaLiga2 | Real Sociedad B–Almería | 2-4 | 23 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868500) |
| 2026-10-10 | LaLiga2 | Celta Fortuna–Real Sociedad B | 4-1 | 12 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868485) |
| 2026-10-04 | LaLiga2 | Real Sociedad B–Granada | 2-3 | 28 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868478) |
| 2026-09-26 | LaLiga2 | AD Ceuta FC–Real Sociedad B | 3-1 | 48 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868461) |
| 2026-09-19 | LaLiga2 | Real Sociedad B–Mallorca | 0-1 | 81 | fail | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868457) |
| 2026-09-12 | LaLiga2 | FC Andorra–Real Sociedad B | 1-3 | 11 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868437) |
| 2026-09-05 | LaLiga2 | Real Sociedad B–Tenerife | 1-1 | 60 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868433) |
| 2026-08-31 | LaLiga2 | Burgos CF–Real Sociedad B | 2-2 | 13 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868417) |

## Botoșani–Oțelul Galați — numărători și sensibilitate

| Eșantion | n / lipsă | GF / GA | reușite | tardiv /0–0 | medie /varianță | exact7 /≥8 |
| --- | --- | --- | --- | --- | --- | --- |
| home | 14 /0 | 15 /13 | 11 | 1 /2 | 2 /2.3076923076923075 | — |
| away | 14 /0 | 10 /15 | 13 | 1 /0 | 1.7857142857142858 /1.4120879120879124 | — |
| home_venue | 6 /0 | 8 /4 | 5 | 0 /1 | 2 /2.8 | — |
| away_venue | 6 /0 | 4 /8 | 6 | 0 /0 | 2 /1.6 | — |

λsezon=1.8928571428571428,λteren=1.9999999999999998,λ50/50=1.9464285714285712. Numai-sezon84.94%,numai-teren86.47%,brut85.72%,stres78.93%. Fără recalibrare contextuală.

OpenMeteo la2026-11-09T20:00RO:-0.4°C, ploaie0%,vânt4.1km/h. Prognoză, fără inspecție teren. APIpoate diferi; unitatea vântuluiAPI nu este explicită.

### Botoșani — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-17 | FC Voluntari–Botoșani | afară | 15 | success | 2 | success |
| 2026-07-27 | Botoșani–Rapid București | acasă | 45 | success | 3 | success |
| 2026-08-03 | Universitatea Cluj–Botoșani | afară | 38 | success | 5 | success |
| 2026-08-10 | Botoșani–Corvinul Hunedoara | acasă | 0–0 | fail | 6 | success |
| 2026-08-17 | FCSB–Botoșani | afară | 10 | success | 10 | fail |
| 2026-08-24 | Botoșani–Csikszereda Miercurea Ciuc | acasă | 69 | success | 6 | success |
| 2026-08-30 | FCV Farul Constanța–Botoșani | afară | 0–0 | fail | 5 | success |
| 2026-09-05 | Botoșani–Sepsi OSK | acasă | 8 | success | 4 | success |
| 2026-09-13 | FC Argeș Pitesti–Botoșani | afară | 60 | success | 2 | success |
| 2026-09-19 | CFR Cluj–Botoșani | afară | 24 | success | 4 | success |
| 2026-10-10 | Botoșani–UTA Arad | acasă | 11 | success | 7 | success |
| 2026-10-16 | Dinamo București–Botoșani | afară | 81 | fail | 7 | success |
| 2026-10-23 | Botoșani–Petrolul Ploiești | acasă | 27 | success | 1 | success |
| 2026-11-01 | Universitatea Craiova–Botoșani | afară | 21 | success | 3 | success |

### Botoșani — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-01 | Superliga | Universitatea Craiova–Botoșani | 2-1 | 21 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853543) |
| 2026-10-27 | Cup Grp. A | UTA Arad–Botoșani | 2-2 | 16 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6093727) |
| 2026-10-23 | Superliga | Botoșani–Petrolul Ploiești | 1-3 | 27 | success | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853533) |
| 2026-10-16 | Superliga | Dinamo București–Botoșani | 2-0 | 81 | fail | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853520) |
| 2026-10-10 | Superliga | Botoșani–UTA Arad | 2-0 | 11 | success | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853516) |
| 2026-09-19 | Superliga | CFR Cluj–Botoșani | 3-1 | 24 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853504) |
| 2026-09-13 | Superliga | FC Argeș Pitesti–Botoșani | 1-1 | 60 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853499) |
| 2026-09-05 | Superliga | Botoșani–Sepsi OSK | 5-0 | 8 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853490) |
| 2026-09-02 | Cup Grp. A | Botoșani–Corvinul Hunedoara | 0-1 | 17 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6093725) |
| 2026-08-30 | Superliga | FCV Farul Constanța–Botoșani | 0-0 | 0–0 | fail | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853482) |

### Oțelul Galați — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-18 | Oțelul Galați–CFR Cluj | acasă | 4 | success | 7 | success |
| 2026-07-24 | UTA Arad–Oțelul Galați | afară | 21 | success | 6 | success |
| 2026-07-31 | Oțelul Galați–Dinamo București | acasă | 42 | success | 3 | success |
| 2026-08-09 | Petrolul Ploiești–Oțelul Galați | afară | 34 | success | 5 | success |
| 2026-08-16 | Oțelul Galați–Universitatea Craiova | acasă | 56 | success | 2 | success |
| 2026-08-24 | Oțelul Galați–FC Argeș Pitesti | acasă | 51 | success | 5 | success |
| 2026-08-28 | FC Voluntari–Oțelul Galați | afară | 3 | success | 4 | success |
| 2026-09-06 | Oțelul Galați–Rapid București | acasă | 90+4 | fail | 5 | success |
| 2026-09-14 | Universitatea Cluj–Oțelul Galați | afară | 31 | success | 4 | success |
| 2026-09-21 | Oțelul Galați–Corvinul Hunedoara | acasă | 20 | success | 13 | fail |
| 2026-10-10 | FCSB–Oțelul Galați | afară | 65 | success | 2 | success |
| 2026-10-16 | Oțelul Galați–Csikszereda Miercurea Ciuc | acasă | 23 | success | 3 | success |
| 2026-10-25 | FCV Farul Constanța–Oțelul Galați | afară | 8 | success | 3 | success |
| 2026-11-01 | Oțelul Galați–Sepsi OSK | acasă | 68 | success | 6 | success |

### Oțelul Galați — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-01 | Superliga | Oțelul Galați–Sepsi OSK | 1-1 | 68 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853540) |
| 2026-10-25 | Superliga | FCV Farul Constanța–Oțelul Galați | 4-3 | 8 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853531) |
| 2026-10-16 | Superliga | Oțelul Galați–Csikszereda Miercurea Ciuc | 1-1 | 23 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853523) |
| 2026-10-10 | Superliga | FCSB–Oțelul Galați | 1-0 | 65 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853517) |
| 2026-09-21 | Superliga | Oțelul Galați–Corvinul Hunedoara | 2-3 | 20 | success | 13 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853508) |
| 2026-09-14 | Superliga | Universitatea Cluj–Oțelul Galați | 3-0 | 31 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853503) |
| 2026-09-06 | Superliga | Oțelul Galați–Rapid București | 0-1 | 90+4 | fail | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853492) |
| 2026-08-28 | Superliga | FC Voluntari–Oțelul Galați | 2-2 | 3 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853486) |
| 2026-08-24 | Superliga | Oțelul Galați–FC Argeș Pitesti | 1-0 | 51 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853475) |
| 2026-08-16 | Superliga | Oțelul Galați–Universitatea Craiova | 1-1 | 56 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853470) |

## Las Palmas–Eldense — numărători și sensibilitate

| Eșantion | n / lipsă | GF / GA | reușite | tardiv /0–0 | medie /varianță | exact7 /≥8 |
| --- | --- | --- | --- | --- | --- | --- |
| home | 12 /0 | 13 /15 | 11 | 0 /1 | 2.3333333333333335 /2.0606060606060606 | — |
| away | 12 /0 | 7 /13 | 9 | 2 /1 | 1.6666666666666667 /1.5151515151515154 | — |
| home_venue | 6 /0 | 5 /6 | 5 | 0 /1 | 1.8333333333333333 /2.1666666666666665 | — |
| away_venue | 6 /0 | 2 /7 | 4 | 2 /0 | 1.5 /1.9 | — |

λsezon=2,λteren=1.6666666666666667,λ50/50=1.8333333333333335. Numai-sezon86.47%,numai-teren81.11%,brut84.01%,stres76.93%. Fără recalibrare contextuală.

Prognoză independentă lipsă.

### Las Palmas — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | Las Palmas–Albacete | acasă | 21 | success | 3 | success |
| 2026-08-22 | AD Ceuta FC–Las Palmas | afară | 29 | success | 4 | success |
| 2026-08-29 | Girona–Las Palmas | afară | 22 | success | 4 | success |
| 2026-09-04 | Las Palmas–Leganés | acasă | 0–0 | fail | 4 | success |
| 2026-09-12 | Cádiz–Las Palmas | afară | 60 | success | 6 | success |
| 2026-09-20 | Las Palmas–Burgos CF | acasă | 37 | success | 3 | success |
| 2026-09-27 | Eibar–Las Palmas | afară | 6 | success | 4 | success |
| 2026-10-04 | Las Palmas–Real Valladolid | acasă | 5 | success | 5 | success |
| 2026-10-11 | Mallorca–Las Palmas | afară | 27 | success | 2 | success |
| 2026-10-17 | Las Palmas–Castellón | acasă | 18 | success | 3 | success |
| 2026-10-24 | Las Palmas–Real Sociedad B | acasă | 8 | success | 6 | success |
| 2026-11-02 | Real Oviedo–Las Palmas | afară | 0–0 | success | lipsă | pending |

### Las Palmas — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-02 | LaLiga2 | Real Oviedo–Las Palmas | 2-3 | 0–0 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868521) |
| 2026-10-27 | Copa del Rey | RSD Alcalá–Las Palmas | 1-3 | lipsă | pending | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6386079) |
| 2026-10-24 | LaLiga2 | Las Palmas–Real Sociedad B | 1-0 | 8 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868512) |
| 2026-10-17 | LaLiga2 | Las Palmas–Castellón | 0-3 | 18 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868499) |
| 2026-10-11 | LaLiga2 | Mallorca–Las Palmas | 3-1 | 27 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868488) |
| 2026-10-04 | LaLiga2 | Las Palmas–Real Valladolid | 2-2 | 5 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868477) |
| 2026-09-27 | LaLiga2 | Eibar–Las Palmas | 3-2 | 6 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868462) |
| 2026-09-20 | LaLiga2 | Las Palmas–Burgos CF | 1-2 | 37 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868456) |
| 2026-09-12 | LaLiga2 | Cádiz–Las Palmas | 0-1 | 60 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868439) |
| 2026-09-04 | LaLiga2 | Las Palmas–Leganés | 0-0 | 0–0 | fail | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868431) |

### Eldense — sezonul curent

| Dată | Partidă | teren | primul gol | Gol70 | FirstHalf | cornere |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-17 | Almería–Eldense | afară | 15 | success | 7 | success |
| 2026-08-22 | Eldense–Cádiz | acasă | 49 | success | 7 | success |
| 2026-08-29 | Leganés–Eldense | afară | 90 | fail | 4 | success |
| 2026-09-05 | Eldense–Mallorca | acasă | 0–0 | fail | 9 | fail |
| 2026-09-13 | Sporting Gijón–Eldense | afară | 76 | fail | 4 | success |
| 2026-09-19 | Eldense–Eibar | acasă | 29 | success | 7 | success |
| 2026-09-27 | Burgos CF–Eldense | afară | 45 | success | 4 | success |
| 2026-10-02 | Eldense–Real Oviedo | acasă | 1 | success | 6 | success |
| 2026-10-10 | Eldense–Córdoba | acasă | 11 | success | 5 | success |
| 2026-10-18 | Real Valladolid–Eldense | afară | 23 | success | 3 | success |
| 2026-10-25 | Albacete–Eldense | afară | 20 | success | 7 | success |
| 2026-11-01 | Eldense–Granada | acasă | 60 | success | 1 | success |

### Eldense — ultimele10oficiale (separat de formula campionatului)

| Dată | Competiție | Partidă | FT90 | primul gol | Gol70 | FirstHalf | cornere | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-01 | LaLiga2 | Eldense–Granada | 0-2 | 60 | success | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868519) |
| 2026-10-29 | Copa del Rey | Atlético Baleares–Eldense | 0-1 | 58 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6386047) |
| 2026-10-25 | LaLiga2 | Albacete–Eldense | 2-0 | 20 | success | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868503) |
| 2026-10-18 | LaLiga2 | Real Valladolid–Eldense | 2-3 | 23 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868502) |
| 2026-10-10 | LaLiga2 | Eldense–Córdoba | 2-1 | 11 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868487) |
| 2026-10-02 | LaLiga2 | Eldense–Real Oviedo | 4-1 | 1 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868475) |
| 2026-09-27 | LaLiga2 | Burgos CF–Eldense | 1-0 | 45 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868459) |
| 2026-09-19 | LaLiga2 | Eldense–Eibar | 1-2 | 29 | success | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868454) |
| 2026-09-13 | LaLiga2 | Sporting Gijón–Eldense | 0-1 | 76 | fail | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868445) |
| 2026-09-05 | LaLiga2 | Eldense–Mallorca | 0-0 | 0–0 | fail | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868430) |

## Surse și limite

JSON/API cu timestamp înMatchData,OfficialRecent,Context șiStatusAtSelection; HTMLpublic înSources/, index[sources.json](../Goal70_2026-11-09/sources.json). Istoric complet[historical_observations.csv](../Goal70_2026-11-09/historical_observations.csv); probabilitățile originale și reviziile inventarului sunt păstrate. FotMob este sursa principală a observațiilorFirstHalf; reconciliereaFirstHalf+SecondHalf=All nu echivalează cu verificare independentă. Nu există preț disponibil/profit demonstrat, pariu plasat sau nouă alegere în numele utilizatorului.

## Cupe și verificarea programului

În programul global sunt inventariate FA Cup: PlymouthParkway–Berkhamsted; celelalte competiții numiteCup sunt listate cu identități și status încups_coverage.json și excluse explicit din universul campionatelor. Nu există model intern comparabil colectat pentru aceste echipe inferioare. EvenimentulUSLLeagueOne apare prin filtrul textualChampionship, dar nu este cupă europeană și rămâne exclus. Nicio proiecție peste competiții pentru cei4candidați de campionat. LasPalmas are acum status individualFotMob reconciliat cuESPNși clubul; doar lista globală a omis partida.

## Verificări finale și limite

Ultimele10oficiale: Kristiansund10/10Gol70 șiVålerenga10/10, dar acest istoric anterior focarului medical nu confirmă recuperarea actuală. Sabadell9/10Gol70, cu0–0Andorra; Sanse8/10, cuGirona86 șiMallorca81. Pe cornereSabadell/Sanse au10/10observațiiFirstHalffavorabile fiecare, față de câte2FirstHalflipsă în ultimele10ale celor două echipe norvegiene. Aceste serii sunt diagnostice, nu procente finale pentru azi.

Toate cele7modele au fost recalculate independent din numărători. Snapshoturile inițiale, rangurile și timestampurile sunt identice; alegerile utilizatorului păstrate. Patru teste de arhivă și11verificări înbrowser au trecut, zeroexcepții și fărăoverflowmobil. Browserul în sandbox s-a închis cu eroareGPU; aceeași verificare cu profil izolat a trecut în execuția locală aprobată. Nu a fost schimbat browserul implicit sau serviciul utilizatorului.

Dashboard8zile/688rapoarte. Azi toate rezultatelepending, fără rată preluată deieri; probabilitateaFirstHalflipsă rămâneNULL. Researchpartial, dashboardcomplete; a încheia această predare nu înseamnă că toate verificările independente sunt complete.
