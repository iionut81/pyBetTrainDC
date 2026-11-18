# Cornere 1H sub 7,5 — 14 noiembrie 2026

**Candidat și preferința individuală principală: Real Valladolid–Castellón, 15:00 RO — 95,72% brut / 90,29% stres; fair matematic 1,0448.** Istoric 24/25 general și 13/13 pe teren. Riscuri: totalurile 7/9 din istoric, dispersia Castellónului în deplasare, absențele și rafalele prognozate. Este o estimare Poisson necalibrată, fără confirmarea vreunui preț disponibil.

| Rang | Meci | Ora RO | Brut | Stres | Fair matematic | Istoric / teren |
|---|---|---|---|---|---|---|
| 1 | Real Valladolid–Castellón | 15:00 | 95.72% | 90.29% | 1.0448 | 24/25; 13/13 |
| 2 | Almería–FC Andorra | 19:30 | 95.24% | 89.36% | 1.0500 | 25/25; 12/12 |
| 3 | Real Oviedo–Girona | 22:00 | 95.10% | 89.08% | 1.0516 | 24/26; 9/11 |
| 4 | Mallorca–Córdoba | 19:30 | 94.49% | 87.91% | 1.0583 | 21/25; 13/13 |
| 5 | Real Sociedad B–Sporting Gijón | 17:15 | 93.23% | 85.56% | 1.0726 | 25/26; 12/12 |

[Context și alternative](../Goal70_2026-11-14/CONTEXT.md) · [Calcule complete](../Goal70_2026-11-14/CALCULATIONS.md) · [Inventar](../Goal70_2026-11-14/INVENTORY.md). Datele folosite au FirstHalf explicit și reconciliere FirstHalf+SecondHalf=All. FT nu substituie prima repriză.

## Acoperire și limite

Inventarul are **164 de înregistrări**, dintre care 38 cu identitate FotMob și 126 de suplimente: 51 de partide din Coupe de France, 59 de calificări U17/U19 și 16 alte semnalări. Acesta nu este un total de 164 de partide confirmate independent: există un conflict England U18–Norway U19 versus England U18–Sweden U18. Istoricul exact și minimele 5/3 nu au fost validate pentru suplimente. Nu substituim alte generații, cluburi sau ediții vechi.

Sunt verificate 29 de ligi: primele 20 de asociații din [clasamentul final UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), calculat pe 2021/22–2025/26, plus România, Serbia și cele șapte secunde urmărite. În 28 de ligi nu au fost identificate partide; în Segunda sunt cinci. Am confruntat programul global FotMob, calendarele edițiilor actuale și ESPN. Patru ligi nu au confirmare ESPN: Polonia, Croația, Serbia și secunda belgiană. Un răspuns nesuportat nu dovedește absența meciurilor.

Separat sunt trei partide de cupă din Slovacia/Croația și cele 51 din Coupe de France. Nu au istoric bilateral validat pentru proiecția de campionat. Calendarele UCL/UEL/UECL și Copa del Rey consultate nu indică partide azi. Lista completă, sursele și excluderile sunt în [INVENTORY.md](../Goal70_2026-11-14/INVENTORY.md), universe_coverage.csv și candidate_fixtures.json.

Am colectat **304 istorice unice** în edițiile actuale, cu zece liste de goluri și 36 de observații FirstHalf lipsă sau nereconciliate. Suplimentele fără istoric nu intră în acest numitor. Sunt calculate și comparate toate cele **șase modele Gol70 și cinci modele de cornere eligibile în datele validate**. Modelarea întregului inventar și confirmarea independentă mondială rămân parțiale; nu prezentăm un top global complet.

Cele opt partide UEFA Nations League au numai 4 generale/2 pe teren pentru fiecare echipă, sub 5/3. Edițiile AFCON/CONCACAF sunt și ele limitate. Amicalele folosesc numai anul 2026, separat de competițiile oficiale. Identitățile seniori/juniori/masculin/feminin sunt distincte. Meciurile începute sau terminate în dimineața RO sunt inventariate fără predicții retrospective și fără rezultate fabricate azi.

Două listări duble Friendly/King’s Cup, Oman–Tajikistan și Thailand–Brunei, sunt deduplicate și excluse până la clarificarea competiției. Basque Country–Venezuela este o selecție regională/non-FIFA, distinctă și fără model. Haiti–USA necesită confirmarea gazdei desemnate și a terenului; nu este finalist.

Japan–Brazil este **neutru, la Singapore**, cu Japonia gazdă desemnată. Baseline-ul 87,13/80,60% folosește părțile desemnate ale amicalelor 2026, fără a pretinde avantaj propriu real sau calibrare pe neutru. Frecvențele sunt 11/14 general și 6/8 pe părțile desemnate. Nu este candidat preferat.

[AUF confirmă Uruguay U17 feminin–Peru U17 feminin](https://www.auf.org.uy/la-sub-17-femenina-disputara-dos-amistosos-ante-peru/) la 10:00 local = 15:00 RO, pe teren propriu în Uruguay; istoricul exact al amicalelor 2026 nu este validat. Nu a fost identificată o partidă de naționale feminine de seniori, fără a demonstra absența mondială. Panama–El Salvador și Costa Rica–Guatemala U19 feminin, afișate 14 nov 04:00 în UTC+8, aparțin zilei 13 nov, 22:00 RO, conform surselor secundare UTC.

Snapshotul numeric inițial și reviziile sunt arhivate în Revisions/. Procentele, rangurile originale și momentele inițiale ale predicțiilor numerice sunt păstrate. Nicio alegere nouă a utilizatorului nu a fost creată.

## Lecții aplicate din review

[Lecțiile pentru azi](../Review_2026-11-13/LESSONS_FOR_NEXT_SCAN.md) au fost predate la **2026-11-14T08:50:20.097+02:00**, înaintea scanării începute la **2026-11-14T08:50:56.420+02:00**.

| Lecție | Verificare și dovadă nouă | Stare / efect |
|---|---|---|
| Model, frecvență și stres separate | Valladolid Gol70 91,42/85,98%, cornere 95,72/90,29%; 24/25 general și 13/13 teren | Aplicată: preferință individuală pentru cornere, fără bonus procentual |
| GF70 propriu, 0–0 și concentrarea golurilor | Valladolid 13/6, Castellón 9/7 pe teren; Castellón–Sabadell 0–0; diagnostic fără diferențe FT≥4: 89,50% | Aplicată: intensitatea nu este o rată binară validată |
| FirstHalf exact 7/≥8 și dispersie | Eibar–Valladolid 9; Valladolid–Albacete 7; Sociedad B–Castellón 7; Castellón afară varianță 5,33 versus medie 4 | Aplicată: coada rămâne în istoric și scenariul +20% |
| Campionat versus cupă/Europa | Cluburile folosesc Segunda 2026/27; ultimele zece oficiale sunt diagnostic separat; naționalele au ediții și categorii separate | Aplicată: fără pooling pentru minime |
| Lot actual versus XI vechi | Conferințele Escribá/Hernández; buletinul Rêgo din 13 nov contrazice simpla etichetă „convocat” | Aplicată: API lastStarting11 nu este XI confirmat |
| Odihnă, vreme, teren și status | Trofeul din cursul săptămânii la Valladolid; amânarea Eibar la Castellón; rafale prognozate 40,7 km/h; toate cele șase modele neîncepute la 08:58 RO | Aplicată; minutele individuale și gazonul efectiv rămân neverificate |
| Neutralitate și identitate | Japonia–Brazilia la Singapore; U17/U19/U21/U23 și feminin distincte; conflict King’s Cup | Aplicată: baseline-ul neutru nu este preferat; lipsurile rămân explicite |
| Eliminări și cauze tactice | Extragerile cardurilor nu au fost auditate exhaustiv cronologic | Neverificată parțial: nu pretindem o sensibilitate completă fără roșu sau cauze demonstrate |

Cele trei succese explicite de ieri nu autorizează modificarea formulei, ponderilor, pragurilor sau modelelor. Metoda și scenariile rămân cele salvate.
