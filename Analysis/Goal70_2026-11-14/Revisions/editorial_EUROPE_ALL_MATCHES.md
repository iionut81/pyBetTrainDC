# Gol70 — 14 noiembrie 2026

**Candidat al pieței: Real Valladolid–Castellón, 15:00 RO — 91,42% brut / 85,98% stres; fair matematic 1,0938.** Preferința individuală principală este același meci la cornere 1H. Evenimentele nu se combină. Toate cele șase modele Gol70 și cinci de cornere au fost comparate; status neînceput reconfirmat la 08:58 RO.

| Rang | Meci | Ora RO | Brut | Stres | Fair matematic | Istoric / teren |
|---|---|---|---|---|---|---|
| 1 | Real Valladolid–Castellón | 15:00 | 91.42% | 85.98% | 1.0938 | 24/25; 13/13 |
| 2 | Almería–FC Andorra | 19:30 | 89.24% | 83.19% | 1.1206 | 24/25; 11/12 |
| 3 | Japan–Brazil | 12:15 | 87.13% | 80.60% | 1.1478 | 11/14; 6/8 |
| 4 | Mallorca–Córdoba | 19:30 | 85.50% | 78.66% | 1.1696 | 21/25; 12/13 |
| 5 | Real Sociedad B–Sporting Gijón | 17:15 | 82.06% | 74.70% | 1.2187 | 20/26; 9/12 |
| 6 | Real Oviedo–Girona | 22:00 | 81.28% | 73.83% | 1.2303 | 22/26; 8/11 |

[Context și contraargumente](CONTEXT.md) · [Calcule și numărători](CALCULATIONS.md) · [Ultimele zece oficiale](LAST10_OFFICIAL.md) · [Inventar și excluderi](INVENTORY.md). Cercetarea este **parțială**.

## Acoperire și limite

Inventarul are **164 de înregistrări**, dintre care 38 cu identitate FotMob și 126 de suplimente: 51 de partide din Coupe de France, 59 de calificări U17/U19 și 16 alte semnalări. Acesta nu este un total de 164 de partide confirmate independent: există un conflict England U18–Norway U19 versus England U18–Sweden U18. Istoricul exact și minimele 5/3 nu au fost validate pentru suplimente. Nu substituim alte generații, cluburi sau ediții vechi.

Sunt verificate 29 de ligi: primele 20 de asociații din [clasamentul final UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), calculat pe 2021/22–2025/26, plus România, Serbia și cele șapte secunde urmărite. În 28 de ligi nu au fost identificate partide; în Segunda sunt cinci. Am confruntat programul global FotMob, calendarele edițiilor actuale și ESPN. Patru ligi nu au confirmare ESPN: Polonia, Croația, Serbia și secunda belgiană. Un răspuns nesuportat nu dovedește absența meciurilor.

Separat sunt trei partide de cupă din Slovacia/Croația și cele 51 din Coupe de France. Nu au istoric bilateral validat pentru proiecția de campionat. Calendarele UCL/UEL/UECL și Copa del Rey consultate nu indică partide azi. Lista completă, sursele și excluderile sunt în [INVENTORY.md](../Goal70_2026-11-14/INVENTORY.md), universe_coverage.csv și candidate_fixtures.json.

Am colectat **306 istorice unice** în edițiile actuale, cu zece liste de goluri și 36 de observații FirstHalf lipsă sau nereconciliate. Suplimentele fără istoric nu intră în acest numitor. Sunt calculate și comparate toate cele **șase modele Gol70 și cinci modele de cornere eligibile în datele validate**. Modelarea întregului inventar și confirmarea independentă mondială rămân parțiale; nu prezentăm un top global complet.

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

## Ligile și programele verificate

| Liga | Rang UEFA 2026 / supliment | Global | Ediție | ESPN | Acoperire |
|---|---|---|---|---|---|
| England / eng.1 | 1 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Italy / ita.1 | 2 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Spain / esp.1 | 3 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Germany / ger.1 | 4 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| France / fra.1 | 5 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Portugal / por.1 | 6 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Netherlands / ned.1 | 7 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Belgium / bel.1 | 8 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Turkey / tur.1 | 9 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Czechia / cze.1 | 10 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Greece / gre.1 | 11 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Poland / pol.1 | 12 | 0 | 0 | indisponibil | FotMob_global+season;independent_missing |
| Denmark / den.1 | 13 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Norway / nor.1 | 14 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Cyprus / cyp.1 | 15 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Switzerland / sui.1 | 16 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Austria / aut.1 | 17 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Scotland / sco.1 | 18 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Sweden / swe.1 | 19 | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Croatia / cro.1 | 20 | 0 | 0 | indisponibil | FotMob_global+season;independent_missing |
| Romania / rou.1 | supplement | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Serbia / srb.1 | supplement | 0 | 0 | indisponibil | FotMob_global+season;independent_missing |
| England 2 / eng.2 | supplement | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Germany 2 / ger.2 | supplement | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Spain 2 / esp.2 | supplement | 5 | 5 | 5 | FotMob_global+season+ESPN |
| Italy 2 / ita.2 | supplement | 0 | 0 | 0 | FotMob_global+season+ESPN |
| France 2 / fra.2 | supplement | 0 | 0 | 0 | FotMob_global+season+ESPN |
| Belgium 2 / bel.2 | supplement | 0 | 0 | indisponibil | FotMob_global+season;independent_missing |
| Turkey 2 / tur.2 | supplement | 0 | 0 | 0 | FotMob_global+season+ESPN |
