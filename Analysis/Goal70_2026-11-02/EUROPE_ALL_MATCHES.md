# Gol pana la 70: comparatia completa Europa, 2 noiembrie 2026

Calcul si verificari salvate la **2026-11-02T11:36:58+02:00**, Europe/Bucharest. Extragerile de program sunt anterioare acestei ore; toate cele 27 de ore de start sunt in viitor si statusurile salvate sunt SCHEDULED/pre. Acesta este un instantaneu, nu monitorizare live.
Statusul celor doua finaliste a fost reverificat separat la 11:36:57-58 RO: ambele pre-meci, cu orele 19:30 si 20:00 RO. Dovada: Europe_Combined/finalists_live_status.json.

**Rezultat:** topul strict al formulei este Sassuolo-Fiorentina, Sonderjyske-Copenhaga, Benfica-Alverca. Primele doua raman cele doua analize individuale cerute. Nu sunt un bilet combinat. Lorient-Brest coboara pe locul 8; alegerea initiala din doua ligi era incompleta.

Procentele sunt rezultate Poisson brute, necalibrate; nu sunt rate de reusita demonstrate. Nicio cota de bookmaker sau predictie de tipster nu a intrat in selectie sau calcule. Cota echitabila nu confirma o oferta disponibila de minimum 1,10. Nu a fost plasat niciun pariu.

## Universul verificat

Primele 20 de asociatii din [clasamentul UEFA 2026, tabelul Kassies](https://kassiesa.net/uefa/data/method5/crank2026.html), numai prima liga masculina. Separat: Romania, Serbia si diviziile secunde urmarite deja in proiect, plus Ligue 2. Clasamentul fix 2026 nu este clasamentul provizoriu 2027. Arabia Saudita din vechea lista a proiectului nu este Europa. Nu pretindem ca am cercetat toate diviziile inferioare ale acestor tari.

| Loc UEFA 2026 | Tara | Meciuri azi, prima liga | Verificarea programului |
|---:|---|---:|---|
| 1 | England | 1 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 2 | Italy | 2 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 3 | Spain | 1 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 4 | Germany | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 5 | France | 1 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 6 | Portugal | 2 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 7 | Netherlands | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 8 | Belgium | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 9 | Turkey | 1 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 10 | Czechia | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 11 | Greece | 2 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 12 | Poland | 3 | FotMob + program oficial Ekstraklasa |
| 13 | Denmark | 1 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 14 | Norway | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 15 | Cyprus | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 16 | Switzerland | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 17 | Austria | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 18 | Scotland | 0 | Niciun meci gasit in programele ESPN/FotMob |
| 19 | Sweden | 2 | Program ESPN/FotMob; calcul pe minutele golurilor |
| 20 | Croatia | 0 | FotMob: niciun meci gasit; endpoint ESPN nesuportat |

Total: **16 meciuri din primele ligi ale topului 20 + 11 suplimentare = 27 evaluate**. In 10 dintre primele ligi nu am gasit meciuri azi. Absenta dintr-un agregator nu este o garantie oficiala impotriva unei modificari ulterioare de program.

[Program global FotMob, 2 noiembrie](https://www.fotmob.com/api/data/matches?date=20261102). [Program ESPN, exemplu Serie A](https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates=20261102&limit=1000). [Program oficial polonez, etapa 13](https://ekstraklasa.org/en/news/schedule-of-the-202627-matchweek-13/). Pentru ligile fara meciuri nu a fost necesara descarcarea istoricului complet.

## Toate partidele, aceeasi formula

Frecventa este numarul partidelor din sezonul curent ale ORICAREIA dintre cele doua echipe cu primul gol <=70, fara dublarea unui meci direct. Este descriptiva si nu este o a doua probabilitate validata. N sezon = numarul meciurilor gazdei/oaspetilor; N teren = gazda acasa/oaspete in deplasare.

| Rang brut | Liga | Meci | Ora RO | P brut | Stres -20% | Echitabila | Istoric primul gol <=70 | N sezon; N teren |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | Serie A | [Sassuolo - Fiorentina](https://www.espn.com/soccer/match/_/gameId/401874963) | 19:30 | 94,40% | 90,03% | 1,0594 | 18/18 (100,00%) | 9/9; 4/4 |
| 2 | Superliga | [Sønderjyske Fodbold - F.C. København](https://www.espn.com/soccer/match/_/gameId/401874494) | 20:00 | 94,36% | 89,97% | 1,0598 | 20/23 (86,96%) | 12/12; 6/6 |
| 3 | Liga Portugal | [Benfica - Alverca](https://www.espn.com/soccer/match/_/gameId/401885403) | 22:15 | 92,45% | 87,34% | 1,0817 | 18/18 (100,00%) | 9/9; 4/4 |
| 4 | LaLiga | [Deportivo - Osasuna](https://www.espn.com/soccer/match/_/gameId/401882824) | 22:00 | 91,37% | 85,91% | 1,0944 | 19/20 (95,00%) | 10/10; 5/5 |
| 5 | Serie A | [Atalanta - Parma](https://www.espn.com/soccer/match/_/gameId/401874971) | 21:45 | 90,92% | 85,32% | 1,0999 | 15/18 (83,33%) | 9/9; 5/4 |
| 6 | Ekstraklasa | [Radomiak Radom - Śląsk Wrocław](https://www.fotmob.com/matches/radomiak-radom-vs-slask-wrocaw/1knfv0#5804026) | 19:00 | 89,89% | 84,01% | 1,1125 | 21/24 (87,50%) | 12/12; 6/6 |
| 7 | SuperLiga Romania | [Universitatea Cluj - FCSB](https://www.fotmob.com/matches/fcsb-vs-universitatea-cluj/28mnlio#5853542) | 20:30 | 89,76% | 83,85% | 1,1141 | 23/26 (88,46%) | 13/13; 7/6 |
| 8 | Ligue 1 | [Lorient - Brest](https://www.espn.com/soccer/match/_/gameId/401876419) | 21:45 | 89,46% | 83,47% | 1,1178 | 14/16 (87,50%) | 8/8; 4/4 |
| 9 | Ligue 2 | [Stade de Reims - Sochaux](https://www.espn.com/soccer/match/_/gameId/401876702) | 21:45 | 88,35% | 82,09% | 1,1318 | 15/20 (75,00%) | 10/10; 5/5 |
| 10 | 1. Lig | [Kayserispor - Muğlaspor](https://www.fotmob.com/matches/kayserispor-vs-muglaspor/861edw8#5908687) | 19:00 | 87,91% | 81,55% | 1,1376 | 22/24 (91,67%) | 12/12; 5/6 |
| 11 | SuperLiga Romania | [FC Voluntari - FCV Farul Constanța](https://www.fotmob.com/matches/fcv-farul-constanta-vs-fc-voluntari/2f0b31c8#5853539) | 17:30 | 87,30% | 80,81% | 1,1455 | 22/26 (84,62%) | 13/13; 7/7 |
| 12 | LaLiga 2 | [Real Valladolid - Leganés](https://www.espn.com/soccer/match/_/gameId/401883123) | 19:30 | 87,26% | 80,77% | 1,1460 | 19/22 (86,36%) | 11/11; 5/6 |
| 13 | SuperLiga Serbia | [Radnički Niš - Železničar Pančevo](https://www.fotmob.com/matches/radnicki-nis-vs-zeleznicar-pancevo/2z0du52p#1000015166) | 19:00 | 87,10% | 80,57% | 1,1481 | 21/25 (84,00%) | 13/13; 6/6 |
| 14 | Ekstraklasa | [Raków Częstochowa - Pogoń Szczecin](https://www.fotmob.com/matches/rakow-czestochowa-vs-pogon-szczecin/177mfj#5804027) | 21:30 | 86,99% | 80,44% | 1,1495 | 20/23 (86,96%) | 12/11; 6/4 |
| 15 | Super League | [Kalamata - Iraklis](https://www.espn.com/soccer/match/_/gameId/401896813) | 19:30 | 86,12% | 79,40% | 1,1611 | 15/16 (93,75%) | 8/8; 4/5 |
| 16 | Premier League | [Newcastle United - Everton](https://www.espn.com/soccer/match/_/gameId/401878762) | 22:00 | 86,04% | 79,30% | 1,1623 | 15/16 (93,75%) | 8/8; 4/4 |
| 17 | LaLiga 2 | [Tenerife - FC Andorra](https://www.espn.com/soccer/match/_/gameId/401883144) | 15:00 | 85,25% | 78,37% | 1,1731 | 21/22 (95,45%) | 11/11; 5/5 |
| 18 | Super League | [Atromitos - Volos NFC](https://www.espn.com/soccer/match/_/gameId/401896739) | 18:00 | 84,66% | 77,69% | 1,1811 | 13/15 (86,67%) | 7/8; 3/4 |
| 19 | Allsvenskan | [Mjällby AIF - Kalmar FF](https://www.espn.com/soccer/match/_/gameId/401842867) | 20:00 | 84,06% | 76,99% | 1,1896 | 43/51 (84,31%) | 26/26; 13/13 |
| 20 | LaLiga 2 | [Real Oviedo - Las Palmas](https://www.espn.com/soccer/match/_/gameId/401883146) | 17:15 | 83,55% | 76,39% | 1,1970 | 19/22 (86,36%) | 11/11; 5/5 |
| 21 | LaLiga 2 | [Mallorca - Cádiz](https://www.espn.com/soccer/match/_/gameId/401883147) | 22:00 | 82,66% | 75,39% | 1,2097 | 17/22 (77,27%) | 11/11; 5/4 |
| 22 | Ligue 2 | [Red Star FC 93 - AS Nancy Lorraine](https://www.espn.com/soccer/match/_/gameId/401876703) | 21:45 | 82,18% | 74,84% | 1,2168 | 15/20 (75,00%) | 10/10; 4/5 |
| 23 | Allsvenskan | [GAIS - AIK](https://www.espn.com/soccer/match/_/gameId/401842870) | 20:00 | 81,94% | 74,57% | 1,2204 | 44/51 (86,27%) | 26/26; 13/13 |
| 24 | Super Lig | [Genclerbirligi - Alanyaspor](https://www.espn.com/soccer/match/_/gameId/401888260) | 19:00 | 81,53% | 74,10% | 1,2266 | 15/18 (83,33%) | 9/9; 5/5 |
| 25 | SuperLiga Serbia | [Mladost Lučani - FK Radnik Surdulica](https://www.fotmob.com/matches/mladost-lucani-vs-fk-radnik-surdulica/1dpj6qnx#1000015164) | 19:00 | 80,43% | 72,89% | 1,2432 | 20/25 (80,00%) | 13/13; 6/6 |
| 26 | Ekstraklasa | [Zagłębie Lubin - Korona Kielce](https://www.fotmob.com/matches/zagebie-lubin-vs-korona-kielce/26rtbg#5804030) | 19:00 | 76,74% | 68,86% | 1,3031 | 18/24 (75,00%) | 12/12; 6/6 |
| 27 | Liga Portugal | [Gil Vicente - Vitória de Guimaraes](https://www.espn.com/soccer/match/_/gameId/401885407) | 22:15 | 75,06% | 67,08% | 1,3322 | 13/18 (72,22%) | 9/9; 5/4 |

Tenerife-Andorra este inclus la 15:00 RO: partida amanata duminica din cauza ploii a fost reprogramata luni. Acest lucru corecteaza excluderea provizorie din raportul initial. [RTVC, stirea reprogramarii](https://rtvc.es/entradas/). Universitatea Cluj-FCSB: providerul si SuperLiga indica 18:30 UTC / 20:30 RO; pagina clubului arata 21:00. Marchez conflictul de ora, fara a afecta numaratorile istorice. [SuperLiga](https://www.superliga.ro/meci/superliga-26-27-regular-ucj-fcsb?sectiune=live-blog), [club](https://www.fcucluj.ro/meciuri).

## Formula si ipoteze

A = gazda, B = oaspete; GF/GA sunt numai goluri valide cu minutul de baza afisat <=70, inclusiv 45+. Nu se inmulteste din nou cu 70/90.

```text
lambda_sezon = ((GF_A/n_A + GA_B/n_B) + (GF_B/n_B + GA_A/n_A))/2
lambda_teren = aceeasi formula, numai A acasa si B in deplasare
lambda = 0,5*lambda_sezon + 0,5*lambda_teren
p = 1 - exp(-lambda); p_stres = 1 - exp(-0,8*lambda); echitabila = 1/p
```

Ponderarea 50/50 este euristica; esantioanele se suprapun. Minimul tehnic pentru calcul, 5 meciuri de sezon si 3 pe teren pentru fiecare echipa, nu este o garantie de precizie. Nu exista calibrare pe un esantion independent. Multe goluri ulterioare deschiderii scorului si meciurile cu eliminari pot creste artificial lambda pentru evenimentul PRIMULUI gol. Stresul nu este un interval de incredere. Nu aplic bonusuri sau penalizari procentuale pentru tactica.

## Inputuri si calcule pentru primele trei

### Sassuolo - Fiorentina

| Esantion | N | GF pana la 70 | GA pana la 70 | Partide cu gol <=70 |
|---|---:|---:|---:|---:|
| Gazda, sezon | 9 | 10 | 19 | 9 |
| Oaspete, sezon | 9 | 9 | 14 | 9 |
| Gazda acasa | 4 | 5 | 8 | 4 |
| Oaspete deplasare | 4 | 4 | 6 | 4 |

lambda_sezon = ((10/9 + 14/9) + (9/9 + 19/9))/2 = 2,888889.
lambda_teren = ((5/4 + 6/4) + (4/4 + 8/4))/2 = 2,875000.
lambda_combinat = 2,881944; numai sezon = 94,44%; numai teren = 94,36%; combinat = 94,40%; stres = 90,03%; echitabila = 1,0594.

### Sønderjyske Fodbold - F.C. København

| Esantion | N | GF pana la 70 | GA pana la 70 | Partide cu gol <=70 |
|---|---:|---:|---:|---:|
| Gazda, sezon | 12 | 13 | 24 | 10 |
| Oaspete, sezon | 12 | 19 | 12 | 11 |
| Gazda acasa | 6 | 9 | 10 | 5 |
| Oaspete deplasare | 6 | 9 | 7 | 5 |

lambda_sezon = ((13/12 + 12/12) + (19/12 + 24/12))/2 = 2,833333.
lambda_teren = ((9/6 + 7/6) + (9/6 + 10/6))/2 = 2,916667.
lambda_combinat = 2,875000; numai sezon = 94,12%; numai teren = 94,59%; combinat = 94,36%; stres = 89,97%; echitabila = 1,0598.

### Benfica - Alverca

| Esantion | N | GF pana la 70 | GA pana la 70 | Partide cu gol <=70 |
|---|---:|---:|---:|---:|
| Gazda, sezon | 9 | 20 | 7 | 9 |
| Oaspete, sezon | 9 | 10 | 11 | 9 |
| Gazda acasa | 4 | 6 | 3 | 4 |
| Oaspete deplasare | 4 | 5 | 6 | 4 |

lambda_sezon = ((20/9 + 11/9) + (10/9 + 7/9))/2 = 2,666667.
lambda_teren = ((6/4 + 6/4) + (5/4 + 3/4))/2 = 2,500000.
lambda_combinat = 2,583333; numai sezon = 93,05%; numai teren = 91,79%; combinat = 92,45%; stres = 87,34%; echitabila = 1,0817.

## Istoricul verificat al finalistilor

Mai jos sunt ultimele maximum 10 meciuri DE CAMPIONAT ale fiecarui finalist, din sezonul actual. Sassuolo si Fiorentina au cate 9; cele 10 oficiale incluzand cupa sunt deja documentate in raportul initial. Pentru danezi si portughezi, aceste tabele NU sunt prezentate ca ultimele 10 din toate competitiile: cupa si meciurile europene sunt excluse din modelul de campionat si nu avem aici o serie completa a minutelor lor. Limitarea este explicita.

### Sassuolo - campionat

| Data | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---:|---|
| 2026-10-27 | [Sassuolo - Lazio 2-6](https://www.espn.com/soccer/match/_/gameId/401874980) | 2' | DA |
| 2026-10-24 | [Como - Sassuolo 2-0](https://www.espn.com/soccer/match/_/gameId/401874961) | 49' | DA |
| 2026-10-16 | [Frosinone - Sassuolo 4-2](https://www.espn.com/soccer/match/_/gameId/401874959) | 9' | DA |
| 2026-10-11 | [Sassuolo - AC Milan 1-1](https://www.espn.com/soccer/match/_/gameId/401874789) | 63' | DA |
| 2026-09-18 | [Monza - Sassuolo 2-1](https://www.espn.com/soccer/match/_/gameId/401874756) | 51' | DA |
| 2026-09-13 | [Sassuolo - Juventus 3-2](https://www.espn.com/soccer/match/_/gameId/401874750) | 65' | DA |
| 2026-09-06 | [Bologna - Sassuolo 2-2](https://www.espn.com/soccer/match/_/gameId/401874751) | 19' | DA |
| 2026-08-29 | [Sassuolo - Torino 2-1](https://www.espn.com/soccer/match/_/gameId/401874743) | 32' | DA |
| 2026-08-23 | [Atalanta - Sassuolo 2-1](https://www.espn.com/soccer/match/_/gameId/401874748) | 6' | DA |

### Fiorentina - campionat

| Data | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---:|---|
| 2026-10-29 | [Fiorentina - Atalanta 2-2](https://www.espn.com/soccer/match/_/gameId/401874969) | 41' | DA |
| 2026-10-25 | [Internazionale - Fiorentina 2-1](https://www.espn.com/soccer/match/_/gameId/401875036) | 65' | DA |
| 2026-10-18 | [Fiorentina - Como 1-3](https://www.espn.com/soccer/match/_/gameId/401874774) | 32' | DA |
| 2026-10-10 | [Genoa - Fiorentina 2-1](https://www.espn.com/soccer/match/_/gameId/401874947) | 7' | DA |
| 2026-09-20 | [Fiorentina - Napoli 1-1](https://www.espn.com/soccer/match/_/gameId/401874934) | 23' | DA |
| 2026-09-11 | [Venezia - Fiorentina 2-4](https://www.espn.com/soccer/match/_/gameId/401874759) | 22' | DA |
| 2026-09-05 | [Fiorentina - Torino 1-2](https://www.espn.com/soccer/match/_/gameId/401874939) | 47' | DA |
| 2026-08-29 | [Fiorentina - Frosinone 0-3](https://www.espn.com/soccer/match/_/gameId/401874942) | 26' | DA |
| 2026-08-24 | [AS Roma - Fiorentina 4-0](https://www.espn.com/soccer/match/_/gameId/401874928) | 27' | DA |

### Sønderjyske Fodbold - campionat

| Data | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---:|---|
| 2026-10-25 | [AC Horsens - Sønderjyske Fodbold 5-1](https://www.espn.com/soccer/match/_/gameId/401874546) | 12' | DA |
| 2026-10-18 | [Sønderjyske Fodbold - Silkeborg IF 4-0](https://www.espn.com/soccer/match/_/gameId/401874565) | 1' | DA |
| 2026-10-12 | [AGF - Sønderjyske Fodbold 3-0](https://www.espn.com/soccer/match/_/gameId/401874516) | 9' | DA |
| 2026-09-20 | [Sønderjyske Fodbold - Randers FC 4-2](https://www.espn.com/soccer/match/_/gameId/401874534) | 25' | DA |
| 2026-09-13 | [Lyngby Boldklub - Sønderjyske Fodbold 1-1](https://www.espn.com/soccer/match/_/gameId/401874525) | 6' | DA |
| 2026-09-06 | [Sønderjyske Fodbold - AC Horsens 2-5](https://www.espn.com/soccer/match/_/gameId/401874485) | 12' | DA |
| 2026-08-31 | [F.C. København - Sønderjyske Fodbold 3-1](https://www.espn.com/soccer/match/_/gameId/401874513) | 7' | DA |
| 2026-08-23 | [Sønderjyske Fodbold - FC Nordsjælland 0-3](https://www.espn.com/soccer/match/_/gameId/401874515) | 44' | DA |
| 2026-08-17 | [Brøndby IF - Sønderjyske Fodbold 3-2](https://www.espn.com/soccer/match/_/gameId/401874477) | 18' | DA |
| 2026-08-07 | [Sønderjyske Fodbold - Viborg FF 0-0](https://www.espn.com/soccer/match/_/gameId/401874473) | fara gol (0-0) | NU |

### F.C. København - campionat

| Data | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---:|---|
| 2026-10-26 | [F.C. København - Odense Boldklub 1-1](https://www.espn.com/soccer/match/_/gameId/401874531) | 34' | DA |
| 2026-10-18 | [F.C. København - AGF 1-1](https://www.espn.com/soccer/match/_/gameId/401874551) | 35' | DA |
| 2026-10-11 | [FC Midtjylland - F.C. København 2-1](https://www.espn.com/soccer/match/_/gameId/401874498) | 12' | DA |
| 2026-09-20 | [Brøndby IF - F.C. København 0-1](https://www.espn.com/soccer/match/_/gameId/401874554) | 79' | NU |
| 2026-09-11 | [F.C. København - AC Horsens 2-0](https://www.espn.com/soccer/match/_/gameId/401874540) | 7' | DA |
| 2026-09-06 | [Odense Boldklub - F.C. København 0-5](https://www.espn.com/soccer/match/_/gameId/401874532) | 5' | DA |
| 2026-09-03 | [F.C. København - FC Nordsjælland 2-0](https://www.espn.com/soccer/match/_/gameId/401874523) | 8' | DA |
| 2026-08-31 | [F.C. København - Sønderjyske Fodbold 3-1](https://www.espn.com/soccer/match/_/gameId/401874513) | 7' | DA |
| 2026-08-23 | [Viborg FF - F.C. København 4-0](https://www.espn.com/soccer/match/_/gameId/401874522) | 1' | DA |
| 2026-08-16 | [Randers FC - F.C. København 0-4](https://www.espn.com/soccer/match/_/gameId/401874482) | 2' | DA |

### Benfica - campionat

| Data | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---:|---|
| 2026-10-26 | [Santa Clara - Benfica 1-2](https://www.espn.com/soccer/match/_/gameId/401885414) | 32' | DA |
| 2026-10-11 | [Benfica - Vitória de Guimaraes 2-0](https://www.espn.com/soccer/match/_/gameId/401885421) | 24' | DA |
| 2026-09-20 | [FC Porto - Benfica 3-1](https://www.espn.com/soccer/match/_/gameId/401885433) | 31' | DA |
| 2026-09-13 | [Benfica - Gil Vicente 3-1](https://www.espn.com/soccer/match/_/gameId/401885439) | 7' | DA |
| 2026-09-09 | [Moreirense - Benfica 0-4](https://www.espn.com/soccer/match/_/gameId/401885470) | 45'+4' | DA |
| 2026-09-05 | [Maritimo - Benfica 0-3](https://www.espn.com/soccer/match/_/gameId/401885452) | 7' | DA |
| 2026-08-31 | [Benfica - Estoril 2-1](https://www.espn.com/soccer/match/_/gameId/401885457) | 40' | DA |
| 2026-08-17 | [Casa Pia - Benfica 0-7](https://www.espn.com/soccer/match/_/gameId/401885479) | 3' | DA |
| 2026-08-09 | [Benfica - Académico de Viseu 2-2](https://www.espn.com/soccer/match/_/gameId/401885485) | 10' | DA |

### Alverca - campionat

| Data | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---:|---|
| 2026-10-24 | [Alverca - Arouca 0-1](https://www.espn.com/soccer/match/_/gameId/401885420) | 24' | DA |
| 2026-10-12 | [FC Famalicao - Alverca 1-4](https://www.espn.com/soccer/match/_/gameId/401885425) | 14' | DA |
| 2026-09-19 | [Alverca - Rio Ave 1-0](https://www.espn.com/soccer/match/_/gameId/401885438) | 3' | DA |
| 2026-09-12 | [C.D. Nacional - Alverca 1-3](https://www.espn.com/soccer/match/_/gameId/401885441) | 34' | DA |
| 2026-09-05 | [Alverca - Braga 1-2](https://www.espn.com/soccer/match/_/gameId/401885456) | 27' | DA |
| 2026-08-29 | [Alverca - Santa Clara 1-1](https://www.espn.com/soccer/match/_/gameId/401885464) | 69' | DA |
| 2026-08-22 | [Sporting CP - Alverca 3-1](https://www.espn.com/soccer/match/_/gameId/401885467) | 2' | DA |
| 2026-08-15 | [Alverca - Estrela 2-2](https://www.espn.com/soccer/match/_/gameId/401885482) | 51' | DA |
| 2026-08-09 | [FC Porto - Alverca 2-0](https://www.espn.com/soccer/match/_/gameId/401885487) | 9' | DA |

## Analiza contextuala si decizia

### 1. Sassuolo-Fiorentina, 19:30 RO

Serie A, etapa 10, campionat regulat, fara calificare pe un scor agregat. Ambele au nevoie de puncte: Sassuolo avea 8, Fiorentina 5 dupa 9 meciuri. Nu rezulta ca trebuie sa atace nechibzuit de la start; un egal poate fi acceptabil intr-un meci intre echipe aflate in dificultate. Prima deschidere a scorului <=70 apare in toate cele 18 partide distincte ale sezonului celor doua, cu esantion pe teren de numai 4+4 partide. Acesta este principalul argument pentru pastrarea candidatului dupa cautarea completa.

Aquilani vs Vanoli; formatii PROBABILE Sky 4-3-3 vs 4-3-2-1. Berardi si Lauriente pe laterale pot crea dueluri directe, iar intermediarii Fiorentinei pot ataca spatiul din fata fundasilor: acestea sunt scenarii tactice, nu bonusuri numerice. Sassuolo nu ii are pe Thorstvedt (suspendat), Kone, Leysen si Boloca; la Fiorentina sunt listati indisponibili Parisi, Atta, Ranieri si Pongracic. Absentele defensive pot ajuta producerea ocaziilor, dar niciun efect cauzal nu a fost cuantificat. [Sky, loturi si module probabile](https://sport.sky.it/calcio/serie-a/partite/2026/giornata-10/sassuolo-fiorentina/formazioni).

**Corectie de sursa:** Sassuolo-Lazio 2-6, 27 octombrie: al doilea gol Noslin a fost la 65, nu 75. Toti cei sase marcati de Lazio sunt <=70. Sursa oficiala Lazio si ESPN se aliniaza pentru aceasta clasificare; vechiul raport se baza pe o transcriere eronata a minutului. De aceea P brut trece de la 93,87% la 94,40%. Diferentele de 1-2 minute dintre furnizori pentru alte goluri din acel meci nu trec pragul de 70. [Tablou oficial Lazio](https://sslazio.hiwaymedia.io/it/news/match-report/serie-a-enilive-or-sassuolo-lazio-le-formazioni-ufficiali-a).

**Riscul decisiv:** acel 2-6 a avut eliminarea lui Thorstvedt la 31, iar sase goluri primite pana la 70 umfla intensitatea. Eliminand INTREG meciul numai ca diagnostic, N Sassuolo devine 8 sezon/3 acasa si P scade la **90,34%**. Nu inlocuiesc estimarea cu acest rezultat ales dupa observarea unui outlier. Arata fragilitatea celor 94,40%, nu o probabilitate reala alternativa validata. Acesta este un contraargument mai relevant decat diferenta de 0,04 puncte procentuale fata de meciul danez.

### 2. Sonderjyske-Copenhaga, 20:00 RO

Superliga daneza, etapa 13, faza regulata de 22 etape; puncte in campionat, fara departajare la finalul serii. Sonderjyske este ultima, cu 8 puncte dupa 12 jocuri; Copenhaga a doua, cu 26. La actualizarea clasamentului dupa duminica, Midtjylland are 31 din 13; nu mai folosesc diferenta de doua puncte din avancronica publicata sambata. FCK cauta revenirea dupa trei etape fara victorie. Un egal ar ajuta relativ mai mult gazdele, astfel ca blocul jos este un scenariu plauzibil. [Avancronica oficiala FCK](https://www.fck.dk/nyhed/fokus-paa-soenderjyske-fc-koebenhavn), [clasament actualizat in avancronica Sportportalen](https://sportportalen.dk/nyheder/fodbold/soenderjyske-fck-optakt-superligaen-runde-13).

Gazdele lui Fatah Abdirahman au primit cele mai multe goluri din liga, dar ultimele doua acasa au fost victorii 4-2 si 4-0: argumentul este ca pot contribui si ele la gol, nu doar ca FCK trebuie sa marcheze. Au urmat 1-5 la Horsens si eliminare in cupa dupa prelungiri, miercuri. FCK a jucat joi in cupa si mai are Iberia pe 5 noiembrie si derby cu Brondby pe 8: program incarcat si risc de rotatie, nu dovada unei oboseli inevitabile. Copenhaga trebuie sa desfaca eventualul bloc jos prin latime, combinatii intre linii si prezenta in careu; daca incepe lent, volumul total istoric de goluri poate sa nu se traduca intr-un gol devreme.

Bo Svensson a convocat aceiasi 20 de jucatori ca in cupa. Sunt confirmati ABSENTI Magnus Mattsson, Thapelo Maseko, Marvin Nasnas si Robert Silva. Elyounoussi, Cornelius si Mads Emil Madsen sunt convocati. Acesta este lotul convocat, NU un 11 de start confirmat. Nu am verificat o formatie probabila actuala suficient de solida pentru a o publica drept fapt si nu am reconciliat integral lotul gazdelor. [Lot oficial FCK, 1 noiembrie 18:25](https://www.fck.dk/nyhed/truppen-mod-soenderjyske-50).

**Contradictia importanta:** 94,36% din Poisson fata de 20/23 = 86,96% in istoricul sezonului fara dublarea meciului direct. Numai configuratia de teren are 10/12 = 83,33%. Esecurile sunt OB-Sonderjyske 1-0 (primul gol 81), Sonderjyske-Viborg 0-0 si Brondby-FCK 0-1 (79). Inclusiv o favorita care castiga poate rata evenimentul <=70. Modelul foloseste goluri dupa primul; nu avem justificare pentru a considera cei 94% bine calibrati. Pastrez partida drept a doua in topul FORMULEI, cu aceasta rezerva substantiala.

### Benfica-Alverca, a treia alternativa, 22:15 RO

Model 92,45%, stres 87,34%, istoric actual 18/18 partide distincte cu primul gol <=70. Benfica are 20 goluri marcate pana la 70 in 9 jocuri, Alverca 11 primite. Sunt numai cate 4 meciuri relevante pe teren, deci 18/18 nu inseamna certitudine. Marco Silva si Sergio Ferreira sunt antrenorii mentionati in avancronica, iar antrenorul gazdelor avertizeaza asupra dificultatii, fara a se baza pe diferenta din clasament. [Declaratie pre-meci, SPORT TV](https://www.sporttv.pt/noticias/857511/marco-silva-alverca-vai-ser-dificil-nao-olhamos-para-tabela).

Benfica este alternativa cu frecventa istorica mai curata decat meciul danez, dar raw si stres sunt mai mici. Nu am verificat integral loturile actuale; listele publice gasite erau contradictorii sau aveau date vechi. Nu mut automat meciul pe locul 2 doar ca istoricul are 100%, nici nu pretind ca are o cota oferita de minimum 1,10. Poate intra in comparatia finala la confirmarea loturilor; nu este o a treia selectie de jucat.

### De ce nu celelalte

Deportivo-Osasuna este urmatoarea dupa model (91,37%, 19/20 istoric), dar fara verificare integrala a loturilor nu pretind o analiza contextuala identica cu cea a finalistilor. Atalanta-Parma are 90,92%, insa Parma a marcat ZERO goluri <=70 in cele patru deplasari si istoricul sezonului celor doua este 15/18. Radomiak-Slask are 89,89%, U Cluj-FCSB 89,76%, Lorient-Brest 89,46%: sunt alternative reale, dar nu conduc formula aplicata uniform. U Cluj-FCSB are 23/26 in istoric, cu doua esecuri in deplasarile FCSB (0-0 cu Sepsi si primul gol 77 la CFR) si primul gol 73 in Voluntari-U Cluj. Reims-Sochaux are numai 15/20; liga sau numele favoritei nu elimina aceste esecuri. Suedia ofera un sezon mai lung, dar ambele meciuri au raw sub 85%. Lista completa de mai sus pastreaza restul comparatiei, fara selectie timpurie pe tari.

Nu am validat prognoze meteo orare, starea gazonului sau toate deplasarile pentru cele 27; acestea nu intra ca date inventate si nu justific o precizie numerica suplimentara. Cercetarea este completa ca PROGRAM si CALCUL pentru universul definit; verificarea contextuala amanuntita se concentreaza pe finalisti si nu este echivalenta pentru toate cele 27. Nu declar un avantaj financiar sau o rata reala de 90%.

## Controlul datelor si reproducere

Sunt 1306 partide istorice distincte validate: toate golurile reconciliate cu scorul final pe fiecare echipa, minutele si etichetele <=70 recalculate, autogolurile atribuite beneficiarului. Un meci suedez fara lista completa a golurilor a fost respins (Orgryte-Elfsborg, 29 mai); nu implica nicio echipa suedeza care joaca azi. 27/27 modele trec verificarea inputurilor, formulei, pretului echitabil, statusului salvat si orei de start. Nu sunt rezultate de backtest.

Datele ESPN pentru sezonul calendaristic 2026 au fost filtrate dupa season.year=2026, status final si data anterioara zilei analizei. Pentru Suedia: sezon 2026; pentru celelalte: 2026/27. Nu se amesteca sezonul 2025/26. Pentru Polonia, Romania, Serbia si Turcia 2 s-a folosit FotMob, toate partidele sezonului echipelor de azi. Alte ligi ESPN contin istoric si pentru echipe care nu joaca azi; nu intra in numaratorile candidatilor.

Fisiere: Europe_Combined/all_models.csv (toate inputurile), comparison_with_first_goal.csv (comparatia finala), historical_validated.csv (1306 partide), historical_rejected.csv, league_coverage.csv, today_fixtures.json. JSON-urile ESPN/FotMob sunt selectii de campuri sportive publice; cotele au fost omise. Datele initiale si jurnalul initial nu sunt suprascrise.

Reproducere offline: powershell -NoProfile -File Analysis/Goal70_2026-11-02/evaluate_europe_espn.ps1 -DataDirectory Analysis/Goal70_2026-11-02/Europe_Combined; apoi verify_and_report_europe.ps1 -RecordedAtUtc cu ORA ANALIZEI SALVATE, nu cu o data ulterioara meciurilor. Scriptul refuza un kickoff deja trecut la ora introdusa.
