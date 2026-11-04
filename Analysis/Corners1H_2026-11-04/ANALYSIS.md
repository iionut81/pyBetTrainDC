# Cornere1H — 4 noiembrie 2026

Salvat 2026-11-04T08:12:57.873+02:00. Research partial: 16/17 modele si verificare independenta incompleta.

Candidatul analistului: **Kortrijk - KV Mechelen, 21:30 RO, CORNERS_1H_UNDER_7_5**, maximum7 cornere cumulate FirstHalf. Model brut 95.78%, stres 90.42%, cota matematica 1.0440, rang1/16 calculabile. Estimarea finala ramane modelul brut necalibrat; contextul nu adauga procente.

18/18 observatii distincte au <=7; exact7: 2, >=8: 0. Pe teren8/8. Numaratori: Kortrijk9 meciuri (16 obtinute/19 cedate), Mechelen9 (13/25); acasa4 (9/7), deplasare4 (5/8). Lambda sezon 4.0556, teren 3.6250, combinat 3.8403; numai-sezon 94.55%, numai-teren 96.81%. E un esantion mic; 18/18 nu inseamna risc zero. Dispersia si toate totalurile sunt mai jos/in CSV.

[Preview oficial Kortrijk, 2 noiembrie](https://www.kvk.be/nieuws/voorbeschouwing-kvk-kvm/) descrie presing inalt si defensiva intarita la Standard, cu Beichler nou pe banca; mentioneaza absentele de atunci ale lui Dejaegere, Roche, Mehssatou si Campbell, fara a confirma automat absenta lor azi. Vanderbiest este suspendat, asistentul Van Handenhoven conduce Mechelen. Aceste schimbari limiteaza comparabilitatea mediei de sezon. Inferenta contrara underului: presingul si jocul pe benzi pot aduce suturi blocate si serii de cornere; un gol timpuriu nu reduce automat ritmul. [Conferinta din21 octombrie](https://www.kvk.be/nieuws/persconferentie-kvkkvm/) confirma intentia noului presing; nu este o declaratie noua de azi.

Partida este reprogramata dupa furtuna din23 octombrie; data/ora sunt confirmate de [Mechelen](https://kvmechelen.be/kvkkvmafgelast/) si [Kortrijk](https://www.kvk.be/nieuws/thuis-tegen-mechelen/). [Programul Mechelen](https://kvmechelen.be/2627debusop/) confirma deplasari Kortrijk miercuri si Leuven sambata. Ambele au jucat sambata31 octombrie, recuperare aproximativ4 zile, apoi calendar strans. Miza punctelor nu justifica automat un plan prudent.

Loturi: sursele secundare nu sunt in acord cu lista oficiala istorica; nu prezint absente de la31 octombrie sau formule5-4-1/3-4-1-2 drept titulari confirmati azi. Lotul complet si unsprezecele raman neconfirmate la08:00. [Prognoza locala](https://weer-meteo.be/en/cities/kortrijk.html) indica uscat, circa6 grade la21:00, vant17 km/h, in timp ce [alta prognoza](https://www.wetterkontor.de/wetter-vorhersage/belgien/kortrijk) arata seara vant24 km/h/rafale48: pastrez diferenta; nu presupun vreme fara risc sau teren inspectat.

De ce inaintea alternativelor: Artis-Bohemians are 89.14% brut, 78.41% stres si3 cazuri >=8/24. Fenerbahce-Liverpool are 89.11% / 78.37%, dar proiectie UCL. Cardiff-West Ham are5 cazuri >=8/26 si schimbare recenta de antrenor. Kortrijk-Mechelen conduce clar screeningul, cu riscurile noului regim tactic, esantionului mic si verificarii independente incomplete.

Limita de sursa: totalurile FotMob FirstHalf se reconciliaza individual cu SecondHalf si All. ESPN nu livreaza cronologie completa a cornerelor pentru toate partidele belgiene; nu transform o cronologie lipsa in0. Verificarea independenta ramane partiala. Recomandarea principala individuala a zilei este acest candidat la cornere, cu prudenta explicita; PSV este candidatul separat cerut pentru cealalta piata. Nu sunt doua alegeri ale utilizatorului si nu se propune combinatie.

## Acoperire si lipsuri

Univers: primele ligi masculine din top20 asociatii [UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), plus Romania si Serbia, plus 7 divizii secunde urmarite: Anglia, Germania, Spania, Italia, Franta, Belgia, Turcia. Total 29 ligi. Inventar: 7 meciuri interne (1 Cehia, 1 Belgia, 3 Championship, 2 Belgia 2), 9 UCL si 1 UEL = 17 partide. [Program FotMob](https://www.fotmob.com/api/data/matches?date=20261104), programe de sezon individuale in Schedules/, verificari ESPN in ESPN/. Toate cele 29 programe de sezon concorda la numarul zilnic cu inventarul global; ESPN nu suporta toate ligile. Absenta este raportata din programe valide, nu din raspunsuri goale.

Clasamentul include 17/17 modele Gol70 si 16/17 cornere. Viking-Sabah are model Gol70, dar cornere incomplete. Date: 343 partide istorice distincte, fiecare cu identitate, scor, minute si perioade in historical_observations.json/MatchData. Istoricul a fost taiat la 2026-11-04 00:00 RO. Sezon 2026 pentru Norvegia/Suedia; 2026/27 pentru celelalte ligi. Proiectiile UCL/UEL din ligi nationale sunt marcate separat; intensitatea adversarilor nu este ajustata arbitrar.

Sunt 103 partide in afara universului comparabil in exclusions.csv: alte continente, fotbal feminin/juniori, ligi inferioare si cupe fara istoric curent comparabil colectat (inclusiv Copa del Rey: 7, Cupa Ciprului: 1, preliminarii Cupa Turciei: 1, Cupa Slovaciei: 2). Cupele sunt inventariate distinct, dar necalculate; nu este un top global al tuturor competitiilor. Limita explicita: istoricul echipelor inferioare din aceste cupe nu este colectat.

Corectie de sursa inainte de publicare: catalogul FotMob a confirmat ID Cipru 136, Turcia 2 165, Azerbaidjan 262. Raspunsurile initiale pentru Cipru/Azerbaidjan desemnau alte competitii si au fost respinse. Instantaneele initiale, diferentele si motivul sunt in Revisions/initial_screening/. Recalcularea a completat Viking-Sabah Gol70; clasamentul initial este pastrat.

Viking-Sabah FK: cornere 1H 25/25 si 0/9; observatiile lipsa nu sunt zero.
## Metoda

Metoda prospectiva salvata: numai campionatul curent; minimum 5 meciuri/echipa si 3 pe teren. Lambda sezon = (GF_A/n_A + GA_B/n_B + GF_B/n_B + GA_A/n_A)/2; aceeasi formula pe teren; lambda = 50% sezon + 50% teren. La Gol70, GF/GA numara numai goluri valide cu minut afisat <=70, inclusiv 45+, fara factor 70/90. P = 1-exp(-lambda), stres -20% intensitate. La cornere se folosesc cornerele obtinute/cedate FirstHalf: P=PoissonCDF(7,lambda), stres +20%. Cota matematica=1/P. Fara calibrare, ajustari contextuale de procente, preturi sau pariuri. Frecventele istorice si modelul sunt masuri diferite; esantioanele sezon/teren se suprapun.

## Clasament complet

| Rang | Meci | Ora RO | Competitie | Model | Stres | Cota matematica | Baza | Limita |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kortrijk - KV Mechelen | 21:30 | bel.1 | 95.78% | 90.42% | 1.0440 | same_league_current_season |  |
| 2 | Artis Brno - Bohemians 1905 | 19:00 | cze.1 | 89.14% | 78.41% | 1.1218 | same_league_current_season |  |
| 3 | Fenerbahçe - Liverpool | 19:45 | uefa.champions | 89.11% | 78.37% | 1.1222 | domestic_league_projection_cross_competition |  |
| 4 | Cardiff City - West Ham United | 21:45 | eng.2 | 88.69% | 77.66% | 1.1276 | same_league_current_season |  |
| 5 | Anderlecht Futures - K. Lierse SK | 21:00 | bel.2 | 88.55% | 77.43% | 1.1294 | same_league_current_season |  |
| 6 | Wrexham - Middlesbrough | 21:45 | eng.2 | 88.53% | 77.41% | 1.1295 | same_league_current_season |  |
| 7 | Slavia Prague - Arsenal | 22:00 | uefa.champions | 87.65% | 75.97% | 1.1410 | domestic_league_projection_cross_competition |  |
| 8 | Borussia Dortmund - Real Betis | 22:00 | uefa.champions | 87.11% | 75.11% | 1.1480 | domestic_league_projection_cross_competition |  |
| 9 | AEK Athens - Real Madrid | 19:45 | uefa.champions | 86.50% | 74.15% | 1.1560 | domestic_league_projection_cross_competition |  |
| 10 | Lens - Como | 22:00 | uefa.champions | 85.10% | 71.96% | 1.1751 | domestic_league_projection_cross_competition |  |
| 11 | Bristol City - Wolverhampton Wanderers | 22:00 | eng.2 | 84.94% | 71.71% | 1.1774 | same_league_current_season |  |
| 12 | Rennes - Olympiacos | 17:30 | uefa.europa | 82.89% | 68.64% | 1.2064 | domestic_league_projection_cross_competition |  |
| 13 | RB Leipzig - Manchester City | 22:00 | uefa.champions | 82.79% | 68.49% | 1.2078 | domestic_league_projection_cross_competition |  |
| 14 | FC Porto - Napoli | 22:00 | uefa.champions | 79.38% | 63.59% | 1.2598 | domestic_league_projection_cross_competition |  |
| 15 | Francs Borains - FCV Dender EH | 21:00 | bel.2 | 77.76% | 61.36% | 1.2860 | same_league_current_season |  |
| 16 | PSV Eindhoven - Club Brugge | 22:00 | uefa.champions | 76.73% | 59.97% | 1.3033 | domestic_league_projection_cross_competition |  |
| 17 | Viking - Sabah FK | 22:00 | uefa.champions | lipsa | lipsa | lipsa | domestic_league_projection_cross_competition | missing_observations_or_sample_below_5_3 |

## Numaratori si dispersie

| Esantion | N | Obtinut | Cedat | Succes | Media | Dispersie | Dupa70 / 0-0 | Exact7 / >=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| home | 9 | 16 | 19 | 9 | 3.888888888888889 | 1.6111111111111112 | n/a | 0 / 0 |
| away | 9 | 13 | 25 | 9 | 4.222222222222222 | 4.694444444444444 | n/a | 2 / 0 |
| home_venue | 4 | 9 | 7 | 4 | 4 | 0.6666666666666666 | n/a | 0 / 0 |
| away_venue | 4 | 5 | 8 | 4 | 3.25 | 2.25 | n/a | 0 / 0 |

## Sensibilitate

| Scenariu | Sezon | Teren | Model | Nota |
| --- | --- | --- | --- | --- |
| Fara meciuri cu rosu | 6/8 | 2/4 | 96.86% | diagnostic; ponderile originale raman neschimbate |

## Istoric complet FirstHalf al finalistului

| Data | Meci | Gazda | Oaspete | Total | Rezultat | Sursa |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-07 | Club Brugge - Kortrijk | 3 | 0 | 3 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811710) |
| 2026-08-15 | Kortrijk - Royal Antwerp | 2 | 1 | 3 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811721) |
| 2026-08-29 | Kortrijk - Sporting Charleroi | 1 | 3 | 4 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811738) |
| 2026-09-03 | Anderlecht - Kortrijk | 3 | 2 | 5 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811733) |
| 2026-09-06 | Kortrijk - Zulte Waregem | 4 | 1 | 5 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811752) |
| 2026-09-13 | RAAL La Louvière - Kortrijk | 0 | 2 | 2 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811763) |
| 2026-09-20 | Kortrijk - SK Beveren | 2 | 2 | 4 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811772) |
| 2026-10-10 | Genk - Kortrijk | 2 | 1 | 3 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811775) |
| 2026-10-31 | Standard Liège - Kortrijk | 4 | 2 | 6 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811797) |
| 2026-08-09 | Gent - KV Mechelen | 4 | 1 | 5 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811715) |
| 2026-08-16 | KV Mechelen - Standard Liège | 0 | 1 | 1 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811726) |
| 2026-08-22 | Sporting Charleroi - KV Mechelen | 2 | 0 | 2 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811730) |
| 2026-08-29 | RAAL La Louvière - KV Mechelen | 0 | 2 | 2 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811737) |
| 2026-09-05 | KV Mechelen - Westerlo | 3 | 4 | 7 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811748) |
| 2026-09-11 | KV Mechelen - Anderlecht | 2 | 3 | 5 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811755) |
| 2026-09-19 | Lommel - KV Mechelen | 2 | 2 | 4 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811768) |
| 2026-10-11 | KV Mechelen - St.Truiden | 1 | 4 | 5 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811776) |
| 2026-10-31 | KV Mechelen - Genk | 2 | 5 | 7 | success | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811795) |

## Ultimele10 oficiale Kortrijk

| Data | Competitie | Meci | FT | Primul gol | Gol70 | Cornere 1H | Surse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-31 | Belgian Pro League | Standard Liège - Kortrijk | 1-1 | 45+1 | success | 6 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811797) |
| 2026-10-10 | Belgian Pro League | Genk - Kortrijk | 3-0 | 3 | success | 3 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811775) |
| 2026-09-20 | Belgian Pro League | Kortrijk - SK Beveren | 1-0 | 52 | success | 4 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811772) |
| 2026-09-13 | Belgian Pro League | RAAL La Louvière - Kortrijk | 3-0 | 7 | success | 2 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811763) |
| 2026-09-06 | Belgian Pro League | Kortrijk - Zulte Waregem | 0-2 | 90+3 | fail | 5 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811752) |
| 2026-09-03 | Belgian Pro League | Anderlecht - Kortrijk | 1-0 | 43 | success | 5 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811733) |
| 2026-08-29 | Belgian Pro League | Kortrijk - Sporting Charleroi | 1-3 | 34 | success | 4 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811738) |
| 2026-08-15 | Belgian Pro League | Kortrijk - Royal Antwerp | 0-3 | 31 | success | 3 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811721) |
| 2026-08-07 | Belgian Pro League | Club Brugge - Kortrijk | 3-0 | 30 | success | 3 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811710) |

## Ultimele10 oficiale Mechelen

| Data | Competitie | Meci | FT | Primul gol | Gol70 | Cornere 1H | Surse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-31 | Belgian Pro League | KV Mechelen - Genk | 1-0 | 22 | success | 7 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811795) |
| 2026-10-18 | Cup | KV Mechelen - Eupen | 5-1 | 5 | success | lipsa | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=6361391) |
| 2026-10-11 | Belgian Pro League | KV Mechelen - St.Truiden | 4-3 | 24 | success | 5 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811776) |
| 2026-09-19 | Belgian Pro League | Lommel - KV Mechelen | 0-0 | 0-0 | fail | 4 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811768) |
| 2026-09-11 | Belgian Pro League | KV Mechelen - Anderlecht | 0-1 | 43 | success | 5 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811755) |
| 2026-09-05 | Belgian Pro League | KV Mechelen - Westerlo | 0-4 | 7 | success | 7 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811748) |
| 2026-08-29 | Belgian Pro League | RAAL La Louvière - KV Mechelen | 2-2 | 6 | success | 2 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811737) |
| 2026-08-22 | Belgian Pro League | Sporting Charleroi - KV Mechelen | 2-0 | 35 | success | 2 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811730) |
| 2026-08-16 | Belgian Pro League | KV Mechelen - Standard Liège | 3-3 | 9 | success | 1 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811726) |
| 2026-08-09 | Belgian Pro League | Gent - KV Mechelen | 2-0 | 37 | success | 5 | [FotMob](https://www.fotmob.com/api/data/matchDetails?matchId=5811715) |

Ultima verificare a identitatii/kickoffului: 2026-11-04T06:08:11.064Z UTC. Toate cele17 partide sunt neincepute, identitatea ambelor echipe concorda; live_status.csv. [Program UEFA](https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-phase-fixtures/) confirma perechile UCL. Formatiile nu sunt inca anuntate. Nu exista rezultate fabricate pentru4 noiembrie.


## Completare finala a surselor

Actualizare de sursa la 2026-11-04T08:15:43.080+02:00: cele7 intrari Copa del Rey din programul global sunt **amanate**, conform [RFEF, comunicat3 noiembrie17:00CET](https://rfef.es/es/noticias/la-rfef-acuerda-ampliar-la-suspension-de-sus-competiciones-hasta-el-6-de-noviembre); aceeasi suspendare acopera Copa de la Reina. Lista globala era depasita pentru aceste cupe. Excluderile au primit statusul oficial si sursa; nu sunt prezentate ca meciuri confirmate azi. Cele17 partide modelate raman neafectate.

Verificarea independenta a golurilor pentru PSV/Brugge si Kortrijk/Mechelen a asociat38 partide distincte intre ESPN si FotMob: toate38 au aceleasi numaratori GF70/GA70. Cele18 cronologii belgiene de cornere ESPN sunt incomplete; totalurile FirstHalf nu au primit o a doua confirmare independenta. Acest lucru ramane limita, nu dezacord numeric sau zero.

Datele complete ale perioadelor sunt salvate si local in historical_corners_1h.csv; dovezile detaliate comune sunt in ../Goal70_2026-11-04/MatchData/.
