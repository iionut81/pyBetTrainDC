# Review 7 noiembrie 2026

Salvat 2026-11-08T10:32:31.267+02:00, înaintea oricărui inventar din 8 noiembrie. Surse prospective: [Gol70](../Goal70_2026-11-07/EUROPE_ALL_MATCHES.md), [cornere](../Corners1H_2026-11-07/ANALYSIS.md), CSV canonice 7 noiembrie; [comparație Blackburn/Málaga](../Goal70_2026-11-07/Cardiff_Malaga_Comparison/COMPARISON.md), [alternative cornere](../Corners1H_2026-11-07/Alternative_Comparison/COMPARISON.md). Rezultate publice FotMob, JSON salvate în FotMob/, identitate echipe verificată; toate golurile reconciliate cu FT90 și cornerele FirstHalf+SecondHalf=All pe echipă. Lipsurile/conflictele rămân pending.

## Alegerile explicite actuale

| Meci | Piață | Observație exactă | Rezultat | Sursă |
| --- | --- | --- | --- | --- |
| Blackburn Rovers–West Ham United | GOAL_BY_70 | 54 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836923) |
| Excelsior–Go Ahead Eagles | CORNERS_1H_UNDER_7_5 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781800) |

Blackburn–West Ham: FT2–1, primul gol54, succes. Excelsior–Go Ahead: FT0–1, FirstHalf3 cornere, succes. Primul gol72 în același meci este FAIL pentru Gol70, dar nu schimbă succesul cornere. Sporting a fost revocat explicit în dashboard; nu îl recreăm. Ambele alegeri sunt păstrate fără deducerea unui pariu/profit.

## GOAL_BY_70

Univers verificat și lipsuri: {"total":102,"modeled":101,"success":86,"fail":14,"unscored":[{"id":"5887676","home":"Nacional","away":"Académico Viseu","result":"pending","reason":"not_final"},{"id":"5868531","home":"Granada","away":"Tenerife","result":"cancelled","reason":"provider_cancelled"}]}. Separat modele prospective: {"total":101,"modeled":101,"success":86,"fail":14,"unscored":[{"id":"5887676","home":"Nacional","away":"Académico Viseu","result":"pending","reason":"not_final"}]}.

| Rang original | Meci | Competiție | Brut / AR20 | Istoric / teren | Observație | Rezultat | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Excelsior–Go Ahead Eagles | ned.1 | 96.47% / 93.11% | 21/22; 11/11 | 72 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781800) |
| 2 | FC Vaduz–Sion | sui.1 | 95.24% / 91.25% | 23/23; 11/11 | 11 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804297) |
| 3 | RB Leipzig–Augsburg | ger.1 | 95.07% / 91.00% | 15/16; 7/7 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881222) |
| 4 | Luzern–Young Boys | sui.1 | 94.92% / 90.78% | 23/23; 11/11 | 22 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804293) |
| 5 | Cardiff City–Burnley | eng.2 | 94.84% / 90.67% | 26/28; 14/14 | 15 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836925) |
| 6 | Westerlo–Sporting Charleroi | bel.1 | 94.63% / 90.37% | 20/20; 10/10 | 58 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811808) |
| 7 | Austria Lustenau–Grazer AK | aut.1 | 94.58% / 90.29% | 19/21; 11/11 | 9 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5846017) |
| 8 | ADO Den Haag–Sparta Rotterdam | ned.1 | 94.55% / 90.24% | 22/22; 10/10 | 22 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781799) |
| 9 | Mainz 05–Bayern München | ger.1 | 94.36% / 89.97% | 14/16; 6/8 | 14 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881220) |
| 10 | Roma–Sassuolo | ita.1 | 94.22% / 89.77% | 19/20; 10/10 | 27 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749740) |
| 11 | Keçiörengücü–Manisa Futbol Kulübü | tur.2 | 93.46% / 88.72% | 22/26; 11/12 | 38 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908694) |
| 12 | 1. FC Nürnberg–Darmstadt | ger.2 | 93.45% / 88.70% | 19/20; 9/9 | 50 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881723) |
| 13 | Olympiakos Nicosia–Apollon Limassol | cyp.1 | 93.40% / 88.64% | 14/16; 7/8 | 32 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5975447) |
| 14 | Málaga–Racing Santander | esp.1 | 93.09% / 88.21% | 19/22; 8/10 | 28 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868125) |
| 15 | Blackburn Rovers–West Ham United | eng.2 | 92.36% / 87.22% | 27/28; 14/14 | 54 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836923) |
| 16 | Korona Kielce–Radomiak Radom | pol.1 | 92.30% / 87.14% | 23/26; 11/12 | 69 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804033) |
| 17 | VfB Stuttgart–Werder Bremen | ger.1 | 92.21% / 87.02% | 14/16; 6/7 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881223) |
| 18 | RAAL La Louvière–Cercle Brugge | bel.1 | 91.79% / 86.47% | 20/20; 10/10 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811804) |
| 19 | Dundee United–Motherwell | sco.1 | 91.71% / 86.36% | 20/22; 11/11 | 9 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793972) |
| 20 | Pendikspor–İstanbulspor | tur.2 | 91.56% / 86.16% | 25/26; 12/13 | 69 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908695) |
| 21 | Rosenborg–Start | nor.1 | 91.22% / 85.72% | 47/49; 24/24 | 27 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105048) |
| 22 | Burgos CF–Celta Fortuna | esp.2 | 91.19% / 85.68% | 22/24; 11/12 | 12 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868526) |
| 23 | Nacional–Académico Viseu | por.1 | 91.15% / 85.63% | 20/20; 10/10 | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887676) |
| 24 | Örgryte–Västerås SK | swe.1 | 90.90% / 85.31% | 49/53; 24/26 | 0–0 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5107656) |
| 25 | Hertha BSC–Wolfsburg | ger.2 | 90.70% / 85.04% | 15/20; 6/9 | 43 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881720) |
| 26 | Galatasaray–Amed Sportif | tur.1 | 90.70% / 85.04% | 17/20; 8/9 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904800) |
| 27 | Birmingham City–Swansea City | eng.2 | 90.70% / 85.04% | 24/28; 13/14 | 29 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836922) |
| 28 | Anderlecht Futures–Club Brugge NXT | bel.2 | 90.52% / 84.82% | 14/16; 8/9 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015504) |
| 29 | Sporting CP–Moreirense | por.1 | 90.46% / 84.74% | 18/20; 10/10 | 61 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887679) |
| 30 | Royal Antwerp–Lommel | bel.1 | 90.22% / 84.43% | 17/20; 10/10 | 40 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811800) |
| 31 | Córdoba–AD Ceuta FC | esp.2 | 90.02% / 84.17% | 23/24; 10/10 | 23 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868529) |
| 32 | Francs Borains–RFC Seraing | bel.2 | 89.96% / 84.10% | 17/17; 8/8 | 67 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015502) |
| 33 | Fulham–Newcastle United | eng.1 | 89.82% / 83.92% | 15/18; 8/8 | 80 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795509) |
| 34 | NK Istra 1961–Rijeka | cro.1 | 89.77% / 83.87% | 18/22; 10/12 | 30 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5765955) |
| 35 | Ried–LASK | aut.1 | 89.74% / 83.83% | 19/21; 9/10 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5846019) |
| 36 | Cagliari–Frosinone | ita.1 | 89.72% / 83.80% | 19/20; 8/8 | 38 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749741) |
| 37 | Bristol City–Norwich City | eng.2 | 89.46% / 83.47% | 26/28; 14/14 | 22 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836924) |
| 38 | Watford–Middlesbrough | eng.2 | 89.27% / 83.23% | 28/28; 14/14 | 0–0 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836931) |
| 39 | Samsunspor–Konyaspor | tur.1 | 89.19% / 83.14% | 16/20; 7/9 | 39 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904802) |
| 40 | Paderborn–Eintracht Frankfurt | ger.1 | 89.13% / 83.05% | 11/16; 6/8 | 23 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881221) |
| 41 | West Bromwich Albion–Millwall | eng.2 | 89.08% / 82.99% | 24/28; 13/14 | 50 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836932) |
| 42 | Göztepe–Başakşehir | tur.1 | 88.64% / 82.45% | 18/20; 6/8 | 18 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904798) |
| 43 | Portsmouth–Stoke City | eng.2 | 88.48% / 82.25% | 25/28; 14/14 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836928) |
| 44 | Arsenal–Hull City | eng.1 | 88.46% / 82.23% | 15/18; 7/8 | 72 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795505) |
| 45 | AEL Limassol–Krasava Ypsonas FC | cyp.1 | 88.42% / 82.18% | 15/16; 7/7 | 33 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5975443) |
| 46 | FCV Farul Constanța–Rapid București | rou.1 | 88.27% / 81.99% | 21/28; 8/12 | 32 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853546) |
| 47 | Wrexham–Wolverhampton Wanderers | eng.2 | 88.06% / 81.73% | 26/28; 13/14 | 25 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836933) |
| 48 | Fortuna Sittard–Telstar | ned.1 | 87.75% / 81.36% | 19/22; 8/10 | 15 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781801) |
| 49 | Auxerre–Paris FC | fra.1 | 87.72% / 81.32% | 15/18; 6/8 | 63 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802983) |
| 50 | Csikszereda Miercurea Ciuc–Universitatea Cluj | rou.1 | 87.32% / 80.84% | 23/28; 9/12 | 0–0 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853545) |
| 51 | Mantova–Hellas Verona | ita.2 | 87.29% / 80.80% | 19/20; 8/9 | 0–0 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967146) |
| 52 | Aris Thessaloniki–Panetolikos | gre.1 | 87.29% / 80.80% | 15/18; 6/8 | 11 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5943289) |
| 53 | Hradec Králové–Sigma Olomouc | cze.1 | 87.12% / 80.59% | 23/26; 13/14 | 45+3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831214) |
| 54 | Grasshopper–Lausanne | sui.1 | 87.02% / 80.47% | 20/23; 10/12 | 29 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804292) |
| 55 | OH Leuven–KV Mechelen | bel.1 | 86.80% / 80.21% | 19/20; 9/10 | 50 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811803) |
| 56 | Saint-Étienne–Laval | fra.2 | 86.77% / 80.17% | 20/22; 9/10 | 45 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5803316) |
| 57 | St. Johnstone–Dundee FC | sco.1 | 86.75% / 80.15% | 20/22; 11/11 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793973) |
| 58 | St. Mirren–Rangers | sco.1 | 86.72% / 80.11% | 18/22; 10/12 | 18 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793974) |
| 59 | Queens Park Rangers–Lincoln City | eng.2 | 86.71% / 80.10% | 24/28; 12/14 | 11 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836930) |
| 60 | Torino–Lecce | ita.1 | 86.47% / 79.81% | 20/20; 10/10 | 60 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749748) |
| 61 | Čukarički–Radnički Niš | srb.1 | 86.47% / 79.81% | 23/26; 12/14 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015171) |
| 62 | FC Zbrojovka Brno–Baník Ostrava | cze.1 | 86.33% / 79.65% | 23/25; 12/13 | 23 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831213) |
| 63 | FC Zlín–Slovácko | cze.1 | 86.28% / 79.59% | 24/26; 14/14 | 61 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831220) |
| 64 | Preston North End–Southampton | eng.2 | 86.22% / 79.52% | 24/28; 11/14 | 33 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836929) |
| 65 | Van Spor Kulübü–Sivasspor | tur.2 | 85.97% / 79.22% | 24/26; 12/13 | 14 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908701) |
| 66 | Widzew Łódź–Pogoń Szczecin | pol.1 | 85.89% / 79.13% | 20/25; 9/12 | 33 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804038) |
| 67 | Clermont Foot–Rodez | fra.2 | 85.77% / 78.99% | 18/22; 7/11 | 76 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5803309) |
| 68 | Cesena–Catanzaro | ita.2 | 85.77% / 78.99% | 18/20; 10/11 | 83 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967142) |
| 69 | Charlton Athletic–Sheffield United | eng.2 | 85.46% / 78.62% | 23/28; 11/14 | 12 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836926) |
| 70 | Pisa–Südtirol | ita.2 | 85.41% / 78.56% | 17/20; 9/9 | 26 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967148) |
| 71 | OB–Viborg | den.1 | 85.34% / 78.47% | 21/25; 9/12 | 11 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739542) |
| 72 | Muğlaspor–Esenler Erokspor | tur.2 | 85.11% / 78.21% | 20/26; 9/13 | 26 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908703) |
| 73 | FCSB–Corvinul Hunedoara | rou.1 | 85.03% / 78.11% | 21/28; 10/13 | 41 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853549) |
| 74 | Degerfors–GAIS | swe.1 | 84.89% / 77.95% | 47/53; 24/26 | 57 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5107658) |
| 75 | Leganés–Albacete | esp.2 | 84.41% / 77.39% | 21/24; 9/11 | 20 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868532) |
| 76 | Cádiz–Real Oviedo | esp.2 | 84.39% / 77.37% | 20/24; 11/13 | 22 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868527) |
| 77 | HNK Gorica–Slaven | cro.1 | 84.34% / 77.31% | 19/23; 10/12 | 17 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5765954) |
| 78 | Śląsk Wrocław–Piast Gliwice | pol.1 | 84.32% / 77.28% | 23/26; 9/12 | 0–0 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804037) |
| 79 | Padova–Empoli | ita.2 | 84.28% / 77.24% | 18/20; 9/10 | 14 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967147) |
| 80 | Virton–Beerschot | bel.2 | 84.12% / 77.06% | 17/17; 8/8 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015505) |
| 81 | Górnik Zabrze–Wieczysta Kraków | pol.1 | 83.83% / 76.72% | 22/26; 13/14 | 31 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804031) |
| 82 | Paris Saint-Germain–Troyes | fra.1 | 83.79% / 76.67% | 15/18; 5/7 | 39 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802987) |
| 83 | Benevento–Avellino | ita.2 | 83.47% / 76.31% | 19/20; 10/10 | 60 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967140) |
| 84 | Osasuna–Athletic Club | esp.1 | 82.86% / 75.61% | 18/21; 7/9 | 84 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868126) |
| 85 | Železničar Pančevo–FK Radnički 1923 | srb.1 | 82.84% / 75.58% | 23/26; 11/14 | 15 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015172) |
| 86 | Alverca–Casa Pia AC | por.1 | 82.62% / 75.34% | 18/20; 9/10 | 36 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887672) |
| 87 | Espanyol–Deportivo A Coruña | esp.1 | 82.38% / 75.07% | 20/22; 8/10 | 0–0 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868124) |
| 88 | Parma–Bologna | ita.1 | 82.18% / 74.84% | 17/20; 7/10 | 53 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749747) |
| 89 | Valencia–Real Madrid | esp.1 | 81.81% / 74.43% | 16/22; 8/12 | 28 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868129) |
| 90 | Eyüpspor–Kocaelispor | tur.1 | 81.73% / 74.33% | 18/20; 9/10 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904795) |
| 91 | Nottingham Forest–Manchester City | eng.1 | 81.50% / 74.08% | 15/18; 5/8 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795513) |
| 92 | Slovan Liberec–Bohemians 1905 | cze.1 | 81.35% / 73.91% | 22/25; 12/12 | 72 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831216) |
| 93 | Holstein Kiel–Arminia Bielefeld | ger.2 | 81.27% / 73.82% | 14/20; 7/10 | 14 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881721) |
| 94 | VfL Osnabrück–Kaiserslautern | ger.2 | 80.31% / 72.75% | 16/20; 8/10 | 55 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881725) |
| 95 | Pau–Nantes | fra.2 | 80.29% / 72.73% | 18/22; 9/11 | 28 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5803315) |
| 96 | Borussia Dortmund–Elversberg | ger.1 | 79.04% / 71.35% | 11/16; 4/8 | 55 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881215) |
| 97 | Aberdeen–Hibernian | sco.1 | 78.39% / 70.64% | 18/21; 8/11 | 58 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793970) |
| 98 | NFC Volos–Kifisia FC | gre.1 | 75.58% / 67.62% | 16/18; 8/8 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5943294) |
| 99 | Juve Stabia–Arezzo | ita.2 | 75.03% / 67.04% | 15/20; 6/9 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967144) |
| 100 | FK Zemun–FK IMT Beograd | srb.1 | 74.72% / 66.71% | 23/27; 12/14 | 51 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015170) |
| 101 | Leeds United–Tottenham Hotspur | eng.1 | 70.95% / 62.80% | 11/18; 4/8 | 79 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795511) |
| 102 | Granada–Tenerife | esp.2 | lipsă / lipsă | 23/24; 11/11 | lipsă | cancelled | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868531) |

**Fail Excelsior–Go Ahead Eagles**: observație 72; rang original 1, ned.1, bază same_league_current_season. Brut 96.47%, AR20 93.11%. Istoric 21/22, teren 11/11; GF70 propriu pe teren 3/5 și 7/6; tardive/0–0 sezon 1/0 și 0/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Örgryte–Västerås SK**: observație 0–0; rang original 24, swe.1, bază same_league_current_season. Brut 90.90%, AR20 85.31%. Istoric 49/53, teren 24/26; GF70 propriu pe teren 14/13 și 16/13; tardive/0–0 sezon 0/1 și 1/2. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Fulham–Newcastle United**: observație 80; rang original 33, eng.1, bază same_league_current_season. Brut 89.82%, AR20 83.92%. Istoric 15/18, teren 8/8; GF70 propriu pe teren 6/4 și 4/4; tardive/0–0 sezon 2/1 și 0/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Watford–Middlesbrough**: observație 0–0; rang original 38, eng.2, bază same_league_current_season. Brut 89.27%, AR20 83.23%. Istoric 28/28, teren 14/14; GF70 propriu pe teren 6/7 și 8/7; tardive/0–0 sezon 0/0 și 0/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Arsenal–Hull City**: observație 72; rang original 44, eng.1, bază same_league_current_season. Brut 88.46%, AR20 82.23%. Istoric 15/18, teren 7/8; GF70 propriu pe teren 9/4 și 3/4; tardive/0–0 sezon 1/0 și 1/1. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Csikszereda Miercurea Ciuc–Universitatea Cluj**: observație 0–0; rang original 50, rou.1, bază same_league_current_season. Brut 87.32%, AR20 80.84%. Istoric 23/28, teren 9/12; GF70 propriu pe teren 5/6 și 3/6; tardive/0–0 sezon 2/1 și 1/1. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Mantova–Hellas Verona**: observație 0–0; rang original 51, ita.2, bază same_league_current_season. Brut 87.29%, AR20 80.80%. Istoric 19/20, teren 8/9; GF70 propriu pe teren 5/5 și 4/4; tardive/0–0 sezon 0/0 și 1/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Clermont Foot–Rodez**: observație 76; rang original 67, fra.2, bază same_league_current_season. Brut 85.77%, AR20 78.99%. Istoric 18/22, teren 7/11; GF70 propriu pe teren 1/6 și 4/5; tardive/0–0 sezon 1/3 și 0/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Cesena–Catanzaro**: observație 83; rang original 68, ita.2, bază same_league_current_season. Brut 85.77%, AR20 78.99%. Istoric 18/20, teren 10/11; GF70 propriu pe teren 5/5 și 5/6; tardive/0–0 sezon 0/0 și 2/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Śląsk Wrocław–Piast Gliwice**: observație 0–0; rang original 78, pol.1, bază same_league_current_season. Brut 84.32%, AR20 77.28%. Istoric 23/26, teren 9/12; GF70 propriu pe teren 3/6 și 5/6; tardive/0–0 sezon 0/1 și 2/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Osasuna–Athletic Club**: observație 84; rang original 84, esp.1, bază same_league_current_season. Brut 82.86%, AR20 75.61%. Istoric 18/21, teren 7/9; GF70 propriu pe teren 3/5 și 3/4; tardive/0–0 sezon 0/1 și 0/2. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Espanyol–Deportivo A Coruña**: observație 0–0; rang original 87, esp.1, bază same_league_current_season. Brut 82.38%, AR20 75.07%. Istoric 20/22, teren 8/10; GF70 propriu pe teren 3/5 și 4/5; tardive/0–0 sezon 2/0 și 0/0. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Slovan Liberec–Bohemians 1905**: observație 72; rang original 92, cze.1, bază same_league_current_season. Brut 81.35%, AR20 73.91%. Istoric 22/25, teren 12/12; GF70 propriu pe teren 4/5 și 7/7; tardive/0–0 sezon 0/1 și 0/2. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Leeds United–Tottenham Hotspur**: observație 79; rang original 101, eng.1, bază same_league_current_season. Brut 70.95%, AR20 62.80%. Istoric 11/18, teren 4/8; GF70 propriu pe teren 5/4 și 0/4; tardive/0–0 sezon 1/2 și 2/2. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

Reușite aproape de prag: Korona Kielce–Radomiak Radom (69); Pendikspor–İstanbulspor (69); Francs Borains–RFC Seraing (67).

Finalist analist original **Luzern–Young Boys**: success, observație 22; brut 94.92%, AR20 90.78%. Argumentele și riscurile rămân cele din raportul prospectiv, nu sunt rescrise după rezultat.

Toate succesele sunt prezentate în tabel, alături de avertismentele prospective; niciun succes nu validează probabilitatea estimată sau o cauză tactică. Un eșec nu demonstrează singur model greșit, iar un succes nu demonstrează calibrare.

## CORNERS_1H_UNDER_7_5

Univers verificat și lipsuri: {"total":102,"modeled":83,"success":84,"fail":14,"unscored":[{"id":"5887676","home":"Nacional","away":"Académico Viseu","result":"pending","reason":"not_final"},{"id":"5975443","home":"AEL Limassol","away":"Krasava Ypsonas FC","result":"pending","reason":"missing_or_unreconciled_FirstHalf"},{"id":"5975447","home":"Olympiakos Nicosia","away":"Apollon Limassol","result":"pending","reason":"missing_or_unreconciled_FirstHalf"},{"id":"5868531","home":"Granada","away":"Tenerife","result":"cancelled","reason":"provider_cancelled"}]}. Separat modele prospective: {"total":83,"modeled":83,"success":69,"fail":13,"unscored":[{"id":"5887676","home":"Nacional","away":"Académico Viseu","result":"pending","reason":"not_final"}]}.

| Rang original | Meci | Competiție | Brut / AR20 | Istoric / teren | Observație | Rezultat | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Parma–Bologna | ita.1 | 96.92% / 92.75% | 17/20; 10/10 | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749747) |
| 2 | NK Istra 1961–Rijeka | cro.1 | 96.65% / 92.19% | 22/22; 12/12 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5765955) |
| 3 | Juve Stabia–Arezzo | ita.2 | 96.36% / 91.59% | 18/20; 8/9 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967144) |
| 4 | Birmingham City–Swansea City | eng.2 | 96.15% / 91.16% | 26/28; 13/14 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836922) |
| 5 | HNK Gorica–Slaven | cro.1 | 96.03% / 90.92% | 22/23; 11/12 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5765954) |
| 6 | Excelsior–Go Ahead Eagles | ned.1 | 95.86% / 90.59% | 21/22; 11/11 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781800) |
| 7 | Anderlecht Futures–Club Brugge NXT | bel.2 | 95.49% / 89.85% | 16/16; 9/9 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015504) |
| 8 | Sporting CP–Moreirense | por.1 | 95.46% / 89.79% | 19/20; 9/10 | 10 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887679) |
| 9 | Leeds United–Tottenham Hotspur | eng.1 | 94.46% / 87.85% | 16/18; 8/8 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795511) |
| 10 | Čukarički–Radnički Niš | srb.1 | 94.45% / 87.83% | 25/26; 13/14 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015171) |
| 11 | Auxerre–Paris FC | fra.1 | 94.33% / 87.60% | 16/18; 7/8 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802983) |
| 12 | Torino–Lecce | ita.1 | 94.11% / 87.18% | 19/20; 10/10 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749748) |
| 13 | Göztepe–Başakşehir | tur.1 | 94.11% / 87.18% | 18/20; 8/8 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904798) |
| 14 | FK Zemun–FK IMT Beograd | srb.1 | 93.87% / 86.74% | 25/27; 12/14 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015170) |
| 15 | Paris Saint-Germain–Troyes | fra.1 | 93.79% / 86.58% | 16/18; 5/7 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802987) |
| 16 | Eyüpspor–Kocaelispor | tur.1 | 93.61% / 86.24% | 20/20; 10/10 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904795) |
| 17 | Pisa–Südtirol | ita.2 | 93.43% / 85.92% | 18/20; 8/9 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967148) |
| 18 | Železničar Pančevo–FK Radnički 1923 | srb.1 | 93.09% / 85.29% | 23/26; 12/14 | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015172) |
| 19 | Padova–Empoli | ita.2 | 93.08% / 85.27% | 19/20; 10/10 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967147) |
| 20 | Cardiff City–Burnley | eng.2 | 93.00% / 85.13% | 24/28; 12/14 | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836925) |
| 21 | Luzern–Young Boys | sui.1 | 92.93% / 84.99% | 21/23; 11/11 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804293) |
| 22 | Van Spor Kulübü–Sivasspor | tur.2 | 92.80% / 84.77% | 23/26; 12/13 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908701) |
| 23 | Virton–Beerschot | bel.2 | 92.73% / 84.63% | 17/17; 8/8 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015505) |
| 24 | FC Zlín–Slovácko | cze.1 | 92.37% / 83.98% | 26/26; 14/14 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831220) |
| 25 | Wrexham–Wolverhampton Wanderers | eng.2 | 92.34% / 83.92% | 24/28; 14/14 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836933) |
| 26 | Alverca–Casa Pia AC | por.1 | 91.75% / 82.88% | 17/20; 9/10 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887672) |
| 27 | Burgos CF–Celta Fortuna | esp.2 | 91.69% / 82.77% | 22/24; 11/12 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868526) |
| 28 | Valencia–Real Madrid | esp.1 | 91.53% / 82.49% | 20/22; 10/12 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868129) |
| 29 | Preston North End–Southampton | eng.2 | 91.04% / 81.65% | 23/28; 11/14 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836929) |
| 30 | Aberdeen–Hibernian | sco.1 | 91.03% / 81.62% | 19/21; 10/11 | 13 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793970) |
| 31 | Clermont Foot–Rodez | fra.2 | 90.97% / 81.52% | 22/22; 11/11 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5803309) |
| 32 | Hradec Králové–Sigma Olomouc | cze.1 | 90.76% / 81.17% | 25/26; 14/14 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831214) |
| 33 | Fortuna Sittard–Telstar | ned.1 | 90.46% / 80.63% | 20/22; 9/10 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781801) |
| 34 | Pendikspor–İstanbulspor | tur.2 | 90.28% / 80.34% | 24/26; 12/13 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908695) |
| 35 | Bristol City–Norwich City | eng.2 | 90.28% / 80.33% | 25/28; 13/14 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836924) |
| 36 | Galatasaray–Amed Sportif | tur.1 | 90.05% / 79.95% | 16/20; 8/9 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904800) |
| 37 | Westerlo–Sporting Charleroi | bel.1 | 89.83% / 79.57% | 20/20; 10/10 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811808) |
| 38 | FC Vaduz–Sion | sui.1 | 89.68% / 79.32% | 20/23; 11/11 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804297) |
| 39 | Nottingham Forest–Manchester City | eng.1 | 89.46% / 78.95% | 17/18; 7/8 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795513) |
| 40 | RAAL La Louvière–Cercle Brugge | bel.1 | 89.37% / 78.80% | 17/20; 9/10 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811804) |
| 41 | Watford–Middlesbrough | eng.2 | 89.31% / 78.69% | 23/28; 11/14 | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836931) |
| 42 | Keçiörengücü–Manisa Futbol Kulübü | tur.2 | 89.20% / 78.51% | 23/26; 11/12 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908694) |
| 43 | Córdoba–AD Ceuta FC | esp.2 | 89.18% / 78.48% | 22/24; 8/10 | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868529) |
| 44 | Mainz 05–Bayern München | ger.1 | 89.14% / 78.41% | 14/16; 7/8 | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881220) |
| 45 | Blackburn Rovers–West Ham United | eng.2 | 88.97% / 78.14% | 22/28; 12/14 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836923) |
| 46 | Charlton Athletic–Sheffield United | eng.2 | 88.97% / 78.14% | 25/28; 11/14 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836926) |
| 47 | Górnik Zabrze–Wieczysta Kraków | pol.1 | 88.72% / 77.72% | 22/26; 12/14 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804031) |
| 48 | Royal Antwerp–Lommel | bel.1 | 88.67% / 77.63% | 17/20; 8/10 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811800) |
| 49 | Mantova–Hellas Verona | ita.2 | 88.67% / 77.63% | 17/20; 7/9 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967146) |
| 50 | Aris Thessaloniki–Panetolikos | gre.1 | 88.55% / 77.43% | 17/18; 7/8 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5943289) |
| 51 | Francs Borains–RFC Seraing | bel.2 | 87.94% / 76.44% | 15/17; 6/8 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=1000015502) |
| 52 | FC Zbrojovka Brno–Baník Ostrava | cze.1 | 87.84% / 76.27% | 20/25; 11/13 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831213) |
| 53 | VfB Stuttgart–Werder Bremen | ger.1 | 87.83% / 76.27% | 15/16; 6/7 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881223) |
| 54 | Grasshopper–Lausanne | sui.1 | 87.73% / 76.10% | 20/23; 10/12 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804292) |
| 55 | Benevento–Avellino | ita.2 | 87.56% / 75.83% | 16/20; 8/10 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967140) |
| 56 | Ried–LASK | aut.1 | 87.46% / 75.66% | 20/21; 9/10 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5846019) |
| 57 | Nacional–Académico Viseu | por.1 | 87.43% / 75.63% | 16/20; 9/10 |  | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887676) |
| 58 | Borussia Dortmund–Elversberg | ger.1 | 87.31% / 75.42% | 13/16; 7/8 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881215) |
| 59 | Samsunspor–Konyaspor | tur.1 | 86.92% / 74.81% | 17/20; 7/9 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5904802) |
| 60 | West Bromwich Albion–Millwall | eng.2 | 86.29% / 73.81% | 23/28; 11/14 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836932) |
| 61 | Slovan Liberec–Bohemians 1905 | cze.1 | 86.26% / 73.76% | 23/25; 10/12 | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5831216) |
| 62 | OH Leuven–KV Mechelen | bel.1 | 86.14% / 73.57% | 18/20; 9/10 | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811803) |
| 63 | Portsmouth–Stoke City | eng.2 | 86.10% / 73.51% | 23/28; 10/14 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836928) |
| 64 | St. Mirren–Rangers | sco.1 | 85.84% / 73.11% | 19/22; 11/12 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793974) |
| 65 | Örgryte–Västerås SK | swe.1 | 85.76% / 72.97% | 44/53; 23/26 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5107656) |
| 66 | Holstein Kiel–Arminia Bielefeld | ger.2 | 85.60% / 72.73% | 17/20; 9/10 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881721) |
| 67 | ADO Den Haag–Sparta Rotterdam | ned.1 | 85.10% / 71.96% | 16/22; 9/10 | 10 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5781799) |
| 68 | Queens Park Rangers–Lincoln City | eng.2 | 84.93% / 71.70% | 21/28; 11/14 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5836930) |
| 69 | Fulham–Newcastle United | eng.1 | 84.24% / 70.65% | 16/18; 6/8 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795509) |
| 70 | 1. FC Nürnberg–Darmstadt | ger.2 | 84.21% / 70.60% | 18/20; 9/9 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881723) |
| 71 | Málaga–Racing Santander | esp.1 | 83.72% / 69.86% | 20/22; 9/10 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868125) |
| 72 | Rosenborg–Start | nor.1 | 83.54% / 69.60% | 34/49; 18/24 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5105048) |
| 73 | Hertha BSC–Wolfsburg | ger.2 | 83.49% / 69.53% | 16/20; 7/9 | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881720) |
| 74 | Cesena–Catanzaro | ita.2 | 82.56% / 68.15% | 15/20; 7/11 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5967142) |
| 75 | Paderborn–Eintracht Frankfurt | ger.1 | 82.46% / 68.01% | 13/16; 6/8 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881221) |
| 76 | Cagliari–Frosinone | ita.1 | 82.17% / 67.57% | 15/20; 6/8 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749741) |
| 77 | VfL Osnabrück–Kaiserslautern | ger.2 | 81.56% / 66.69% | 17/20; 7/10 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881725) |
| 78 | Austria Lustenau–Grazer AK | aut.1 | 81.52% / 66.64% | 17/21; 10/11 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5846017) |
| 79 | Dundee United–Motherwell | sco.1 | 80.79% / 65.58% | 20/22; 10/11 | 10 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793972) |
| 80 | RB Leipzig–Augsburg | ger.1 | 79.78% / 64.15% | 14/16; 6/7 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5881222) |
| 81 | Arsenal–Hull City | eng.1 | 78.94% / 62.97% | 14/18; 5/8 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5795505) |
| 82 | OB–Viborg | den.1 | 76.44% / 59.58% | 20/25; 9/12 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739542) |
| 83 | St. Johnstone–Dundee FC | sco.1 | 76.23% / 59.30% | 17/22; 8/11 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5793973) |
| 84 | Espanyol–Deportivo A Coruña | esp.1 | lipsă / lipsă | 20/21; 9/10 | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868124) |
| 85 | Osasuna–Athletic Club | esp.1 | lipsă / lipsă | 19/20; 8/9 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868126) |
| 86 | Roma–Sassuolo | ita.1 | lipsă / lipsă | 17/19; 8/10 | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5749740) |
| 87 | AEL Limassol–Krasava Ypsonas FC | cyp.1 | lipsă / lipsă | 0/0; 0/0 |  | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5975443) |
| 88 | Olympiakos Nicosia–Apollon Limassol | cyp.1 | lipsă / lipsă | 0/0; 0/0 |  | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5975447) |
| 89 | Korona Kielce–Radomiak Radom | pol.1 | lipsă / lipsă | 18/24; 9/12 | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804033) |
| 90 | Śląsk Wrocław–Piast Gliwice | pol.1 | lipsă / lipsă | 19/25; 10/12 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804037) |
| 91 | Widzew Łódź–Pogoń Szczecin | pol.1 | lipsă / lipsă | 20/24; 8/11 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5804038) |
| 92 | Cádiz–Real Oviedo | esp.2 | lipsă / lipsă | 20/22; 13/13 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868527) |
| 93 | Leganés–Albacete | esp.2 | lipsă / lipsă | 23/23; 11/11 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868532) |
| 94 | Granada–Tenerife | esp.2 | lipsă / lipsă | 23/24; 10/11 |  | cancelled | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868531) |
| 95 | Pau–Nantes | fra.2 | lipsă / lipsă | 19/21; 9/10 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5803315) |
| 96 | Saint-Étienne–Laval | fra.2 | lipsă / lipsă | 19/21; 8/9 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5803316) |
| 97 | NFC Volos–Kifisia FC | gre.1 | lipsă / lipsă | 13/17; 6/8 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5943294) |
| 98 | Csikszereda Miercurea Ciuc–Universitatea Cluj | rou.1 | lipsă / lipsă | 27/27; 12/12 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853545) |
| 99 | FCV Farul Constanța–Rapid București | rou.1 | lipsă / lipsă | 25/27; 12/12 | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853546) |
| 100 | FCSB–Corvinul Hunedoara | rou.1 | lipsă / lipsă | 23/27; 10/13 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5853549) |
| 101 | Muğlaspor–Esenler Erokspor | tur.2 | lipsă / lipsă | 21/25; 11/13 | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5908703) |
| 102 | Degerfors–GAIS | swe.1 | lipsă / lipsă | 49/52; 26/26 | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5107658) |

**Fail Parma–Bologna**: observație 8; rang original 1, ita.1, bază same_league_current_season. Brut 96.92%, AR20 92.75%. Istoric 17/20, teren 10/10; exact7=2, ≥8=3; varianță teren 1.3/2.3000000000000003. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Sporting CP–Moreirense**: observație 10; rang original 8, por.1, bază same_league_current_season. Brut 95.46%, AR20 89.79%. Istoric 19/20, teren 9/10; exact7=1, ≥8=1; varianță teren 11/3.5. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Železničar Pančevo–FK Radnički 1923**: observație 8; rang original 18, srb.1, bază same_league_current_season. Brut 93.09%, AR20 85.29%. Istoric 23/26, teren 12/14; exact7=0, ≥8=3; varianță teren 1.9047619047619044/6.476190476190477. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Cardiff City–Burnley**: observație 8; rang original 20, eng.2, bază same_league_current_season. Brut 93.00%, AR20 85.13%. Istoric 24/28, teren 12/14; exact7=0, ≥8=4; varianță teren 5.666666666666667/2.5714285714285716. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Aberdeen–Hibernian**: observație 13; rang original 30, sco.1, bază same_league_current_season. Brut 91.03%, AR20 81.62%. Istoric 19/21, teren 10/11; exact7=0, ≥8=2; varianță teren 2.6666666666666665/4.5. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Watford–Middlesbrough**: observație 9; rang original 41, eng.2, bază same_league_current_season. Brut 89.31%, AR20 78.69%. Istoric 23/28, teren 11/14; exact7=1, ≥8=5; varianță teren 9.904761904761905/7.9523809523809526. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Córdoba–AD Ceuta FC**: observație 9; rang original 43, esp.2, bază same_league_current_season. Brut 89.18%, AR20 78.48%. Istoric 22/24, teren 8/10; exact7=1, ≥8=2; varianță teren 18.799999999999997/1.7. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Mainz 05–Bayern München**: observație 9; rang original 44, ger.1, bază same_league_current_season. Brut 89.14%, AR20 78.41%. Istoric 14/16, teren 7/8; exact7=1, ≥8=2; varianță teren 1.6666666666666667/8.25. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Slovan Liberec–Bohemians 1905**: observație 9; rang original 61, cze.1, bază same_league_current_season. Brut 86.26%, AR20 73.76%. Istoric 23/25, teren 10/12; exact7=2, ≥8=2; varianță teren 0.5/9.61904761904762. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail OH Leuven–KV Mechelen**: observație 9; rang original 62, bel.1, bază same_league_current_season. Brut 86.14%, AR20 73.57%. Istoric 18/20, teren 9/10; exact7=4, ≥8=2; varianță teren 8.7/3.8. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail ADO Den Haag–Sparta Rotterdam**: observație 10; rang original 67, ned.1, bază same_league_current_season. Brut 85.10%, AR20 71.96%. Istoric 16/22, teren 9/10; exact7=3, ≥8=6; varianță teren 5.7/3.8. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Hertha BSC–Wolfsburg**: observație 9; rang original 73, ger.2, bază same_league_current_season. Brut 83.49%, AR20 69.53%. Istoric 16/20, teren 7/9; exact7=2, ≥8=4; varianță teren 6.25/12.7. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Dundee United–Motherwell**: observație 10; rang original 79, sco.1, bază same_league_current_season. Brut 80.79%, AR20 65.58%. Istoric 20/22, teren 10/11; exact7=2, ≥8=2; varianță teren 11.8/1.6. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

**Fail Roma–Sassuolo**: observație 8; rang original 86, ita.1, bază same_league_current_season. Brut lipsă, AR20 lipsă. Istoric 17/19, teren 8/10; exact7=1, ≥8=2; varianță teren 7.3/4.7. Acestea sunt semnale cunoscute înaintea meciului, nu explicații cauzale. Context aprofundat nu exista pentru întreg universul; cauza tactică a acestui rezultat rămâne necunoscută.

Reușite aproape de prag: Čukarički–Radnički Niš (7); Göztepe–Başakşehir (7); Virton–Beerschot (7); Galatasaray–Amed Sportif (7); Blackburn Rovers–West Ham United (7); Royal Antwerp–Lommel (7); Benevento–Avellino (7); St. Mirren–Rangers (7); Rosenborg–Start (7); Korona Kielce–Radomiak Radom (7).

Finalist analist original **NK Istra 1961–Rijeka**: success, observație 4; brut 96.65%, AR20 92.19%. Argumentele și riscurile rămân cele din raportul prospectiv, nu sunt rescrise după rezultat.

Toate succesele sunt prezentate în tabel, alături de avertismentele prospective; niciun succes nu validează probabilitatea estimată sau o cauză tactică. Un eșec nu demonstrează singur model greșit, iar un succes nu demonstrează calibrare.

## Ipoteze prospective, rezultate și limite

Luzern–Young Boys fusese preferat pentru GF70 propriu pe teren8/6 și13/5 și23/23istoric; riscurile erau reacția defensivăYB, loturi neconfirmate și Luzern90+2în cupă. Istra–Rijeka fusese preferat pentru22/22 și dispersie mică pe teren, cu riscul derbyului și presiunii ofensive declarat explicit. Rezultatul fiecăruia este în tabel; nu atribuim automat realizarea sau ratarea acelui scenariu unei tactici observate doar din scor.

Blackburn avea27/28 și14/14, GF70propriu9/7 și10/7, dar Parker instalat4noiembrie și recuperare3zileWestHam erau avertismente. Golul54 este compatibil cu ipoteza, nu dovadă de stabilitate sub noul antrenor. Excelsior avea21/22,11/11 și deplasăriGoAhead4,1,4,3,4,4; cele3cornere sunt compatibile cu acel profil. Riscul de reacție după înfrângeri mari și loturile probabile nu dispare după succes. Același Excelsior, lider brutGol70, a pierdutGol70la72: GF70propriu3/5 și0–9afectat de roșu fuseseră avertismente, fără demonstrație a cauzei.

Restanțele au fost recerute în older_retries.csv: Serbia nu este finală, douăFirstHalfCipru lipsesc. Nu există alte alegeri explicite vechi fără rezultat în CSV. Sursele independente nu confirmă fiecareFirstHalf; o singură sursă cu reconciliere internă nu devine confirmare independentă. Nu se completează rezultate din scorHT/FT și nu se fac modificări ale modelelor sau probabilităților arhivate.

## Coroborare și clarificări la 2026-11-08T10:44:54.006+02:00

[Sky Sports Blackburn](https://www.skysports.com/football/blackburn-rovers-vs-west-ham-united/8067885432187075232) confirmă primul gol54. [Cronologia Excelsior](https://www.skysports.com/football/excelsior-vs-go-ahead-eagles/556791) confirmă trei cornere în prima repriză, la4,40și41, și primul gol72. Aceasta coroborează independent ambele alegeri explicite; nu extindem verificarea la toate cele102partide. Copiile publice HTML și timestampurile sunt în Goal70_2026-11-08/Sources/.

Nacional–Académico Viseu este acum programat8noiembrie17:30RO în furnizor, nu rezultat verificat din7noiembrie. Snapshotul7noiembrie rămâne păstrat cu rezultatpending/not_final, fără rată imputată. Modele cornere:69/82,13failși1pending; toate observațiile84/98,14fail,3pendingși1anulat. Roma–Sassuolo8este fail verificat fără model, nu dispare din total. Finaliștii originali: Luzern primul22succes; Istra4cornere succes. Parma8șiSporting10sunt avertismente de dispersie și schimbări de context, nu dovadă de cauză tactică.
