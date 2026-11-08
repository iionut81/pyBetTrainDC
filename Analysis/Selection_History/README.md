# Istoricul selectiilor: Gol70 si cornere 1H

Punct de reluare pentru analizele discutate cu utilizatorul.

- `user_selections.csv`: alegerile declarate de utilizator, cu probabilitatile
  originale, ora inregistrarii, piata si campuri separate pentru rezultat.
- `YYYY-MM-DD.md`: rezumatul zilei, alegerile utilizatorului, recomandarile
  analistului si lista completa a candidatilor pentru gol pana la 70.
- `YYYY-MM-DD_goal70_candidates.csv`: instantaneul tuturor candidatilor zilei,
  cu ordinea modelului si campuri necompletate pentru rezultatele viitoare.

Prima zi: [2 noiembrie 2026](2026-11-02.md).

Analiza prospectiva urmatoare: [3 noiembrie 2026](2026-11-03.md).
Sunt salvati 20 candidati calculati si o singura selectie a analistului,
QPR-Burnley GOAL_BY_70, confirmata si de utilizator la 10:47 RO.
Rezultatele raman necompletate; estimarile originale sunt pastrate.

La cererea suplimentara a utilizatorului, aceeasi zi include si analiza
cornerelor 1H: Derby-Norwich este candidatul analistului, cu o observatie
conflictuala exclusa din estimarea finala. Cele 20 de partide de screening
sunt in `2026-11-03_corners1h_candidates.csv`; 19 au model calculabil.
La 11:59 RO utilizatorul a confirmat Derby-Norwich ca alegerea sa pentru
cornere. Este inregistrata separat in `user_selections.csv`, cu estimarea
finala originala si rezultatul inca necompletat. QPR-Burnley ramane alegerea
sa pentru Gol70.

Rezultate verificate la 3 noiembrie:
[retrospectiva completa a zilei de 2 noiembrie](../Review_2026-11-02/REVIEW.md).
Alegerile utilizatorului sunt ambele reusite; lista Gol70 are 22/26 succese,
4 esecuri si o amanare exclusa din rata. Procentele initiale sunt pastrate.

O intentie declarata nu inseamna pariu plasat. Nu inventa miza, cota disponibila,
rezultat sau profit. Nu confunda alegerea utilizatorului cu prima recomandare
a analistului. Rezultatele se completeaza ulterior din surse verificabile,
pastrand separat observatia reala, eticheta succes/esec si sursa/momentul verificarii.
Nu suprascrie procentele salvate inainte de meci dupa aflarea rezultatului.

Definitii: GOAL_BY_70 = minimum un gol valid al oricarei echipe cu minutul
afisat <=70, inclusiv timpul aditional al primei reprize. CORNERS_1H_UNDER_7_5
= totalul cornerelor ambelor echipe la pauza <=7, inclusiv 45+.
Un camp lipsa nu inseamna zero. Daca partida nu se joaca sau sursa este
insuficienta, nu o eticheta automat succes/esec.

Continuitatea necesita acces la acest proiect si la AGENTS.md. Istoricul in
sine nu programeaza executii; ulterior utilizatorul a autorizat explicit
sarcina `Pred_Daily_Research_0800`, activa zilnic la 08:00 RO din 4 noiembrie.
Detalii si conditii in `automation/README_daily_research.md`.

Fluxul zilnic si regulile dashboardului, confirmate la 3 noiembrie, sunt in
[DAILY_WORKFLOW.md](DAILY_WORKFLOW.md). Interfata locala este in
`Previews/Pred_Research_Terminal/dist/index.html`; selectorul de data descopera
toate zilele salvate la reconstruire. Dupa fiecare analiza, alegere comunicata
sau verificare de rezultate, reconstruim interfata in aceeasi sesiune.


5 noiembrie2026: [jurnal](2026-11-05.md), [review4 noiembrie](../Review_2026-11-04/REVIEW.md), [Gol70](../Goal70_2026-11-05/EUROPE_ALL_MATCHES.md), [cornere](../Corners1H_2026-11-05/ANALYSIS.md).35 partide,33/21 modele; research parțial. Candidați analistPalace–Hoffenheim șiCelta–Union; alegerile expliciteMjällby–Borac/Celta–Union sunt distincte, pending.


8 noiembrie 2026: [jurnal](2026-11-08.md), [review7noiembrie](../Review_2026-11-07/REVIEW.md), [Gol70](../Goal70_2026-11-08/EUROPE_ALL_MATCHES.md), [cornere](../Corners1H_2026-11-08/ANALYSIS.md).105partide,103/81modele; research parțial. Candidați Cambuur–PSV/Girona–Almería, preferință individuală Girona; nicio alegere nouă inferată.
