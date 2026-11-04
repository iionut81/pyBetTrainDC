# Flux zilnic salvat — PRED, 3 noiembrie 2026

Cerinta utilizatorului: dashboard dinamic, cu analize si rezultate pastrate
pentru fiecare zi. Utilizatorul va comunica alegerea sa pentru cornere;
aceasta trebuie inregistrata si afisata ca **Alegerea ta**, urmata de **-Succes**
sau **-Fail** doar dupa verificarea rezultatului. Pana atunci: **-In asteptare**;
amanarile/anularile au etichete distincte. Raspunde in romana.
Nu cere un nou prompt, cote, lista de meciuri sau confirmarea inceperii.

## Analiza si inregistrare

Ordine obligatorie din5 noiembrie2026, clarificata explicit la4 noiembrie:
INAINTE de scanarea meciurilor curente, verifica rezultatele precedente,
analizeaza succesele/esecurile si reusitele la limita fata de predictiile
originale, apoi salveaza REVIEW.md si LESSONS_FOR_NEXT_SCAN.md in directorul
Review al zilei analizate. Lectiile includ dovezi, semnale cunoscute anterior,
cauze necunoscute, ipoteze de testat si verificari concrete pentru noul research.
Citeste si lectiile relevante acumulate. Abia apoi urmeaza pasii de research
de mai jos. Noul raport explica, cu dovezi, ce lectii au fost aplicate.
Timestampurile predarii review-ului si inceperii scanarii raman in run_report.
Daca exista lipsuri, salveaza review partial si limitele inainte de scanare;
nu inventa rezultate sau cauze. Fara schimbari automate de model/procente.

1. Citeste AGENTS.md, CLAUDE.md si promptul master. Pastreaza metoda Gol70 si
   cornere 1H, acoperirea integrala convenita, contextul finalistelor, sursele,
   excluderile si datele lipsa. Nu opri cercetarea la primul meci acceptabil.
2. Salveaza prospectiv in `Analysis/Goal70_YYYY-MM-DD/` si
   `Analysis/Corners1H_YYYY-MM-DD/`: numaratori, formule, istoric, dovezi,
   data/ora RO si starea meciurilor. Gol70 inseamna minut afisat <=70, inclusiv
   45+; cornere inseamna totalul FirstHalf <=7. Nu folosi cote de bookmaker.
3. Arhiveaza lista completa in `YYYY-MM-DD_goal70_candidates.csv` si
   `YYYY-MM-DD_corners1h_candidates.csv` din acest director. Copiaza schema
   zilei precedente, identificatorii furnizorilor, kickoff RO si ora snapshotului.
   Completeaza separat jurnalul `YYYY-MM-DD.md`, inclusiv NO BET si lipsurile.
   Constructorul poate citi alternativ `Corners1H_YYYY-MM-DD/all_models.csv`.
4. Pastreaza distinct screeningul, estimarea finala si frecventele istorice.
   Finalistii cornerelor sunt cititi din `prospective_selection.csv` si, daca
   exista, `prospective_alternative.csv`. Nu rescrie procentele initiale.
5. Cand utilizatorul comunica o alegere, adauga idempotent un rand in
   `user_selections.csv`: selection_id unic (data + piata + match_id), momentul
   inregistrarii RO, match_id/provider, piata, kickoff, snapshotul estimarii,
   probabilitate/stres/cota matematica, analysis_path si sursa exacta existenta.
   Foloseste estimarea finala verificata daca exista. Nu inventa ora istorica
   a mesajului: timestampul este momentul inregistrarii in proiect.
   Rezultatul, miza, pretul disponibil si profitul raman goale daca nu sunt cunoscute.
6. Alegerea utilizatorului si candidatul analistului sunt roluri diferite.
   Cand utilizatorul alege explicit cate un eveniment pe ambele piete, salveaza
   ambele separat. Aceasta nu este autorizare de pariere sau bilet combinat.
   Recomandarea principala implicita ramane una; NO BET este permis.

## Rezultate si lectii

7. La revenire, verifica rezultatele din surse publice identificabile: primul
   gol valid/minutul sau 0-0 pentru Gol70; totalul FirstHalf pentru cornere.
   Pastreaza minutele aditionale, amanarile si conflictele. Nu deduce cornerele
   1H din FT. Missing nu este zero. Nu converti relatarea unui succes in rezultat
   verificat fara sursa; noteaz-o separat pana la verificare.
8. Salveaza rezultatele universului in
   `Analysis/Review_YYYY-MM-DD/goal70_outcomes.csv` si
   `corners1h_outcomes.csv`, cu id-ul meciului/crosswalk, result, observatia,
   result_source si momentul verificarii. Pentru cornere foloseste explicit
   `actual_corners_1h`; pentru gol `first_goal_minute`. Rezultate acceptate:
   success, fail, pending, postponed_not_scored, void/cancelled.
   Actualizeaza separat campurile de rezultat din jurnalul alegerilor; pastreaza
   probabilitatile si timestampurile prospective. Conflictele de rezultat opresc
   construirea dashboardului, in loc sa fie ascunse.
9. Compara succesele, esecurile si rezultatele aproape de prag. Distinge motivul
   tactic, fluctuatiile normale si datele slabe. Datele aflate dupa meci nu devin
   predictori pentru ziua respectiva. O zi buna/slaba nu valideaza/invalideaza
   modelul; schimbarile de model cer evaluare prospectiva si autorizare separata.

## Dashboard si continuitate

10. Dupa analiza, alegere noua sau rezultate, ruleaza:

    `node Previews/Pred_Research_Terminal/build.mjs`

    Constructorul descopera toate jurnalele datate si listele canonice, fara
    editarea codului pentru o zi noua. Produce interfata cu date incluse,
    `dist/snapshot.json` si cate un `dist/data/YYYY-MM-DD.json`.
    Nu ruleaza modele, nu descarca date si nu modifica sursele analitice.
11. Verifica numaratorile si rolurile afisate; cand schimbi interfata, ruleaza
    `node Previews/Pred_Research_Terminal/check-preview.mjs`. Deschide sau
    reincarca `Previews/Pred_Research_Terminal/dist/index.html` in browser.
    Selectorul principal schimba ziua analizata; Istoric are zi si piata
    separate. Evaluarea agrega numai rezultate verificate, separat pe piete;
    alegerile utilizatorului sunt raportate separat de universul candidatilor.
12. Alegerile utilizatorului apar PRIMELE in lista zilei/pietei respective,
    apoi restul candidatilor in ordinea modelului. Pastreaza rangul original
    al modelului: de exemplu Mallorca poate fi prima afisata si avea rang 21.
    Filtrele active se respecta; nu rescrie clasamentul sau probabilitatile.
    Dupa fiecare actualizare manuala lasa in raspuns linkul absolut clicabil
    catre `Previews/Pred_Research_Terminal/dist/index.html`.

Pe 3 noiembrie: QPR-Burnley Gol70 si Derby County-Norwich City cornere sunt
alegerile utilizatorului, ambele cu rezultat necompletat. Pe 2 noiembrie:
Mallorca-Cadiz Gol70 si Sassuolo-Fiorentina cornere sunt succese verificate.
Universul cornerelor de pe 2 noiembrie este verificat doar partial; nu declara
toata lista castigatoare pe baza rezultatului Sassuolo.

Actualizare explicita din 3 noiembrie: utilizatorul a autorizat rularea zilnica
la 08:00 Europe/Bucharest. Sarcina Windows `Pred_Daily_Research_0800` a fost
inregistrata si urmatoarea rulare verificata este 4 noiembrie 2026, 08:00 RO.
Runner: `automation/run_daily_research.ps1`; instructiuni executabile pentru
agent: `automation/daily_research_prompt.md`; detalii si conditii in
`automation/README_daily_research.md`. Aceasta cerere inlocuieste limitarea
anterioara la lucru manual pentru fluxul respectiv.

La 08:00 incepe verificarea alegerilor si analizelor zilei precedente, apoi
bilantul si lectiile, apoi cercetarea AMBELOR piete pentru ziua curenta si
actualizarea dashboardului. Ora 08:00 este inceputul lucrului, nu promisiunea
ca toate cercetarile sunt deja terminate atunci. Lipsurile raman explicite.
Calculatorul trebuie pornit, online si utilizatorul autentificat in Windows;
Codex CLI foloseste autentificarea existenta. Daca ora este ratata, aceeasi
sarcina are StartWhenAvailable plus declansatoare la logon, deblocare, revenire
din sleep si conectarea retelei. Nu ruleaza research inainte de 08:00 si nu
repeta un research deja incheiat in acea zi. La final continua conversatia
existenta a utilizatorului pentru a publica rezumatul si linkul catre HTML.
Un esec de livrare se noteaza si permite reincearcarea livrarii fara repetarea
cercetarii. Nu exista mesaje catre alte persoane sau pariuri automate. Un HTML
deja deschis necesita reincarcare.
Continuitatea pe alt calculator cere copierea proiectului si reinstalarea
sarcinii; nu este memorie universala intre chaturi.
