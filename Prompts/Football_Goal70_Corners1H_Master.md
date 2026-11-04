# Model portabil: gol pana la 70 si sub 7,5 cornere prima repriza

Versiune: 1.6. Data: 2026-11-03. Fus orar: Europe/Bucharest.

Clarificare de acoperire, 2 noiembrie: verifica integral primele ligi masculine
ale primelor 20 de asociatii UEFA, precizand anul clasamentului. Adauga separat
Romania, Serbia si diviziile secunde europene urmarite in proiect; nu confunda
vechea lista de 20 ligi (care include Arabia Saudita si divizii secunde) cu
primele 20 de tari UEFA. Calculeaza toate partidele eligibile inainte de top.
Raporteaza separat inventarul programului, calculul statistic complet si
verificarea contextuala aprofundata a finalistelor. Pentru ligi fara partide
azi nu este necesara colectarea istoricului intreg in acea zi.

Exemplu complet si reproductibil, fara schimbarea modelului de productie:
`Analysis/Goal70_2026-11-02/EUROPE_ALL_MATCHES.md`. Sursele publice ESPN si FotMob
au functionat la aceasta analiza ca alternative dupa blocarea Sofascore.
Verifica actualitatea si completitudinea lor la fiecare reluare; niciun endpoint
gol sau HTTP 400 nu dovedeste singur ca liga nu are partide.

## Scop si continuitate

Clarificare4 noiembrie2026, aplicabila din5 noiembrie: inaintea scanarii si
pickurilor noi urmeaza obligatoriu review-ul rezultatelor, analiza succeselor,
esecurilor si reusitelor la limita fata de snapshoturile prospective, apoi
salvarea lectiilor verificabile si a limitelor in
Analysis/Review_YYYY-MM-DD/LESSONS_FOR_NEXT_SCAN.md (data predictiilor revizuite).
Researchul zilei curente citeste aceste lectii si documenteaza verificarile
concrete aplicate candidatilor. Detalii si ordinea in
automation/daily_research_prompt.md. Nu modifica automat ponderi, procente,
calibrare sau filtre dupa rezultate; ipotezele necesita testare separata.

Actualizare de functionare confirmata la 3 noiembrie: urmeaza
`Analysis/Selection_History/DAILY_WORKFLOW.md`. Salveaza fiecare zi si toate
listele analizate, actualizeaza alegerile declarate de utilizator (inclusiv
alegerea sa distincta pentru cornere), apoi rezultatele verificate si bilantul.
Dupa fiecare actualizare regenereaza dashboardul local cu
`node Previews/Pred_Research_Terminal/build.mjs`. Nu hardcoda ziua sau alegerile.
Pe 3 noiembrie utilizatorul a confirmat QPR-Burnley Gol70 si Derby-Norwich
cornere 1H. Inregistreaza ambele alegeri explicite separat; limita implicita
a recomandarii principale nu justifica omiterea unei alegeri declarate.
Nu rescrie estimarile initiale dupa rezultate. Cererea explicita ulterioara
din 3 noiembrie autorizeaza sarcina zilnica Windows `Pred_Daily_Research_0800`,
incepand cu 4 noiembrie, 08:00 RO. Urmeaza `automation/daily_research_prompt.md`:
verificarea alegerilor si rezultatelor de ieri, lectii, apoi research AMBELE
piete si dashboard. Nu crea duplicate sau alte programari. Etichetele alegerii
utilizatorului primesc -Succes/-Fail numai dupa verificare; altfel raman in
asteptare, amanate sau anulate. Un candidat analist nu devine automat alegere.

Stil de discutie confirmat la 2 noiembrie: raspunde in romana, autonom, cu
alegerea si explicatia concreta a datelor, pozitiei in clasament, limitelor si
contraargumentelor. Nu cere utilizatorului cote, indicii de ligi sau un nou
prompt pentru fiecare analiza. Cand solicita o alternativa, compara candidatii
deja verificati si actualizeaza dovezile necesare, fara a inventa disponibilitatea
unei piete. Pastreaza aceeasi formula si separarea intre frecvente si model.

Istoricul zilnic este indexat in `Analysis/Selection_History/README.md`.
Pastreaza lista completa analizata si alegerile utilizatorului separat de
recomandarile analistului. Pentru 2 noiembrie utilizatorul a ales explicit
Mallorca-Cadiz Gol70 si Sassuolo-Fiorentina sub 7,5 cornere 1H; cele doua alegeri
ale zilei nu modifica implicit protocolul general. Inregistreaza intentia,
ora, estimarile originale si rezultatele initial necompletate. Nu presupune
pariu plasat, miza, pret, bilet combinat sau profit. Nu rescrie clasamentul ori
procentele dupa aflarea rezultatelor.

Acest document pastreaza metoda ceruta de utilizator, separat de o selectie
concreta sau de afirmatiile numerice dintr-o conversatie. Copiaza proiectul,
inclusiv AGENTS.md, acest document, scriptul de audit si istoricul, pe noul
calculator. Intr-o conversatie fara acces la proiect trebuie atasate sau puse
la dispozitie aceste fisiere. Nu promite memorie automata intre conversatii.

Cerere scurta de reluare:

> Citeste AGENTS.md si Prompts/Football_Goal70_Corners1H_Master.md. Verifica
> acoperirea si actualitatea datelor, apoi aplica metoda tuturor ligilor cu
> istoric utilizabil. Pentru azi, verifica integral programul acelor ligi,
> clasifica separat cele doua evenimente si documenteaza toate excluderile.
> Nu cere cote si nu opri cercetarea dupa primul candidat acceptabil.

## Decizia utilizatorului: protocol zilnic din 2026-11-02

Decizie confirmata in conversatia din 2026-11-01, de pastrat pentru reluare:

- Maximum UN meci / O selectie individuala pe zi, in total pe cele doua piete:
  minimum un gol pana la minutul 70 inclusiv SAU sub 7,5 cornere in prima
  repriza inclusiv timpul aditional. Nu recomanda doua conditii combinate,
  nici in acelasi meci, nici in meciuri diferite, doar pentru cresterea cotei.
- NO BET daca datele sau argumentele sunt insuficiente. Nu exista obligatia
  unei selectii zilnice. Compara tot universul eligibil inainte de alegere.
- Preferinta utilizatorului: cota disponibila peste 1,10 si sanse mari de
  reusita. Nu cere cote utilizatorului si nu folosi cote de bookmaker pentru
  selectie sau calculul probabilitatilor. Fara observarea unui pret real,
  nu confirma disponibilitatea cotei si nu declara avantaj de pariere.
- Judeca relatia probabilitate-pret: pragul matematic este p > 1/cota,
  inainte de costuri. Exemple: 1,10 cere peste 90,91%; 1,12 peste 89,29%;
  1,15 peste 86,96%. Aceste praguri nu sunt probabilitati ale unui meci.
  Cota echitabila 1/p nu este o oferta disponibila. Incertitudinea estimarii
  poate elimina un avantaj aparent foarte mic; nu inventa o marja validata.
- Nu afirma o rata demonstrata de 90% pentru Gol70. Nu transforma o medie
  ipotetica de 3 esecuri la 30 pariuri intr-un plafon lunar sau o promisiune.
  Reusitele pot coexista cu pierdere neta la cote prea mici.
- Etapa propusa ramane validarea fara bani reali: salveaza predictia, datele,
  ipotezele, sursele si scenariul nefavorabil inainte de kickoff, apoi rezultatul.
  Nu recomanda all-in, progresii de recuperare sau venit zilnic garantat.
- Documentarea preferintei nu programeaza o executie automata. Nu promite
  analiza sau notificare de maine fara o programare efectiv creata si verificata.
  La reluarea solicitarii, aplica direct protocolul fara a cere din nou cote,
  lista de meciuri, indicii de ligi sau confirmarea inceperii analizei.

Comparatia si dovezile existente sunt pastrate in
`Analysis/Football70_Corners1H_Audit_2026-11-01/RESEARCH_SINGLE_VS_DOUBLE.md`.

## Date obligatorii si identitate

- match_id stabil, sursa, data extragerii, competitie, liga, sezon, gazda,
  oaspete, kickoff cu fus orar, status si momentul verificarii.
- Pentru goluri: fiecare gol valid, echipa creditata, minutul de baza,
  timpul aditional si repriza; scorul la 90 minute pentru reconciliere.
  Autogolul se atribuie echipei beneficiare. Se exclud golurile anulate,
  reprizele suplimentare eliminatorii si loviturile de departajare.
- Pentru cornere: obtinute si cedate de fiecare echipa la sfarsitul primei
  reprize, inclusiv 45+; sursa perioadei trebuie explicita. Cand exista,
  se pastreaza si cronologia cornerelor si totalurile la 90 minute.
- Lipsa datelor este NULL, nu zero. Se exclud duplicatele si conflictele
  nerezolvate; aceeasi partida apare o singura data la agregare.
- Nu se confunda FHG (gol in prima repriza) cu gol pana la 70. Nu se deduc
  cornerele primei reprize prin impartirea totalului la doi.
- Fisierele vechi trebuie auditate: existenta lor nu demonstreaza actualitate,
  acoperire completa a sezonului sau verificarea independenta a sursei.

## Protocol de cautare si selectie

1. Inventariaza toate ligile cu date; raporteaza numarul de meciuri, perioada,
   ultima actualizare si completitudinea campurilor, separat pe eveniment.
2. Pentru o analiza zilnica, obtine programul complet de azi in Romania pentru
   toate ligile inventariate; exclude partidele incepute si orele neconfirmate.
3. Calculeaza pentru fiecare partida eligibila, apoi formeaza lista scurta.
   Candidatii exclusi primesc un motiv explicit. Evalueaza si contraargumentele.
4. Verifica independent competitia, miza, clasamentul, egalul acceptabil,
   antrenorii, tactica, loturile, odihna, deplasarea, vremea si terenul.
   Separarea titularilor probabili de cei confirmati este obligatorie.
5. Stabileste topul dupa compararea finalistelor. Nu eticheta o selectie drept
   cea mai buna a zilei fara sa precizezi universul cercetat si limitarile.
6. Pastreaza modelul statistic separat de analiza contextuala. Fara bonusuri
   arbitrare pentru accidentari, motivatie, tactica sau vreme.

## Lectii documentate dupa verificarea zilei de 2 noiembrie

Retrospectiva: `Analysis/Review_2026-11-02/REVIEW.md`. Lista avea 27 de
candidati: 26 jucate, 22 succese Gol70, 4 esecuri si o amanare. Cele doua
alegeri ale utilizatorului au fost reusite. Acesta este bilantul unei zile,
nu o rata validata sau dovada profitabilitatii.

- Eticheteaza separat inventarul complet, finalistii analistului si alegerea
  utilizatorului. Nu numi toate partidele inventariate recomandari ferme.
- La finalisti, arata GF70/meci al fiecarei echipe pe teren, frecventa
  partidelor fara gol la 70 si deschiderile tarzii. Semnaleaza diferentele
  dintre model si frecventa observata, chiar daca partida a castigat anterior.
- Afiseaza sensibilitatea numai-sezon/numai-teren si la meciuri cu eliminari.
  Nu schimba ponderile 50/50 dupa rezultatul unei zile. Compara variantele
  cronologic pe o perioada separata inainte de orice schimbare de model.
- Marcheaza reusitele apropiate de prag si amanarile. Un meci amanat fara
  rezultat nu este un 0-0 jucat si nu intra in numitorul succes/esec.
- Golurile si statisticile culese dupa partida explica rezultatul; nu sunt
  date predictive disponibile ieri. FT nu descrie exclusiv intervalul 0-70.
- Un viitor model pentru timpul PRIMULUI gol ramane ipoteza de cercetare,
  fara modificarea productiei sau procente noi arbitrare.

## Gol pana la minutul 70 inclusiv: baseline Brentford

A = gazda; B = oaspete. Foloseste numai campionatul actual pentru numaratori.
GF70 si GA70 numara golurile valide cu minutul afisat <=70, inclusiv cele
din 45+; nu includ 71 sau mai tarziu. Nu mai aplica un factor 70/90.

```
lambda_sezon = ((GF70_A/n_A + GA70_B/n_B)
              + (GF70_B/n_B + GA70_A/n_A)) / 2
lambda_teren = ((GF70_A_acasa/n_A_acasa + GA70_B_deplasare/n_B_deplasare)
              + (GF70_B_deplasare/n_B_deplasare + GA70_A_acasa/n_A_acasa)) / 2
lambda_combinat = 0.5 * lambda_sezon + 0.5 * lambda_teren
p_model = 1 - exp(-lambda_combinat)
p_stres = 1 - exp(-0.8 * lambda_combinat)
cota_echitabila = 1 / p_model
```

Arata numaratorile, numitorii, rezultatele numai-sezon si numai-teren.
Ponderile sunt euristice, iar esantioanele se suprapun. Tinta orientativa
85-90% nu este un prag care autorizeaza modificarea calculelor.

Prezinta ultimele zece meciuri oficiale verificabile ale fiecarei echipe:
data, competitie, gazda/oaspete, scor la 90 minute, primul gol, rezultat <=70.
Include explicit 0-0 si deschiderile dupa 70; separa cupa de campionat.
Compara frecventa primului gol cu modelul. Golurile de dupa deschiderea
scorului pot umfla lambda fara sa imbunatateasca predictia primului gol.

Doar minutul primului gol permite evaluarea evenimentului binar. Formula
Brentford necesita TOATE minutele golurilor, cu atribuire pe echipe.
Scorul la pauza si scorul final nu pot reconstrui aceste numaratori.

## Sub 7,5 cornere in prima repriza

Succes = suma cornerelor ambelor echipe la pauza <=7. Esuare = >=8.
Sapte cornere sunt castigatoare, dar trebuie semnalate ca apropiere de limita.

Calculeaza baseline-ul de mai sus inlocuind GF70/GA70 cu cornerele obtinute/
cedate in prima repriza. Foloseste campionatul actual si configuratia terenului.

```
p_model = exp(-lambda_combinat) * sum(lambda_combinat**k / k!, k=0..7)
p_stres = PoissonCDF(7, 1.2 * lambda_combinat)
cota_echitabila = 1 / p_model
```

Arata istoricul totalurilor la pauza, mediile obtinute/cedate, frecventa >=8,
frecventa exact 7, dispersia si diferentele acasa/deplasare. Golul timpuriu
poate reduce sau amplifica presiunea; nu presupune automat scaderea cornerelor.
Modelele Poisson si binomial negativ pot fi comparate prin backtest; nu se
schimba distributia sau parametrii dupa observarea rezultatului dorit.

## Testare istorica si evaluare

- Prezice cronologic: pentru fiecare meci foloseste numai date dinaintea
  acelei zile. Calculeaza toate predictiile zilei inainte de actualizarea
  istoricului cu rezultatele zilei. Izoleaza liga si sezonul.
- Pentru baseline-ul exploratoriu se fixeaza dinainte minimum 5 meciuri de
  sezon per echipa si minimum 3 in configuratia terenului. Se raporteaza
  excluderile; pragurile nu sunt o garantie de suficienta statistica.
- Raporteaza numarul de predictii, hit rate, probabilitatea medie, Brier,
  log loss, grupe de probabilitate si rezultate pe liga. Compara top 1/top 3
  zilnic cu universul eligibil, nu numai meciurile care au castigat.
- Distingerea calibrarii pe aceleasi predictii de validarea independenta
  este obligatorie. Calibrarea si pragurile se invata exclusiv pe trecut,
  apoi se evalueaza pe o perioada ulterioara neatinsa.
- Nu prezenta rata de reusita ca profit. Fara cote reale nu exista ROI
  masurat si nici confirmare ca piata ofera minimum 1,10.
- Tactica si accidentarile istorice necapturate la momentul respectiv nu
  se adauga retrospectiv pentru a cosmetiza testul statistic.

## Starea constatata la 2026-11-01

Audit reproductibil: `audit_goal70_corners_1h.ps1 -AsOf 2026-11-01`.
Rapoarte: `Analysis/Football70_Corners1H_Audit_2026-11-01/`.

Analiza esecurilor si corectia exploratorie a probabilitatilor sunt documentate
in `Analysis/Football70_Corners1H_Audit_2026-11-01/FAILURE_ANALYSIS.md`.
Recalibrarea globala monotona nu imbunatateste ordinea Top 1; nu se confunda
procentele afisate mai realiste cu mai multe selectii castigate. Strategia
GOAL_BY_70 nu are inca o rata de succes validata pe istoricul complet.

- Istoric cornere 1H existent: `simulations/Corners 1H/data/corners_1h_history.csv`.
- Model 1H existent: `train_corners_1h.py`, `run_corners_1h_daily.py`.
- Istoricul FHG existent retine scoruri FT/HT, nu toate minutele golurilor.
- `football-goal70-2026-10-14.csv` este un esantion manual cu sase echipe,
  fara liga/sezon/perioada explicite; nu este o baza completa pe campionate.
- `sofascore_loader.fetch_goal_minutes` poate cere evenimente, dar existenta
  functiei nu inseamna ca evenimentele sunt deja stocate. La import trebuie
  validate perioada, golurile anulate si reconcilierea scorului la 90 minute.
- Codul calibrarii existente a cornerelor ajusteaza Platt pe aceleasi
  predictii walk-forward pe care raporteaza scorurile calibrate. Aceste
  scoruri calibrate nu reprezinta o validare independenta out-of-sample.
- Nu rula automat joburi de productie sau retraining; nu modifica modelele
  existente sau motorul inghetat de selectie pentru a face acest audit.

## Raport pentru utilizator

Incepe cu meciul, ora Romaniei, evenimentul si probabilitatea modelului.
Precizeaza momentul actualizarii, numarul de ligi si partide cercetate,
numitorile, frecventele observate, scenariul nefavorabil, cota matematica,
sursele, riscul principal si alternativele respinse. Daca lipsesc date,
spune exact ce lipseste si ce se poate calcula deja. Nu forta selectii.
