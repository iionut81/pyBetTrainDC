# Audit al istoricului: gol pana la 70 si cornere sub 7,5 la pauza

Data de referinta: 1 noiembrie 2026, Europe/Bucharest.
Aceasta este o evaluare offline; nu contine selectii noi pentru astazi.

Analiza ulterioara a celor 46 de esecuri Top 1 si testul unei calibrari
invatate exclusiv pe perioada anterioara sunt in `FAILURE_ANALYSIS.md`.
Comparatia single/dubla pe aceleasi 323 zile si cercetarea externa despre
cele trei combinatii sunt in `RESEARCH_SINGLE_VS_DOUBLE.md`.

## Ce s-a facut

S-au verificat fisierele locale si s-a aplicat baseline-ul solicitat, cu 50%
sezon si 50% configuratie acasa/deplasare, tuturor ligilor cu date eligibile
de cornere in prima repriza. Nu s-au folosit cote, nu s-au ajustat praguri pe
rezultate si nu s-au modificat modelele existente sau calibrarea de productie.
Istoricul local nu fusese folosit in calculele manuale prezentate anterior
in conversatia din 1 noiembrie; acele calcule nu erau rezultate ale acestui test.

## Acoperire constatata

| Istoric | Observatii | Limita |
|---|---:|---|
| Cornere 1H brute | 10.506, in 20 ligi | 29 randuri au inconsistente intre reprize si total |
| Cornere 1H utilizabile | 10.477 meciuri unice | Acoperire partiala; nu sunt sezoane complete demonstrate |
| Gol70 manual | 56 observatii, 55 partide distincte, sase echipe | Fara liga/sezon/perioada explicite si fara istoricul complet al adversarilor |
| FHG | 11.596 randuri, 11 ligi | Scor FT/HT; nu furnizeaza toate minutele golurilor |

Cele 20 ligi: B1, B2, D1, D2, DK1, E0, E1, F1, I1, I2, N1, P1,
RO1, RS1, SA1, SP1, SP2, SW1, TR1, TR2.

Istoricul cornerelor se intinde intre 2021-08-27 si 2026-05-20, cu ultimele
date pe liga in aprilie-mai 2026. Nu a fost actualizat pentru noiembrie.
Cele sase echipe din fisierul manual sunt Bodo, City, Dortmund, Norwich,
PSG si Southampton. Numaratorile structurale nu constituie o reverificare
independenta a veridicitatii meciurilor sau a surselor externe.

Vezi `corners_1h_coverage_by_league.csv`, `goal70_manual_coverage.csv`,
`fhg_coverage_not_goal70.csv` si `source_manifest.csv` pentru detalii.
Manifestul pastreaza dimensiunile, antetele si hashurile SHA256 ale intrarilor.

## Test nou pe cornere: rezultate

Conditii fixate inainte de rulare: minimum cinci meciuri anterioare in sezon
pentru fiecare echipa si minimum trei in configuratia terenului. Sezonul si
liga sunt izolate. Toate predictiile unei zile sunt calculate inainte de
introducerea oricarui rezultat din acea zi. Au fost excluse 5.914 meciuri
pentru istoric anterior insuficient; nu s-au completat lipsurile cu zero.

| Populatie | Predictii | Reusite | Rata observata | Probabilitate medie estimata |
|---|---:|---:|---:|---:|
| Toate meciurile eligibile | 4.563 | 4.078 | 89,37% | 89,42% |
| Prima alegere a fiecarei zile eligibile | 413 | 367 | 88,86% | 94,27% |
| Pana la trei alegeri pe zi | 1.006 | 908 | 90,26% | 93,40% |

Top 3 inseamna pana la trei meciuri individuale, nu un bilet combinat.
Clasamentul a folosit probabilitatea bruta descrescator; departajare
determinista dupa liga si echipe pentru probabilitati egale la 12 zecimale.
Aceasta precizie evita schimbarea ordinii la export/import CSV; nu modifica
probabilitatile calculate sau pragurile. Nu include filtrul tactic manual din chat.

Prima alegere nu a imbunatatit rata de succes fata de universul eligibil.
In plus, modelul a supraestimat probabilitatea primelor alegeri cu aproximativ
5,4 puncte procentuale. Asadar, o valoare calculata de 94-97% nu poate fi
prezentata ca probabilitate reala validata pentru un meci nou.

In grupa 95-100%, 728 din 804 predictii au reusit: 90,55% fata de probabilitatea
medie estimata de 96,59%. O medie globala bine aliniata poate ascunde erori
importante chiar in zona din care sunt alese primele recomandari.

Metrici globale: Brier 0,09682, log loss 0,35062. Rezultatele pe liga si
grupele de probabilitate sunt salvate separat. Nu s-a cautat retrospectiv
un prag care sa faca rezultatele sa para mai bune.

## Limitari ale interpretarii

- Acesta este un backtest cronologic exploratoriu, necalibrat. Nu este un
  studiu prospectiv si nu demonstreaza profitabilitate.
- Universul topului zilnic cuprinde doar meciurile prezente in istoricul
  disponibil si eligibile statistic, nu programul complet reconstruit al zilei.
- Numele echipelor sunt normalizate exact; nu s-au ghicit aliasuri. Asta poate
  reduce esantionul. Este nevoie de o verificare a acoperirii si a selectiei
  sursei inainte de concluzii despre performanta pe intreaga liga.
- Pragurile minime sunt reguli exploratorii, nu garantii de stabilitate.
- Nu au fost reconstruite accidentari, vreme, loturi sau tactica istorice.
- Datele nu au fost reverificate meci cu meci la un furnizor independent.
- Pentru un model imbunatatit trebuie rezervata o perioada ulterioara neatinsa;
  nu se ajusteaza si apoi valideaza modelul pe aceleasi rezultate.

## Auditul modelului existent de cornere

`train_corners_1h.py` produce predictii cronologice, dar apoi potriveste
calibrarea Platt pe toate aceste predictii si raporteaza metricile calibrate
pe aceleasi randuri. Acele metrici calibrate nu sunt o evaluare independenta
out-of-sample a calibrarii. Codul existent nu a fost modificat in acest audit.

## Ce lipseste pentru gol pana la 70

Nu exista in fisierele inspectate un istoric complet, verificat, pe ligi,
cu toate minutele golurilor si apartenenta la echipe. Fisierul manual de
56 observatii trece verificarile structurale ale listelor de minute si ale
etichetei <=70, dar nu poate alimenta un backtest complet pe campionate.
Exista o partida comuna, care trebuie numarata o singura data la agregare.

Pentru fiecare meci sunt necesare identitatea evenimentului, liga, sezonul,
data/ora, toate golurile valide din timpul regulamentar, echipa creditata,
repriza, minutul si timpul aditional. Trebuie pastrate distinct meciurile
0-0 confirmate si cazurile unde nu s-au obtinut evenimentele.

Functia existenta `sofascore_loader.fetch_goal_minutes` este un punct de
plecare pentru colectare, nu dovada ca istoricul a fost deja salvat.
Importul trebuie sa confirme perioada, golurile anulate si scorul la 90 minute.
Nu s-au lansat mii de cereri API si nu s-a prezentat un backtest fictiv Gol70.

## Reproducere si transfer

Din radacina proiectului, cu PowerShell:

```powershell
.\audit_goal70_corners_1h.ps1 -AsOf 2026-11-01
```

Scriptul citeste numai fisierele de intrare si scrie rapoarte in acest
director. Nu actualizeaza date online, nu trimite notificari si nu ruleaza
joburile zilnice. Python nu a putut fi executat in mediul acestei sesiuni;
auditul este reproductibil in PowerShell, fara dependinte Python.

Verificare executata cu succes:

```powershell
.\tests\test_goal70_corners_audit.ps1 -AsOf 2026-11-01
```

Rezultat PASS: cinci hashuri de intrare neschimbate, 4.563 predictii verificate
pentru praguri/rezultate/scenariu de stres, toate cele 413 clasamente zilnice
reproduse si sase cazuri verificate independent pentru esantioane strict
anterioare zilei prezise. Verificarile matematice includ lambda zero,
monotonicitatea probabilitatii Under si o valoare Poisson de referinta.

Pentru continuitate copiaza proiectul cu AGENTS.md, CLAUDE.md,
Prompts/Football_Goal70_Corners1H_Master.md, scriptul si datele.
Intr-un chat care nu are acces la proiect, ataseaza explicit documentul metodei
si raportul. Continuitatea depinde de accesul la aceste fisiere, nu de o
promisiune de memorie automata a conversatiilor.
