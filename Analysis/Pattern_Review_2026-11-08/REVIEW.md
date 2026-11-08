# Numitori comuni: audit descriptiv la 8 noiembrie 2026

## Cererea și alegerile actuale

Utilizatorul confirmă că alegerile de ieri au fost decizii personale asistate de analiza din chat, nu preluarea automată a liderilor statistici. Păstrăm distincția dintre această metodă de decizie și ruta tehnică de înregistrare din dashboard. Ieri: Blackburn–West Ham Gol70, rang brut15; Excelsior–Go Ahead cornere, rang6. Candidații inițiali ai analistului și întreg universul sunt cohorte diferite.

Azi CSV/evenimentele canonice confirmă Brighton–Brentford GOAL_BY_70 și Girona–Almería CORNERS_1H_UNDER_7_5, ambele pending. Declarația nu implică plasarea unui pariu, cote, miză sau profit. Nu adăugăm rânduri duplicate și nu schimbăm snapshoturile.

## Eșantion real disponibil

Șase zile de predicții cu rezultate:2–7noiembrie. Ziua8este în așteptare și exclusă. Cohorta de modele include doar probabilitate salvată prospectiv plus rezultat verificat; lipsuri/anulări și meciuri fără model sunt excluse din performanța modelului, nu ascunse din istoricul general.

| Grup | Gol70 | Cornere1H≤7 |
|---|---:|---:|
| Alegerile utilizatorului | 6/6 | 6/6 |
| Toate modelele verificate | 197/229 (86.03%) | 173/195 (88.72%) |
| Eșecuri în cohorta modelelor | 32 | 22 |
| Probabilitate medie salvată, aceeași cohortă | 87.35% | 89.13% |

Modelele nu sunt selecții ferme și nu sunt independente: se repetă echipe/ligi și zile, cu istorice suprapuse. Candidații statistici nu înlocuiesc retrospectiv alegerile utilizatorului. Nu avem încă eșecuri între cele șase alegeri proprii pe fiecare piață; eșecurile analizate mai jos apar în universul mai larg, nu sunt pierderi declarate ale utilizatorului.

## Ce se conturează la succese

Gol70: ipoteza utilă este producție ofensivă proprie pe teren plus frecvență a primului gol, robustețe la scoruri mari/eliminări și context actual verificat. Exemplu prospectiv Blackburn: GF70propriu9/7acasă și10/7WestHamafară,27/28sezon/14/14teren; primul54. Nu este dovadă cauzală, iar Braunschweig arată că producția proprie bună poate fi urmată de0–0.

Cornere: ipoteza utilă este distribuție FirstHalf relativ stabilă pe teren, nu doar medie mică: exact7/≥8, dispersie și adversari. Excelsior–GoAhead: înainte de meci21/22și11/11, cu totaluriGoAheadafară4,1,4,3,4,4; rezultat3cornere. Istra, candidatul inițial, avea22/22și12/12 și a terminat cu4cornere. Aceste rezultate sunt compatibile cu ipoteza, nu o validează.

## Ce au în comun eșecurile — și ce NU putem susține

Nu există un singur profil dovedit. Avertismentele cunoscute înainte de meci se suprapun cu succesele:

- Gol70: λpoate depinde de goluri cedate, scoruri ample sau goluri ulterioare deschiderii. Excelsior, lider96,47%, avea GF70propriuacasă3/5 și istoricul0–9afectat de eliminare; primul gol72, Fail. Semnal prospectiv, nu cauză tactică demonstrată.
- Istoricul aproape perfect nu elimină meciul blocat: Braunschweig–Cottbus92,76%,19/20și10/10 a terminat0–0; Watford–Middlesbrough28/28și14/14 a terminat0–0. Nu există o regulă simplă «istoric perfect=succes».
- La cornere, media poate ascunde reprize extreme: Parma–Bologna, lider96,92%, avea17/20favorabile și3reprize≥8; rezultatul8a fostFail. Sporting–Moreirense95,46% aveaSportingacasă4,1,10,4,6,varianță11; rezultatul10a fostFail. Sporting fusese revocat și nu este alegerea utilizatorului de ieri.
- Schimbările de antrenor/lot și transferul campionat→Europa pot limita folosirea istoricului. Sunt factori de verificat, nu explicații atribuite automat tuturor eșecurilor.

## Comparație efectivă între succese și eșecuri

Mediile de mai jos sunt descriptive, pe rânduri cu câmpul disponibil. Nu sunt praguri, teste cauzale sau filtre noi.

| Indicator anterior meciului | Succese | Eșecuri |
|---|---:|---:|
| Gol70: frecvență istorică medie pe teren | 87.69% (n=197) | 83.89% (n=32) |
| Gol70: rata proprie pe teren a atacului mai slab | 0.76 goluri/meci (n=157) | 0.73 (n=26) |
| Cornere: frecvență istorică favorabilă medie pe teren | 87.56% (n=173) | 86.00% (n=22) |
| Cornere: maximul raportului varianță/medie dintre cele două terenuri | 1.55 (n=173) | 1.76 (n=22) |

Diferențele sunt mici și nu separă clar clasele. De exemplu, rata atacului mai slab0,76vs0,73nu justifică un nou prag magic. La cornere, dispersia ceva mai mare între eșecuri este o ipoteză de verificat ulterior, nu dovadă că eliminarea acelor meciuri îmbunătățește selecția. Nu am optimizat praguri pe aceste rezultate, nu am ales o regulă pe baza randamentului de aici și nu am schimbat modelul.

## Cât mai trebuie urmărit

Nu putem promite o dată la care «găsim formula».30de zile este primul bilanț pilot, nu termen de antrenament care garantează validarea. Avem6zile verificate; mai sunt24zile cu rezultate pentru acel bilanț. Volumul diferă mult între zile:7noiembrie contribuie100/229Gol70și82/195cornere, deci rezultatul nu este distribuit uniform în timp.

După definirea unei ipoteze pe trecut, trebuie evaluată pe zile viitoare nefolosite la alegerea ei. Orientativ,100–200de alegeri comparabile pe fiecare piață ar oferi mai multă informație decât6, dar nu este prag universal și la un singur eveniment pe zi strângerea lor durează luni. Evaluarea pe mai multe ligi/echipe și repetarea în timp rămân necesare. Nu forțăm selecții pentru a crește eșantionul.

Regulile candidate se fixează înainte de rezultate; se compară cu metoda curentă pe aceeași populație eligibilă, inclusiv ce meciuri s-ar fi exclus și câte NO BET ar fi rezultat. Orice modificare de model/ponderi/praguri necesită autorizare separată. [Documentația calibrării: date separate pentru ajustare și evaluare](https://scikit-learn.org/stable/modules/calibration). [NIST: incertitudinea proporțiilor, mai ales cu puține eșecuri](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

## Surse și reproducere

Arhiva canonică2–7noiembrie, CSVprospective și rezultate Review, user_selections.csv și selection_events.jsonl; loadArchive din data.mjs reconciliază identități și exclude probabilitățile inexistente. Exemple detaliate în Review_2026-11-06/REVIEW.md și Review_2026-11-07/REVIEW.md, plus Corners1H_2026-11-07/Alternative_Comparison/COMPARISON.md.

Reproducere: node Analysis/Pattern_Review_2026-11-08/audit.mjs apoi report.mjs. descriptive_audit.json conține agregatele;rows.json păstrează piața,data,id,probabilitatea originală,rezultatul și sursa. Nu se descarcă rezultate noi, nu se pornește taskul zilnic, nu se antrenează modele și nu se fac commit/push.
