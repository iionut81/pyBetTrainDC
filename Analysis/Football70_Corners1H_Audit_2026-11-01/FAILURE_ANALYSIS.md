# Interpretarea esecurilor si o corectie experimentala

Referinta: 2026-11-01. Acest raport priveste exclusiv CORNERS_1H_UNDER_7_5.
Nu exista un backtest complet GOAL_BY_70 si nu se transfera aceste rate la goluri.

## Ce inseamna prima alegere

Locul 1 dupa compararea tuturor candidatilor eligibili din ziua istorica
disponibila. Nu inseamna oprirea cautarii la primul meci care pare acceptabil.
Topul istoric nu include verificarea tactica manuala si nici programul complet
al ligilor, daca meciurile nu sunt prezente in datele locale.

Din 413 prime alegeri, 367 au reusit si 46 au esuat:
88,86% succes, 11,14% esec. Probabilitatea medie a modelului a fost 94,27%,
echivalent cu numai 5,73% esec estimat. Esec efectiv aproape dublu fata de
ce sugera modelul. Rata istorica nu este o probabilitate garantata pentru azi.

## Unde s-a pierdut

| Cornere la pauza | Prime alegeri pierdute |
|---|---:|
| 8 | 24 |
| 9 | 16 |
| 10 | 4 |
| 11 | 1 |
| 16 | 1 |

40 din 46 de esecuri, adica 87%, s-au produs cu 8 sau 9 cornere.
Cronologiile cornerelor si contextul istoric nu au fost reconstruite;
nu putem atribui aceste rezultate unor goluri timpurii, accidentari sau vremii.

Modelul estima in medie 3,94 cornere pentru locul 1; s-au observat 4,49.
Varianta observata a totalului a fost 5,50. Reziduul Pearson patrat mediu a
fost 1,59. Acestea indica o subestimare a intensitatii si/sau a variatiei in
populatia selectata. Varianta agregata include diferentele intre meciuri;
nu reprezinta singura o demonstratie a supradispersiei conditionale.

O simpla regula "mai multe meciuri in esantion" nu este inca demonstrata:
la minimum 3-4 observatii anterioare pe terenul relevant au reusit 276/309
(89,3%); la 5-9 au reusit 89/102 (87,3%); categoria 10+ are numai doua cazuri.
Sunt descrieri retrospective, nu filtre validate pentru imbunatatirea selectiei.

## Ce corectie a fost testata

Script: `analyze_corners_1h_failures.ps1`.

S-a fixat o singura separare cronologica, 2025-07-01. O calibrare logistica
globala (Platt) a fost invatata pe 3.641 predictii anterioare si apoi aplicata
pe 922 predictii ulterioare. Nu s-au cautat praguri folosind rezultatele
perioadei ulterioare si nu s-a actualizat calibrarea de productie.

```
p_corectat = sigmoid(1.6963276621684558
                    + 0.18722589699140135 * logit(p_brut))
```

Coeficientul pozitiv pastreaza ordinea meciurilor; aceasta corectie nu poate
imbunatati rata de succes a aceluiasi top 1. Ajusteaza probabilitatile afisate.
Formula este un experiment istoric, nu o corectie universala de aplicat
automat procentelor manuale ale unui meci curent.

| Perioada ulterioara | N | Succes real | P medie bruta | P medie corectata |
|---|---:|---:|---:|---:|
| Toate eligibile | 922 | 89,48% | 88,83% | 89,17% |
| Aceleasi prime alegeri | 65 | 86,15% | 93,99% | 90,50% |

Pe toate cele 922 meciuri, Brier a scazut de la 0,09587 la 0,09387, iar
log loss de la 0,34384 la 0,33491. Pe primele alegeri, Brier a scazut de la
0,12516 la 0,12018. Corectia imbunatateste evaluarea probabilistica in aceasta
separare, insa ramane supraestimare pentru top 1: 90,50% estimat vs 86,15% real.
Rezultatele castigate/pierdute nu s-au schimbat: 56/65 prime alegeri reusite.

Acest test ramane exploratoriu: auditul agregat al istoricului fusese deja
inspectat. Nu este o validare prospectiva pe date noi complet neatinse.

## Consecinte pentru utilizare

1. Nu prezentam 94-97% din Poisson ca sansa reala validata a primei alegeri.
2. Actualizam si verificam datele; istoric curent nu inseamna doar fisiere existente.
3. Pentru mai putine esecuri trebuie imbunatatita selectia, nu doar procentul
   afisat: estimarea intensitatii, regularizarea esantioanelor mici si modele
   care reprezinta variatia cornerelor trebuie comparate cronologic.
4. Orice filtru de liga, tactica sau esantion se decide pe perioada de dezvoltare
   si se verifica ulterior. Nu eliminam retrospectiv cazurile pierdute pentru
   a obtine o rata convenabila.
5. Pentru GOAL_BY_70 completam separat toate minutele golurilor pe ligi;
   pana atunci nu avem o rata de succes validata pentru acea strategie.

Reproducere:

```powershell
.\analyze_corners_1h_failures.ps1 -AsOf 2026-11-01 -CalibrationCutoff 2025-07-01
```

Fisierele `top1_failure_matches.csv`, `top1_failure_totals.csv`,
`top1_failure_by_league.csv`, `top1_failure_by_prior_venue_sample.csv`,
`calibration_chronological_comparison.csv`, `calibration_later_period_predictions.csv`
si `failure_analysis_summary.json` permit urmarirea calculelor.
