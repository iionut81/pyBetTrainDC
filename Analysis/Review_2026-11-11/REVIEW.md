# Review — 11 noiembrie 2026

Verificare 2026-11-12T09:20:29.470+02:00, înaintea scanării din 12 noiembrie. Zero alegeri explicite ieri și zero alegeri active mai vechi fără rezultat în CSV-ul canonic; evenimentele dashboard au fost citite, alegerile revocate nu sunt recreate. Candidatul analistului era Artis–Jablonec la cornere; Gol70 NO BET. Nu au fost pariuri înregistrate sau deduse.

Surse prospective: [Gol70](../Goal70_2026-11-11/EUROPE_ALL_MATCHES.md), [cornere](../Corners1H_2026-11-11/ANALYSIS.md), [context](../Goal70_2026-11-11/CONTEXT.md), CSV canonice din 11 noiembrie. Date finale: FotMob/*.json, identitate și liste de goluri reconciliate cu scorul, cornere explicit FirstHalf+SecondHalf=All.

## goal70

{"total":5,"verified":5,"success":5,"fail":0,"pending":0,"modeled_verified":3,"modeled_success":3}. Numitorul ratei include exclusiv rezultate verificate. Alegeri utilizator: niciuna.

| Meci / competiție | Brut / AR20 / rang | Istoric sezon / teren | Observație | Rezultat | Sursă |
| --- | --- | --- | --- | --- | --- |
| Sheffield Wednesday–Accrington Stanley / eng.efl_trophy | 88.68% / 82.50% / 1 | 22/27; 11/14 | 39 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6014408) |
| Artis Brno–Jablonec / cze.cup | 88.48% / 82.25% / 2 | 22/28; 10/13 | 32 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6420192) |
| Barnsley–Oldham Athletic / eng.efl_trophy | 86.26% / 79.57% / 3 | 24/26; 13/13 | 25 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6015570) |
| FKM Nové Zámky–Zemplín Michalovce / svk.cup | indisponibil / indisponibil / fără model | 12/14; 5/6 | 59 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6530609) |
| Zeleziarne Podbrezová–Zlaté Moravce / svk.cup | indisponibil / indisponibil / fără model | 17/22; 9/11 | 44 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6486793) |

Nu există fail verificat în această cohortă la această predare. Reușite aproape de prag (gol65–70 / exact7cornere): niciuna verificată.

## corners1h

{"total":5,"verified":2,"success":2,"fail":0,"pending":3,"modeled_verified":2,"modeled_success":2}. Numitorul ratei include exclusiv rezultate verificate. Alegeri utilizator: niciuna.

| Meci / competiție | Brut / AR20 / rang | Istoric sezon / teren | Observație | Rezultat | Sursă |
| --- | --- | --- | --- | --- | --- |
| Artis Brno–Jablonec / cze.cup | 93.09% / 85.29% / 1 | 26/28; 13/13 | indisponibil | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6420192) |
| Sheffield Wednesday–Accrington Stanley / eng.efl_trophy | 87.66% / 75.99% / 2 | 23/27; 11/14 | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6014408) |
| Barnsley–Oldham Athletic / eng.efl_trophy | 74.60% / 57.16% / 3 | 20/26; 8/13 | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6015570) |
| FKM Nové Zámky–Zemplín Michalovce / svk.cup | indisponibil / indisponibil / fără model | 11/14; 5/6 | indisponibil | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6530609) |
| Zeleziarne Podbrezová–Zlaté Moravce / svk.cup | indisponibil / indisponibil / fără model | 12/14; 5/7 | indisponibil | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6486793) |

Nu există fail verificat în această cohortă la această predare. Reușite aproape de prag (gol65–70 / exact7cornere): niciuna verificată.

## Ipoteze prospective versus observații

Gol70: Wednesday–Accrington39, Artis–Jablonec32 și Barnsley–Oldham25 sunt compatibile cu probabilitățile brute88,68/88,48/86,26%, dar nu demonstrează calibrarea. Wednesday avea22/27 și11/14teren versus88,68%; Artis22/28 și10/13, GF70propriu pe teren0,857/0,833, cu trei tardive/0–0 per echipă; Barnsley24/26 și13/13, însă GF70propriu0,833/0,429. Semnalele de prudență erau cunoscute și nu dispar după succes. Cele două succese slovace59/44 sunt ale inventarului fără model, nu validări ale unei predicții. NO BET nu prezicea un eșec.

Cornere: Wednesday1 și Barnsley3 sunt compatibile cu modelele87,66/74,60%, dar sunt numai două observații. Riscurile prospective rămâneau varianța Wednesday acasă11,24 la media5,71 și Barnsley8,67cornere medii acasă,6≥8din26sezon. Nu deducem cauze tactice din aceste rezultate. Artis era rang1,93,09%brut/85,29%AR20,26/28sezon și13/13teren,2exact7/2≥8. FirstHalf lipsește; candidatul NU este succes sau fail verificat. TotalCorner arată6–6cornereFT, fără split1H disponibil în pagina consultată: FT12nu poate puncta piața1H. Surse secundare sunt salvate în independent_sources.json.

Contextul Artis includea cupă eliminatorie, trei zile de odihnă, deplasare europeană Jablonec, convocări și titulari neconfirmați. Formațiile anterioare din API nu erau XI confirmat; sursa post-meci Rotascore indică4-2-3-1/3-4-1-2, ilustrând limita, fără a transforma informația ulterioară în predictor. Eliminarea Glossoa78nu explică prima repriză. Cauzele efective ale distribuției cornerelor rămân necunoscute. EFL relatează primul golOldham72, FotMob63: conflict secundar fără efect asupra primului gol25; păstrat, fără alegerea unei explicații convenabile.

## Restanțe și limite

Trival–Leganés are acum7cornereFirstHalf(2+5), succes aproape de prag; partida inventariată prospectiv pe10 a fost disputată pe11, data efectivă rămâne explicită, reconciliate cu repriza2 șiFT; rândul canonic10noiembrie actualizat, copia anterioară păstrată. Rămân3FirstHalf10noiembrie și7Cipru, plus3FirstHalf11noiembrie. Serbia: cancelledîn furnizor versus reprogramarea veche; Badajoz: cancellednou versusPP/RFEFTBDvechi. Aceste conflicte rămân nepunctate; nu rescriem cohorte de dată pentru LasPalmas–Eldense disputat9noiembrie. older_retries.csv conține fiecare încercare și timestamp. Confirmarea independentă exhaustivă lipsește. Modelul nu este ajustat după o zi.


Corecțieeditorială2026-11-12T09:26:53.625+02:00:nota inițială de predare și primulmesaj au menționat eronat6(4+2)laTrival; dovadaraw șiCSVaveau7(2+5). Corectată narațiunea, fărăschimbarearezultatului saupredictiei. Nu prezentăm aceastăcorecție ca verificareefectuatăînainteascanării.
