# Alternativa ceruta: Sassuolo-Fiorentina, sub 7,5 cornere 1H

2 noiembrie 2026, 19:30 RO / 18:30 CET. Statusul a fost reverificat la
14:07:47 RO: meciul nu a inceput. Este alternativa individuala ceruta la
Radnicki-Zeleznicar; nu un al doilea picior al unui bilet combinat.
Disponibilitatea pietei si pretul nu au fost verificate si nu sunt presupuse.

Selectia provine din comparatia completa a celor 27 de partide documentata in
[ANALYSIS.md](ANALYSIS.md), nu dintr-o cautare limitata la Italia.

| Esantion | N | Cornere obtinute 1H | Cedate 1H | Media totalului | Reprize >=8 | Exact 7 |
|---|---:|---:|---:|---:|---:|---:|
| Sassuolo sezon | 9 | 18 | 21 | 4,3333 | 1/9 | 0/9 |
| Fiorentina sezon | 9 | 18 | 17 | 3,8889 | 0/9 | 1/9 |
| Sassuolo acasa | 4 | 5 | 10 | 3,75 | 0/4 | 0/4 |
| Fiorentina afara | 4 | 4 | 6 | 2,50 | 0/4 | 0/4 |

Totaluri 1H, de la cel mai recent: Sassuolo sezon 1,5,2,4,5,6,8,4,4;
Fiorentina sezon 5,3,7,3,3,2,4,6,2. Configuratia terenului: Sassuolo
acasa 1,4,6,4; Fiorentina afara 3,3,2,2. Sunt toate cele noua meciuri de
campionat ale fiecareia din sezon, nu pretind zece din toate competitiile.
Observatii distincte: 18; sub 7,5: 17/18 (94,44%). Teren: 8/8; exact 7: 0/8.
Esecul a fost Bologna-Sassuolo, 6 septembrie, 8 cornere la pauza.
Fiorentina-Como, 18 octombrie, a avut 7, succes aproape de prag.

```text
lambda_sezon = ((18/9 + 17/9) + (18/9 + 21/9))/2 = 4,111111
lambda_teren = ((5/4 + 6/4) + (4/4 + 10/4))/2 = 3,125
lambda_combinat = 0,5*4,111111 + 0,5*3,125 = 3,618056
PoissonCDF(7, lambda_combinat) = 96,8437%
Stres: PoissonCDF(7, 1,2*lambda_combinat) = 92,5879%
Cota echitabila = 1/0,968437095 = 1,032592
```

Ponderile sunt euristice; esantioanele se suprapun. Probabilitatea bruta nu
este calibrata si stresul nu este un interval de incredere. Patru meciuri
pentru fiecare configuratie de teren reprezinta un esantion mic.

Formatii probabile: Sassuolo 4-3-3, Fiorentina 4-3-2-1, Aquilani/Vanoli.
Thorstvedt suspendat; la gazde Kone, Leysen si Boloca indisponibili, la oaspeti
Parisi, Atta, Ranieri si Pongracic. Sursa [Sky, program si formatii probabile](https://sport.sky.it/calcio/serie-a/partite/2026/giornata-10/sassuolo-fiorentina/formazioni).
Nu sunt echipe de start confirmate. Absentelor nu li s-a atribuit un efect
procentual. Principalul risc: presiune pe benzi, suturi blocate si cornere
grupate dupa un gol timpuriu. Numarul golurilor nu stabileste numarul cornerelor.
Sassuolo-Lazio, ultima observatie acasa, a avut o eliminare la 31 si nu este
un model de meci normal; nu pretind ca seria prezice perfect ritmul de azi.

Datele numerice sunt cele din `all_models.csv` si `historical_corners_1h.csv`,
perioada FirstHalf explicita, reconciliate 1H+2H=FT. Verificarea aritmetica
a intregii comparatii este in `verification.json`. Confirmarea actuala de
program este in `sassuolo_current_status.json`.
[Sursa FotMob a partidei](https://www.fotmob.com/matches/sassuolo-vs-fiorentina/28u7rc#5749738).
Nu au fost consultate cote pentru alegere. Istoricul alternativei este
pastrat separat de prima selectie; rezultatul ramane necompletat.
