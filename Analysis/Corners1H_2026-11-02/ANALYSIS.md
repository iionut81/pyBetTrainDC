# Sub 7,5 cornere prima repriza: 2 noiembrie 2026

**Selectia: Radnicki Nis - Zeleznicar Pancevo, 19:00 RO / 18:00 CET.** Cel mult 7 cornere cumulate pana la pauza, inclusiv timpul aditional. Model Poisson brut **96,84%**; stres intensitate +20% **92,58%**. Actualizare program: 2026-11-02T13:31:35+02:00. Partida nu a inceput. O singura selectie pentru aceasta analiza; nu este combinata cu golul pana la 70 si nu se stabileste o miza.

## Acoperire si sursa

Au fost colectate toate cele 27 de partide din programul Europei stabilit azi: 16 din primele ligi masculine ale primelor 20 tari UEFA 2026, plus 11 din Romania, Serbia si diviziile secunde europene urmarite. Sunt 15 ligi active; celelalte prime ligi nu au partide identificate azi. Toate cele 27 au inputuri complete pe sezonul curent pentru echipele implicate, minimum 5 meciuri pe sezon si 3 in configuratia terenului. 575 meciuri istorice DISTINCTE, toate cu cornere FirstHalf, SecondHalf si All reconciliate separat pe echipa. Niciun rand lipsa nu este inlocuit cu zero.

Istoricul local anterior se oprea in mai 2026 si nu a fost folosit pentru selectia curenta. Sursa noua: [FotMob, programul zilei](https://www.fotmob.com/api/data/matches?date=20261102), endpoint matchDetails, content.stats.Periods.FirstHalf, key=corners. Este sursa numerica principala pentru 1H; reconcilierea reprizelor este control intern, nu verificare independenta a fiecarei observatii. Pentru ultimele doua partide ale finalistilor, totalurile FT au fost confruntate cu un alt furnizor: 3-6 si 8-5, dar aceasta nu constituie verificare separata a tuturor cifrelor 1H. [Nis-Partizan](https://scoremondo.com/en/match/fk-radnicki-nis-fk-partizan-belgrade-31356004), [Radnicki 1923-Zeleznicar](https://scoremondo.com/en/match/fk-radnicki-1923-fk-zeleznicar-pancevo-31356001).

## Inputurile selectiei

| Esantion | N | Cornere obtinute 1H | Cornere cedate 1H | Media TOTALULUI 1H | Exact 7 | Minimum 8 |
|---|---:|---:|---:|---:|---:|---:|
| Radnicki sezon | 13 | 26 | 26 | 4,0000 | 0 | 0 |
| Zeleznicar sezon | 13 | 25 | 18 | 3,3077 | 0 | 0 |
| Radnicki acasa | 6 | 10 | 13 | 3,8333 | 0 | 0 |
| Zeleznicar afara | 6 | 10 | 10 | 3,3333 | 0 | 0 |

Radnicki: 13/13 reusite; Zeleznicar: 13/13. Meciul direct din 17 iulie apare in ambele serii, deci totalul distinct este **25/25**, nu 26. Configuratia terenului: **12/12**, sase pentru fiecare echipa. Nu exista reprize cu exact 7; maximul este 6 in seria Radnicki si 5 in seria Zeleznicar. Un asemenea istoric nu demonstreaza o probabilitate de 100% si nu elimina riscul unei serii neobisnuite de cornere.

## Calculul fix, fara cote

```text
lambda_sezon = ((26/13 + 18/13) + (25/13 + 26/13))/2 = 3,653846
lambda_teren = ((10/6 + 10/6) + (10/6 + 13/6))/2 = 3,583333
lambda_combinat = 0,5*3,653846 + 0,5*3,583333 = 3,618590
P(total <=7) = exp(-lambda) * SUM(lambda^k/k!, k=0..7) = 96,8414%
Stres: lambda*1,2 = 4,342308; P(total <=7) = 92,5831%
Cota echitabila matematica = 1/P = 1,032616
```

Numai sezon: 96,69%; numai teren: 96,99%. Ponderarea 50/50 este euristica; esantioanele se suprapun. Poisson nu este calibrat sau validat pentru acest clasament zilnic, iar stresul nu este un interval de incredere. Cota echitabila nu este un pret disponibil si nu confirma pragul dorit de peste 1,10. Nu au fost folosite cote sau probabilitati ale tipsterilor.

Dispersie observata, varianta de esantion a totalurilor 1H: Radnicki sezon 1,50 fata de media 4,00; Zeleznicar sezon 1,56 fata de media 3,31. Pe teren: 1,77 fata de media 3,83 si 1,47 fata de media 3,33. In aceste serii variatia este mai mica decat media; nu adopt o alta distributie dupa observarea rezultatelor. Un meci cu ritm diferit poate depasi acest istoric restrans.

## Ultimele 10 meciuri de campionat, totaluri la pauza

### Radnicki Nis

| Data | Partida | Cornere gazde-oaspeti 1H | Total 1H |
|---|---|---:|---:|
| 2026-10-28 | [Radnički Niš - Partizan Beograd](https://www.fotmob.com/matches/radnicki-nis-vs-partizan-beograd/1pt0n6#1000015091) | 1-5 | 6 |
| 2026-10-23 | [Radnički Niš - Novi Pazar](https://www.fotmob.com/matches/radnicki-nis-vs-novi-pazar/8o2br12#1000015158) | 3-1 | 4 |
| 2026-10-16 | [FK Radnik Surdulica - Radnički Niš](https://www.fotmob.com/matches/radnicki-nis-vs-fk-radnik-surdulica/1epn4no2#1000015151) | 4-1 | 5 |
| 2026-10-11 | [Radnički Niš - Vojvodina](https://www.fotmob.com/matches/vojvodina-vs-radnicki-nis/1cwhw2#1000015144) | 2-1 | 3 |
| 2026-09-20 | [FK Crvena Zvezda - Radnički Niš](https://www.fotmob.com/matches/radnicki-nis-vs-fk-crvena-zvezda/1vuwec#1000015135) | 3-1 | 4 |
| 2026-09-13 | [Radnički Niš - OFK Beograd](https://www.fotmob.com/matches/radnicki-nis-vs-ofk-beograd/1vwinp#1000015131) | 2-2 | 4 |
| 2026-09-05 | [Mladost Lučani - Radnički Niš](https://www.fotmob.com/matches/mladost-lucani-vs-radnicki-nis/jrvrl#1000015120) | 1-5 | 6 |
| 2026-08-29 | [Radnički Niš - FK Radnički 1923](https://www.fotmob.com/matches/fk-radnicki-1923-vs-radnicki-nis/jgqwh#1000015118) | 1-3 | 4 |
| 2026-08-22 | [Mačva Šabac - Radnički Niš](https://www.fotmob.com/matches/radnicki-nis-vs-macva-sabac/22irmbp3#1000015104) | 2-2 | 4 |
| 2026-08-16 | [FK IMT Beograd - Radnički Niš](https://www.fotmob.com/matches/radnicki-nis-vs-fk-imt-beograd/23xrs368#1000015103) | 3-1 | 4 |

### Zeleznicar Pancevo

| Data | Partida | Cornere gazde-oaspeti 1H | Total 1H |
|---|---|---:|---:|
| 2026-10-28 | [FK Radnički 1923 - Železničar Pančevo](https://www.fotmob.com/matches/fk-radnicki-1923-vs-zeleznicar-pancevo/2xj4bz8p#1000015079) | 2-3 | 5 |
| 2026-10-23 | [FK IMT Beograd - Železničar Pančevo](https://www.fotmob.com/matches/fk-imt-beograd-vs-zeleznicar-pancevo/9vvc9qmq#1000015157) | 4-0 | 4 |
| 2026-10-18 | [Železničar Pančevo - Partizan Beograd](https://www.fotmob.com/matches/partizan-beograd-vs-zeleznicar-pancevo/2zibieog#1000015153) | 2-2 | 4 |
| 2026-10-11 | [FK Zemun - Železničar Pančevo](https://www.fotmob.com/matches/fk-zemun-vs-zeleznicar-pancevo/2yyd3t9z#1000015141) | 1-1 | 2 |
| 2026-09-20 | [Železničar Pančevo - Čukarički](https://www.fotmob.com/matches/cukaricki-vs-zeleznicar-pancevo/2yzz7ht7#1000015139) | 2-0 | 2 |
| 2026-09-17 | [Železničar Pančevo - FK Crvena Zvezda](https://www.fotmob.com/matches/fk-crvena-zvezda-vs-zeleznicar-pancevo/2zq4anet#1000015099) | 1-2 | 3 |
| 2026-09-11 | [Železničar Pančevo - Mačva Šabac](https://www.fotmob.com/matches/macva-sabac-vs-zeleznicar-pancevo/9ssnaprm#1000015126) | 5-0 | 5 |
| 2026-09-04 | [Novi Pazar - Železničar Pančevo](https://www.fotmob.com/matches/novi-pazar-vs-zeleznicar-pancevo/4rgsb5pz#1000015125) | 1-2 | 3 |
| 2026-08-29 | [Železničar Pančevo - FK Radnik Surdulica](https://www.fotmob.com/matches/fk-radnik-surdulica-vs-zeleznicar-pancevo/8ab0sqs8#1000015113) | 4-0 | 4 |
| 2026-08-23 | [Vojvodina - Železničar Pančevo](https://www.fotmob.com/matches/vojvodina-vs-zeleznicar-pancevo/2z0c7mfv#1000015110) | 2-0 | 2 |

Sunt meciuri de campionat, nu pretind o serie completa din toate competitiile. Restantele din 28 octombrie sunt incluse pe data REALA a disputarii, nu dupa numarul etapei. Ultima deplasare Zeleznicar a avut 13 cornere FT, dar 5 la pauza (2-3): nu am dedus 1H impartind totalul la doi.

## Context si contraargumente

Superliga Serbia, etapa 14, campionat regulat; fara calificare pe scor agregat sau prelungiri. Radnicki are 8 puncte si cinci infrangeri consecutive, Zeleznicar 26 si cinci victorii consecutive. Scorul clasamentului nu impune automat multe cornere. [Program si statistici oficiale ale ligii](https://www.superliga.rs/utakmica/14-kolo-radnicki-zeleznicar/). Numele comercial al competitiei nu inseamna ca folosim cote de bookmaker.

Gazdele sunt antrenate acum de Nikola Drincic: declaratia din avancronica vorbeste despre noua antrenamente si trei partide de la venire. Acest lucru reduce comparabilitatea cu o parte din cele 13 meciuri. Sunt raportati absenti Bojan Kovacevic si Ranko Jokic. Nu exista inca un 11 confirmat. [Relatarea conferintei in presa locala](https://glasjuga.rs/radnicki-u-ponedeljak-docekuje-zeleznicar-ulaz-na-cair-besplatan-drincic-veruje-u-opstanak/).

Radomir Kokovic confirma al treilea meci consecutiv in deplasare intr-un interval scurt, oboseala si jucatori suspendati sau accidentati; nu nominalizeaza complet indisponibilii. Nu inventez o lista de absenti si nu presupun ca oboseala ajuta automat underul: poate reduce atacul sau poate provoca greseli, respingeri si presiune defensiva. [Conferinta oficiala Zeleznicar](https://fkzeleznicar.rs/kokovic-moramo-da-izadjemo-iz-zone-komfora/).

Modulele 4-2-3-1 Radnicki si 4-1-4-1 Zeleznicar apar ca ultime formatii cunoscute la meciurile din 28 octombrie in agregatoare; nu le prezint drept module confirmate pentru azi. Scenariul favorabil este o prima repriza cu atacuri limitate pe benzi si volume apropiate de istoricul sezonului. Scenariul nefavorabil: gol timpuriu, gazda nevoita sa urmareasca scorul, centrari repetate si suturi blocate, generand cornere in grupuri. Un gol nu reduce automat cornerele.

Ambele au jucat pe 28 octombrie, deci cinci zile calendaristice pana azi. S-a confirmat calatoria oaspetilor din Pancevo spre Nis. Ploaia la antrenamentul oaspetilor din 1 noiembrie este confirmata, dar NU este o prognoza de vreme la Nis in timpul partidei. Nu exista aici un raport verificat al gazonului sau prognoza orara; nu modific procentul pe aceste presupuneri. [Anuntul oficial despre deplasare](https://fkzeleznicar.rs/poslednji-trening-pred-polazak-u-nis/).

## De ce aleg acest meci, desi este al treilea dupa procentul brut

Atromitos-Volos are raw 97,46%, stres 93,90%, dar numai 3 meciuri ale gazdelor acasa, 15 observatii distincte si un esec >=8. Volos si-a schimbat antrenorul pe 26 octombrie: intreg istoricul sau de campionat din inputuri este anterior lui Garrido. [Numire oficiala](https://www.volosfc.com/o-juan-carlos-garrido-neos-proponitis-toy-voloy-elabet/).

Sassuolo-Fiorentina are 96,8437%, practic egal cu 96,8414% al partidei alese: diferenta de 0,0023 puncte procentuale nu este relevanta pentru o asemenea precizie a datelor. Acolo sunt 18 observatii distincte si un esec >=8; aici sunt 25, niciun esec >=8 si nicio repriza de exact 7. Lorient-Brest are 96,81%, 16 observatii distincte si doar 8 pe teren. Voluntari-Farul are 96,45%, 26 observatii, dar doua reprize de exact 7 pe configuratia terenului; nu e la fel de confortabil fata de prag. Mallorca-Cadiz este alternativa cu raw 95,72%, stres 90,30%, dar un esec >=8 in sezon.

Alegerea finala foloseste aceasta judecata despre cantitatea datelor, proximitatea pragului si incertitudinile contextuale. Nu am modificat formula, procentele sau adaugat bonusuri. Nu afirm ca alegerea este demonstrata superior printr-un backtest al acestei reguli de preferinta.

## Comparatia integrala, ordonata strict dupa model

| Rang raw | Partida | P brut | Stres +20% | >=8 in sezon, meciuri distincte | Exact 7, sezon distinct |
|---:|---|---:|---:|---:|---:|
| 1 | [Atromitos - NFC Volos](https://www.fotmob.com/matches/atromitos-vs-nfc-volos/546bpkva#5943281) | 97,46% | 93,90% | 1/15 | 1/15 |
| 2 | [Sassuolo - Fiorentina](https://www.fotmob.com/matches/sassuolo-vs-fiorentina/28u7rc#5749738) | 96,84% | 92,59% | 1/18 | 1/18 |
| 3 | [Radnički Niš - Železničar Pančevo](https://www.fotmob.com/matches/radnicki-nis-vs-zeleznicar-pancevo/2z0du52p#1000015166) | 96,84% | 92,58% | 0/25 | 0/25 |
| 4 | [Lorient - Brest](https://www.fotmob.com/matches/brest-vs-lorient/2g6hwg#5802975) | 96,81% | 92,53% | 0/16 | 1/16 |
| 5 | [Tenerife - FC Andorra](https://www.fotmob.com/matches/tenerife-vs-fc-andorra/1mbsjawm#5868523) | 96,80% | 92,50% | 1/22 | 0/22 |
| 6 | [Kalamata - Iraklis](https://www.fotmob.com/matches/kalamata-vs-iraklis/vcf5h#5943282) | 96,65% | 92,18% | 0/16 | 0/16 |
| 7 | [FC Voluntari - FCV Farul Constanța](https://www.fotmob.com/matches/fcv-farul-constanta-vs-fc-voluntari/2f0b31c8#5853539) | 96,45% | 91,77% | 0/26 | 3/26 |
| 8 | [Real Valladolid - Leganés](https://www.fotmob.com/matches/leganes-vs-real-valladolid/2pwv36#5868524) | 96,10% | 91,05% | 1/22 | 2/22 |
| 9 | [Mallorca - Cádiz](https://www.fotmob.com/matches/cadiz-vs-mallorca/2eiadu#5868520) | 95,72% | 90,30% | 1/22 | 2/22 |
| 10 | [Raków Częstochowa - Pogoń Szczecin](https://www.fotmob.com/matches/rakow-czestochowa-vs-pogon-szczecin/177mfj#5804027) | 95,27% | 89,42% | 2/23 | 0/23 |
| 11 | [Sønderjyske - FC København](https://www.fotmob.com/matches/fc-kobenhavn-vs-sonderjyske/2ct7eo#5739537) | 94,11% | 87,18% | 2/23 | 1/23 |
| 12 | [Kayserispor - Muğlaspor](https://www.fotmob.com/matches/kayserispor-vs-muglaspor/861edw8#5908687) | 93,46% | 85,98% | 2/24 | 2/24 |
| 13 | [Deportivo A Coruña - Osasuna](https://www.fotmob.com/matches/osasuna-vs-deportivo-coruna/2q4ala#5868113) | 93,43% | 85,92% | 0/20 | 2/20 |
| 14 | [Gençlerbirliği - Alanyaspor](https://www.fotmob.com/matches/alanyaspor-vs-genclerbirligi/1acwgp#5904789) | 93,30% | 85,67% | 2/18 | 2/18 |
| 15 | [Reims - Sochaux](https://www.fotmob.com/matches/reims-vs-sochaux/37o4k2#5803306) | 93,08% | 85,27% | 1/20 | 2/20 |
| 16 | [Universitatea Cluj - FCSB](https://www.fotmob.com/matches/fcsb-vs-universitatea-cluj/28mnlio#5853542) | 91,77% | 82,92% | 2/26 | 3/26 |
| 17 | [Red Star - Nancy](https://www.fotmob.com/matches/red-star-vs-nancy/1tub79#5803305) | 91,55% | 82,52% | 3/20 | 0/20 |
| 18 | [GAIS - AIK](https://www.fotmob.com/matches/gais-vs-aik/2ahulu#5107646) | 91,42% | 82,30% | 1/51 | 6/51 |
| 19 | [Real Oviedo - Las Palmas](https://www.fotmob.com/matches/las-palmas-vs-real-oviedo/2dsrk6#5868521) | 91,27% | 82,03% | 2/22 | 0/22 |
| 20 | [Zagłębie Lubin - Korona Kielce](https://www.fotmob.com/matches/zagebie-lubin-vs-korona-kielce/26rtbg#5804030) | 88,55% | 77,43% | 3/24 | 3/24 |
| 21 | [Mjällby - Kalmar FF](https://www.fotmob.com/matches/mjallby-vs-kalmar-ff/2onym2#5107648) | 87,84% | 76,28% | 8/51 | 4/51 |
| 22 | [Mladost Lučani - FK Radnik Surdulica](https://www.fotmob.com/matches/mladost-lucani-vs-fk-radnik-surdulica/1dpj6qnx#1000015164) | 86,76% | 74,56% | 3/25 | 1/25 |
| 23 | [Gil Vicente - Vitória de Guimarães](https://www.fotmob.com/matches/vitoria-de-guimaraes-vs-gil-vicente/2kb160#5887667) | 86,62% | 74,33% | 5/18 | 0/18 |
| 24 | [Benfica - Alverca](https://www.fotmob.com/matches/benfica-vs-alverca/35t7xg#5887671) | 84,94% | 71,72% | 4/18 | 2/18 |
| 25 | [Atalanta - Parma](https://www.fotmob.com/matches/atalanta-vs-parma/2w0b82#5749731) | 84,23% | 70,63% | 3/18 | 2/18 |
| 26 | [Radomiak Radom - Śląsk Wrocław](https://www.fotmob.com/matches/radomiak-radom-vs-slask-wrocaw/1knfv0#5804026) | 76,38% | 59,50% | 5/24 | 2/24 |
| 27 | [Newcastle United - Everton](https://www.fotmob.com/matches/everton-vs-newcastle-united/2yo9qd#5795503) | 72,22% | 54,12% | 3/16 | 2/16 |

## Dovezi si reproducere

Date: historical_corners_1h.csv (575 meciuri), all_models.csv (27 inputuri si calcule), coverage.csv (15 ligi active), current_status.json (status actual pentru toate 27), FotMob/*.json (campuri sportive publice selectate, fara cote). Reproducere offline: powershell -NoProfile -File Analysis/Corners1H_2026-11-02/evaluate_corners.ps1; apoi verify_and_report.ps1. Nu se ruleaza retraining si nu se modifica productia.

Verificare PASS: 575 observatii unice in trecut, perioada FirstHalf explicita, reconcilierea 1H+2H=FT pe fiecare echipa, inputuri si probabilitati recalculate independent din CSV, 27 statusuri pre-meci si kickoff-uri in viitor la ora salvarii. Verificarea este a datelor salvate, nu monitorizare dupa analiza.
