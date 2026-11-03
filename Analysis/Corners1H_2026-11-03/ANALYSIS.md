# Derby-Norwich: sub 7,5 cornere prima repriza, 3 noiembrie 2026

**Candidatul ales pentru analiza cornerelor: Derby County - Norwich City, 21:45 RO / 19:45 GMT, Pride Park, Championship etapa 14.** Eveniment: maximum 7 cornere cumulate in prima repriza, inclusiv 45+. Analiza salvata la 2026-11-03T11:18:49+02:00. Model brut pe esantionul final verificat: 96,61%; stres intensitate +20%: 92,11%. Probabilitatile nu sunt calibrate; stresul nu este un interval de incredere.

Cererea curenta autorizeaza aceasta analiza suplimentara. QPR-Burnley GOAL_BY_70 ramane alegerea deja confirmata de utilizator. Derby-Norwich la cornere este alegerea analistului pentru cererea curenta, nu o noua selectie confirmata de utilizator, nu o inlocuire a QPR si nu un bilet combinat. Protocolul general nu este modificat permanent; nu presupun miza ori pariu plasat.

## Acoperirea cercetarii

Am pornit de la inventarul zilei pentru primele ligi masculine ale topului UEFA 2026, Romania, Serbia si cele 7 divizii secunde urmarite: 29 ligi. Sunt 10 partide in universul de baza (9 Championship si una Allsvenskan); am adaugat 9 Champions League si una Cupa Danemarcei. Inventarul si limitarile lui sunt in `fixture_universe_reference.csv` si `fixture_exclusions_reference.csv`, cu sursele din analiza Gol70 a aceleiasi zile. Pentru patru ligi fara suport ESPN, absenta partidelor este bazata pe FotMob, nu pe HTTP 400.

Au fost cercetate toate cele 20 de partide identificate, inaintea alegerii. Au rezultat 19 calcule de screening din statistici explicite FirstHalf; Shakhtar-Sporting este exclus numeric, fiindca in cele 11 meciuri Shakhtar lipsesc cornerele 1H. Nu inlocuiesc lipsurile cu zero. 417 partide istorice distincte asteptate din 15 ligi pentru cele 40 echipe: 406 au 1H, 2H si FT reconciliate separat pe echipa; 11 nu au 1H. Nu numesc aceste 19 modele un top al tuturor diviziilor europene.

[Program FotMob](https://www.fotmob.com/api/data/matches?date=20261103). Statusurile tuturor celor 20 de meciuri au fost reverificate la 11:15:27 RO: toate neincepute, ore neschimbate, `current_status.json`. Prima selectie rezulta dupa compararea celor 19 modele eligibile; verificarile contextuale aprofundate se concentreaza pe finaliste.

## Controlul surselor si observatia exclusa

FotMob este sursa statistica principala: content.stats.Periods.FirstHalf, cheia corners, incluzand timpul aditional. FirstHalf + SecondHalf = All este o verificare interna; nu demonstreaza independent corectitudinea cifrelor.

Pentru toate cele 26 de meciuri de campionat Derby/Norwich am colectat si cronologia ESPN. Evenimentele unice corner-awarded sunt numarate pe perioada 1, inclusiv 45+, si reconciliate cu totalurile FT ale ESPN. In 25/26 totalurile 1H coincid. In 23/26 coincid si ambele reprize si FT; la Derby-Wrexham si Norwich-WBA diferentele sunt exclusiv in a doua repriza/FT, iar 1H coincide. Sunt pastrate in `provider_comparison.csv` si `ESPN/*.json`; doua fluxuri concordante pot avea surse comune.

La Derby-Stoke, 17 octombrie, FotMob are 2-3 la pauza, ESPN 3-3. [Cronologia PlayerStats/Sportmonks](https://playerstats.football/fixture/derby-county/stoke-city/2026-10-17) arata al cincilea corner la 45+3 si apoi repriza a doua, sustinand totalul 5. ESPN include un corner suplimentar Derby la 45+4. Nu am verificat video daca a fost acordat si neexecutat. **Exclud aceasta partida din estimarea finala**, fara sa aleg valoarea mai avantajoasa. Pastrez separat screeningul FotMob, varianta ESPN si varianta cu observatia exclusa. Clasamentul de screening de mai jos nu este rescris retrospectiv.

## Datele folosite in estimarea finala

Dupa excludere: Derby 12/13 meciuri de sezon verificabile fara conflict, Norwich 13/13; pe teren Derby 5/6 acasa, Norwich 6/6 deplasare. Acestea depasesc minimul exploratoriu fixat de 5 pe sezon si 3 pe teren, dar sezonul Derby nu este complet in estimarea finala. Valorile lipsa nu sunt zero.

| Esantion | N | Obtinute 1H | Cedate 1H | Media obtinute | Media cedate | Media TOTALULUI | >=8 | Exact 7 | Varianta totalului |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Derby sezon verificat | 12 | 21 | 24 | 1,75 | 2,00 | 3,75 | 1/12 | 1/12 | 7,66 |
| Norwich sezon | 13 | 23 | 19 | 1,77 | 1,46 | 3,23 | 0/13 | 0/13 | 1,86 |
| Derby acasa verificat | 5 | 10 | 11 | 2,00 | 2,20 | 4,20 | 1/5 | 0/5 | 13,20 |
| Norwich deplasare | 6 | 10 | 11 | 1,67 | 1,83 | 3,50 | 0/6 | 0/6 | 1,90 |

Meciuri distincte finale: 24/25 sub 7,5 (96,00%), unul cu minimum 8 si unul cu exact 7. Configuratia terenului: 10/11 sub 7,5 (90,91%), un esec, niciun total exact 7. Acestea sunt frecvente observate, nu probabilitati pentru azi. Derby-Norwich nu s-a jucat in acest sezon, deci nu exista o intalnire directa dubla in agregare.

Singurul esec este Derby-Cardiff, 22 august: 5-5 la pauza, total 10. Sheff Utd-Derby, 24 octombrie, a avut 1-6 la pauza, total 7, deci succes aproape de limita. Aceste partide raman in date; nu elimin esecul sau succesul fragil pentru a imbunatati procentul.

## Calculul fix si toate sensibilitatile

```text
lambda_sezon = ((21/12 + 19/13) + (23/13 + 24/12))/2 = 3,490384615
lambda_teren = ((10/5 + 11/6) + (10/6 + 11/5))/2 = 3,85
lambda_combinat = 0,5*lambda_sezon + 0,5*lambda_teren = 3,670192308
P(total 1H <=7) = exp(-lambda)*SUM(lambda^k/k!, k=0..7) = 96,6128976%
Stres: lambda*1,2 = 4,404230769; P(total 1H <=7) = 92,1091154%
Cota echitabila matematica = 1/P = aproximativ 1,0351
```

Numai sezon: 97,36%; numai teren: 95,73%. Ponderarea 50/50 este euristica, esantioanele se suprapun. Cota echitabila nu este o oferta disponibila. Nu am consultat cote sau probabilitati ale tipsterilor.

| Varianta, aceeasi formula | N sezon A/B | N teren A/B | Model brut | Stres +20% |
|---|---:|---:|---:|---:|
| baseline_fotmob | 13/13 | 6/6 | 96,35% | 91,56% |
| use_espn_derby_stoke_6_corners | 13/13 | 6/6 | 96,05% | 90,96% |
| exclude_derby_stoke_conflicting_observation | 12/13 | 5/6 | 96,61% | 92,11% |
| exclude_first_half_red_card_matches | 12/13 | 5/6 | 96,28% | 91,43% |
| exclude_all_red_card_matches | 11/11 | 5/5 | 97,09% | 93,11% |

Randul oficial pentru estimarea finala este exclude_derby_stoke_conflicting_observation. Celelalte sunt diagnostice. Excluderea observatiei creste numeric modelul de la 96,35% la 96,61%; **nu creste automat increderea** si nu este o imbunatatire demonstrata a modelului. Folosind valoarea ESPN, rezultatul ar fi 96,05% si stresul 90,96%. Selectia ramane prima numeric si in aceasta varianta fata de QPR-Burnley (95,93%), dar diferenta este prea mica pentru a demonstra superioritate reala.

Sensibilitatile la eliminari din tabel folosesc screeningul FotMob ca baza. Unicul rosu in prima repriza a fost WBA la 34 in Derby-WBA, care a avut total 4 la pauza. Eliminarea de la 66 din Sheff Utd-Derby si cele din Norwich-Burnley/Millwall-Norwich, la 77/64, sunt dupa pauza; nu pot explica cauzal cornerele primei reprize. Nu modific procentul printr-un bonus/penalizare arbitrar.

**Riscul de model:** la Derby acasa, dupa excludere, varianta totalurilor este 13,20 fata de media 4,20. Seria este mai dispersata decat presupune Poisson; o medie mica poate ascunde reprize cu multe cornere grupate. Modelul 96,61% este peste frecventa pe teren de 90,91%, iar esantionul este mic. Nu pot echivala aceste calcule cu o rata reala validata de 96%.

## Istoric la pauza: ultimele 10 meciuri de campionat

### Derby

| Data | Partida | Cornere 1H gazda-oaspete | Total 1H | Incadrare |
|---|---|---:|---:|---|
| 2026-10-31 | [Millwall - Derby County](https://www.fotmob.com/matches/millwall-vs-derby-county/3d61hf#5836902) | 0-1 | 1 | Sub 7,5 |
| 2026-10-24 | [Sheffield United - Derby County](https://www.fotmob.com/matches/sheffield-united-vs-derby-county/2xizwz#5836894) | 1-6 | 7 | Sub 7,5 |
| 2026-10-17 | [Derby County - Stoke City](https://www.fotmob.com/matches/derby-county-vs-stoke-city/3fgl40#5836877) | 2-3 / 3-3 | 5 / 6 | Exclus: conflict intre surse |
| 2026-10-14 | [Queens Park Rangers - Derby County](https://www.fotmob.com/matches/derby-county-vs-queens-park-rangers/3f6zl5#5836873) | 1-0 | 1 | Sub 7,5 |
| 2026-10-10 | [Derby County - Wrexham](https://www.fotmob.com/matches/wrexham-vs-derby-county/3b7u3n#5836853) | 0-4 | 4 | Sub 7,5 |
| 2026-09-19 | [Burnley - Derby County](https://www.fotmob.com/matches/burnley-vs-derby-county/2sd9kk#5836840) | 3-2 | 5 | Sub 7,5 |
| 2026-09-12 | [Derby County - Birmingham City](https://www.fotmob.com/matches/birmingham-city-vs-derby-county/2xjeg0#5836829) | 2-1 | 3 | Sub 7,5 |
| 2026-09-09 | [Derby County - West Bromwich Albion](https://www.fotmob.com/matches/west-bromwich-albion-vs-derby-county/2xjsz2#5836823) | 3-1 | 4 | Sub 7,5 |
| 2026-09-05 | [West Ham United - Derby County](https://www.fotmob.com/matches/west-ham-united-vs-derby-county/2xhsc2#5836813) | 1-1 | 2 | Sub 7,5 |
| 2026-09-01 | [Portsmouth - Derby County](https://www.fotmob.com/matches/portsmouth-vs-derby-county/2vcple#5836792) | 4-0 | 4 | Sub 7,5 |

### Norwich

| Data | Partida | Cornere 1H gazda-oaspete | Total 1H | Incadrare |
|---|---|---:|---:|---|
| 2026-10-31 | [Norwich City - Watford](https://www.fotmob.com/matches/watford-vs-norwich-city/375k2g#5836903) | 5-0 | 5 | Sub 7,5 |
| 2026-10-24 | [Charlton Athletic - Norwich City](https://www.fotmob.com/matches/charlton-athletic-vs-norwich-city/2rpq65#5836889) | 0-2 | 2 | Sub 7,5 |
| 2026-10-17 | [Norwich City - Portsmouth](https://www.fotmob.com/matches/portsmouth-vs-norwich-city/2ru1jy#5836880) | 1-1 | 2 | Sub 7,5 |
| 2026-10-14 | [Norwich City - Southampton](https://www.fotmob.com/matches/southampton-vs-norwich-city/2rvm2w#5836871) | 0-1 | 1 | Sub 7,5 |
| 2026-10-10 | [Swansea City - Norwich City](https://www.fotmob.com/matches/norwich-city-vs-swansea-city/39cc2d#5836858) | 2-1 | 3 | Sub 7,5 |
| 2026-09-20 | [Norwich City - Bolton Wanderers](https://www.fotmob.com/matches/bolton-wanderers-vs-norwich-city/2sw7sn#5836844) | 1-1 | 2 | Sub 7,5 |
| 2026-09-12 | [Middlesbrough - Norwich City](https://www.fotmob.com/matches/middlesbrough-vs-norwich-city/2ss9sa#5836830) | 0-4 | 4 | Sub 7,5 |
| 2026-09-09 | [Norwich City - Birmingham City](https://www.fotmob.com/matches/birmingham-city-vs-norwich-city/2tzduw#5836824) | 0-4 | 4 | Sub 7,5 |
| 2026-09-05 | [Sheffield United - Norwich City](https://www.fotmob.com/matches/sheffield-united-vs-norwich-city/2tyzks#5836809) | 3-2 | 5 | Sub 7,5 |
| 2026-09-01 | [Stoke City - Norwich City](https://www.fotmob.com/matches/norwich-city-vs-stoke-city/3bm02w#5836795) | 1-1 | 2 | Sub 7,5 |

In ultimele 10 de campionat Derby exista 9 observatii utilizate si una exclusa; cele 9 au reusit, una cu exact 7. Norwich are 10/10 sub prag, maxim 5. Nu prezint aceste meciuri drept ultimele 10 din toate competitiile: Norwich a jucat si in EFL Cup. Predictia foloseste numai Championship 2026/27.

Seria Derby acasa de la cel mai recent, cu lipsa marcata: [exclus 5/6], 4, 3, 4, 0, 10. Norwich deplasare: 2, 3, 4, 5, 2, 5. Observatia 0 a lui Derby este confirmata de perioada FirstHalf si cronologia ESPN, nu un camp lipsa.

## Context, tactica, loturi si conditii

Campionat regulat, etapa 14: Derby locul 23, 7 puncte; Norwich locul 15, 15 puncte. Gazdele cauta puncte pentru salvare, oaspetii a treia victorie consecutiva. Nu exista retur eliminatoriu, scor agregat sau prelungiri; egalul este posibil, dar nu o calificare. [Context si clasament contemporan](https://stadiro.com/matches/england/championship/2026-11-03/derby-v-norwich/). Nici motivatia, nici golurile nu determina singure numarul cornerelor.

John Eustace/Philippe Clement sunt antrenorii actuali. Ultimele module verificate sunt 4-2-3-1 pentru ambele; titularii de azi nu sunt confirmati. Clement descrie Derby ca echipa directa, cu dueluri importante, si cere control cu mingea; recunoaste nevoia evaluarii prospetimii si posibilitatea schimbarilor dupa Watford. Papa Diallo lipseste, Crnac nu este pregatit sa inceapa dupa accidentarea lunga, iar Brooks are un volum neobisnuit de minute. [Conferinta oficiala Norwich, 2 noiembrie](https://www.canaries.co.uk/news/press-conference-clement-im-looking-forward-game-its-going-be-hard-battle).

Eustace anunta ca Ward, Ozoh si Meister necesita evaluare, iar Clarke si Agyemang au revenit la antrenament complet; revenirea la antrenament nu confirma titularizarea. [Relatarea actualizata cu declaratiile antrenorului](https://staging-aws.sports.yahoo.com/articles/derby-county-duo-closing-returns-170400754.html). Agregatoarele care ii dau in continuare pe toti absentii drept cert indisponibili sunt in conflict cu aceasta actualizare; nu folosesc o lista veche ca dovada a unei reduceri sigure a cornerelor.

Interpretare tactica, fara ajustari procentuale: jocul direct Derby poate trimite atacurile in dueluri centrale si faze relativ scurte; controlul Norwich ar putea tempera schimburile. Invers, daca Norwich fixeaza gazdele jos si foloseste repetat benzile, ori Derby urmareste scorul prin centrari, respingerile pot produce cornere in rafale. Nu presupun ca 4-2-3-1 garanteaza underul si nu confund un gol timpuriu cu reducerea presiunii.

Insight verificabil din singurul esec: [cronologia ESPN Derby-Cardiff](https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2/summary?event=401880326) arata cornere la 1, 3, 4, 7, 15, 17, 18, 18, 44 si 45+3. Pragul era depasit la 18, inaintea primului gol de la 19. Riscul nu se reduce la scenariul unui gol timpuriu: cornerele se pot grupa chiar la 0-0.

Ambele echipe au jucat pe 31 octombrie, aproximativ trei zile pana azi; meci intern, fara schimbare de fus orar. Pride Park are gazon natural; nu am inspectie a starii suprafetei ori itinerarul exact al oaspetilor. [Met Office, Derby, 3 noiembrie 04:00 UTC](https://weather.metoffice.gov.uk/forecast/gcqvn6pq4?new-design=false) indica 11 C maxima, 7 C minima, seara adesea uscata, posibilitate de averse izolate. Nu exista un motiv meteo verificat de veto; prognoza nu garanteaza conditiile efective.

## De ce aceasta partida fata de alternative

Derby-Norwich conducea deja screeningul la 96,35%, inaintea filtrarii observatiei conflictuale. Are 25/26 sub prag in screening si 24/25 dupa excludere; QPR-Burnley are 23/26, dar 12/12 pe teren fata de 10/11 la Derby-Norwich. Acesta este un contraargument pentru QPR, nu il ascund. QPR are si schimbarea recenta de antrenor Burnley, care reduce comparabilitatea istoricului. Diferenta de model este mica, nu o dovada de superioritate reala a lui Derby-Norwich.

Manchester United-Roma: 94,51% / 87,94%, 2/19 reprize >=8; Atletico-Bayern 93,62% / 86,27%, 3/19 >=8. Sunt proiectii ale datelor interne in Champions League, cu esantioane mici pe teren, nu probabilitati UCL validate. Nu am facut pentru ele aceeasi verificare contextuala aprofundata ca pentru Derby. Portsmouth-Swansea: 92,06% / 83,44%, 4/26 esecuri si 3/12 in configuratia terenului. Sirius-Halmstad: 75,45% / 58,27%, 9/51 esecuri si 14/51 reprize cu exact 7; nu este o alternativa buna in aceasta comparatie.

## Screening integral: statistica principala FotMob, inaintea verificarii finale

UCL/cupa folosesc datele campionatelor interne, nu un model validat pentru competitia zilei. Toate calculele eligibile sunt facute; doar finalistul are filtrul suplimentar documentat. Procentele sunt conditionate de sursa si de model, nu recomandari multiple.

| Rang | Partida | Competitie | Ora RO | Model brut | Stres +20% | >=8 distincte | Exact 7 distincte |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Derby County - Norwich City | eng.2 | 21:45 | 96,35% | 91,56% | 1/26 | 1/26 |
| 2 | Queens Park Rangers - Burnley | eng.2 | 21:45 | 95,93% | 90,73% | 3/26 | 0/26 |
| 3 | Manchester United - Roma | uefa.champions | 22:00 | 94,51% | 87,94% | 2/19 | 1/19 |
| 4 | Atlético Madrid - Bayern München | uefa.champions | 22:00 | 93,62% | 86,27% | 3/19 | 1/19 |
| 5 | LASK - Slovan Bratislava | uefa.champions | 22:00 | 93,45% | 85,95% | 2/24 | 2/24 |
| 6 | Feyenoord - Inter | uefa.champions | 22:00 | 93,22% | 85,54% | 3/21 | 1/21 |
| 7 | Portsmouth - Swansea City | eng.2 | 21:45 | 92,06% | 83,44% | 4/26 | 0/26 |
| 8 | Bodø/Glimt - Lille | uefa.champions | 22:00 | 91,12% | 81,79% | 4/34 | 4/34 |
| 9 | Villarreal - Paris Saint-Germain | uefa.champions | 22:00 | 89,61% | 79,21% | 4/20 | 2/20 |
| 10 | Blackburn Rovers - Stoke City | eng.2 | 21:45 | 89,32% | 78,71% | 6/26 | 2/26 |
| 11 | Birmingham City - Millwall | eng.2 | 22:00 | 88,06% | 76,64% | 5/26 | 2/26 |
| 12 | West Bromwich Albion - Southampton | eng.2 | 22:00 | 87,55% | 75,81% | 4/26 | 2/26 |
| 13 | Watford - Lincoln City | eng.2 | 21:45 | 87,03% | 74,98% | 6/26 | 2/26 |
| 14 | Preston North End - Sheffield United | eng.2 | 21:45 | 86,80% | 74,61% | 3/26 | 4/26 |
| 15 | Galatasaray - VfB Stuttgart | uefa.champions | 19:45 | 84,84% | 71,57% | 4/18 | 2/18 |
| 16 | Charlton Athletic - Bolton Wanderers | eng.2 | 21:45 | 84,69% | 71,33% | 4/26 | 2/26 |
| 17 | Lyngby - Randers FC | den.cup | 19:00 | 84,19% | 70,57% | 8/24 | 1/24 |
| 18 | Barcelona - Aston Villa | uefa.champions | 22:00 | 79,82% | 64,21% | 5/20 | 1/20 |
| 19 | Sirius - Halmstads BK | swe.1 | 20:00 | 75,45% | 58,27% | 9/51 | 14/51 |
| 20 | Shakhtar Donetsk - Sporting CP | uefa.champions | 19:45 | N/A: lipsa 1H Shakhtar | N/A | N/A | N/A |

## Evidenta prospectiva si limite

Estimarea finala oficiala este pe cele 25 observatii fara conflict 1H, cu Derby-Stoke exclus: 96,61% brut, 92,11% stres. Nu este un procent calibrat; frecventa pe teren este 90,91%, varianta mare Derby ramane riscul principal. Cota echitabila este circa 1,0351; disponibilitatea unui pret si profitabilitatea nu sunt demonstrate. Nu stabilesc miza si nu combin cu GOAL_BY_70.

Fisiere: all_models.csv (screening initial al celor 20), historical_corners_1h.csv (406 statistici reconciliate intern), rejected_corners.csv (11 lipsuri Shakhtar), coverage.csv, selected_history.csv (25 observatii finale), selected_exclusions.csv (conflict Derby-Stoke), provider_comparison.csv, sensitivity.csv, current_status.json, verification.json. Nicio observatie ori selectie nu contine rezultat viitor; productia si selection_engine raman neschimbate.
