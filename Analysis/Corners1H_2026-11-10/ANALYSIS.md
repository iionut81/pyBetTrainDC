# CORNERS_1H_UNDER_7_5 — 10 noiembrie 2026

**NO BET. Niciun candidat analist.** Actualizat 2026-11-10T09:58:31.534+02:00, Europe/Bucharest. Review/lecții predate 2026-11-10T09:55:11.585+02:00; scanarea începută 2026-11-10T09:55:30.138+02:00. [Review](../Review_2026-11-09/REVIEW.md), [lecții](../Review_2026-11-09/LESSONS_FOR_NEXT_SCAN.md).

## Acoperire și limite

29ligi: primele20asociații după clasamentul final2026, plusRomânia/Serbia și7secunde urmărite (Anglia,Germania,Spania,Italia,Franța,Belgia,Turcia). [Tabel2026](https://kassiesa.net/uefa/data/method5/crank2026.html), [UEFA](https://www.uefa.com/nationalassociations/uefarankings/?year=2026); nu confundăm anul2026 cu lista de acces2026/27 bazată pe2025.

Zero partide identificate în aceste29ligi: globalFotMob,29calendare de sezon valide,25programeESPN valide; Polonia,Croația,Serbia,Belgia2 nu au confirmare independentăESPN. Un endpoint403/gol nu este tratat ca dovadă. Calendarele sezonului au fost confruntate separat; niciun meci suplimentar nu apare. Absența în cele4ligi are confirmare de la un singur furnizor, explicit limitată.

Separat24înregistrări de cupă:7Copa (6programate șiBadajoz–Tenerife fără dată),17EFLTrophy. NicioUCL/UEL/UECL masculină identificată. [RFEF](https://rfef.es/es/noticias/actualizado-el-calendario-de-los-partidos-aplazados-de-la-copa-del-rey-mapfre-y-de-la-copa) confirmă6partide10noiembrie șiBadajoz–TenerifeTBD; [EFL](https://www.efl.com/competitions/jersey-mikes-trophy/) are pagină dinamică, lista17nu este integral reconciliată independent. Restul49partide globale sunt înexclusions.csv (ligi inferioare/neurmărite, feminine, extraEuropa). Niciuna nu este declarată inexistentă.

**Cercetare parțială, nu top global.** Fiecare partidă de cupă are cel puțin un adversar fără istoric curent verificat de goluri≤70 șiFirstHalf în universul calendarelor colectate. Nu afirmăm că aceste date nu există public: colectarea lor pentru ligile inferioare/U21 nu este realizată aici.58istorice de campionat disponibile pentru echipele mapate sunt salvate; nu reprezintă istoric pentru ambii adversari și nu produc modele eligibile. Extinderea inventarului cupelor este arhivată înRevisions; inventarul inițial avea0partide de ligă, fără probabilități publicate.

## Metodă și eligibilitate

Numai sezon intern actual, minimum5observații sezon și3înconfigurația terenului pentru FIECARE echipă și fără observații cerute lipsă. λsezon=(GF_A/n_A+GA_B/n_B+GF_B/n_B+GA_A/n_A)/2, analog teren; λ=0,5λsezon+0,5λteren. Gol70:1−exp(−λ), stres0,8λ. Cornere:PoissonCDF(7,λ),stres1,2λ. Cota matematică1/p, fără preț bookmaker. Numărăm toate golurile valide≤70 inclusiv45+, fără70/90; cornereFirstHalf≤7 incluzând45+. Nicio substituție dinHT/FT. Modelul rămâne exploratoriu necalibrat.

Astăzi0modele eligibile pe fiecare piață; probabilitățile, stresul și rangurile rămân goale. Frecvențele parțiale dinCSV sunt exclusiv ale echipei cu istoric, nu ale duelului. Proiecția intern→cupă nu este validată și nu primește procente arbitrare. Nu folosim istoricul primei echipe pentruU21. Nu alegem meciuri începute și nu producem estimări retrospective.

## Lecții aplicate din review

| Lecție | Verificare / dovadă | Statut / efect |
| --- | --- | --- |
| Brut/AR20 versus frecvențe | 24identități evaluate,58istorice; cel puțin un adversar nemapat la fiecare | Aplicată: fără p sau rang inventat; NO BET |
| GF70 propriu,>70/0–0 | IeriSabadell0–0,LasPalmas90+5,Botoșani69; azi fără istoric bilateral complet | Aplicabilă; verificare finalist neaplicabilă, niciun finalist |
| Intern versusEuropa/cupă | 24cupe separate de29ligi;U21nu primește date de seniori | Aplicată: fără transfer automat |
| FirstHalf exact7/≥8,dispersie | Ieri6/1/2/2, niciunexact7; azi istoricul bilateral incomplet | Aplicabilă, calcul comparativ neverificat |
| Loturi/antrenori,odihnă/meteo | Nu există finalist eligibil; nu s-a efectuat verificare aprofundată24loturi | Neverificată; fără explicații inventate |
| Program/reprogramări | 29calendare sezon;RFEF6meciuri+BadajozTBD | Aplicată, corectează tratamentul placeholderului |
| Cauze/calibrare | Rezultatele nu explică tactic mecanismul; minute26/27conflictKBK | Aplicată: arhive și modele păstrate |

## Cele29ligi

| Țară/ligă | Rang2026 | Global | Sezon | ESPN | Limită |
| --- | --- | --- | --- | --- | --- |
| England /eng.1 | 1 | 0 | 0 | 0 | programe confruntate |
| Italy /ita.1 | 2 | 0 | 0 | 0 | programe confruntate |
| Spain /esp.1 | 3 | 0 | 0 | 0 | programe confruntate |
| Germany /ger.1 | 4 | 0 | 0 | 0 | programe confruntate |
| France /fra.1 | 5 | 0 | 0 | 0 | programe confruntate |
| Portugal /por.1 | 6 | 0 | 0 | 0 | programe confruntate |
| Netherlands /ned.1 | 7 | 0 | 0 | 0 | programe confruntate |
| Belgium /bel.1 | 8 | 0 | 0 | 0 | programe confruntate |
| Turkey /tur.1 | 9 | 0 | 0 | 0 | programe confruntate |
| Czechia /cze.1 | 10 | 0 | 0 | 0 | programe confruntate |
| Greece /gre.1 | 11 | 0 | 0 | 0 | programe confruntate |
| Poland /pol.1 | 12 | 0 | 0 | nesuportat | un furnizor, două programe |
| Denmark /den.1 | 13 | 0 | 0 | 0 | programe confruntate |
| Norway /nor.1 | 14 | 0 | 0 | 0 | programe confruntate |
| Cyprus /cyp.1 | 15 | 0 | 0 | 0 | programe confruntate |
| Switzerland /sui.1 | 16 | 0 | 0 | 0 | programe confruntate |
| Austria /aut.1 | 17 | 0 | 0 | 0 | programe confruntate |
| Scotland /sco.1 | 18 | 0 | 0 | 0 | programe confruntate |
| Sweden /swe.1 | 19 | 0 | 0 | 0 | programe confruntate |
| Croatia /cro.1 | 20 | 0 | 0 | nesuportat | un furnizor, două programe |
| Romania /rou.1 | supplement | 0 | 0 | 0 | programe confruntate |
| Serbia /srb.1 | supplement | 0 | 0 | nesuportat | un furnizor, două programe |
| England 2 /eng.2 | supplement | 0 | 0 | 0 | programe confruntate |
| Germany 2 /ger.2 | supplement | 0 | 0 | 0 | programe confruntate |
| Spain 2 /esp.2 | supplement | 0 | 0 | 0 | programe confruntate |
| Italy 2 /ita.2 | supplement | 0 | 0 | 0 | programe confruntate |
| France 2 /fra.2 | supplement | 0 | 0 | 0 | programe confruntate |
| Belgium 2 /bel.2 | supplement | 0 | 0 | nesuportat | un furnizor, două programe |
| Turkey 2 /tur.2 | supplement | 0 | 0 | 0 | programe confruntate |

## Inventar cupă și excluderi din model

| ID | Competiție | Meci | KickoffRO furnizor | Stare | Lipsă |
| --- | --- | --- | --- | --- | --- |
| 6386040 | esp.copa_del_rey | Anaitasuna FT–Deportivo Alavés | 2026-11-10T20:00:00.000+02:00 | neînceput în inventar | one_current_domestic_history_unavailable |
| 6386069 | esp.copa_del_rey | Pamplona–Eibar | 2026-11-10T20:00:00.000+02:00 | neînceput în inventar | one_current_domestic_history_unavailable |
| 6386050 | esp.copa_del_rey | CD Badajoz–Tenerife | TBD oficial | amânat; fără dată | postponed_RFEF_date_TBD |
| 6386066 | esp.copa_del_rey | Mijas-Las Lagunas–Real Jaén | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6386039 | esp.copa_del_rey | SD Amorebieta–Logroñés | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6386086 | esp.copa_del_rey | Trival Valderas–Leganés | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | one_current_domestic_history_unavailable |
| 6386084 | esp.copa_del_rey | Tavernes de la Valldigna–Valencia | 2026-11-10T22:00:00.000+02:00 | neînceput în inventar | one_current_domestic_history_unavailable |
| 6014863 | eng.efl_trophy | Crewe Alexandra–Wigan Athletic | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6015009 | eng.efl_trophy | Mansfield Town–Chesterfield | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6015015 | eng.efl_trophy | Stockport County–Tranmere Rovers | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6015567 | eng.efl_trophy | Burton Albion–Notts County | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6015568 | eng.efl_trophy | Grimsby Town–Nottingham Forest U21 | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6015886 | eng.efl_trophy | Huddersfield Town–Liverpool U21 | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016270 | eng.efl_trophy | Oxford United–Cheltenham Town | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016299 | eng.efl_trophy | Leicester City–Walsall | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016300 | eng.efl_trophy | Stevenage–Fulham U21 | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016308 | eng.efl_trophy | Bristol Rovers–Reading | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016320 | eng.efl_trophy | Plymouth Argyle–Swindon Town | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016342 | eng.efl_trophy | Barnet–Leyton Orient | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016326 | eng.efl_trophy | Northampton Town–Gillingham | 2026-11-10T21:00:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6015001 | eng.efl_trophy | Bradford City–York City | 2026-11-10T21:30:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016333 | eng.efl_trophy | Bromley–Milton Keynes Dons | 2026-11-10T21:30:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016334 | eng.efl_trophy | Crawley Town–Brentford U21 | 2026-11-10T21:30:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |
| 6016313 | eng.efl_trophy | Colchester United–Ipswich Town U21 | 2026-11-10T21:45:00.000+02:00 | neînceput în inventar | both_current_domestic_histories_unavailable |

Surse brute: fixtures_global.json, ESPN/, Schedules/, MatchData/, historical_observations.csv și JSON, universe_coverage.csv, cups_coverage.json, sources.json înGoal70_2026-11-10. Contextul aprofundat al finalistelor este neaplicabil deoarece nu există candidat cu model eligibil. Nu au fost adăugate alegeri ale utilizatorului. Pas următor: reîncercarea restanțelor și inventarul11noiembrie dupăreview; eventualele date inferioare trebuie verificate înaintea unui model, niciodată reconstruite după rezultat.
