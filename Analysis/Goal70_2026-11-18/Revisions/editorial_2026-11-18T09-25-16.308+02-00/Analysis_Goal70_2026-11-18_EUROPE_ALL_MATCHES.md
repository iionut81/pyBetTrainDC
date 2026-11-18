# GOAL_BY_70 —18 noiembrie2026

**NO BET.** Șasemodeleeligibilecalculatecomparativ,niciunselectatferm. LiderdescreeningPortugalete–RealUnión20RO,84,61%brut/77,62%stres,fairmatematic1,1819;17/23general și8/12teren. Frecvențesubmodel,transfercampionat–cupă și loturineconfirmate:nuconvertimrangulînrecomandare. Badajoz–Tenerife83,45/76,29%,Minera–Cartagena83,43/76,26%:alternativecomparate,fărăavantajvalidat. Nuestepragautomatnou.

|Rang brut|Meci / ora RO|Brut / AR20|Fair matematic|Istoric / teren|GF70 propriu pe teren A/B|
|---|---|---|---|---|---|
|1|Portugalete–Real Unión / 20:00|84.61% / 77.62%|1.1819|17/23;8/12|8/6;5/6|
|2|CD Badajoz–Tenerife / 21:00|83.45% / 76.29%|1.1983|20/24;11/12|6/6;7/6|
|3|Deportiva Minera–Cartagena / 21:30|83.43% / 76.26%|1.1987|17/23;12/13|8/6;5/7|
|4|Reus–Barakaldo / 21:30|75.08% / 67.09%|1.3320|17/23;8/10|2/5;3/5|
|5|Utebo–Sabadell / 20:00|75.02% / 67.04%|1.3329|19/25;9/13|4/5;2/8|
|6|Alcoyano–Mallorca / 21:00|72.03% / 63.91%|1.3884|18/24;8/12|7/6;2/6|

[Context și decizie](CONTEXT.md),[numărători și calcule](CALCULATIONS.md),[ultimeleoficiale](LAST10_OFFICIAL.md),[inventar](INVENTORY.md),[surse](sources.json).

**Cercetare parțială**, 2026-11-18T09:20:48.542+02:00. 22 partide identificate:9 naționale masculine seniori deja încheiate în noaptea RO,6CupaSpaniei,3CupaSloveniei,4UWCL feminin club. Nu există meci de națională feminină/juniori confirmat pentru18RO în sursele verificate; aceasta nu demonstrează absența mondială. 29 ligi convenite verificate:top20asociații UEFA clasamentul final2026 utilizat în proiect,plus România/Serbia și șapte secunde(Anglia,Germania,Spania,Italia,Franța,Belgia,Turcia). Zero partide identificate în aceste29, 25confirmări ESPN și4lipsă(Polonia,Croația,Serbia,Belgia2). Listele complete de sezonFotMob au fost consultate; răspunsurileESPN lipsă nu sunt absență. Pentru cupă s-au adăugat separat SegundaFederación,PrimeraFederación și PrvaLiga în verificarea istoricelor. Pagina UEFA dinamică nu a reprodus tabelul clasamentului, lista explicită este înuniverse_coverage.csv. Nu este top global complet.

273 partide istorice distincte cerute,4liste de goluri și138FirstHalf lipsă/neconciliate. Pentru cele6modeleGol70 ambele istorice sunt complete și au5/3. Toate6au fost calculate înaintea deciziei;zeromodelecornereeligibile. Amicale2026separat deNationsLeagueediția2026/27;seniori/juniori/masculin/feminin și cluburi distincte. Cele9naționale încheiate nu primesc probabilitate retrospectivă. UWCL are4/4general și2/2teren în ediția curentă,subminime;nu am completat cu istoricul masculin sau de națională. Lacluburi,CupaSpaniei folosește proiecție declarată între campionate și cupă,nu istoric propriu cupă validat. PentruCupaSloveniei datelebilaterale rămân insuficiente.

Surseprogram:FotMobglobal17/18/19UTC filtrat numeric înEurope/Bucharest+calendarecomplete ligi,ESPN,Footynomad181rândurimondiale(cu cluburi și în afara domeniului),FootballMania și căutări federative. UEFAWU17/WU19 pagini dinamice fără listă utilă:golul paginii nu este dovadădeabsență. [FIFA](https://www.fifa.com/en/tournaments/mens/u17worldcup/qatar-2026/articles/score-fixtures-match-schedule) confirmă startulCMU17pe19nov,nu18. SemnalareaOlandaU19–IrlandaU19 are conflict2025/2026;adversarii ediției2026dinUEFA/FAIdiferă,nu e meci confirmat azi. BraziliaU20–PeruU20 apare16nov23UTC/17ROîncoroborare,contrazicândetichetaSNA;nu îl mutăm automatpe18. USU20–Paraguay17nov17UTC confirmat calendarEintracht,nu18RO. WesternSprings–Tonga esteclubvsnațională,în afara domeniuluinaționalevsnaționale. Conflictele rămân însources/coverage_gaps.json.

Arhivă înRevisions/before_extension și before_identity_reconciliation,înainte deextindere/corecție. ȘaptecluburiU19ucrainene au fost eliminate din suplimentul naționalelor:identități de club,nu naționale. Endpoint9137s-a doveditChinaLeagueOne și nu este folosit îneligibilitate/proiecție;9138/8968suntligile spanioleconfirmateprinAPIechipe. Datele și sursele brute sunt păstrate.

## Lectii aplicate din review

|Lecție|Verificare / dovadă nouă|Statut și efect asupra preferinței|
|---|---|---|
|Model versus eveniment binar|Portugalete84,61%vs17/23=73,91% și8/12=66,67%;șase modele și120apariții oficiale recente|Aplicată:rang1este screening,fără recomandarefermă|
|GF70propriu,0–0/>70,scoruriample|Numărători toate6;Badajoz are0–0cuLasPalmasB/PuenteGenil și90+7cuBetisB;diagnostic81,53%fărăscoruriample|Aplicată:nu promovăm rang2doar pentru11/12teren;baselinepăstrat|
|Intern versus cupă|Toate6proiecții din campionate diferite înprima rundăCopa;fărăistoriccupăpentruaceeașiediție|Aplicată:incertitudinetransfer,NO BETfărăajustaredeprocent|
|Cornereexact7/>=8,dispersie|138FirstHalflipsăîn273;numărători și coziobservateînCSV/diagnostics|Aplicată:NULLnu0,nu modeldinFT;dispersiabilateralăincompletă,NO BET|
|90minute versusET|ReviewIrlanda2–1la90/2–2AET;gol62succes,gol108exclus|Aplicatălareview;istoriceAETnereconciliaterămânlipsă,fărăumplere|
|Loturi/antrenori/program/meteo|EgoitzBilbaooficial;Cerverapreview/plan;IñigoVélezpreview;4vs4zilePortugalete,3vs3Badajoz/Minera;rafale60,1Badajoz|Parțialaplicată:XI/loturibilaterale,tacticărecentă/minute/gazonneconfirmate;nicio preferințăfermă|
|Categorii/ediții/teren/status|UWCL4/2subminim;naționale9încheiate;șasecupeverificateneînceputeînfinal_status.json|Aplicată:fărăpooling,fărăpickretrospectiv;femininmondial/junioriincomplet|

Predarelecții2026-11-18T09:10:57.650+02:00;începutscanare2026-11-18T09:11:14.341+02:00. Auditulsistemelor17și protocolulchallengerilor rămânstudiuizolat;nuauînlocuitformula sau clasamentul brut.
