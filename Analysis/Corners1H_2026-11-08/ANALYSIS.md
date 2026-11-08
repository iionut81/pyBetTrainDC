# CORNERS_1H_UNDER_7_5 — 8 noiembrie 2026

Actualizare editorială și verificare finală: 2026-11-08 T 10:44:54.006+02:00, Europe/Bucharest. Valorile, rangurile și timestampurile predicțiilor inițiale rămân neschimbate; variantele anterioare ale raportului sunt păstrate în Revisions/.

[Review 7 noiembrie](../Review_2026-11-07/REVIEW.md) și [lecțiile predate](../Review_2026-11-07/LESSONS_FOR_NEXT_SCAN.md) au fost salvate la 10:32:31 RO, înaintea cercetării de azi, începută la 10:32:50 RO. Cele două alegeri explicite de ieri au reușit: Blackburn–West Ham primul gol 54, Excelsior–Go Ahead 3 cornere. Sporting a fost revocat și nu este recreat ca alegere.

Sunt inventariate **105 partide în 29 de ligi**: primele 20 de asociații UEFA din clasamentul încheiat în 2026, plus România, Serbia și cele 7 divizii secunde urmărite. [Clasament UEFA 2026](https://www.uefa.com/nationalassociations/uefarankings/?year=2026). Anglia 2 și Franța 2 nu au partide în cele două programe consultate și calendarul sezonului; celelalte 27 de ligi au meciuri. Sunt **103 modele Gol 70 și 81 de cornere**, din 1974 de partide istorice unice anterioare zilei predicției. Toate partidele eligibile au fost calculate înaintea finalistelor.

FCAndorra–Mallorca și Estoril–Marítimo sunt anulate și excluse din selecție. Celelalte 103 partide au fost verificate neîncepute. Lipsesc 24 de modele de cornere, inclusiv cele 2 partide anulate; motivele și numărătorile sunt în CSV. Patru partide FA Cup cu echipe din ligi inferioare sunt inventariate separat și excluse deoarece nu am colectat istoricul comparabil. Nu apar partide UCL/UEL/UECL în programul global consultat. Nu generalizăm această constatare la toate cupele lumii.

**Cercetare parțială.** FotMob global și calendarul sezonului acoperă cele 29 de ligi, dar confirmarea independentă nu este completă. ESPN nu este suportat pentru Polonia, Croația, Serbia și Belgia 2; răspunsurile goale din Cehia, Cipru, Elveția, România și Turcia 2 contrazic FotMob și sunt marcate ca diferențe între surse. Nu reprezintă dovadă de absență a meciurilor. Opt partide au context API extins,16 echipe și 159 de rânduri cu ultimele meciuri oficiale: Le Mans are numai 9 meciuri disponibile, iar 6 observații FirstHalf și o listă de goluri nu pot fi reconciliate. Lipsa nu este zero. Titularii și unele informații medicale rămân neconfirmate.

Metoda este cea salvată: campionatul actual, minimum 5 meciuri de sezon și 3 pe teren per echipă, fără observații obligatorii lipsă; λ combină 50/50 intensitatea pe sezon și pe teren. Gol 70 folosește 1−exp(−λ), stres 0,8 λ; cornerele PoissonCDF(7,λ), stres 1,2 λ. Procentele sunt exploratorii, necalibrate. Cota 1/p este exclusiv matematică. Nu am folosit cote bookmaker, xG inventat sau ajustări procentuale din context.

[Acoperire](../Goal 70_2026-11-08/universe_coverage.csv), [excluderi](../Goal 70_2026-11-08/exclusions.csv), [cupe](../Goal 70_2026-11-08/cups_coverage.json), [statusuri](../Goal 70_2026-11-08/status_at_selection.csv), [surse publice arhivate](../Goal 70_2026-11-08/sources.json), [detalii contextuale](../Goal 70_2026-11-08/CONTEXT_SOURCES.md).

## Candidat analist și preferință individuală: Girona–Almería,17:15 RO

Eveniment: maximum 7 cornere cumulate FirstHalf, inclusiv 45+. **Rang brut 3;96,97%brut,92,84%în stres, cotă matematică 1,0313.** Istoric 24/24 partide unice favorabile,13/13 pe teren, o repriză exact 7 și zero≥8. Serii cronologice: Girona acasă 1,4,6,5,4,3,4; Almería afară 3,3,5,3,3,3. Medii 3,86/3,33 și varianțe 2,48/0,67. Cele 7 și 6 observații sunt puține; zero eșecuri istorice nu înseamnă că 8 cornere sunt imposibile azi.

Segunda, etapa 13, Montilivi, gazon natural. Fișa oficială Almería arată 21 puncte Girona și 26 Almería după 12 meciuri; locul Girona diferă 7/8 între avancronica din 4 noiembrie și tabelul curent, o diferență de moment nefolosită în formulă. Nu este eliminatorie. API și ESPN indică 15:15 UTC=17:15 RO. Fișa clubului arată 15:15 fără fus explicit, iar Mundo Deportivo 16:15 local; folosim UTC explicit și păstrăm ambiguitatea afișării clubului.

Contraargumentul principal este ofensiv: [Quique Álvarez](https://www.gironafc.cat/es/noticias/estoy-convencido-de-que-el-equipo-dara-manana-la-mejor-version) cere presiune mai agresivă, iar [García Pimienta](https://www.udalmeriasad.com/noticias/nos-enfrentamos-a-un-equipo-que-ha-jugado-champions) descrie combinațiile și contrapresingul Girona. Presiunea și blocajele pot produce cornere în serie. Scenariul favorabil este că posesia construită și ieșirile controlate limitează atacurile repetate, însă nu avem măsurători comparabile care să valideze această explicație tactică.

Girona API lastStarting 11=4-3-3 și Almería 4-2-3-1; nu sunt titulari confirmați. David López apare în XI vechi și suspendat: suspendarea are prioritate. Clubul Girona confirmă problema lui Jastin. [Convocarea oficială Almería](https://www.udalmeriasad.com/noticias/convocatoria-para-el-partido-en-montilivi) confirmă 22 jucători și absențele Chirino, Ely și Melamed; convocarea rezolvă incertitudinea anterioară Chirino. Identitatea antrenorilor este confirmată primar, dar cronologia completă a instalării și schimbărilor rămâne limitată.

Ultimul campionat pe 31 octombrie pentru ambele,8 zile între date; cupa 27 octombrie separată. Almería are al treilea meci oficial consecutiv afară, inclusiv cupa, ceea ce nu implică automat lipsă de odihnă. Clubul confirmă zbor charter la Girona pe 7 noiembrie; nu cunoaștem orele de somn. Ultimele 10 oficiale au 9/9 FirstHalfverificabile fiecare, câte o partidă de cupă lipsă; nu sunt 10/10. Gol 70 Girona include Mallorca 0–0 și Real Sociedad B 86, fără transferarea acelor rezultate spre piața cornerelor.

Meteo independent la 17 RO:18,8°C, ploaie 0%, vânt 18,3 km/h; API 19°C/ploaie 2%, unitatea vântului nespecificată. Nu avem inspecție a gazonului. Diagnosticul fără istorice cu roșu în prima repriză produce 96.81% / 92.53%, cu 2 partide unice eliminate; nu modifică predicția 96,97%/92,84%. Numărătorile sunt în red_card_sensitivity.json.

### Alternative

Le Mans–Monaco conduce brut cu 99,23%/97,93%,17/18 sezon și 9/9 teren. Le Mans acasă are seria 2,7,0,1, media 2,50 și varianța 9,67 în numai 4 meciuri, față de Monaco afară media 2,60/varianța 1,30 în 5. [Programul oficial Monaco](https://www.asmonaco.com/fr/news/le-planning-des-rouge-et-blanc-avant-nordsjaelland-et-le-mans) confirmă Europa 5 noiembrie și deplasarea 8 noiembrie. Intensitatea medie mică produce un Poisson foarte ridicat, fără să facă 99,23%o rată validată.

Celta–Levante 97,02%/92,96%,21/22 și 12/12 tereneste foarte apropiat statistic; varianțe 4,67/4 față de medii 2,67/4 și program european Celta pe 5 noiembrie. Rio Ave–Braga 96,92%/92,75%,18/20 și 9/10 are două≥8, cu loturi probabile și fără confirmare primară completă. Atlético–Barcelona 96,37%are 17/22 și cinci≥8, o divergență importantă model/frecvență, fără context aprofundat azi. Prefer Girona prin seria completă și dispersie, păstrând riscurile pressingului și suspendării López. Nu există test care să demonstreze superioritatea acestei preferințe.

Aceasta este singura preferință individuală implicită a zilei. Publicăm separat candidatul Gol 70 conform cererii, fără combinație și fără a adăuga o alegere a utilizatorului.

| Eșantion | n | GF70 / cornere obținute | GA70 / cornere cedate | Reușite | Tardiv / 0–0 | Total mediu / varianță | Exact7 / ≥8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| home | 12 | 20 | 20 | 12 | — | 3.3333333333333335/2.6060606060606064 | 0/0 |
| away | 12 | 20 | 26 | 12 | — | 3.8333333333333335/2.333333333333333 | 1/0 |
| home_venue | 7 | 13 | 14 | 7 | — | 3.857142857142857/2.4761904761904763 | 0/0 |
| away_venue | 6 | 3 | 17 | 6 | — | 3.3333333333333335/0.6666666666666666 | 0/0 |

λ sezon=(20/12+26/12+20/12+20/12)/2=3.5833.

λ teren=(13/7+17/6+3/6+14/7)/2=3.5952.

λ combinat=0,5λ sezon+0,5λ teren=3.5893. Brut=96.97%, numai sezon=96.99%, numai teren=96.94%, AR20=92.84%, cotă exclusiv matematică=1.0313.

P=exp(−λ)Σλ^k/k!, k=0..7; AR20 folosește1,2λ. Ponderarea este euristică, eșantioanele se suprapun și procentele nu sunt calibrate. Golurile de după deschiderea scorului pot umfla intensitatea fără să valideze primul gol. Nu modificăm procentele din context.
## Lecții aplicate din review

| Lecție | Verificare nouă | Statut și efect |
| --- | --- | --- |
| Model versus frecvență și AR 20 | Excelsior 96,47%a pierdut Gol 70 la 72; Parma 96,92%a avut 8 cornere și Sporting 95,46%a avut 10. Azi afișăm model, stres și numitori | Aplicată: liderii bruți nu sunt aleși automat; fără procente modificate |
| GF 70 propriu, deschideri tardive și 0–0 | Cambuur 8/6 șiPSV 9/5, față deAtalanta 3/4; PEC 83 șiLeipzig 75 păstrate | Aplicată: susține preferința contextuală, fără garanție |
| Campionat separat de Europa/cupă | Formula doar campionatul;159 rânduri oficiale diagnostice, șase FirstHalf lipsă | Aplicată; lipsurile și transferabilitatea rămân explicite |
| Dispersie și exact 7/≥8 | Girona 24/24, un exact 7,zero≥8; varianțe 2,48/0,67, versusLeMans 9,67 | Aplicată: preferință Girona, fără excluderi automate dintr-o zi |
| Antrenori și loturi actuale | Bosz,Quique șiGarcíaPimienta confirmați; Chirino absent în convocare; López dinXIvechi suspendat | Aplicată parțial: titularii și cronologia completă rămân neconfirmate |
| Status, odihnă și vreme |103 neîncepute,2 anulate; verificare finală a finalistelor, charterAlmería și Open-Meteo | Aplicată; vremea este prognoză, fără bonus numeric |
| Cauze tactice și calibrare | Fără dovadă cauzală exhaustivă sau test cronologic nou | Neverificate; fără tuning sau promisiunea eliminării eșecurilor |

## Comparația integrală

| Rang brut original | Partidă / ora RO | Brut | AR20 | Istoric unic | Teren | Rol / lipsă |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Le Mans–Monaco 18:15 | 99.23% | 97.93% | 17/18 | 9/9 | comparat |
| 2 | Celta Vigo–Levante 19:30 | 97.02% | 92.96% | 21/22 | 12/12 | comparat |
| 3 | Girona–Almería 17:15 | 96.97% | 92.84% | 24/24 | 13/13 | candidat analist |
| 4 | Rio Ave–Braga 22:30 | 96.92% | 92.75% | 18/20 | 9/10 | comparat |
| 5 | Atlético Madrid–Barcelona 22:00 | 96.37% | 91.61% | 17/22 | 10/10 | comparat |
| 6 | Rizespor–Erzurumspor FK 12:30 | 96.24% | 91.34% | 18/20 | 7/9 | comparat |
| 7 | Partizan Beograd–Mačva Šabac 18:00 | 95.88% | 90.62% | 26/27 | 13/14 | comparat |
| 8 | Brighton & Hove Albion–Brentford 16:00 | 95.82% | 90.49% | 18/18 | 8/8 | comparat |
| 9 | Osijek–Hajduk Split 16:00 | 95.80% | 90.45% | 23/23 | 11/11 | comparat |
| 10 | Mardin 1969 Spor–Batman Petrolspor 18:00 | 95.50% | 89.87% | 25/26 | 13/13 | comparat |
| 11 | Carrarese–Sampdoria 18:15 | 95.18% | 89.23% | 18/20 | 11/11 | comparat |
| 12 | LR Vicenza–Cremonese 20:30 | 94.96% | 88.81% | 19/20 | 8/9 | comparat |
| 13 | Cambuur–PSV Eindhoven 17:45 | 94.85% | 88.59% | 21/22 | 11/11 | comparat |
| 14 | CFR Cluj–Universitatea Craiova 19:00 | 94.34% | 87.61% | 26/28 | 11/12 | comparat |
| 15 | Ipswich Town–AFC Bournemouth 16:00 | 94.33% | 87.60% | 18/18 | 8/8 | comparat |
| 16 | Virtus Entella–Modena 16:00 | 94.11% | 87.18% | 20/20 | 10/10 | comparat |
| 17 | Magdeburg–St. Pauli 14:30 | 93.78% | 86.56% | 19/20 | 10/10 | comparat |
| 18 | Boluspor–Bodrum FK 15:00 | 93.70% | 86.42% | 23/25 | 10/12 | comparat |
| 19 | Genoa–Milan 16:00 | 93.61% | 86.24% | 18/20 | 10/11 | comparat |
| 20 | K. Lierse SK–Eupen 17:00 | 93.33% | 85.73% | 15/16 | 6/7 | comparat |
| 21 | Bandırmaspor–Fatih Karagümrük 12:30 | 93.32% | 85.71% | 24/26 | 13/13 | comparat |
| 22 | Vojvodina–FK Crvena Zvezda 20:00 | 93.26% | 85.60% | 24/27 | 13/14 | comparat |
| 23 | Monza–Atalanta 16:00 | 93.17% | 85.43% | 17/20 | 8/8 | comparat |
| 24 | FC Midtjylland–Sønderjyske 19:00 | 92.84% | 84.84% | 22/25 | 9/12 | comparat |
| 25 | Çorum FK–Fenerbahçe 15:00 | 92.53% | 84.27% | 19/20 | 9/9 | comparat |
| 26 | Tromsø–Bodø/Glimt 15:30 | 92.18% | 83.65% | 44/49 | 23/24 | comparat |
| 27 | Estrela da Amadora–Benfica 22:30 | 92.14% | 83.58% | 18/20 | 10/10 | comparat |
| 28 | Hartberg–Altach 15:30 | 91.93% | 83.21% | 19/21 | 10/11 | comparat |
| 29 | Olympiacos–PAOK Thessaloniki 21:00 | 91.85% | 83.06% | 16/18 | 8/8 | comparat |
| 30 | Gent–Kortrijk 17:00 | 91.75% | 82.88% | 20/20 | 10/10 | comparat |
| 31 | Union St.Gilloise–Genk 19:30 | 91.75% | 82.88% | 20/20 | 10/10 | comparat |
| 32 | Angers–Nice 18:15 | 91.74% | 82.86% | 16/18 | 8/9 | comparat |
| 33 | AZ Alkmaar–FC Groningen 17:45 | 91.67% | 82.75% | 20/22 | 9/10 | comparat |
| 34 | NK Varaždin–Dinamo Zagreb 19:00 | 91.58% | 82.59% | 21/22 | 10/11 | comparat |
| 35 | Freiburg–Union Berlin 18:30 | 91.43% | 82.32% | 15/16 | 7/7 | comparat |
| 36 | Lens–Marseille 21:45 | 91.40% | 82.27% | 18/18 | 8/8 | comparat |
| 37 | Crystal Palace–Liverpool 16:00 | 91.11% | 81.76% | 18/18 | 8/8 | comparat |
| 38 | SC Heerenveen–Feyenoord 13:15 | 90.77% | 81.17% | 20/22 | 11/12 | comparat |
| 39 | St. Gallen–Lugano 15:00 | 90.09% | 80.01% | 21/23 | 9/11 | comparat |
| 40 | Servette–FC Zürich 17:30 | 90.09% | 80.01% | 21/23 | 11/12 | comparat |
| 41 | Manchester United–Aston Villa 18:30 | 90.03% | 79.91% | 15/18 | 7/8 | comparat |
| 42 | Hoffenheim–Schalke 04 20:30 | 90.00% | 79.86% | 16/16 | 8/8 | comparat |
| 43 | Thun–Basel 17:30 | 89.98% | 79.82% | 21/23 | 11/11 | comparat |
| 44 | Club Brugge–Anderlecht 14:30 | 89.60% | 79.19% | 18/20 | 10/10 | comparat |
| 45 | Inter–Como 19:00 | 89.53% | 79.06% | 17/20 | 10/11 | comparat |
| 46 | Napoli–Lazio 13:30 | 89.49% | 79.00% | 17/20 | 8/9 | comparat |
| 47 | Rennes–Lille 16:00 | 89.44% | 78.91% | 17/18 | 8/9 | comparat |
| 48 | Elche–Real Betis 15:00 | 89.01% | 78.20% | 19/22 | 8/10 | comparat |
| 49 | FC Utrecht–NEC Nijmegen 21:00 | 88.93% | 78.06% | 19/22 | 9/10 | comparat |
| 50 | Sunderland–Chelsea 16:00 | 88.61% | 77.54% | 16/18 | 8/8 | comparat |
| 51 | Villarreal–Getafe 19:30 | 88.49% | 77.34% | 19/22 | 9/10 | comparat |
| 52 | FC Twente–Ajax 15:30 | 88.18% | 76.84% | 22/22 | 10/10 | comparat |
| 53 | WSG Tirol–Austria Wien 15:30 | 88.17% | 76.82% | 19/21 | 11/11 | comparat |
| 54 | AGF–AC Horsens 15:00 | 88.09% | 76.69% | 22/25 | 12/12 | comparat |
| 55 | 1. FC Köln–Bayer Leverkusen 16:30 | 87.62% | 75.93% | 15/16 | 8/8 | comparat |
| 56 | Nacional–Académico Viseu 17:30 | 87.43% | 75.63% | 16/20 | 9/10 | comparat |
| 57 | Slavia Prague–Artis Brno 16:00 | 87.35% | 75.49% | 22/26 | 10/13 | comparat |
| 58 | Hamarkameratene–Sarpsborg 08 18:00 | 87.21% | 75.26% | 42/49 | 19/24 | comparat |
| 59 | Brommapojkarna–Mjällby 17:30 | 86.87% | 74.73% | 45/53 | 21/26 | comparat |
| 60 | FC København–Brøndby IF 13:00 | 86.86% | 74.72% | 20/25 | 11/12 | comparat |
| 61 | Jagiellonia Białystok–GKS Katowice 15:45 | 86.39% | 73.97% | 20/25 | 10/13 | comparat |
| 62 | Halmstads BK–Djurgården 15:00 | 86.30% | 73.82% | 47/53 | 25/26 | comparat |
| 63 | Teplice–Pardubice 14:00 | 85.82% | 73.07% | 20/26 | 11/14 | comparat |
| 64 | AEK Athens–Levadiakos 17:00 | 85.70% | 72.89% | 13/17 | 5/8 | comparat |
| 65 | Sandefjord–Molde 18:00 | 85.28% | 72.24% | 45/49 | 22/24 | comparat |
| 66 | Viking–KFUM 18:00 | 84.82% | 71.53% | 41/49 | 21/24 | comparat |
| 67 | Fredrikstad–Brann 20:15 | 84.23% | 70.63% | 38/49 | 19/24 | comparat |
| 68 | Viktoria Plzeň–Sparta Prague 19:00 | 84.02% | 70.32% | 21/26 | 12/14 | comparat |
| 69 | Jablonec–Mladá Boleslav 16:00 | 83.89% | 70.13% | 23/26 | 12/14 | comparat |
| 70 | Bochum–Karlsruher SC 14:30 | 83.49% | 69.53% | 18/20 | 8/9 | comparat |
| 71 | IFK Göteborg–Malmö FF 15:00 | 82.86% | 68.58% | 45/53 | 22/26 | comparat |
| 72 | Randers FC–Nordsjælland 17:00 | 81.89% | 67.16% | 19/25 | 10/14 | comparat |
| 73 | St.Truiden–Zulte Waregem 20:15 | 81.86% | 67.13% | 17/20 | 8/10 | comparat |
| 74 | Kalmar FF–Sirius 17:30 | 80.23% | 64.79% | 46/53 | 22/26 | comparat |
| 75 | Lillestrøm–Aalesund 18:00 | 79.66% | 63.98% | 42/49 | 21/24 | comparat |
| 76 | KSC Lokeren–FCV Dender EH 20:15 | 78.53% | 62.42% | 13/17 | 5/8 | comparat |
| 77 | FC Heidenheim–Hannover 96 14:30 | 78.41% | 62.25% | 18/20 | 9/10 | comparat |
| 78 | Real Sociedad–Rayo Vallecano 17:15 | 78.24% | 62.01% | 19/21 | 8/10 | comparat |
| 79 | Sturm Graz–Salzburg 18:00 | 76.25% | 59.33% | 16/21 | 7/10 | comparat |
| 80 | Celtic–Kilmarnock 18:00 | 74.40% | 56.89% | 18/23 | 9/12 | comparat |
| 81 | Heart of Midlothian–Falkirk 17:00 | 73.63% | 55.91% | 18/22 | 7/11 | comparat |
| 82 | Brest–Lyon 18:15 | lipsă | lipsă | 15/17 | 7/8 | missing_observations_or_sample_below_5_3 |
| 83 | Le Havre–Lorient 18:15 | lipsă | lipsă | 17/17 | 9/9 | missing_observations_or_sample_below_5_3 |
| 84 | Fiorentina–Juventus 21:45 | lipsă | lipsă | 16/19 | 8/10 | missing_observations_or_sample_below_5_3 |
| 85 | Anorthosis–Omonia Nicosia 16:00 | lipsă | lipsă | 0/0 | 0/0 | missing_observations_or_sample_below_5_3 |
| 86 | APOEL Nicosia–Aris Limassol 19:00 | lipsă | lipsă | 0/0 | 0/0 | missing_observations_or_sample_below_5_3 |
| 87 | Pafos FC–Omonia 29 Maiou 19:00 | lipsă | lipsă | 0/0 | 0/0 | missing_observations_or_sample_below_5_3 |
| 88 | Legia Warszawa–Raków Częstochowa 18:30 | lipsă | lipsă | 24/25 | 11/12 | missing_observations_or_sample_below_5_3 |
| 89 | Lech Poznań–Zagłębie Lubin 21:15 | lipsă | lipsă | 20/24 | 11/13 | missing_observations_or_sample_below_5_3 |
| 90 | FC Andorra–Mallorca 15:00 | lipsă | lipsă | 22/23 | 11/12 | missing_observations_or_sample_below_5_3 |
| 91 | Las Palmas–Eldense 19:30 | lipsă | lipsă | 22/23 | 12/12 | missing_observations_or_sample_below_5_3 |
| 92 | Sporting Gijón–Real Valladolid 19:30 | lipsă | lipsă | 21/23 | 10/12 | missing_observations_or_sample_below_5_3 |
| 93 | Estoril–Marítimo 17:30 | lipsă | lipsă | 19/20 | 10/10 | missing_observations_or_sample_below_5_3 |
| 94 | Santa Clara–Gil Vicente 17:30 | lipsă | lipsă | 16/19 | 7/9 | missing_observations_or_sample_below_5_3 |
| 95 | Vitória de Guimarães–FC Porto 20:00 | lipsă | lipsă | 14/19 | 8/10 | missing_observations_or_sample_below_5_3 |
| 96 | Asteras Tripolis–Kalamata 17:00 | lipsă | lipsă | 17/17 | 9/9 | missing_observations_or_sample_below_5_3 |
| 97 | Iraklis–Panathinaikos 17:30 | lipsă | lipsă | 15/17 | 6/7 | missing_observations_or_sample_below_5_3 |
| 98 | OFI Crete–Atromitos 18:30 | lipsă | lipsă | 15/16 | 8/8 | missing_observations_or_sample_below_5_3 |
| 99 | Alanyaspor–Trabzonspor 17:30 | lipsă | lipsă | 18/19 | 9/9 | missing_observations_or_sample_below_5_3 |
| 100 | Beşiktaş–Gençlerbirliği 20:00 | lipsă | lipsă | 16/19 | 7/9 | missing_observations_or_sample_below_5_3 |
| 101 | Sepsi OSK–FC Voluntari 16:00 | lipsă | lipsă | 26/27 | 12/12 | missing_observations_or_sample_below_5_3 |
| 102 | Ümraniyespor–Kayserispor 15:00 | lipsă | lipsă | 24/24 | 13/13 | missing_observations_or_sample_below_5_3 |
| 103 | AIK–Häcken 17:30 | lipsă | lipsă | 48/52 | 22/26 | missing_observations_or_sample_below_5_3 |
| 104 | Novi Pazar–Mladost Lučani 14:00 | lipsă | lipsă | 23/26 | 12/14 | missing_observations_or_sample_below_5_3 |
| 105 | FK Radnik Surdulica–OFK Beograd 14:00 | lipsă | lipsă | 25/26 | 14/14 | missing_observations_or_sample_below_5_3 |

### Ultimele zece oficiale: Girona

| Data UTC | Competiție | Meci | FT90 | Primul gol | Gol70 | Cornere1H | Under7,5 | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-31 | LaLiga2 | Real Sociedad B–Girona | 1-0 | 86 | fail | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868522) |
| 2026-10-27 | Copa del Rey | CD Cuarte–Girona | 0-2 | 23 | success | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6386056) |
| 2026-10-24 | LaLiga2 | Girona–Celta Fortuna | 5-1 | 17 | success | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868509) |
| 2026-10-18 | LaLiga2 | Girona–FC Andorra | 2-0 | 5 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868496) |
| 2026-10-12 | LaLiga2 | Tenerife–Girona | 1-1 | 4 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868490) |
| 2026-10-04 | LaLiga2 | Girona–Mallorca | 0-0 | fără gol | fail | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868476) |
| 2026-09-25 | LaLiga2 | Girona–Albacete | 2-0 | 6 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868463) |
| 2026-09-19 | LaLiga2 | Cádiz–Girona | 1-2 | 35 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868451) |
| 2026-09-12 | LaLiga2 | Girona–Castellón | 1-2 | 7 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868442) |
| 2026-09-05 | LaLiga2 | Sporting Gijón–Girona | 0-2 | 27 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868435) |

### Ultimele zece oficiale: Almería

| Data UTC | Competiție | Meci | FT90 | Primul gol | Gol70 | Cornere1H | Under7,5 | Sursă |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-31 | LaLiga2 | Eibar–Almería | 1-1 | 18 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868518) |
| 2026-10-27 | Copa del Rey | Castellonense–Almería | 1-2 | fără gol | pending | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6386049) |
| 2026-10-24 | LaLiga2 | Almería–Real Oviedo | 2-2 | 16 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868504) |
| 2026-10-17 | LaLiga2 | Real Sociedad B–Almería | 2-4 | 23 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868500) |
| 2026-10-11 | LaLiga2 | Almería–Leganés | 3-0 | 52 | success | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868481) |
| 2026-10-03 | LaLiga2 | Almería–Burgos CF | 2-1 | 40 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868471) |
| 2026-09-27 | LaLiga2 | Mallorca–Almería | 0-1 | 35 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868466) |
| 2026-09-20 | LaLiga2 | Almería–Celta Fortuna | 2-0 | 6 | success | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868449) |
| 2026-09-12 | LaLiga2 | Córdoba–Almería | 0-2 | 52 | success | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868441) |
| 2026-09-06 | LaLiga2 | Almería–Cádiz | 3-2 | 13 | success | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868426) |

## Seria sezonului curent

| Data | Meci | Primul gol | Cornere1H | Rezultat eveniment | Sursă |
| --- | --- | --- | --- | --- | --- |
| 2026-08-16 | Girona–Leganés | 48 | 1 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868398) |
| 2026-08-17 | Almería–Eldense | 15 | 7 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868393) |
| 2026-08-21 | Córdoba–Girona | 12 | 2 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868408) |
| 2026-08-23 | Tenerife–Almería | 32 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868414) |
| 2026-08-29 | Sabadell–Almería | 53 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868424) |
| 2026-08-29 | Girona–Las Palmas | 22 | 4 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868421) |
| 2026-09-05 | Sporting Gijón–Girona | 27 | 2 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868435) |
| 2026-09-06 | Almería–Cádiz | 13 | 2 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868426) |
| 2026-09-12 | Girona–Castellón | 7 | 6 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868442) |
| 2026-09-12 | Córdoba–Almería | 52 | 5 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868441) |
| 2026-09-19 | Cádiz–Girona | 35 | 5 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868451) |
| 2026-09-20 | Almería–Celta Fortuna | 6 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868449) |
| 2026-09-25 | Girona–Albacete | 6 | 5 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868463) |
| 2026-09-27 | Mallorca–Almería | 35 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868466) |
| 2026-10-03 | Almería–Burgos CF | 40 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868471) |
| 2026-10-04 | Girona–Mallorca | fără gol | 4 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868476) |
| 2026-10-11 | Almería–Leganés | 52 | 6 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868481) |
| 2026-10-12 | Tenerife–Girona | 4 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868490) |
| 2026-10-17 | Real Sociedad B–Almería | 23 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868500) |
| 2026-10-18 | Girona–FC Andorra | 5 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868496) |
| 2026-10-24 | Girona–Celta Fortuna | 17 | 4 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868509) |
| 2026-10-24 | Almería–Real Oviedo | 16 | 5 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868504) |
| 2026-10-31 | Real Sociedad B–Girona | 86 | 1 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868522) |
| 2026-10-31 | Eibar–Almería | 18 | 3 | succes | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868518) |

## Surse și limite finale

SurseHTMLprimarecuHTTP/timestampîn[sources.json](../Goal70_2026-11-08/sources.json)șiSources/;APIindividualContext/șiStatusAtSelection/,istoriceMatchData/. Prognoza independentă înWeather/,furnizorhttps://open-meteo.com/. Calculelefolosescdatecurenteanterioarezilei,fărăxGfabricat. Celeoptpartidecontextualizatenureprezintăverificareaprofundatăatuturor105. PatruFAcupligainferioarănecalculate;inventarulteritoriuluiconvenitnuesteinventarultuturorligilordincealume. LipsurileșiunfurnizorunicFirstHalfimpunstatuspartial.
