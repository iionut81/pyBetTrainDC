# Cornere prima repriză≤7 — 5 noiembrie2026

Raport prospectiv completat2026-11-05T09:20:31.458+02:00. **Research parțial**:21/35 modele,14 lipsuri explicite, nu top global.

## Lectii aplicate din review

[Predarea salvată](../Review_2026-11-04/LESSONS_FOR_NEXT_SCAN.md) 2026-11-05T08:03:59.216+02:00; începutul scanării2026-11-05T08:04:13.715+02:00. Reluarea continuă acea scanare, fără predicții retrospective.

| Lecție | Verificare | Statut | Efect |
| --- | --- | --- | --- |
| Model vs frecvență și AR20% | Toate35 în tabel,33/21 modele. Palace94,06% vs15/17=88,24%; Celta97,47% vs21/21, eșantion mic. | aplicabilă | Nu numesc frecvența sau Poisson o rată calibrată; nu cresc/scad procente. |
| Intern vs Europa/cupă | Toate partidele sunt proiectate din ligi naționale. Ultimele10 oficiale separate; Thun fără gol propriu în două partide Conference, ambele au însă gol advers. | aplicabilă | Prefer Palace față de Thun pentru Gol70; proiecția rămâne incertă pentru ambele. |
| GF70 propriu pe teren și goluri tardive/0–0 | Thun3/5, Palace5/4, Hoffenheim5/4; contraexemple complete și sensibilități sezon/teren. Mjällby7/14 și două deschideri74/75, trei0–0. | aplicabilă | Producția slabă proprie și stadionul diferit limitează argumentul pentru alegerea Mjällby; snapshotul ei rămâne intact. |
| Dispersie cornere, exact7/≥8 | Serii FirstHalf naționale și ultime10. Celta/Union21/21, exact7 unul; în Europa Omonia–Celta7; Union–Beveren7 în campionat. | aplicabilă | Celta preferată față de Monaco20/22 cu2 depășiri și AGF22/23 cu1; nu garantează repetarea. |
| Antrenori/loturi actuale | Sage/Ilzer din Bundesliga; Giráldez/convocare oficială; Hubert din conferință; Vrancken și accidentări recente. Preview-uri Hearts contradictorii. | parțial verificată | Thun nu este finalist doar fiindcă are rang1. Absențele nu primesc ajustări procentuale. |
| Lipsa nu este zero; rezultate exacte | Review17/17 ambele piețe;14 modele cornere lipsă rămân goale.0 Kortrijk este observație FirstHalf reconciliată. | aplicabilă | Acoperire parțială; azi fără rezultate completate. |
| Cauze și calibrare din succese/failuri | Rennes0–0 și Fenerbahce8 sunt observații; fără audit de faze nu pot explica tactica. | cauze neverificate | Fără ponderi, praguri, filtre sau calibrări noi. |


## Metodă și interpretare

Snapshot numeric original: 08:09 RO, cutoff istoric 2026-11-05 00:00 RO; 844 partide istorice distincte. Sunt folosite numai observații finalizate înaintea cutoffului. Sezonul curent național și separarea acasă/deplasare; minimum5 observații pe echipă, minimum3 pe teren și lipsă de observații nereconciliate în eșantionul folosit, conform metodei salvate.

Pentru A acasă/B deplasare: λ_sezon=(GF_A/n_A+GA_B/n_B+GF_B/n_B+GA_A/n_A)/2; λ_teren folosește numai A acasă și B deplasare; λ=0,5λ_sezon+0,5λ_teren. La goluri GF/GA sunt NUMĂRUL golurilor valide până la minutul afișat70 inclusiv45+, nu indicatorul binar și nu scorul pauzei. P=1−exp(−λ); AR20%=1−exp(−0,8λ). La cornere CF/CA sunt exclusiv FirstHalf; P=PoissonCDF(7;λ), stres PoissonCDF(7;1,2λ). FirstHalf+SecondHalf=All este reconciliat pe echipă.

Frecvențele binare observate au numitor distinct și deduplică întâlnirile comune. Nu sunt probabilități calibrate. Valorile din scenariul de stres nu sunt limite de încredere. Estimarea finală numerică păstrează screeningul, fără ajustare contextuală; calibrată=false. Cota matematică=1/P; niciun preț de piață observat sau folosit. O singură recomandare individuală principală implicită: **Celta–Union cornere1H≤7**; candidatul Gol70 este comparație separată, fără combinație.

## Candidatul analistului și alternative

**Celta Vigo–Union St.Gilloise,22:00RO, CORNERS_1H_UNDER_7_5**. Rang1,97.47% brut/93.91% stres, cota numai matematică1.0260. Principalul individual al analistului, aceeași partidă aleasă explicit de utilizator la09:03:49; rolurile și snapshoturile sunt distincte. Nu există rezultat de azi.

21/21 evenimente interne au≤7; pe teren11/11. λ_sezon3,5909 și λ_teren3,3333 dau3,4621. La Celta acasă totalul are media2,6667 și varianța4,6667: supra-dispersia este un avertisment pentru Poisson. Union deplasare media4,0000/varianța3,0000. Un singur exact7 intern, niciun≥8; în cele6 partide europene recente ale celor două, toate sunt≤7, dar Omonia–Celta a avut exact7. Eșantionul european este diagnostic separat, nu supliment pentru formula națională.

[Convocarea oficială Celta](https://rccelta.es/en/equipo/actualidad/celta-squad-list-for-the-uel-match-against-union-saint-gilloise/) confirmă antrenorul și cinci indisponibili; nu confirmă titularii. [Conferința Union relatată de Walfoot](https://www.walfoot.be/news/2026-11-04/blesses-de-retour-attente-dun-appel-de-van-bommel-lunion-fait-le-point-sur-les-dossiers-chauds-a-vigo) descrie dorința de victorie și o partidă posibil deschisă. Acest context nu permite argumentul «se vor mulțumi să conserve». Intenția ofensivă, rotația și absențele defensive pot crește presiunea și blocajele; sensul efectului asupra cornerelor nu este măsurat. Celta are3 puncte europene față de7 Union; ambele au motivație.

Îl prefer față de **Monaco–Nordsjælland** (96,70%/92,29%,20/22;2 depășiri≥8), unde Nordsjælland deplasare are varianță7,2857 vs medie4,5714, iar verificarea loturilor are contradicții. **AGF–Braga** (96,59%/92,06%,22/23;1 depășire) are media4,8333 a totalurilor AGF acasă și3 observații exact7 în univers; nu există un audit primar complet al loturilor. Nu transform diferențele mici de model în superioritate demonstrată. Seria Celta/Union este favorabilă, însă istoricul mic și confirmarea independentă incompletă a FirstHalf justifică statutul parțial.

## Lista integrală

| Rang | Meci | Ora RO | Competiție | λ sezon | λ teren | λ50/50 | Brut | AR20 | Istoric | Teren | 1/P | Statut | Sursa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Celta Vigo – Union St.Gilloise | 22:00 | uefa.europa | 3.5909 | 3.3333 | 3.4621 | 97.47% | 93.91% | 21/21 | 11/11 | 1.0260 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112347) |
| 2 | Monaco – Nordsjælland | 22:00 | uefa.conference | 3.6410 | 3.6607 | 3.6509 | 96.70% | 92.29% | 20/22 | 10/11 | 1.0341 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112401) |
| 3 | AGF – Braga | 19:45 | uefa.conference | 3.6346 | 3.7167 | 3.6756 | 96.59% | 92.06% | 22/23 | 11/11 | 1.0353 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112532) |
| 4 | Crystal Palace – Hoffenheim | 19:45 | uefa.europa | 3.7500 | 3.8750 | 3.8125 | 95.93% | 90.71% | 17/17 | 8/8 | 1.0425 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112325) |
| 5 | FC Midtjylland – Ajax | 19:45 | uefa.conference | 3.8322 | 4.4333 | 4.1328 | 94.06% | 87.08% | 23/24 | 10/11 | 1.0632 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112405) |
| 6 | Trabzonspor – Freiburg | 19:45 | uefa.conference | 4.2250 | 4.2750 | 4.2500 | 93.26% | 85.60% | 17/18 | 8/9 | 1.0723 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112501) |
| 7 | Sunderland – Dinamo Zagreb | 22:00 | uefa.europa | 4.2576 | 4.4750 | 4.3663 | 92.40% | 84.04% | 19/20 | 8/9 | 1.0822 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112331) |
| 8 | Hapoel Beer Sheva – OFI Crete | 22:00 | uefa.europa | 4.1667 | 4.5750 | 4.3708 | 92.37% | 83.98% | 16/18 | 8/9 | 1.0826 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112365) |
| 9 | Riga FC – Jablonec | 22:00 | uefa.conference | 4.4110 | 4.3922 | 4.4016 | 92.13% | 83.56% | 43/48 | 21/23 | 1.0854 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112522) |
| 10 | Bayer Leverkusen – Marseille | 22:00 | uefa.europa | 4.4861 | 4.3750 | 4.4306 | 91.90% | 83.15% | 17/17 | 8/8 | 1.0881 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112242) |
| 11 | Getafe – Brighton & Hove Albion | 22:00 | uefa.conference | 4.1667 | 4.7333 | 4.4500 | 91.75% | 82.88% | 20/20 | 11/11 | 1.0899 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112474) |
| 12 | Thun – Heart of Midlothian | 19:45 | uefa.conference | 5.1439 | 4.1500 | 4.6470 | 90.08% | 80.00% | 19/23 | 11/11 | 1.1101 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112539) |
| 13 | KuPS – Gent | 19:45 | uefa.conference | 5.0274 | 4.4000 | 4.7137 | 89.48% | 78.98% | 37/41 | 18/20 | 1.1176 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112477) |
| 14 | Anderlecht – Salzburg | 19:45 | uefa.europa | 5.1364 | 4.7000 | 4.9182 | 87.50% | 75.74% | 18/21 | 9/10 | 1.1428 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112311) |
| 15 | Sparta Prague – AFC Bournemouth | 19:45 | uefa.europa | 4.6709 | 5.1667 | 4.9188 | 87.50% | 75.73% | 20/22 | 9/10 | 1.1429 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112304) |
| 16 | Lillestrøm – Viktoria Plzeň | 19:45 | uefa.europa | 5.1538 | 5.0000 | 5.0769 | 85.85% | 73.12% | 32/38 | 15/18 | 1.1649 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112375) |
| 17 | AZ Alkmaar – Juventus | 22:00 | uefa.europa | 5.1773 | 5.2000 | 5.1886 | 84.62% | 71.23% | 16/21 | 7/10 | 1.1818 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112262) |
| 18 | Real Sociedad – Lyon | 19:45 | uefa.europa | 5.5051 | 5.6250 | 5.5650 | 80.14% | 64.66% | 17/20 | 7/9 | 1.2478 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112270) |
| 19 | Celtic – Beşiktaş | 22:00 | uefa.europa | 5.7417 | 5.6500 | 5.6958 | 78.47% | 62.33% | 15/22 | 8/11 | 1.2744 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112301) |
| 20 | Benfica – Lech Poznań | 22:00 | uefa.europa | 5.4750 | 6.6000 | 6.0375 | 73.88% | 56.23% | 16/22 | 6/10 | 1.3535 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112248) |
| 21 | St.Truiden – Brann | 22:00 | uefa.conference | 5.7000 | 6.7667 | 6.2333 | 71.13% | 52.76% | 26/35 | 11/17 | 1.4059 | calculat | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112490) |
| 22 | Milan – Ferencváros | 19:45 | uefa.europa | lipsă | lipsă | lipsă | lipsă | lipsă | 9/10 | 4/5 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112255) |
| 23 | Levski Sofia – Jagiellonia Białystok | 19:45 | uefa.europa | lipsă | lipsă | lipsă | lipsă | lipsă | 9/12 | 4/6 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112380) |
| 24 | NEC Nijmegen – Omonia Nicosia | 19:45 | uefa.europa | lipsă | lipsă | lipsă | lipsă | lipsă | 10/11 | 5/6 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112368) |
| 25 | Sturm Graz – NK Celje | 22:00 | uefa.europa | lipsă | lipsă | lipsă | lipsă | lipsă | 9/11 | 3/5 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112316) |
| 26 | Torreense – Ararat Armenia | 22:00 | uefa.europa | lipsă | lipsă | lipsă | lipsă | lipsă | 0/0 | 0/0 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112361) |
| 27 | Atalanta – Kairat Almaty | 19:45 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 7/10 | 3/6 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112388) |
| 28 | FK Crvena Zvezda – Inter Club d'Escaldes | 19:45 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 13/14 | 6/7 | lipsă | unresolved_domestic_league_mapping | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112410) |
| 29 | Lincoln Red Imps FC – Hajduk Split | 19:45 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 12/12 | 5/5 | lipsă | unresolved_domestic_league_mapping | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112485) |
| 30 | Mjällby – Borac Banja Luka | 19:45 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 24/27 | 12/14 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112548) |
| 31 | FC København – Iberia 1999 | 22:00 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 13/13 | 6/6 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112404) |
| 32 | FC Twente – Pafos FC | 22:00 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 11/11 | 5/5 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112480) |
| 33 | Lugano – FK Kauno Žalgiris | 22:00 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 11/12 | 7/7 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112471) |
| 34 | Panathinaikos – CSKA Sofia | 22:00 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 7/9 | 4/5 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112416) |
| 35 | Universitatea Craiova – Egnatia | 22:00 | uefa.conference | lipsă | lipsă | lipsă | lipsă | lipsă | 13/14 | 8/8 | lipsă | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112506) |

## Serii, dispersie și Europa separat

### Celta Vigo – Union St.Gilloise · rang brut1

Brut97.47%, stres93.91%, istoric≤7:21/21; teren11/11; exact7:1; ≥8:0. CF/CA sezon: gazda14/21 în11, oaspeți22/18 în10; teren gazda7/9 în6, oaspeți8/12 în5. Media/varianța totalului1H pe teren: 2.6667/4.6667 și 4.0000/3.0000. Numai sezon96.96%, numai teren97.92%.

| Data | Meci | Cornere1H | Prag | Sursa |
| --- | --- | --- | --- | --- |
| 2026-09-03 | Real Sociedad – Celta Vigo | 6 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868070) |
| 2026-08-22 | Valencia – Celta Vigo | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868030) |
| 2026-08-27 | Celta Vigo – Osasuna | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868013) |
| 2026-08-30 | Celta Vigo – Athletic Club | 6 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868032) |
| 2026-09-07 | Getafe – Celta Vigo | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868046) |
| 2026-09-13 | Celta Vigo – Málaga | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868052) |
| 2026-09-19 | Celta Vigo – Racing Santander | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868073) |
| 2026-10-11 | Elche – Celta Vigo | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868083) |
| 2026-10-18 | Celta Vigo – Deportivo Alavés | 0 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868092) |
| 2026-10-25 | Celta Vigo – Real Betis | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868104) |
| 2026-10-31 | Rayo Vallecano – Celta Vigo | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5868119) |
| 2026-08-30 | Union St.Gilloise – Anderlecht | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811743) |
| 2026-08-08 | Westerlo – Union St.Gilloise | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811714) |
| 2026-08-15 | Union St.Gilloise – Zulte Waregem | 6 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811720) |
| 2026-09-02 | St.Truiden – Union St.Gilloise | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811745) |
| 2026-09-05 | Sporting Charleroi – Union St.Gilloise | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811747) |
| 2026-09-12 | Union St.Gilloise – Lommel | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811759) |
| 2026-09-20 | Royal Antwerp – Union St.Gilloise | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811769) |
| 2026-10-11 | Union St.Gilloise – OH Leuven | 6 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811778) |
| 2026-10-25 | Union St.Gilloise – Cercle Brugge | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811789) |
| 2026-10-31 | SK Beveren – Union St.Gilloise | 7 | exact7 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5811792) |

Ultimele10 oficiale pe fiecare echipă (inclusiv Europa/cupă separat):

| Echipa | Data | Competiție | Meci | Cornere1H | Rezultat | Sursa |
| --- | --- | --- | --- | --- | --- | --- |
| Celta Vigo | 2026-10-31 | LaLiga | Rayo Vallecano – Celta Vigo | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868119) |
| Celta Vigo | 2026-10-25 | LaLiga | Celta Vigo – Real Betis | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868104) |
| Celta Vigo | 2026-10-22 | Europa League | Celtic – Celta Vigo | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112300) |
| Celta Vigo | 2026-10-18 | LaLiga | Celta Vigo – Deportivo Alavés | 0 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868092) |
| Celta Vigo | 2026-10-15 | Europa League | Celta Vigo – Juventus | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112346) |
| Celta Vigo | 2026-10-11 | LaLiga | Elche – Celta Vigo | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868083) |
| Celta Vigo | 2026-09-19 | LaLiga | Celta Vigo – Racing Santander | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868073) |
| Celta Vigo | 2026-09-16 | Europa League | Omonia Nicosia – Celta Vigo | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112344) |
| Celta Vigo | 2026-09-13 | LaLiga | Celta Vigo – Málaga | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868052) |
| Celta Vigo | 2026-09-07 | LaLiga | Getafe – Celta Vigo | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5868046) |
| Union St.Gilloise | 2026-10-31 | Belgian Pro League | SK Beveren – Union St.Gilloise | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811792) |
| Union St.Gilloise | 2026-10-25 | Belgian Pro League | Union St.Gilloise – Cercle Brugge | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811789) |
| Union St.Gilloise | 2026-10-22 | Europa League | Union St.Gilloise – Hapoel Beer Sheva | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112289) |
| Union St.Gilloise | 2026-10-18 | Cup | Union St.Gilloise – KFC Nijlen | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6361395) |
| Union St.Gilloise | 2026-10-15 | Europa League | Union St.Gilloise – Real Sociedad | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112286) |
| Union St.Gilloise | 2026-10-11 | Belgian Pro League | Union St.Gilloise – OH Leuven | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811778) |
| Union St.Gilloise | 2026-09-20 | Belgian Pro League | Royal Antwerp – Union St.Gilloise | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811769) |
| Union St.Gilloise | 2026-09-17 | Europa League | Viktoria Plzeň – Union St.Gilloise | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112283) |
| Union St.Gilloise | 2026-09-12 | Belgian Pro League | Union St.Gilloise – Lommel | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811759) |
| Union St.Gilloise | 2026-09-05 | Belgian Pro League | Sporting Charleroi – Union St.Gilloise | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5811747) |


### Monaco – Nordsjælland · rang brut2

Brut96.70%, stres92.29%, istoric≤7:20/22; teren10/11; exact7:1; ≥8:2. CF/CA sezon: gazda9/15 în9, oaspeți38/22 în13; teren gazda5/6 în4, oaspeți20/12 în7. Media/varianța totalului1H pe teren: 2.7500/1.5833 și 4.5714/7.2857. Numai sezon96.74%, numai teren96.66%.

| Data | Meci | Cornere1H | Prag | Sursa |
| --- | --- | --- | --- | --- |
| 2026-08-30 | Monaco – Marseille | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802914) |
| 2026-08-23 | Le Havre – Monaco | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802902) |
| 2026-09-04 | Paris Saint-Germain – Monaco | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802925) |
| 2026-09-12 | Strasbourg – Monaco | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802936) |
| 2026-09-18 | Monaco – Lens | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802941) |
| 2026-10-10 | Monaco – Toulouse | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802950) |
| 2026-10-18 | Lorient – Monaco | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802959) |
| 2026-10-25 | Monaco – Lille | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802969) |
| 2026-11-01 | Paris FC – Monaco | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5802979) |
| 2026-09-13 | Nordsjælland – AGF | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739511) |
| 2026-09-07 | FC Midtjylland – Nordsjælland | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739505) |
| 2026-11-01 | Nordsjælland – FC Midtjylland | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739541) |
| 2026-09-03 | FC København – Nordsjælland | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739498) |
| 2026-07-26 | AC Horsens – Nordsjælland | 9 | ≥8 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739467) |
| 2026-08-02 | Nordsjælland – Randers FC | 8 | ≥8 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739471) |
| 2026-08-16 | Nordsjælland – Silkeborg | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739481) |
| 2026-08-23 | Sønderjyske – Nordsjælland | 5 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739486) |
| 2026-08-30 | Nordsjælland – Brøndby IF | 6 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739495) |
| 2026-09-20 | Viborg – Nordsjælland | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739516) |
| 2026-10-09 | Nordsjælland – OB | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739523) |
| 2026-10-18 | Lyngby – Nordsjælland | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739527) |
| 2026-10-25 | Silkeborg – Nordsjælland | 7 | exact7 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739534) |

Ultimele10 oficiale pe fiecare echipă (inclusiv Europa/cupă separat):

| Echipa | Data | Competiție | Meci | Cornere1H | Rezultat | Sursa |
| --- | --- | --- | --- | --- | --- | --- |
| Monaco | 2026-11-01 | Ligue 1 | Paris FC – Monaco | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802979) |
| Monaco | 2026-10-25 | Ligue 1 | Monaco – Lille | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802969) |
| Monaco | 2026-10-22 | Conference League | Monaco – Freiburg | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112399) |
| Monaco | 2026-10-18 | Ligue 1 | Lorient – Monaco | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802959) |
| Monaco | 2026-10-15 | Conference League | CSKA Sofia – Monaco | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112541) |
| Monaco | 2026-10-10 | Ligue 1 | Monaco – Toulouse | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802950) |
| Monaco | 2026-09-18 | Ligue 1 | Monaco – Lens | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802941) |
| Monaco | 2026-09-12 | Ligue 1 | Strasbourg – Monaco | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802936) |
| Monaco | 2026-09-04 | Ligue 1 | PSG – Monaco | 1 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802925) |
| Monaco | 2026-08-30 | Ligue 1 | Monaco – Marseille | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5802914) |
| Nordsjælland | 2026-11-01 | Superligaen | Nordsjælland – FC Midtjylland | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739541) |
| Nordsjælland | 2026-10-28 | DBU Pokalen | Esbjerg fB – Nordsjælland | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6298362) |
| Nordsjælland | 2026-10-25 | Superligaen | Silkeborg – Nordsjælland | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739534) |
| Nordsjælland | 2026-10-22 | Conference League | Nordsjælland – CSKA Sofia | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112531) |
| Nordsjælland | 2026-10-18 | Superligaen | Lyngby – Nordsjælland | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739527) |
| Nordsjælland | 2026-10-15 | Conference League | Hearts – Nordsjælland | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112497) |
| Nordsjælland | 2026-10-09 | Superligaen | Nordsjælland – OB | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739523) |
| Nordsjælland | 2026-09-20 | Superligaen | Viborg – Nordsjælland | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739516) |
| Nordsjælland | 2026-09-16 | DBU Pokalen | Vatanspor – Nordsjælland | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6133878) |
| Nordsjælland | 2026-09-13 | Superligaen | Nordsjælland – AGF | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739511) |


### AGF – Braga · rang brut3

Brut96.59%, stres92.06%, istoric≤7:22/23; teren11/11; exact7:3; ≥8:1. CF/CA sezon: gazda43/19 în13, oaspeți16/9 în10; teren gazda26/3 în6, oaspeți7/6 în5. Media/varianța totalului1H pe teren: 4.8333/3.3667 și 2.6000/1.3000. Numai sezon96.77%, numai teren96.40%.

| Data | Meci | Cornere1H | Prag | Sursa |
| --- | --- | --- | --- | --- |
| 2026-07-25 | AGF – Brøndby IF | 7 | exact7 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739464) |
| 2026-08-01 | Lyngby – AGF | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739469) |
| 2026-08-14 | Viborg – AGF | 7 | exact7 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739479) |
| 2026-08-23 | AGF – OB | 5 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739485) |
| 2026-08-30 | Randers FC – AGF | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739494) |
| 2026-09-02 | AGF – FC Midtjylland | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739497) |
| 2026-09-05 | AGF – Silkeborg | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739501) |
| 2026-09-13 | Nordsjælland – AGF | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739511) |
| 2026-09-20 | AC Horsens – AGF | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739517) |
| 2026-10-12 | AGF – Sønderjyske | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739521) |
| 2026-10-18 | FC København – AGF | 6 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739526) |
| 2026-10-25 | AGF – Viborg | 7 | exact7 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739533) |
| 2026-11-01 | Brøndby IF – AGF | 9 | ≥8 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5739539) |
| 2026-08-09 | Moreirense – Braga | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887562) |
| 2026-08-31 | Braga – Vitória de Guimarães | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887612) |
| 2026-09-05 | Alverca – Braga | 1 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887618) |
| 2026-09-10 | Estrela da Amadora – Braga | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887600) |
| 2026-09-14 | Braga – Estoril | 3 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887629) |
| 2026-09-20 | Santa Clara – Braga | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887642) |
| 2026-10-09 | Braga – Sporting CP | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887647) |
| 2026-10-19 | Braga – Gil Vicente | 2 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887594) |
| 2026-10-25 | Nacional – Braga | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887658) |
| 2026-11-01 | Braga – Famalicão | 4 | ≤6 | [FirstHalf](https://www.fotmob.com/api/data/matchDetails?matchId=5887665) |

Ultimele10 oficiale pe fiecare echipă (inclusiv Europa/cupă separat):

| Echipa | Data | Competiție | Meci | Cornere1H | Rezultat | Sursa |
| --- | --- | --- | --- | --- | --- | --- |
| AGF | 2026-11-01 | Superligaen | Brøndby IF – AGF | 9 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739539) |
| AGF | 2026-10-28 | DBU Pokalen | FC Roskilde – AGF | 10 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6298363) |
| AGF | 2026-10-25 | Superligaen | AGF – Viborg | 7 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739533) |
| AGF | 2026-10-22 | Conference League | AGF – Egnatia | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112534) |
| AGF | 2026-10-18 | Superligaen | FC København – AGF | 6 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739526) |
| AGF | 2026-10-15 | Conference League | Gent – AGF | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112413) |
| AGF | 2026-10-12 | Superligaen | AGF – Sønderjyske | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739521) |
| AGF | 2026-09-20 | Superligaen | AC Horsens – AGF | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739517) |
| AGF | 2026-09-16 | DBU Pokalen | Skjold – AGF | lipsă | pending | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6133879) |
| AGF | 2026-09-13 | Superligaen | Nordsjælland – AGF | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5739511) |
| Braga | 2026-11-01 | Liga Portugal | Braga – Famalicão | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887665) |
| Braga | 2026-10-29 | League Cup | Braga – Famalicão | 8 | fail | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5985966) |
| Braga | 2026-10-25 | Liga Portugal | Nacional – Braga | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887658) |
| Braga | 2026-10-22 | Conference League | Braga – Gent | 5 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112390) |
| Braga | 2026-10-19 | Liga Portugal | Braga – Gil Vicente | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887594) |
| Braga | 2026-10-15 | Conference League | FC København – Braga | 4 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112402) |
| Braga | 2026-10-09 | Liga Portugal | Braga – Sporting CP | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887647) |
| Braga | 2026-09-20 | Liga Portugal | Santa Clara – Braga | 2 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887642) |
| Braga | 2026-09-14 | Liga Portugal | Braga – Estoril | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887629) |
| Braga | 2026-09-10 | Liga Portugal | Estrela da Amadora – Braga | 3 | success | [date](https://www.fotmob.com/api/data/matchDetails?matchId=5887600) |


## Vreme și program

Prognoză orară la cea mai apropiată oră de kickoff, nu observație în stadion:

| Ora | °C | Ploaie mm | Vânt km/h | Rafale km/h | Sursa |
| --- | --- | --- | --- | --- | --- |
| London 2026-11-05T19:00 RO | 9.4 | 0 | 11.5 | 24.8 | [prognoză](https://api.open-meteo.com/v1/forecast?latitude=51.398&longitude=-0.085&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m&start_date=2026-11-05&end_date=2026-11-05&timezone=Europe%2FBucharest) |
| Vigo 2026-11-05T22:00 RO | 11.6 | 0 | 3.9 | 6.5 | [prognoză](https://api.open-meteo.com/v1/forecast?latitude=42.21&longitude=-8.74&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m&start_date=2026-11-05&end_date=2026-11-05&timezone=Europe%2FBucharest) |
| Thun 2026-11-05T19:00 RO | 8.3 | 0 | 2.3 | 5.4 | [prognoză](https://api.open-meteo.com/v1/forecast?latitude=46.76&longitude=7.6&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m&start_date=2026-11-05&end_date=2026-11-05&timezone=Europe%2FBucharest) |

Nu apare un semnal meteo sever în aceste ferestre; starea gazonului nu este inspectată. Palace/Hoffenheim au jucat31 octombrie după cupe28 octombrie; cinci zile între campionat și azi, fără a presupune că oboseala este absentă. Thun și Hearts au jucat1 noiembrie; Celta și Union31 octombrie. Minutele individuale, deplasările și antrenamentele nu sunt auditate integral.

## Verificări de sursă și limite

127 verificări istorice independente pentru cele12 echipe din top3 al ambelor piețe:67 cronologii de goluri ESPN concordante,60 identități/cronologii nerezolvate. Dintre meciurile Celta/Union,13 au identitate/număr goluri concordante, dar cronologia cornerelor ESPN este incompletă; nu o număr drept confirmare independentă a cornerelor și nu transform comentariile absente în zero. Nu există conflict numeric identificat în verificările complete. Vezi independent_provider_checks.csv și Corners1H/ESPN. FotMob rămâne sursa statistică principală.

Contextul aprofundat este limitat la finaliști și alternativele explicit descrise. Restul35 sunt comparate statistic, nu auditate tactic integral. Loturile de start nu sunt confirmate. Cauzele rezultatelor viitoare sunt necunoscute. Nicio valoarexG nu este inventată. Pentru gol lipsesc mapările naționale Andorra/Gibraltar în două partide; pentru cornere14 partide nu îndeplinesc acoperirea curentă. Lista exactă este mai jos.

| Meci | Gazda | Oaspeți | Motiv | Sursa |
| --- | --- | --- | --- | --- |
| Milan – Ferencváros | 10/10 | 0/11 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112255) |
| Levski Sofia – Jagiellonia Białystok | 0/12 | 12/12 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112380) |
| NEC Nijmegen – Omonia Nicosia | 11/11 | 0/8 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112368) |
| Sturm Graz – NK Celje | 11/11 | 0/13 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112316) |
| Torreense – Ararat Armenia | 0/9 | 0/12 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112361) |
| Atalanta – Kairat Almaty | 10/10 | 0/30 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112388) |
| FK Crvena Zvezda – Inter Club d'Escaldes | 14/14 | 0/0 | unresolved_domestic_league_mapping | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112410) |
| Lincoln Red Imps FC – Hajduk Split | 0/0 | 12/12 | unresolved_domestic_league_mapping | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112485) |
| Mjällby – Borac Banja Luka | 27/27 | 0/11 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112548) |
| FC København – Iberia 1999 | 13/13 | 0/31 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112404) |
| FC Twente – Pafos FC | 11/11 | 0/8 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112480) |
| Lugano – FK Kauno Žalgiris | 12/12 | 0/31 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112471) |
| Panathinaikos – CSKA Sofia | 9/9 | 0/13 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112416) |
| Universitatea Craiova – Egnatia | 14/14 | 0/10 | missing_observations_or_sample_below_5_3 | [date](https://www.fotmob.com/api/data/matchDetails?matchId=6112506) |

Inventarul29 ligi, sursele de program și excluderile sunt în [raportul comun de acoperire](../Goal70_2026-11-05/EUROPE_ALL_MATCHES.md). [Surse contextuale](../Goal70_2026-11-05/CONTEXT_SOURCES.md). [Lecții](../Review_2026-11-04/LESSONS_FOR_NEXT_SCAN.md).
