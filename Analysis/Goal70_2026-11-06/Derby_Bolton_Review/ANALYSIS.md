# Derby County–Bolton Wanderers: analiza completă Gol70

Actualizare: 2026-11-06 T 10:49:28.056+02:00, Europe/Bucharest. Meci: 6 noiembrie 2026,22:00 RO/20:00 GMT, Pride Park, Championship 2026/27,etapa 15. Status neînceput verificat 2026-11-06 T 08:42:07.211 Z. Analiză suplimentară solicitată de utilizator după screeningul 08:31; nu rescrie acel snapshot.

## Verdict și rol în listă

Gol până la minutul afișat 70 inclusiv, oricine marchează: model brut 88.27%, AR 2081.99%, cotă echitabilă matematică 1.1329. Rangul brut este 13 din 34 partide inventariate/33 modele eligibile. NUeste candidatul principal al analistului și nu este înregistrat ca alegere a utilizatorului la verificare. Este candidat comparat. Nu îi atribuim retrospectiv o selecție pe care nu am făcut-o.

Partida este în listă pentru că are istoric complet, comparabil din același campionat,25/28 deschideri≤70 și defensiva Derby vulnerabilă. Este o alternativă explicabilă, însă nu aș promova-o peste candidatulGol 70 al dimineții doar pentru 14/14 al Derby. Atacul modest al gazdei, indisponibilitatea confirmatăSzmodics și incertitudinile din mijlocul Bolton pot împiedica deschiderea scorului. Procentele sunt necalibrate;81,99% AR20 este scenariu de sensibilitate, nu probabilitate reală validată. Nu s-au folosit cote de bookmaker și nu se garantează prețul 1,10.

## Datele și calculele originale

GF70/GA70 înseamnă toate golurile proprii/primite la minutul afișat≤70, inclusiv 45+, fără prelungiri eliminatorii/departajare. Frecvența se referă separat la primul gol al partidei.

| Eșantion | n | GF70 | GA70 | Primul gol≤70 | Primul gol>70 | 0–0 |
| --- | --- | --- | --- | --- | --- | --- |
| Derby sezon | 14 | 8 | 22 | 14/14 | 0 | 0 |
| Bolton sezon | 14 | 14 | 16 | 11/14 | 1 | 2 |
| Derby acasă | 7 | 4 | 12 | 7/7 | 0 | 0 |
| Bolton deplasare | 7 | 7 | 7 | 5/7 | 0 | 2 |

λsezon=((8/14+16/14)+(14/14+22/14))/2=2,142857.

λteren=((4/7+7/7)+(7/7+12/7))/2=2,142857.

λcombinat=0,5 λsezon+0,5 λteren=2,142857.

P=1−exp(−2,142857)=88,268083%. Numai sezon=88,27%; numai teren=88,27%. Egalitatea este rezultatul numărătorilor, nu un bug. Intensitate−20%:1−exp(−0,8×2,142857)=81,990769%. Cota matematică 1/P=1,132912, nu ofertă disponibilă.

Ponderarea este euristică, eșantioanele se suprapun și nu există calibrare independentă. Nu se multiplică din nou cu 70/90. Istoric deduplicat 25/28=89,29%; teren 12/14=85,71%. Nu transformăm aceste frecvențe în probabilitatea individuală de azi.

Pe teren, contribuțiile modelului sunt: componenta Derby=(4/7+7/7)/2=0,785714; componenta Bolton=(7/7+12/7)/2=1,357143. Nu reprezintă probabilități despre cine deschide scorul. Ele arată că scenariul modelului se sprijină mai mult pe Bolton contra defensivei Derby.

## De ce există argument pentru gol

1. Derby a avut primul gol≤70 în 14/14 meciuri, inclusiv 7/7 acasă; toate 14 s-au deschis cel târziu 49. Este regularitatea evenimentului în meciurile Derby, NU 14 meciuri în care Derby a marcat.
2. Acasă, Derby a primit gol≤70 în 6/7 meciuri și a marcat≤70 în 3/7. Ratele proprii sunt 4/7=0,57 GF70/meci versus 12/7=1,71 GA70/meci. Principalul argument este vulnerabilitatea defensivei, nu o ofensivă puternică a gazdei.
3. Bolton a marcat≤70 în 4/7 deplasări și a primit în 4/7; cinci meciuri au avut gol≤70. Două sunt 0–0:QPR 22 august șiSouthampton 17 octombrie. Există contraexemple clare pentru a nu trata adversarul vulnerabil ca gol sigur.
4. Bolton a luat 10 puncte în ultimele 4 deplasări și Dalby are 6 goluri de campionat conform[EFL](https://www.efl.com/news/2026/november/05/derby-county-v-bolton-wanderers--pre-match-stat-pack/);[raportul oficial Charlton](https://www.bwfc.co.uk/news/match-report-charlton-athletic-0-1-wanderers) confirmă golul 18 din ultima deplasare. Această formă susține o cale ofensivă, nu un bonus procentual.

## Context, miză și avantajul terenului

Tabelul actual al furnizorului, salvat în team_10170.json șiteam_8559.json: Derby 23/7 puncte după 14 meciuri(1 V 4 E 9 Î),golaveraj 11–29; Bolton 17/17 puncte(5 V 2 E 7 Î),19–26. Previewul secundar publicat 4 noiembrie scrie 16 pentruBolton; păstrăm diferența și folosim tabelul actual 17 la extragere. [Pagina oficială Derby](https://www.dcfc.co.uk/news/2026/11/interview-we-want-to-put-pride-back-in-the-shirt-eustace) confirmă 23/7 și decalajul de 7 puncte față de locul 21.

Este meci de campionat pentru puncte, nu retur/eliminatoriu: victoria aduce 3,egalul 1, fără scor agregat sau prelungiri. Nu pretind verificarea integrală a tuturor criteriilor de departajare 2026/27; documentul EFL dinamic nu a fost extras integral. Aceste criterii nu produc o obligație de gol sau atac în această partidă.

Derby are presiune pentru puncte și reacție după 0–4 cuNorwich. Un egal ar opri seria de înfrângeri și poate fi acceptabil în cursul jocului; nu există dovadă că Eustace trebuie să atace cu orice risc din minutul 1. Bolton poate beneficia de fragilitatea adversarului, dar poate accepta un început controlat într-o deplasare. Niciun antrenor nu a declarat verificabil un plan«jucăm pentru 0–0».

Pride Park este terenul obișnuit Derby, suprafață naturală. Avantajul publicului poate coexista cu presiune/frustrare; nu îi atribuim efect numeric. H 2 Hvechi favorabil Derby, menționat deEFL, nu este folosit în formulă sau ca dovadă despre loturile de azi.

## Tactică, antrenori și mindset — fapte separate de interpretare

John Eustace cere o reacție colectivă și responsabilitate după ultimele prestații. [Interviul oficial5noiembrie](https://www.dcfc.co.uk/news/2026/11/interview-we-want-to-put-pride-back-in-the-shirt-eustace) confirmă obiectivul și Szmodicsoperat,absent 6 săptămâni. Acesta nu este un angajament că echipa va ataca agresiv imediat sau o dovadă privind starea mentală individuală.

API etichetează explicit formațiile ca lastStarting 11: Derby 4-4-2,Bolton 4-2-3-1. NUeste un 11 probabil de azi și, cu atât mai puțin, confirmat. Același câmp includeSzmodics,SheehanșiIto. Previewuri secundare propun Derby 4-2-3-1 și diferă la Bolton; forma finală rămâne necunoscută. Nu afișăm ultima echipă ca titulari actuali.

Interpretare tactică pentruGol 70:

- Dacă Derby împinge fundașii laterali și pierde mingi în construcție, Bolton poate găsi spații pentruDalbyși jucătorii care îl sprijină. Golul Bolton de laCharlton a venit din exploatarea unei pase greșite; este o rută observată, nu garanție că Derby repetă eroarea.
- Dacă gazda joacă mai direct, Morris/Meisterpot oferi prezență pentru centrări și faze fixe. [Raportul oficial contraNorwich](https://www.dcfc.co.uk/news/2026/11/report-derby-county-vs-norwich-city) descrie și șansa luiMorrisla un corner. Șansele izolate nu demonstrează un volum ofensiv mare.
- Dacă Eustace prioritizează distanțe mici și protecția careului după cele 8 goluri primite în două meciuri, începutul poate deveni prudent. Un 4-2-3-1 cu doi mijlocași de protecție poate limita tranzițiile; schimbarea nu este confirmată.
- [Schumacher,2noiembrie](https://www.bwfc.co.uk/news/schumacher-capable-winning-anywhere) vorbește despre încredere în deplasare și reducerea erorilor. Este o declarație recentă înainteaCharlton, nu o conferință nouă pentruDerby. Absențele din mijloc pot limita progresia și calitatea ultimei pase chiar dacă Derby continuă să se apere slab.

Nu avem xG comparabil pe starea 0–0, PPDA sau cronologie de ocazii pentru toate meciurile. Nu inventăm pressing sus, posesie dominantă sau un volum numeric de ocazii pentru această seară.

## Loturi și contradicții

- Derby: Szmodicseste CONFIRMAT absent 6 săptămâni după operația la genunchi, sursă primară. API îl prezintă doarDoubtfulși îl păstrează în ultimul 11; pentru disponibilitate prioritizăm clubul.
- API listează probleme pentruHedges,Ward,Ozoh,Travis,Langås,Cashin,Taylor,Forsyth,Murkin,Clarke,Agyemang. Nu sunt 12 absențe confirmate independent azi. Sports Mole relatează căClarkeșiAgyemangau revenit la antrenament; asta nu confirmă capacitatea de titularizare. Timpul revenirilor rămâne incert.
- Bolton: Bazunu,Toal,Erhahon,Hardie,McAtee,Nelsonapar indisponibili în sursele secundare/API. SheehanșiItonecesită evaluare conform relatării[BoltonNews/Yahoo6noiembrie](https://uk.sports.yahoo.com/news/xav-simons-eyeing-another-memorable-050000390.html).[Raportul oficial Charlton](https://www.bwfc.co.uk/news/match-report-charlton-athletic-0-1-wanderers) confirmă accidentarea luiSheehanîn timpul meciului și înlocuirea luiIto, fără să confirme individual disponibilitatea pentru vineri. Nu extrapolăm lista«fără probleme noi»din 2 noiembrie peste evenimentele din 3 noiembrie.

Impact: problemele defensive Derby favorizează ruta spre golul adversarului, iar lipsaSzmodicsreduce alternativa ofensivă a gazdei. DacăBoltonpierde creativitate prinSheehan/Ito, ambele efecte trebuie evaluate împreună; nu aplicăm ajustări−5% sau+3% arbitrare. Revenirea portaruluiBazununu este confirmată,Bonhameste portarul ultimului 11.

## Odihnă, deplasare și vreme

Ambele au jucat 31 octombrie și 3 noiembrie; acesta este al treilea meci în 7 zile. Intervalul între starturile 3 și 6 noiembrie este 72 h 15 min. Derby a jucat marți acasă, Bolton laCharlton; vineri Bolton are din nou deplasare. Nu cunoaștem itinerarul, ora plecării sau dacă echipa a revenit acasă între meciuri. Nu presupunem că Bolton este automat epuizată sau că oboseala produce goluri.

Open-Meteoextras 2026-11-06 T 08:44:12.097 Z,pentru 20:00 Europe/London:temperatură 8.8°C,probabilitate precipitații 49%,vânt 4 km/h. Sunt prognoze, nu măsurători ale terenului. FotMobmai vechi indica 9°C/16% ploaie; probabilitatea ploii diferă între surse/momente. Nu apare vânt puternic în prognoza independentă, dar nu există inspecție actuală a gazonului. [Sursa orară](https://api.open-meteo.com/v1/forecast?latitude=52.9149&longitude=-1.4472&hourly=temperature_2m,precipitation_probability,wind_speed_10m&timezone=Europe%2FLondon&forecast_days=1).

## Ultimele 10 oficiale — Derby

| Data | Competiție | Meci | Scor 90 | Primul gol | Gol≤70 | Dovadă |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-03 | Championship | Derby–Norwich | 0 - 4 | 8 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836912) |
| 2026-10-31 | Championship | Millwall–Derby | 4 - 1 | 3 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836902) |
| 2026-10-24 | Championship | Sheff Utd–Derby | 1 - 1 | 37 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836894) |
| 2026-10-17 | Championship | Derby–Stoke | 0 - 1 | 49 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836877) |
| 2026-10-14 | Championship | QPR–Derby | 4 - 1 | 18 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836873) |
| 2026-10-10 | Championship | Derby–Wrexham | 1 - 1 | 9 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836853) |
| 2026-09-19 | Championship | Burnley–Derby | 1 - 1 | 34 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836840) |
| 2026-09-12 | Championship | Derby–Birmingham | 1 - 2 | 40 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836829) |
| 2026-09-09 | Championship | Derby–West Brom | 0 - 1 | 16 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836823) |
| 2026-09-05 | Championship | West Ham–Derby | 3 - 0 | 6 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836813) |

10/10 Gol70 în aceste 10, dar fără victorie(3 E 7 Î). Nu confundăm succesul evenimentului cu forma echipei.

## Ultimele 10 oficiale — Bolton

| Data | Competiție | Meci | Scor 90 | Primul gol | Gol≤70 | Dovadă |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-03 | Championship | Charlton–Bolton | 0 - 1 | 18 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836920) |
| 2026-10-31 | Championship | Bolton–Bristol City | 2 - 3 | 13 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836898) |
| 2026-10-23 | Championship | Bolton–Middlesbrough | 1 - 5 | 13 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836887) |
| 2026-10-17 | Championship | Southampton–Bolton | 0 - 0 | Fără gol | Fail | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836881) |
| 2026-10-13 | Championship | Wolves–Bolton | 1 - 3 | 28 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836868) |
| 2026-10-10 | Championship | Bolton–Stoke | 1 - 2 | 45+4 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836851) |
| 2026-09-20 | Championship | Norwich–Bolton | 3 - 4 | 5 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836844) |
| 2026-09-12 | Championship | Bolton–Cardiff | 1 - 0 | 90 | Fail | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836827) |
| 2026-09-08 | Championship | Bolton–West Ham | 2 - 3 | 13 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836815) |
| 2026-09-05 | Championship | Millwall–Bolton | 4 - 0 | 38 | Succes | [sursă](https://www.fotmob.com/api/data/matchDetails?matchId=5836805) |

8/10 Gol70 în ultimele 10. În întreg sezonul mai existăQPR–Bolton 0–0 din 22 august. Bolton–Cardiff 1–0 cu primul gol 90 este pierzător pentruGol 70 deșiBolton a câștigat. Tabelele și numărătorile includ 45+,exclud prelungiri/departajare. Datele sunt din 28 partide unice, înaintea zilei 6 noiembrie; fără dublare deH 2 H.

## Sensibilitate la eliminări

Excluderea mecanică a celor 3 istorice cu eliminare până la 70(Derby–West Brom,Sheffield United–Derby,Bolton–Stoke)produceP=89.43%,AR20=83.43%. Numitori rămași:Derby 12/6 pe teren,Bolton 13/7 pe teren. Nu promovăm acest rezultat la estimare finală și nu schimbăm snapshotul. LaSheffield–DerbyșiDerby–West Bromprimul gol precedase eliminarea; efectul asupra primei deschideri nu poate fi atribuit acelui cartonaș. LaBolton–Stokeoponentul a fost eliminat 20,primul gol 45+4. Vezi[sensitivity.json](sensitivity.json).

## Și la cornere?

În aceeași comparație este tot rang 13, brut 92.47%,AR 2084.17%,λ 4,357143. Istoric 25/28,teren 12/14,3 reprizecu≥8 și 2 cu exact 7. Derbyacasămedia 4,57/varianța 9,29;Boltonafaramedia 4,14/varianța 5,81. Varianțele mai mari decât mediile și 12/14 teren sunt contraargumente la o interpretare de«92,47% sigur». Nu îl promovăm peste candidatulcornereSevilla–Alavésși nu combinăm piețele. DateleGol 70 nu sunt substitut pentrucornere 1 H.

## De ce loc 13 și verdictul profesional

Silkeborg–Lyngbyare 93,05% brut/88,16% AR20,24/25 istoric și 7/6 GF70 propriu pentru fiecare pe terenul relevant, conform screeningului original. Derby–Boltonare 88,27%/81,99%,25/28 și 4/7 respectiv 7/7 GF70. Diferența se explică prin intensitate, nu printr-un veto arbitrar. Această comparație nu este o revalidare completă la ora curentă a loturilor tuturor celorlalți candidați.

Pro decisiv: istoricul constant al deschiderii scorului în meciurile Derby și defensiva care cedează devreme oferă Bolton o cale credibilă de gol. Risc decisiv: Boltonare două 0–0 în 7 deplasări și incertitudini în construcție, iar Derbyproduce numai 0,57 GF70/meci acasă și pierdeSzmodics. O reacție defensivă a gazdei poate diminua ritmul. NO BET rămâne permis; pentru prioritate finală ar trebui clarificateSheehan/Itoși formulaDerby. Nu este nevoie de cote pentru această concluzie.

Nu este o selecție rea doar fiindcă nu ocupă primul loc. Este candidat secundar, susținut de date, cu avertismente concrete. Nu îl numesc prima alegere și nu afirm o probabilitate reală validată de 88–90%.

## Lecții aplicate și integritate

Aplicate:GF70 propriu separat deGA 70;prime goluri și 0–0 păstrate;același campionat în formulă;context la 0–0;lot actual versuslastStarting 11;surse contradictorii,prognoză versusgazon;scenariuAR 20 și sensibilitate la eliminări fără procente contextuale. Modelul și toate clasamentele rămân originale.

Intrări numerice:season_evidence.json/.csv,last10_official.csv,sensitivity.json; sursele actuale încontext_sources.jsonși răspunsurileAPI. Copiile CSVcanonice de azi șiuser_selectionsnu sunt scrise. O analiză cerută nu devine alegere sau pariu plasat.


Reverificare finală: 2026-11-06T10:49:28.535+02:00; identitate confirmată, meci neînceput, ora22:00RO, formații încă etichetate lastStarting11. Vezi final_status.json.
