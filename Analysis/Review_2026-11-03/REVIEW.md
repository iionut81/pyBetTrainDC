# Revizuirea zilei3 noiembrie — verificata4 noiembrie2026

Verificare initiala08:02 RO; raport 2026-11-04T08:12:57.873+02:00. Observatiile exacte, identitatile si timestampurile sunt in goal70_outcomes.csv/corners1h_outcomes.csv si FotMob/. Predictiile/rangurile prospective originale nu au fost modificate.

Alegerile explicite: **QPR-Burnley Gol70: Succes**, primul gol38 (Burnley), apoi65 (QPR), FT1-1. **Derby-Norwich cornere1H: Succes**, FirstHalf2-4=6, SecondHalf2-1=3, FT4-5=9; FT0-4. Observatiile sunt [FotMob QPR](https://www.fotmob.com/api/data/matchDetails?matchId=5836915) si [FotMob Derby](https://www.fotmob.com/api/data/matchDetails?matchId=5836912). [Raportul oficial Derby](https://www.dcfc.co.uk/news/2026/11/report-derby-county-vs-norwich-city) confirma dominatia Norwich si cele3 goluri din prima repriza; nu este substitut pentru cornerele FirstHalf. Se pastreaza diferenta de minut27 din FotMob fata de relatarea textuala28 a celui de-al doilea gol; nu afecteaza evenimentul. Nu exista dovada de pariu plasat sau profit.

Univers Gol70: **18/20**, doua esecuri: Bodo/Glimt-Lille0-0 si Lyngby-Randers cu primul gol90+3. Cornere: **18/20** in toate20 partidele inventariate, respectiv **17/19** pentru cele19 modele publicate; Shakhtar-Sporting este observatie ulterioara, fara model prospectiv, si nu se prezinta ca predictie calculata.

| Meci | FT | Primul gol | Model | Stres | Sursa |
| --- | --- | --- | --- | --- | --- |
| Bodo/Glimt - Lille | 0-0 | 0-0 | 83.88% | 76.77% | [rezultat](https://www.fotmob.com/api/data/matchDetails?matchId=6106316) |
| Lyngby - Randers FC | 1-0 | 90+3 | 88.13% | 81.83% | [rezultat](https://www.fotmob.com/api/data/matchDetails?matchId=6298366) |

| Meci | Cornere1H | Sursa |
| --- | --- | --- |
| Villarreal - Paris Saint-Germain | 8 | [rezultat](https://www.fotmob.com/api/data/matchDetails?matchId=6106326) |
| Watford - Lincoln City | 10 | [rezultat](https://www.fotmob.com/api/data/matchDetails?matchId=5836916) |

La limita cornere exact7 (succes): Blackburn Rovers-Stoke City. Primul gol intre65-70: niciunul.

## Comparatia cu analiza si lectii

QPR-Burnley a furnizat golul cautat in38, compatibil cu ipoteza unei partide cu oportunitati. Scorul final1-1 nu valideaza singur toate argumentele ofensive sau modelul93,63%. Derby a primit3 goluri pana la pauza, dar totalul de cornere a ramas6: succesul nu dovedeste ca un gol timpuriu reduce intotdeauna cornerele. Model96,61% vs rezultat binar1 este o singura observatie, nu calibrare.

Bodo-Lille0-0 si Lyngby-Randers90+3 arata riscul timpului PRIMULUI gol si al proiectiei de campionat in Europa/cupa. Fara cronologie verificata a ocaziilor si fara filmul partidelor nu atribui aceste esecuri unei cauze tactice demonstrate. Le clasific drept rezultate compatibile cu variatie aleatoare, cu ipoteza de nepotrivire intre competitii de testat. La Villarreal-PSG8 cornere este depasire cuun singur corner; Watford-Lincoln10 este depasire mai clara. FT nu explica singur presiunea primei reprize.

Ipoteze prospective: compara cronologic ligile interne cu proiectiile in cupe; raporteaza separarea GF70 propriu pe teren si frecventa fara gol70; monitorizeaza variatia FirstHalf, exact7 si >=8; documenteaza schimbarile de antrenor si sursele depasite inainte de selectie. Fara procente ajustate dupa rezultat, fara retraining sau schimbarea ponderilor50/50.

## Restante2 noiembrie

Reincercarea universului de27 meciuri la cornere a verificat **21/25**, cu un meci nefinalizat/reprogramat si unul abandonat, excluse din numitor. Vezi ../Review_2026-11-02/corners1h_outcomes.csv si RETRY_2026-11-04.md. Cele doua alegeri ale utilizatorului din2 noiembrie isi pastreaza succesul verificat. Nu se rescriu estimarile sau rezultatele vechi fara reconcilierea conflictelor de furnizor.


## Confirmari independente si conflicte de minut

[Raportul oficial Burnley](https://www.burnleyfootballclub.com/media-article/match-report-qpr-1-1-burnley) confirma primul gol38; egalarea este64 in raport si65 in FotMob. CriteriulGol70 ramane succes. Pentru Tenerife-Andorra, [pagina oficiala LaLiga](https://www.laliga.com/en-US/match/temporada-2026-2027-laliga-hypermotion-cd-tenerife-fc-andorra-12) confirma abandonul la2-2; eticheta veche de amanare ramane vizibila in arhiva, cu aceasta rectificare explicita. Rezultatul este exclus din rata in continuare. Nu se modifica predictiile inghetate.

Topurile dupa rangul brut original Gol70: top1: 1/1; top3: 3/3; top5: 5/5. Univers intern: 10/10; proiectii Europa/cupa raportate separat in CSV.

## Completare solicitata de utilizator: semnalele de dinaintea esecurilor

Aceasta sectiune compara rezultatele deja verificate cu snapshoturile originale
din3 noiembrie. Nu este research nou si nu adauga cauze tactice nedemonstrate.
Cele patru esecuri provin din lista comparata; nu erau patru alegeri ale
utilizatorului. Cele doua alegeri explicite au fost ambele Succes.

| Piata / meci | Brut original | AR20% original | Istoric distinct inaintea meciului | Rezultat |
| --- | ---: | ---: | --- | --- |
| Gol70 / Bodo-Glimt–Lille | 83,88% | 76,77% | 32/34; teren16/17 | 0–0, Fail |
| Gol70 / Lyngby–Randers | 88,13% | 81,83% | 22/24; teren10/12 | Primul gol90+3, Fail |
| Cornere1H / Villarreal–PSG | 89,61% | 79,21% | 16/20; minimum8 in4/20; exact7 in2/20 | 8, Fail |
| Cornere1H / Watford–Lincoln | 87,03% | 74,98% | 20/26; minimum8 in6/26; exact7 in2/26 | 10, Fail |

Sursele prospective sunt fisierele3 noiembrie din Selection_History si
Corners1H_2026-11-03/all_models.csv; rezultatele si linkurile sunt in tabelele
de mai sus si CSV-urile de rezultate. Procentele AR20% sunt scenarii, nu
rate de succes masurate si nu limite inferioare statistice garantate.

- Bodo-Lille: frecventa istorica32/34 era mare, dar modelul era83,88%,
  rang19/20, cu proiectie din campionatele interne in Europa. Semnalul pentru
  prudenta exista deja; nu putem demonstra din0–0 ca tactica sau motivatia
  au fost cauza. Trebuie urmarita separat performanta proiectiilor europene.
- Lyngby-Randers: rang12/20, cupa, model88,13%, redus la81,83% in scenariu.
  Golul din90+3 arata de ce scorul final cu gol nu valideaza evenimentul
  pana la70. De urmarit frecventa primului gol tardiv, nu numai numarul
  total de goluri sau prezenta unui gol pana la final.
- Villarreal-PSG: istoricul pentru <=7 era80%, sub modelul89,61%.
  Existau deja4 depasiri in20 observatii, iar scenariul scadea la79,21%.
  Diferenta model–istoric si transferul in Europa erau riscuri documentabile.
  Depasirea cuun corner nu justifica schimbarea retrospectiva a pragului.
- Watford-Lincoln: istoricul20/26 inseamna76,92%, fata de87,03% din model;
  scenariul74,98% era mai aproape de frecventa observata. Minimum8 aparuse
  deja in6/26 =23,08%. Acesta este un semnal de prudenta vizibil inaintea
  meciului, nu o regula descoperita doar dupa rezultat. Pentru cauza efectiva
  a celor10 cornere lipseste aici o analiza verificata a cronologiei fazelor.

### Ce a functionat si ce ramane de testat

QPR-Burnley: primul gol38 este compatibil cu selectia prospectiva sustinuta
de productia ofensiva proprie acasa/deplasare. Modelul93,63% era totusi peste
frecventa23/26 =88,46%; reusita nu demonstreaza ca93,63% este calibrat.
Schimbarea antrenorului Burnley si ultimul0–0 figurau deja drept contraargumente.
Vezi analiza prospectiva Goal70_2026-11-03/EUROPE_ALL_MATCHES.md.

Derby-Norwich: totalul1H6 este Succes. Reusita procedurala clara este separarea
cornerelor1H deFT si tratarea explicita a observatiei istorice Derby-Stoke
cu surse contradictorii. Riscul mare de dispersie era deja recunoscut:
varianta13,20 fata de media4,20 pentru Derby acasa in esantionul final.
Cele3 goluri din prima repriza nu dovedesc o relatie universala goluri–cornere.

Diagnostic pentru intreaga lista: suma(1-p brut) anticipa matematic2,20
esecuri Gol70 in20 modele si2,12 esecuri la cornere in19 modele. Au aparut
cate2. Suma este o asteptare interna a modelelor, nu o validare a calibrarii
si nu un numar garantat de esecuri. Nu necesita presupunerea independentei.
Top1/top3/top5 dupa screeningul brut original au fost1/1,3/3,5/5 pe fiecare
piata; sunt subseturi care se suprapun, din aceeasi zi, nu trei teste independente.

Imbunatatiri de proces de aplicat prospectiv: afiseaza impreuna modelul,
frecventa istorica, esantionul si AR20%; semnaleaza diferentele importante
fara penalizari procentuale inventate; separa intern/Europa/cupa; urmareste
primele goluri tardive si lipsa golului; la cornere urmareste varianta,
exact7 si minimum8; verifica antrenorii si loturile inaintea selectiei finale.
Acestea nu sunt schimbari de model testate. Un model alternativ pentru
dispersia cornerelor sau alte ponderi necesita evaluare istorica/prospectiva
separata si autorizare inainte de modificarea productiei.

Limita ramasa: raportul are rezultate si diagnostice statistice verificabile,
dar nu un audit tactic complet pentru fiecare esec. Nu putem declara ca
am eliminat cauzele esecurilor sau ca le vom evita pe toate in viitor.
