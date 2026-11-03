# Gol pana la 70 inclusiv: doua analize pentru 2 noiembrie 2026

> **Actualizare dupa extinderea ceruta de utilizator:** comparatia initiala de mai
> jos este pastrata ca audit. [Comparatia completa Europa](EUROPE_ALL_MATCHES.md)
> evalueaza toate cele 27 de partide identificate in universul extins. Lorient-Brest
> este locul 8 al formulei, nu a doua alegere globala. Sassuolo-Fiorentina are acum
> P brut 94,40%: raportul oficial Lazio confirma golul Noslin la 65, nu 75, deci
> GA70 Sassuolo sezon/acasa devine 19/8. Tenerife-Andorra a fost reprogramat luni,
> la 15:00 RO, si este inclus in comparatia noua. Jurnalul initial ramane nemodificat.

Verificare finala a programului: **2026-11-02, 11:15 EET / Europe-Bucharest**.
Ambele partide sunt programate mai tarziu; nu au inceput la aceasta verificare.
Cererea curenta de DOUA analize prevaleaza asupra preferintei generale anterioare
de maximum o selectie. Sunt evaluate individual, fara bilet combinat.

| Ordine | Partida | Competitie | Ora Romaniei | P Poisson brut | Stres lambda -20% | Cota echitabila bruta |
|---|---|---|---|---:|---:|---:|
| 1 | Sassuolo - Fiorentina | Serie A, etapa 10 | 19:30 | 93,87% | 89,28% | 1,0653 |
| 2 | Lorient - Brest | Ligue 1, etapa 9 | 21:45 | 89,46% | 83,47% | 1,1178 |

Eveniment: cel putin un gol valid al oricarei echipe, cu minutul afisat <=70;
se include timpul aditional al primei reprize. Nu se cer goluri de la ambele
echipe. Nu se includ goluri anulate, prelungiri eliminatorii sau departajare.
Probabilitatile sunt rezultatele unui baseline euristic, NEVALIDATE ca rate reale.
Scenariul de stres nu este limita inferioara a unui interval de incredere.
Nicio cota de bookmaker sau probabilitate de tipster nu a intrat in calcule.

Ora italiana 18:30 este 19:30 in Romania; 20:45 in Franta este 21:45 in Romania.
[Program Italia, Sky](https://sport.sky.it/calcio/serie-a/partite/2026/giornata-10/sassuolo-fiorentina).
[Anuntul oficial Lorient privind mutarea pe 2 noiembrie](https://www.fclorient.bzh/fc-lorient-stade-brestois-29-reporte-au-lundi-2-novembre-a-20h45/).

## Continuitate fata de referintele Lecce si Strasbourg

Lecce-Genoa s-a terminat 1-2 pe 1 noiembrie, cu primul gol la 11'.
[Relatarea oficiala Lega Serie A](https://www.legaseriea.it/serie-a/news/lecce-genoa-2026-2027-1-2-cronaca-risultato-gol).
Reperul francez din discutie este Strasbourg-Auxerre. Nu exista in proiect
doua formule distincte validate, una italiana si alta franceza. Se aplica
aceeasi formula din promptul master si aceleasi verificari, pe datele si
contextul fiecarei partide. Rezultatul de ieri nu majoreaza probabilitatea de azi.

## Acoperire si limite ale cautarii

A fost inventariat programul disponibil pentru cele 20 de ligi ale proiectului.
Au fost identificate 19 partide programate in 12 dintre ligi, dintre care
Kayserispor-Muglaspor apare numai intr-un program consultat si ramane neconfirmata
independent; Tenerife-Andorra apare amanata. Pentru opt ligi nu apar meciuri
in programele consultate, ceea ce nu este confirmare oficiala ca nu exista.
Au fost inspectate suplimentar cele doua partide de Ligue 2 din Franta,
Reims-Sochaux si Red Star-Nancy, din afara listei originale de 20 de ligi.
[Program agregat](https://uniquescores.com/football/scores/2026-11-02/).
[Program suplimentar](https://statsbet.org/football/calendar).
[Reims-Sochaux, clubul oaspete](https://www.fcsochaux.fr/match/reims-vs-sochaux).
[Red Star-Nancy, clubul gazda](https://www.redstar.fr/2026-10-16/billetterie/votre-place-pour-red-star-fc-nancy-3/).

**Calcul complet pentru trei partide:** cele doua din Serie A si Lorient-Brest.
Pentru celelalte nu s-a obtinut in aceasta analiza sezonul complet de minute
ale golurilor. Nu sunt clasate numeric si nu sunt numite mai slabe.
Acesta este un top al finalistelor cu date reconstruite, NU un top global
demonstrat al tuturor ligilor. [Inventarul complet si motivele](fixture_coverage.csv).

Istoricul local Gol70 nu este o baza actuala completa; scorurile HT/FT ale
modelului FHG nu il inlocuiesc. Au fost transcrise din surse publice 57 de
partide distincte: 50 din campionatele actuale ale celor sase echipe analizate,
doua meciuri de cupa si cinci din sezonul francez precedent. Datele au fost
verificate structural fata de scoruri; aceasta nu echivaleaza cu verificarea
independenta a fiecarui eveniment de catre doua surse. Endpointul public
Sofascore a returnat 403, iar sursele alternative si lipsurile sunt documentate.

## Formula aplicata, identica pentru ambele selectii

A = gazda, B = oaspete. GF70 / GA70 sunt NUMERE de goluri pana la 70,
nu numere de meciuri cu gol si nu scoruri FT inmultite cu 70/90.

```
lambda_sezon = ((GF70_A/nA + GA70_B/nB) + (GF70_B/nB + GA70_A/nA)) / 2
lambda_teren = aceeasi formula, A numai acasa si B numai in deplasare
lambda_combinat = 0,5 * lambda_sezon + 0,5 * lambda_teren
p_brut = 1 - exp(-lambda_combinat)
p_stres = 1 - exp(-0,8 * lambda_combinat)
cota_echitabila = 1 / p_brut
```

Ponderarea 50/50 este euristica. Datele sezonului si cele de teren se suprapun.
Golurile multiple aparute dupa deschiderea scorului pot umfla lambda fata de
riscul real al unui 0-0 la 70. Nu s-au adaugat bonusuri procentuale pentru
accidentari, miza sau tactica; nu s-au ajustat rezultatele catre un interval dorit.

## 1. Sassuolo - Fiorentina: prima optiune

### Context si ritm

Campionat regulat, etapa 10, meci unic pentru puncte; fara scor agregat sau
nevoie de departajare la finalul partidei. Sassuolo are 8 puncte dupa noua
meciuri, Fiorentina 5. Gazdele vin dupa trei infrangeri consecutive, inclusiv
2-6 cu Lazio, oaspetii dupa 2-2 cu Atalanta. Ambelor le foloseste victoria,
dar niciuna nu este obligata prin regulament sa atace din primul minut.
Necesarul de puncte nu este in sine dovada de ritm ridicat.
[Rezultate Sassuolo](https://it.wikipedia.org/wiki/Unione_Sportiva_Sassuolo_Calcio_2026-2027).
[Rezultate Fiorentina](https://it.wikipedia.org/wiki/ACF_Fiorentina_2026-2027).

Aquilani cere in primul rand refacerea distantelor dintre compartimente si
organizare defensiva. Acesta este un CONTRAARGUMENT pentru un meci automat
deschis, dupa cele sase goluri primite. Vanoli subliniaza importanta punctelor,
dar cere corectarea erorilor si implicarea varfului; a descris pregatirea
recenta ca fiind axata pe recuperare.
[Declaratii Aquilani](https://www.calcionews24.com/aquilani-parole-vigilia-sassuolo-fiorentina-serie-a-2026-27/).
[Conferinta Vanoli](https://www.lanazione.it/firenze/sport/fiorentina/vanoli-fiorentina-sassuolo-ajwbz8og).

### Tactica si loturi

Sky estimeaza Sassuolo 4-3-3, Fiorentina 4-3-2-1; Eurosport indica 4-3-3
pentru oaspeti. Sunt formatii PROBABILE, nu confirmate. Variatiile pe posturi
nu permit prezentarea unui unic unsprezece ca sigur.
[Sursa Sky pentru formatii](https://sport.sky.it/calcio/serie-a/partite/2026/giornata-10/sassuolo-fiorentina/formazioni).
[Eurosport](https://www.eurosport.it/calcio/serie-a/2026-2027/sassuolo-fiorentina-probabili-formazioni-statistiche-quando-e-dove-vederla_sto23350631/story.shtml).

Nucleul ofensiv probabil Sassuolo este Berardi-Esposito-Lauriente; Fiorentina
poate folosi Mastantuono si Njie in sprijinul lui Pellegrino. Interpretarea mea:
latimea oferita de gazde si primirile dintre linii pot testa o defensiva viola
cu schimbari fortate; o pierdere la constructie poate oferi ocazia decisiva
oricarei echipe. Nu presupun ca ambii antrenori vor accepta tranzitii dese.

Thorstvedt este suspendat; Leysen, Boloca si Ismael Kone apar indisponibili.
La Fiorentina lipsesc Ranieri, Atta si Pongracic; Parisi apare de asemenea
indisponibil in previziuni. Gazzetta pastreaza incertitudini suplimentare si
posibile reveniri, inclusiv Matic/Idzes. Nu tratez lista larga drept diagnostic
medical confirmat. Pierderea lui Thorstvedt poate reduce prezenta ofensiva;
absentele din defensiva viola pot actiona in sens opus.
[Actualizarea Gazzetta](https://www.gazzetta.it/Calcio/prob_form/sassuolo-fiorentina/2638220).
[ANSA, declaratiile lui Vanoli](https://videoembed.ansa.it/toscana/notizie/2026/11/01/fiorentina-vanoli-col-sassuolo-gara-importante-ma-non-spartiacque_6ba80908-72e0-45f6-adff-2dce2609ba33.html).

Pagina generala Sky a meciului inca mentioneaza Grosso; conferinta de presa
din 1 noiembrie si pagina formatiilor confirma Vanoli. Campul vechi nu este
folosit ca sursa pentru antrenor. Sassuolo a jucat ultima data pe 27 octombrie,
Fiorentina pe 29: aproximativ sase versus patru zile calendaristice intre
meciuri. Nu este utilizat un coeficient numeric pentru deplasare sau oboseala.

### Numaratori si calcul

| Echipa / esantion | n | GF70 | GA70 |
|---|---:|---:|---:|
| Sassuolo, sezon | 9 | 10 | 18 |
| Fiorentina, sezon | 9 | 9 | 14 |
| Sassuolo, acasa | 4 | 5 | 7 |
| Fiorentina, deplasare | 4 | 4 | 6 |

```
lambda_sezon = ((10/9 + 14/9) + (9/9 + 18/9))/2 = 2,833333
lambda_teren = ((5/4 + 6/4) + (4/4 + 7/4))/2 = 2,750000
lambda_combinat = (2,833333 + 2,750000)/2 = 2,791667

P numai sezon = 94,12%
P numai teren = 93,61%
P combinat = 93,87%
P stres = 1-exp(-2,233333) = 89,28%
Cota echitabila bruta = 1,0653
```

In sezon, frecventa empirica a primului gol pana la 70 este 9/9 + 9/9,
adica 18/18 partide distincte. In ultimele zece oficiale: 10/10 + 10/10,
inclusiv cate un meci de cupa. Cupele nu intra in lambda.
Aceste frecvente NU inseamna probabilitate viitoare 100%.

**Argument decisiv:** sustinere din ambele echipe, fara a depinde exclusiv
de capacitatea gazdelor de a inscrie. **Riscul principal:** ambii antrenori
incearca sa stabilizeze defensiva, iar modelul este influentat de rezultate
cu multe goluri. Daca eliminam exclusiv Sassuolo-Lazio din istoricul gazdei,
pastrand restul identic, probabilitatea bruta scade la 90,34%. Este o
analiza de sensibilitate, nu noua estimare oficiala. In trei dintre cele
20 de meciuri recente, primul gol a venit abia la 63-65.

## 2. Lorient - Brest: a doua optiune, cu sustinere mai slaba

### Context si tactica

Etapa 9, derby regional, campionat regulat. Lorient are 12 puncte din opt
meciuri, Brest 9 din opt. Niciun regulament nu impune victoria astazi.
Lorient poate accepta o prima parte prudenta; nu traduc eticheta de derby
in obligatia de a ataca. Meciul a fost mutat de sambata pe luni din motive
organizatorice, nu pentru o problema de teren.
[Anuntul oficial al clubului](https://www.fclorient.bzh/fc-lorient-stade-brestois-29-reporte-au-lundi-2-novembre-a-20h45/).
[Bilantul Lorient, cu diferente de teren](https://statz.ai/team/lorient).

Lorient, antrenat de Alexandre Dujeux, foloseste probabil 3-4-2-1:
trei centrali, pistoane si Makengo/Bernardeau sau Faye in sprijinul lui Balde.
Brest, cu Julien Lachuer, este proiectat in 4-2-3-1: Ajorque reper central,
Doumbia intre linii, Mama Balde si Mboup pe laterale. Interpretarea mea:
Brest poate obtine o ocazie din jocul direct si mingea a doua, fara ca
Lorient sa aiba nevoie sa preseze agresiv. Invers, spatiile din jurul
fundasilor laterali brestoni pot permite tranzitii gazdelor.
[Previzualizare a formatiilor, date informative fara utilizarea pronosticului](https://derbista.com/fr/match/lorient-vs-brest-2026-10-31).

Lorient are trei meciuri consecutive fara gol primit, inclusiv 0-0 la Angers.
In victoriile cu Paris FC si Monaco primul gol a venit la 68 si 61.
Raportul oficial al lui Monaco descrie si pressingul mai redus al lui Lorient
inaintea acelei partide; observatia este datata, nu o masuratoare pentru azi.
[Raport Monaco si deschiderea scorului la 61](https://www.asmonaco.com/fr/pros/calendrier/fc-lorient-as-monaco).
[Golul cu Paris la 68](https://www.deux-zero.com/ligue-1/fiche/edition/2026-2027/match/25168).

La Brest, toate cele opt meciuri ale sezonului au fost deschise pana la
minutul 29, indiferent cine a marcat. In cele patru deplasari, primul gol
a venit la 29, 12, 23 si 6. Acesta este argumentul temporal favorabil,
care contrabalanseaza prudenta gazdelor.
[Sezonul Brest si minutele](https://fr.wikipedia.org/wiki/Saison_2026-2027_du_Stade_brestois_29).

### Loturi si recuperare

Katseris este raportat absent. Un articol recent indica posibilitatea
revenirii lui Mbamba, in timp ce unele agregatoare il pastreaza indisponibil;
statutul ramane incert pana la o confirmare mai buna. Chardonnet si Le Guen
sunt raportati accidentati la Brest. Absenta lui Lloris de la un antrenament
a fost descrisa ca gestionare a efortului, nu ca forfait confirmat.
[Relatare preluata din Ouest-France despre Lorient](https://www.collector.com.tr/1024526059).
[Relatare despre Brest si Lloris](https://m40radio.fr/france/stade-brestois-le-seul-defenseur-central-de-metier-absent-de-la-seance-avant-le-derby-a-lorient-julien-lachuer-rassure/).

Lorient a jucat ultima data pe 25 octombrie, Brest pe 23; aproximativ opt
si zece zile calendaristice intre partide. Nu exista in acest esantion o
presiune evidenta a unui meci jucat cu doua-trei zile in urma. Derby-ul nu
presupune o deplasare internationala; nu s-a presupus itinerarul concret.

### Numaratori si calcul

| Echipa / esantion | n | GF70 | GA70 |
|---|---:|---:|---:|
| Lorient, sezon | 8 | 6 | 6 |
| Brest, sezon | 8 | 13 | 9 |
| Lorient, acasa | 4 | 5 | 4 |
| Brest, deplasare | 4 | 6 | 4 |

```
lambda_sezon = ((6/8 + 9/8) + (13/8 + 6/8))/2 = 2,125000
lambda_teren = ((5/4 + 4/4) + (6/4 + 4/4))/2 = 2,375000
lambda_combinat = (2,125000 + 2,375000)/2 = 2,250000

P numai sezon = 88,06%
P numai teren = 90,70%
P combinat = 89,46%
P stres = 1-exp(-1,800000) = 83,47%
Cota echitabila bruta = 1,1178
```

Frecventa sezonului: Lorient 6/8, Brest 8/8; agregat distinct 14/16 = 87,5%.
Configuratia terenului: 4/4 acasa Lorient si 4/4 deplasari Brest.
Ultimele zece oficiale: Lorient 8/10, Brest 10/10; agregat distinct 18/20 = 90%.
Acest ultim calcul include cate doua partide din sezonul precedent, cu alte
loturi/antrenori, care NU intra in lambda. Cele doua esecuri recente ale lui
Lorient sunt 0-0 cu Nice si Angers. Nu sunt eliminate din tabel.

**Argument decisiv:** cronologia constanta a primului gol in meciurile lui
Brest, inclusiv in deplasare. **Riscul principal:** Lorient poate tine meciul
inchis; doua victorii recente au avut golul foarte aproape de prag.
Nu este echivalentul numeric al lui Strasbourg-Auxerre si nu ii atribui un
procent de 92-95 doar pentru ca este tot in Franta.

## Alternative si pretul matematic

Atalanta-Parma a fost calculat complet:

| Esantion | n | GF70 | GA70 |
|---|---:|---:|---:|
| Atalanta sezon | 9 | 13 | 13 |
| Parma sezon | 9 | 2 | 12 |
| Atalanta acasa | 5 | 10 | 7 |
| Parma deplasare | 4 | 0 | 7 |

Lambda sezon 2,222222; teren 2,575000; combinat 2,398611.
P brut 90,92%; stres 85,32%; cota echitabila 1,0999.
Primul gol pana la 70: Atalanta 8/9, Parma 7/9; agregat 15/18.
Parma nu a marcat pana la 70 in niciuna dintre cele patru deplasari.
Atalanta are o lista lunga de absente, inclusiv mai multi atacanti si fundasi.
Prefer Sassuolo-Fiorentina pentru sustinerea din ambele directii, fara a
numi Atalanta o echipa completa sau garantat ofensiva.
[Istoric Atalanta](https://it.wikipedia.org/wiki/Atalanta_Bergamasca_Calcio_2026-2027).
[Istoric Parma](https://it.wikipedia.org/wiki/Parma_Calcio_1913_2026-2027).
[Loturile probabile si absentele](https://www.eurosport.it/calcio/serie-a/2026-2027/atalanta-parma-probabili-formazioni-statistiche-quando-e-dove-vederla_sto23350632/story.shtml).

Celelalte partide inventariate, inclusiv Newcastle-Everton, Benfica-Alverca,
cele din Romania si cele doua din Ligue 2, nu au primit probabilitati
inventate din scoruri FT sau recomandari ale altora.

La p brut 89,46%, 1,10 nu depaseste pragul matematic al lui Lorient-Brest;
pragul brut este 1,1178, fara marja pentru incertitudine. Pentru Sassuolo
pragul brut este 1,0653. Acestea nu confirma cote reale si nu demonstreaza
avantaj: modelul nu este validat. Nu cer utilizatorului cote si nu blochez
analiza statistica pentru lipsa lor.

## Istoricul primului gol

Ordine descrescatoare dupa data reala a meciului, nu dupa numarul etapei.
La Brest, partida restanta cu Strasbourg din 13 mai este mai recenta decat
PSG-Brest din 10 mai; este inclusa corect in ultimele zece. Meciul cu PSG,
deschis abia la 82, este pastrat in CSV ca observatie verificata suplimentara,
dar NU intra in frecventa ultimelor zece sau in sezonul actual.

### Sassuolo: ultimele 10 meciuri oficiale

| Data | Competitie / sezon | Partida | Scor la 90' | Primul gol | Gol70 |
|---|---|---|---|---:|---|
| 2026-10-27 | Serie A 2026/27 | Sassuolo - Lazio | 2-6 | 2' | DA |
| 2026-10-24 | Serie A 2026/27 | Como - Sassuolo | 2-0 | 49' | DA |
| 2026-10-16 | Serie A 2026/27 | Frosinone - Sassuolo | 4-2 | 9' | DA |
| 2026-10-11 | Serie A 2026/27 | Sassuolo - Milan | 1-1 | 63' | DA |
| 2026-09-18 | Serie A 2026/27 | Monza - Sassuolo | 2-1 | 51' | DA |
| 2026-09-12 | Serie A 2026/27 | Sassuolo - Juventus | 3-2 | 65' | DA |
| 2026-09-06 | Serie A 2026/27 | Bologna - Sassuolo | 2-2 | 19' | DA |
| 2026-09-02 | Coppa Italia 2026/27 | Sassuolo - Frosinone | 1-1 | 36' | DA |
| 2026-08-29 | Serie A 2026/27 | Sassuolo - Torino | 2-1 | 32' | DA |
| 2026-08-23 | Serie A 2026/27 | Atalanta - Sassuolo | 2-1 | 6' | DA |

Surse pentru fiecare partida: [CSV cu minutele tuturor golurilor](goal_minutes_verified.csv). Frecventa: 10/10.

### Fiorentina: ultimele 10 meciuri oficiale

| Data | Competitie / sezon | Partida | Scor la 90' | Primul gol | Gol70 |
|---|---|---|---|---:|---|
| 2026-10-29 | Serie A 2026/27 | Fiorentina - Atalanta | 2-2 | 41' | DA |
| 2026-10-25 | Serie A 2026/27 | Inter - Fiorentina | 2-1 | 65' | DA |
| 2026-10-18 | Serie A 2026/27 | Fiorentina - Como | 1-3 | 32' | DA |
| 2026-10-10 | Serie A 2026/27 | Genoa - Fiorentina | 2-1 | 8' | DA |
| 2026-09-20 | Serie A 2026/27 | Fiorentina - Napoli | 1-1 | 23' | DA |
| 2026-09-15 | Coppa Italia 2026/27 | Fiorentina - Pisa | 3-0 | 40' | DA |
| 2026-09-11 | Serie A 2026/27 | Venezia - Fiorentina | 2-4 | 22' | DA |
| 2026-09-05 | Serie A 2026/27 | Fiorentina - Torino | 1-2 | 47' | DA |
| 2026-08-29 | Serie A 2026/27 | Fiorentina - Frosinone | 0-3 | 26' | DA |
| 2026-08-24 | Serie A 2026/27 | Roma - Fiorentina | 4-0 | 27' | DA |

Surse pentru fiecare partida: [CSV cu minutele tuturor golurilor](goal_minutes_verified.csv). Frecventa: 10/10.

### Lorient: ultimele 10 meciuri oficiale

| Data | Competitie / sezon | Partida | Scor la 90' | Primul gol | Gol70 |
|---|---|---|---|---:|---|
| 2026-10-25 | Ligue 1 2026/27 | Angers - Lorient | 0-0 | fara gol | NU |
| 2026-10-18 | Ligue 1 2026/27 | Lorient - Monaco | 1-0 | 61' | DA |
| 2026-10-10 | Ligue 1 2026/27 | Lorient - Paris FC | 1-0 | 68' | DA |
| 2026-09-19 | Ligue 1 2026/27 | Le Mans - Lorient | 2-1 | 3' | DA |
| 2026-09-12 | Ligue 1 2026/27 | Lorient - Toulouse | 2-2 | 37' | DA |
| 2026-09-05 | Ligue 1 2026/27 | Lens - Lorient | 0-1 | 54' | DA |
| 2026-08-29 | Ligue 1 2026/27 | Lorient - Troyes | 1-2 | 29' | DA |
| 2026-08-22 | Ligue 1 2026/27 | Nice - Lorient | 0-0 | fara gol | NU |
| 2026-05-17 | Ligue 1 2025/26 | Lorient - Le Havre | 0-2 | 33' | DA |
| 2026-05-10 | Ligue 1 2025/26 | Metz - Lorient | 0-4 | 24' | DA |

Surse pentru fiecare partida: [CSV cu minutele tuturor golurilor](goal_minutes_verified.csv). Frecventa: 8/10.

### Brest: ultimele 10 meciuri oficiale

| Data | Competitie / sezon | Partida | Scor la 90' | Primul gol | Gol70 |
|---|---|---|---|---:|---|
| 2026-10-23 | Ligue 1 2026/27 | Brest - Nice | 1-2 | 15' | DA |
| 2026-10-17 | Ligue 1 2026/27 | Lille - Brest | 1-1 | 6' | DA |
| 2026-10-10 | Ligue 1 2026/27 | Brest - Angers | 5-1 | 9' | DA |
| 2026-09-20 | Ligue 1 2026/27 | Auxerre - Brest | 2-1 | 23' | DA |
| 2026-09-13 | Ligue 1 2026/27 | Brest - PSG | 0-1 | 5' | DA |
| 2026-09-05 | Ligue 1 2026/27 | Le Havre - Brest | 1-2 | 12' | DA |
| 2026-08-29 | Ligue 1 2026/27 | Brest - Toulouse | 2-2 | 9' | DA |
| 2026-08-22 | Ligue 1 2026/27 | Le Mans - Brest | 2-2 | 29' | DA |
| 2026-05-17 | Ligue 1 2025/26 | Brest - Angers | 1-1 | 55' | DA |
| 2026-05-13 | Ligue 1 2025/26 | Brest - Strasbourg | 1-2 | 9' | DA |

Surse pentru fiecare partida: [CSV cu minutele tuturor golurilor](goal_minutes_verified.csv). Frecventa: 10/10.

Sursele italiene de sezon sunt [Sassuolo](https://it.wikipedia.org/wiki/Unione_Sportiva_Sassuolo_Calcio_2026-2027)
si [Fiorentina](https://it.wikipedia.org/wiki/ACF_Fiorentina_2026-2027).
Cupele au fost confruntate cu [Sky, Sassuolo-Frosinone](https://sport.sky.it/calcio/coppa-italia/partite/2026/giornata-3/sassuolo-frosinone)
si [Pisa, raport oficial](https://pisasportingclub.com/fiorentina-pisa-3-0/).
Pentru Lorient se folosesc si [calendarul oficial](https://www.fclorient.bzh/calendrier-pro-2026-2027/),
[raportul cu Toulouse](https://www.fclorient.bzh/match/fc-lorient-4/),
[raportul cu Le Mans](https://www.fclorient.bzh/match/fc-lorient-5/)
si sursele individuale din CSV. Datele Brest: [sezonul actual](https://fr.wikipedia.org/wiki/Saison_2026-2027_du_Stade_brestois_29),
[ultimul meci din sezonul trecut](https://angers-sco.fr/stade-brestois-angers-sco-un-point-en-bretagne-pour-cloturer-la-saison/),
[restanta cu Strasbourg](https://www.ladepeche.fr/sports/resultats-sportifs/football/ligue-1/competition-reguliere/rencontre/533523/brest-strasbourg).

## Incertitudini si verificare

- Patru meciuri in configuratia acasa/deplasare reprezinta un esantion mic.
  Sezoanele includ adversari de nivel diferit; nu exista aici o ajustare
  invatata si validata pentru forta lor.
- Titularii nu sunt confirmati. Informatiile contradictorii despre lot nu
  sunt transformate in bonusuri procentuale.
- Nu a fost verificata independent starea actuala a gazonului sau prognoza
  exacta la kickoff. Prognoza datata 29 octombrie pentru Lorient este prea
  veche si partiala pentru a justifica un ajustaj numeric:
  [buletinul datat](https://www.meteometeo.fr/lorient/date/2-novembre-2026/).
- xG din agregatoare nu a fost introdus: nu este necesar formulei, iar
  comparabilitatea furnizorilor si a esantioanelor nu a fost stabilita.
- Baseline-ul nu modeleaza explicit timpul pana la primul gol. Frecventa
  istorica si p Poisson nu sunt acelasi lucru.
- Nu sunt cunoscute cotele disponibile, marja operatorului sau profitabilitatea.
  Niciun rezultat de ieri nu este tratat ca validare statistica.

Verificare independenta fata de calculul JavaScript, numai din fisierele salvate:

```powershell
powershell -NoProfile -File .\Analysis\Goal70_2026-11-02\recompute_goal70.ps1
```

Rezultat: PASS pentru 57 de partide unice, reconcilierea tuturor scorurilor,
minutul primului gol, definitia <=70, excluderea sezonului vechi/cupelor din
lambda si reproducerea celor trei probabilitati. Verificarea codului nu
certifica adevarul extern al surselor. Fisierele de productie nu au fost modificate.

Predictiile sunt salvate in [jurnalul prospectiv](prospective_selections.csv);
campul rezultat ramane gol pana dupa meciuri. Prima optiune ramane Sassuolo-Fiorentina;
Lorient-Brest este o a doua analiza individuala cu sustinere mai slaba.

