# Mallorca-Cadiz: de ce este pe locul 21

**2 noiembrie 2026, 22:00 RO / 21:00 CET, Son Moix, LaLiga Hypermotion, etapa 12.** Status pre-meci reverificat la 11:52:34 RO. Nu au fost folosite cote. Este un studiu al unui candidat, nu o noua selectie zilnica sau o schimbare a modelului.

## Concluzia

**82,66% brut, 87,13% numai pe teren, 75,39% in stres, cota echitabila 1,2097.** Locul 21 vine din formula fixa, nu dintr-un veto contextual. Are un semnal favorabil care merita aratat: Mallorca acasa 5/5 si Cadiz afara 4/4 cu gol <=70. Totusi, pe sezon sunt numai 17/22 (77,27%), iar ponderarea 50/50 trage rezultatul in jos. Nici 9/9, nici 82,66% nu reprezinta probabilitati reale validate.

## Inputuri si calcul

| Esantion | N | Goluri marcate <=70 | Goluri primite <=70 | Partide cu primul gol <=70 |
|---|---:|---:|---:|---:|
| Mallorca sezon | 11 | 9 | 3 | 8/11 |
| Cadiz sezon | 11 | 7 | 13 | 9/11 |
| Mallorca acasa | 5 | 7 | 1 | 5/5 |
| Cadiz deplasare | 4 | 2 | 8 | 4/4 |

```text
lambda_sezon = ((9/11 + 13/11) + (7/11 + 3/11))/2 = 1,454545
P_sezon = 1 - exp(-1,454545) = 76,65%
lambda_teren = ((7/5 + 8/4) + (2/4 + 1/5))/2 = 2,05
P_teren = 1 - exp(-2,05) = 87,13%
lambda_combinat = (1,454545 + 2,05)/2 = 1,752273
P_model = 1 - exp(-1,752273) = 82,66%
P_stres = 1 - exp(-0,8 * 1,752273) = 75,39%
Cota echitabila = 1 / 0,82662055 = 1,2097
```

Ponderarea 50/50 este euristica. Esantioanele sezon/teren se suprapun. Diferenta 76,65-87,13% este sensibilitate la alegerea esantionului, nu interval de incredere. Nu schimb ponderile pentru a obtine peste 85%. Modelul nu are ajustare dupa adversari sau schimbarea antrenorului, iar golurile ulterioare deschiderii scorului pot distorsiona interpretarea pentru PRIMUL gol.

## Toate cele 9 observatii relevante pentru teren

| Data | Meci si scor final | Primul gol |
|---|---|---:|
| 2026-08-15 | [Mallorca - Real Valladolid 2-0](https://www.espn.com/soccer/match/_/gameId/401883258) | 62' |
| 2026-08-22 | [Eldense - Cádiz 2-2](https://www.espn.com/soccer/match/_/gameId/401883253) | 49' |
| 2026-08-30 | [Mallorca - Ceuta 3-0](https://www.espn.com/soccer/match/_/gameId/401883244) | 20' |
| 2026-09-06 | [Almería - Cádiz 3-2](https://www.espn.com/soccer/match/_/gameId/401883241) | 13' |
| 2026-09-13 | [Mallorca - CD Sabadell 2-0](https://www.espn.com/soccer/match/_/gameId/401883172) | 10' |
| 2026-09-26 | [Tenerife - Cádiz 1-1](https://www.espn.com/soccer/match/_/gameId/401883163) | 25' |
| 2026-09-27 | [Mallorca - Almería 0-1](https://www.espn.com/soccer/match/_/gameId/401883164) | 35' |
| 2026-10-11 | [Mallorca - Las Palmas 3-1](https://www.espn.com/soccer/match/_/gameId/401883183) | 27' |
| 2026-10-18 | [Granada - Cádiz 2-0](https://www.espn.com/soccer/match/_/gameId/401883156) | 50' |

Toate au primul gol pana la 62 inclusiv. Mallorca a marcat pana la 70 in 4/5 acasa; al cincilea succes a fost golul adversarului Almeria. Cadiz a primit gol pana la 70 in toate cele 4 deplasari. Faptul ca Mallorca este defensiv solida poate proteja scorul mic, dar reduce probabilitatea ca oaspetii sa salveze selectia daca gazdele nu marcheaza.

## Ultimele 10 meciuri oficiale ale fiecarei echipe

Numai meciuri jucate, nu amicalele; la Cadiz includ cupa din 29 octombrie separat de formula de campionat. Mallorca-Alcoyano nu s-a jucat la data initiala; nu inventez un rezultat de cupa.

### Mallorca

| Data | Competitie | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---|---:|---|
| 2026-10-24 | LaLiga Hypermotion | [Leganés - Mallorca 1-1](https://www.espn.com/soccer/match/_/gameId/401883151) | 34' | DA |
| 2026-10-17 | LaLiga Hypermotion | [Eibar - Mallorca 1-1](https://www.espn.com/soccer/match/_/gameId/401883178) | 52' | DA |
| 2026-10-11 | LaLiga Hypermotion | [Mallorca - Las Palmas 3-1](https://www.espn.com/soccer/match/_/gameId/401883183) | 27' | DA |
| 2026-10-04 | LaLiga Hypermotion | [Girona - Mallorca 0-0](https://www.espn.com/soccer/match/_/gameId/401883160) | fara gol (0-0) | NU |
| 2026-09-27 | LaLiga Hypermotion | [Mallorca - Almería 0-1](https://www.espn.com/soccer/match/_/gameId/401883164) | 35' | DA |
| 2026-09-19 | LaLiga Hypermotion | [Real Sociedad II - Mallorca 0-1](https://www.espn.com/soccer/match/_/gameId/401883167) | 81' | NU |
| 2026-09-13 | LaLiga Hypermotion | [Mallorca - CD Sabadell 2-0](https://www.espn.com/soccer/match/_/gameId/401883172) | 10' | DA |
| 2026-09-05 | LaLiga Hypermotion | [Eldense - Mallorca 0-0](https://www.espn.com/soccer/match/_/gameId/401883219) | fara gol (0-0) | NU |
| 2026-08-30 | LaLiga Hypermotion | [Mallorca - Ceuta 3-0](https://www.espn.com/soccer/match/_/gameId/401883244) | 20' | DA |
| 2026-08-24 | LaLiga Hypermotion | [Granada - Mallorca 2-0](https://www.espn.com/soccer/match/_/gameId/401883252) | 20' | DA |

### Cadiz

| Data | Competitie | Meci si scor final | Primul gol | Gol <=70 |
|---|---|---|---:|---|
| 2026-10-29 | Copa del Rey | [Xerez CD - Cadiz 1-3](https://www.cadizcf.com/noticias/el-cadiz-cf-remonta-en-chapin-y-avanza-en-la-copa-del-rey-1-3) | 2' | DA |
| 2026-10-25 | LaLiga Hypermotion | [Cádiz - Ceuta 3-1](https://www.espn.com/soccer/match/_/gameId/401883130) | 45'+1' | DA |
| 2026-10-18 | LaLiga Hypermotion | [Granada - Cádiz 2-0](https://www.espn.com/soccer/match/_/gameId/401883156) | 50' | DA |
| 2026-10-11 | LaLiga Hypermotion | [Cádiz - Sporting Gijón 0-1](https://www.espn.com/soccer/match/_/gameId/401883186) | 87' | NU |
| 2026-10-03 | LaLiga Hypermotion | [Cádiz - Leganés 4-0](https://www.espn.com/soccer/match/_/gameId/401883194) | 6' | DA |
| 2026-09-26 | LaLiga Hypermotion | [Tenerife - Cádiz 1-1](https://www.espn.com/soccer/match/_/gameId/401883163) | 25' | DA |
| 2026-09-19 | LaLiga Hypermotion | [Cádiz - Girona 1-2](https://www.espn.com/soccer/match/_/gameId/401883206) | 35' | DA |
| 2026-09-12 | LaLiga Hypermotion | [Cádiz - Las Palmas 0-1](https://www.espn.com/soccer/match/_/gameId/401883214) | 60' | DA |
| 2026-09-06 | LaLiga Hypermotion | [Almería - Cádiz 3-2](https://www.espn.com/soccer/match/_/gameId/401883241) | 13' | DA |
| 2026-08-30 | LaLiga Hypermotion | [Cádiz - Real Valladolid 1-1](https://www.espn.com/soccer/match/_/gameId/401883222) | 17' | DA |

Esecurile din intreg sezonul: Eldense-Mallorca 0-0, Real Sociedad II-Mallorca 0-1 (primul gol 81), Girona-Mallorca 0-0; Cadiz-Celta Fortuna 0-0 si Cadiz-Sporting 0-1 (87). Niciunul nu apartine configuratiei de teren de azi. Ultimele 10 oficiale: Mallorca 7/10; Cadiz 9/10, cu adversar inferior in cupa. Nu tratez victoria din cupa ca echivalenta cu o deplasare la Mallorca.

## Context, antrenori si loturi

Programul oficial Mallorca confirma etapa 12, luni la 21:00 CET. Sunt puncte in campionat, fara scor agregat sau prelungiri. Obiectivul sezonului este promovarea pentru Mallorca; o victorie ar duce-o de la 19 la 22 puncte, egalul la 20. Clasamentul din match center-ul oficial Cadiz arata Mallorca 8 cu 19 puncte si Cadiz 19 cu 10, dupa 11 meciuri fiecare. Cadiz se afla in zona retrogradarii; un punct in deplasare la o candidata la promovare poate fi util. Aceasta ultima interpretare este analiza, nu o declaratie ca echipa joaca pentru 0-0. [Program Mallorca](https://www.rcdmallorca.es/noticias/entradas-a-la-venta-para-el-partido-contra-el-cadiz-cf), [clasament oficial pre-meci](https://www.cadizcf.com/partidos/temporada-2026-2027-laliga-hypermotion-12-rcd-mallorca-vs-cadiz-cf-3428).

Luis Garcia cere mai multa regularitate si agresivitate, fara ca asta sa insemne un ritm ofensiv nesustenabil. Raportul clubului de la Leganes descrie constructie prin posesie, Torre intre linii, Fuentes cu spatele la poarta, Luvumbo pe spatii si la recuperarea mingilor respinse. Prima repriza a avut putine ocazii, desi primul gol a fost la 34. Interpretarea pentru azi: Mallorca poate controla teritorial fara sa genereze foarte multe ocazii clare; posesia singura nu produce garantia golului. [Raport oficial Leganes-Mallorca](https://www.rcdmallorca.es/noticias/empate-1-1-ante-el-cd-leganes).

Configuratie tactica de lucru: Mallorca 4-2-3-1 contra Cadiz 4-4-2, asa cum apar in pagina AS, tratate numai ca probabile. Mallorca poate cauta superioritate intre liniile de patru, schimbarile de directie si duelurile extremei cu fundasul lateral. Cadiz poate inchide centrul si cauta Izeta/Garcia Pascual la tranzitie. Daca blocul oaspetilor rezista, poate exista dominare fara gol; daca gazdele pierd mingea cu lateralii sus, contraatacul ofera a doua cale spre gol. Acestea sunt scenarii, nu bonusuri procentuale. Nu copiez integral un 11 precompletat de agregator: banca include si David Lopez, pe care antrenorul il declara indisponibil. [AS, module pre-meci](https://as.com/resultados/futbol/segunda/2026_2027/directo/regular_a_12_6a44993ebedbb16/alineaciones/).

Borja Jimenez este la al TREILEA meci pe banca Cadiz: victoriile 3-1 cu Ceuta si 3-1 la Xerez. Declaratia oficiala pune accent pe competitivitate, pastrarea sanselor in meci si putinele antrenamente avute. Este o schimbare de regim: cea mai mare parte din cele 11 observatii de campionat este de dinaintea sa. Nu avem suficiente meciuri sub noul antrenor pentru un model separat; nici doua victorii nu autorizeaza o crestere automata a probabilitatii. [Avancronica oficiala Borja](https://www.cadizcf.com/noticias/borja-jimenez-debemos-estar-vivos-en-los-partidos).

Loturi Mallorca: John Chetauya si David Lopez nu sunt disponibili; Arnau Puigmal si Josep Cerda sunt pregatiti, conform conferintei din 31 octombrie. Nu preiau liste vechi de indisponibili de la alte avancronici. [Conferinta oficiala Luis Garcia](https://www.rcdmallorca.es/noticias/luis-garcia-tenemos-que-encontrar-esa-regularidad-mirarnos-a-nosotros-mismos-y-mejorar).

Cadiz: Jandro este explicit indisponibil luni (aductor); Vladys/Kopotun are leziune miofasciala, disponibilitate dependenta de evolutie; Petit era asteptat sa revina la grup saptamana urmatoare. Ontiveros a lipsit din cupa din motive familiale, nu din cauza unei accidentari comunicate. Buletinul nu este un 11 confirmat si nu presupun automat revenirea pe teren a vreunuia. Match center-ul oficial precizeaza ca formatiile vor aparea cu o ora inainte de start. [Buletin medical oficial](https://www.cadizcf.com/noticias/parte-medico-jandro-y-vladys).

## Recuperare, deplasare, vreme

Mallorca nu a jucat in cupa la Alcoy; antrenorul confirma ca echipa a calatorit, a revenit si s-a pregatit pentru luni. Ultimul meci efectiv a fost pe 24 octombrie: 9 zile calendaristice pana azi. Cadiz a jucat la Xerez JOI 29, nu miercuri 28, dupa amanarea cu 24 ore: 4 zile pana luni. Cadiz are deplasare la Palma; nu am verificat ora zborului si nu ii atribui un efect numeric. Mallorca are avantaj de odihna, dar golul timpuriu nu rezulta automat din aceasta. Surse: conferinta Luis Garcia si [raport oficial cupa Cadiz](https://www.cadizcf.com/noticias/el-cadiz-cf-remonta-en-chapin-y-avanza-en-la-copa-del-rey-1-3).

Nu am o prognoza orara verificata sau un raport actual despre gazon. Nu afirm ca vremea este perfecta si nu modific estimarea pe baza unei presupuneri meteorologice.

## Decizie explicata

Cel mai puternic argument PRO: toate cele 4 deplasari Cadiz au inclus gol primit <=70, iar toate cele 5 meciuri Mallorca acasa au inclus primul gol <=70. Principalul risc: un singur producator ofensiv credibil intr-un meci controlat de Mallorca, cu Cadiz compact si un antrenor nou. Palmaresul defensiv Mallorca (numai 3 goluri primite <=70 in 11 meciuri, unul acasa) reduce contributia probabila a oaspetilor.

Locul 21 este corect pentru formula fixata, dar nu demonstreaza ca are a 21-a probabilitate reala. Aici formula este sensibila la includerea deplasarilor slab productive ale Mallorca si a meciurilor Cadiz acasa, care nu sunt configuratia de azi. Este o limita de evaluat ulterior pe un backtest, nu o justificare sa schimbam astazi ponderile. Il pastrez ca alternativa interesanta pe teren, fara a-l promova peste finalisti si fara a-l eticheta 90%.

Date si reproducere: Europe_Combined/mallorca_cadiz_inputs.json, mallorca_cadiz_season.csv, mallorca_cadiz_live_status.json; powershell -NoProfile -File Analysis/Goal70_2026-11-02/analyze_mallorca_cadiz.ps1. Calculul a fost refacut si verificat fata de modelul din comparatia de 27 de meciuri. Nu a fost modificat topul, jurnalul selectiilor, modelul de productie sau vreun pariu.
