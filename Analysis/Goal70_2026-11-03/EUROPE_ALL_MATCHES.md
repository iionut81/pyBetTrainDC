# Gol pana la 70 inclusiv: 3 noiembrie 2026

**Selectia individuala a analistului: QPR - Burnley, 21:45 Romania, Championship, minimum un gol valid al oricarei echipe pana la minutul afisat 70 inclusiv.** Instantaneu prospectiv salvat la 2026-11-03T10:39:22+02:00. Utilizatorul nu a declarat inca o alegere pentru azi; niciun pariu, pret sau rezultat nu este presupus.

Model brut 93,63%; scenariu intensitate -20%: 88,95%; cota echitabila strict matematica 1,0681. Nu este o probabilitate calibrata sau o cota oferita. La un pret ipotetic 1,10, pragul de rentabilitate este 90,91%; scenariul de stres este sub acest prag. Nu declar un avantaj demonstrat sau profitabilitate.

Actualizare de jurnal, 10:47:30 RO: utilizatorul a confirmat QPR-Burnley pentru GOAL_BY_70. Instantaneul analizei de la 10:39:22, numaratorile si clasamentul sunt pastrate. Confirmarea nu reprezinta dovada unui pariu plasat. Vezi `../Selection_History/2026-11-03.md` si `../Selection_History/user_selections.csv`.

## Acoperire, limite si excluderi

Universul de baza este fixat dupa [clasamentul UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), nu clasamentul provizoriu 2027: primele ligi masculine din 20 asociatii, plus Romania, Serbia si 7 divizii secunde deja urmarite. Sunt 29 ligi inventariate. Am adaugat separat toate cele 9 partide Champions League si Lyngby-Randers din Cupa Danemarcei. Nu numesc acest inventar toate diviziile Europei.

| Rang UEFA 2026 | Tara / liga | Meciuri identificate | Calculate |
|---|---|---:|---:|
| 1 | England / eng.1 | 0 | 0 |
| 2 | Italy / ita.1 | 0 | 0 |
| 3 | Spain / esp.1 | 0 | 0 |
| 4 | Germany / ger.1 | 0 | 0 |
| 5 | France / fra.1 | 0 | 0 |
| 6 | Portugal / por.1 | 0 | 0 |
| 7 | Netherlands / ned.1 | 0 | 0 |
| 8 | Belgium / bel.1 | 0 | 0 |
| 9 | Turkey / tur.1 | 0 | 0 |
| 10 | Czechia / cze.1 | 0 | 0 |
| 11 | Greece / gre.1 | 0 | 0 |
| 12 | Poland / pol.1 | 0 | 0 |
| 13 | Denmark / den.1 | 0 | 0 |
| 14 | Norway / nor.1 | 0 | 0 |
| 15 | Cyprus / cyp.1 | 0 | 0 |
| 16 | Switzerland / sui.1 | 0 | 0 |
| 17 | Austria / aut.1 | 0 | 0 |
| 18 | Scotland / sco.1 | 0 | 0 |
| 19 | Sweden / swe.1 | 1 | 1 |
| 20 | Croatia / cro.1 | 0 | 0 |
| Supliment | Belgium 2 / bel.2 | 0 | 0 |
| Supliment | England 2 / eng.2 | 9 | 9 |
| Supliment | Spain 2 / esp.2 | 0 | 0 |
| Supliment | France 2 / fra.2 | 0 | 0 |
| Supliment | Germany 2 / ger.2 | 0 | 0 |
| Supliment | Italy 2 / ita.2 | 0 | 0 |
| Supliment | Romania / rou.1 | 0 | 0 |
| Supliment | Serbia / srb.1 | 0 | 0 |
| Supliment | Turkey 2 / tur.2 | 0 | 0 |

Rezultat: 1 Allsvenskan + 9 Championship in universul de baza; 9 UCL + 1 Cupa Danemarcei suplimentar = **20 partide identificate si 20 calcule**. Nicio partida de prima liga in Franta sau Romania nu a fost identificata azi in programele verificate. Inventarul programului si calculul tuturor candidatilor sunt separate de verificarile contextuale aprofundate de mai jos.

[Program FotMob](https://www.fotmob.com/api/data/matches?date=20261103), [Championship ESPN](https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2/scoreboard?dates=20261103&limit=1000), [program Allsvenskan oficial](https://allsvenskan.se/nyheter/sa-spelas-omgang-24-30-av-allsvenskan/), [program UEFA](https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-league-phase-fix/). Ultima reverificare pentru toate cele 20 de partide: 10:36:56 RO, toate programate/neincepute, ore neschimbate; `live_status.json`.

Limita inventarului: ESPN nu suporta aici pol.1, cro.1, srb.1 si bel.2 (HTTP 400). Pentru absenta partidelor in aceste patru ligi ma bazez pe inventarul global FotMob, nu pe raspunsul ESPN. Nu prezint aceasta absenta drept confirmare independenta din doua surse. Istoricul integral nu a fost colectat pentru ligile fara candidati azi.

Excluderile observate sunt in `exclusions.csv`: inclusiv 2 meciuri din Polonia liga a doua (neinclusa anterior), 2 din Muntenegru (in afara top20), 16 Challenge Cup Scotia, 2 Copa del Rey, 2 Cupa Ciprului, 2 Cupa Croatiei, League One/Two si cupe inferioare, juniori si fotbal feminin. Nu am calculat aceste meciuri si nu afirm ca sunt mai slabe sportiv; nu au istoricul comparabil colectat in acest audit.

Date: 1.520 partide istorice validate, 4 respinse prin neconcordanta dintre scor si lista golurilor; niciuna dintre cele patru nu implica echipele candidate. Sezoanele sunt 2026 pentru Suedia/Norvegia si 2026/27 pentru restul. Au fost verificate independent in ESPN/FotMob toate cele 78 de observatii pe echipa QPR/Burnley/Sirius/Halmstad; acestea reprezinta 77 partide distincte, deoarece Halmstad-Sirius apare in ambele istorice. Scorurile si numaratorile GF70/GA70 concorda. Suplimentar sunt salvate ultimele 10 oficiale Derby/Norwich.

## Clasamentul complet al modelului brut

Acesta este un clasament al valorilor aceleiasi formule, nu al unor probabilitati comparabile validate. UCL si cupa folosesc proiectii din campionatele interne. Aceasta schimbare de populatie este motivul pentru care nu aleg automat cea mai mare valoare UCL. Cele 20 randuri nu sunt 20 recomandari.

| Rang brut | Meci | Competitie | Ora RO | Model | Stres -20% | Cota echitabila | Prim gol <=70, istoric distinct | Acasa/deplasare |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | Feyenoord Rotterdam - Internazionale | uefa.champions | 22:00 | 94,28% | 89,86% | 1,0607 | 20/21 | 9/10 |
| 2 | Queens Park Rangers - Burnley | eng.2 | 21:45 | 93,63% | 88,95% | 1,0681 | 23/26 | 11/12 |
| 3 | Barcelona - Aston Villa | uefa.champions | 22:00 | 93,48% | 88,74% | 1,0697 | 19/20 | 9/10 |
| 4 | Galatasaray - VfB Stuttgart | uefa.champions | 19:45 | 92,19% | 87,00% | 1,0847 | 15/18 | 8/10 |
| 5 | Birmingham City - Millwall | eng.2 | 22:00 | 91,69% | 86,33% | 1,0907 | 23/26 | 11/12 |
| 6 | Derby County - Norwich City | eng.2 | 21:45 | 91,63% | 86,26% | 1,0913 | 25/26 | 12/12 |
| 7 | LASK Linz - Slovan Bratislava | uefa.champions | 22:00 | 91,50% | 86,08% | 1,0929 | 23/24 | 13/14 |
| 8 | Portsmouth - Swansea City | eng.2 | 21:45 | 90,85% | 85,23% | 1,1008 | 21/26 | 11/12 |
| 9 | IK Sirius - Halmstads BK | swe.1 | 20:00 | 90,52% | 84,81% | 1,1048 | 46/51 | 25/26 |
| 10 | Blackburn Rovers - Stoke City | eng.2 | 21:45 | 89,79% | 83,89% | 1,1137 | 24/26 | 12/12 |
| 11 | Manchester United - AS Roma | uefa.champions | 22:00 | 89,13% | 83,06% | 1,1219 | 17/19 | 7/9 |
| 12 | Lyngby - Randers FC | den.cup | 19:00 | 88,13% | 81,83% | 1,1346 | 22/24 | 10/12 |
| 13 | Atlético Madrid - Bayern Munich | uefa.champions | 22:00 | 87,76% | 81,37% | 1,1395 | 16/19 | 8/9 |
| 14 | West Bromwich Albion - Southampton | eng.2 | 22:00 | 87,55% | 81,11% | 1,1422 | 23/26 | 12/12 |
| 15 | Watford - Lincoln City | eng.2 | 21:45 | 86,38% | 79,71% | 1,1577 | 24/26 | 11/12 |
| 16 | Villarreal - Paris Saint-Germain | uefa.champions | 22:00 | 85,77% | 78,99% | 1,1659 | 18/20 | 10/11 |
| 17 | Charlton Athletic - Bolton Wanderers | eng.2 | 21:45 | 85,52% | 78,69% | 1,1693 | 21/26 | 8/12 |
| 18 | Shakhtar Donetsk - Sporting CP | uefa.champions | 19:45 | 83,91% | 76,81% | 1,1918 | 19/21 | 10/12 |
| 19 | Bodo/Glimt - Lille | uefa.champions | 22:00 | 83,88% | 76,77% | 1,1922 | 32/34 | 16/17 |
| 20 | Preston North End - Sheffield United | eng.2 | 21:45 | 82,34% | 75,02% | 1,2144 | 20/26 | 8/12 |

## Alegerea: QPR - Burnley

Este primul model dintre meciurile cu istoric din aceeasi competitie, dupa compararea tuturor celor 20. Are contributii ofensive de ambele parti si rezista numeric excluderii rezultatelor extreme/eliminarilor. Acesta este motivul alegerii; nu pretind ca tactica elimina riscul schimbarii antrenorului Burnley.

Championship 2026/27, etapa 14, sezon regulat la Loftus Road; nu exista retur eliminatoriu, gol in deplasare ori prelungiri. QPR este pe locul 9 (20p), Burnley pe 22 (9p). Gazdele au interes pentru apropierea de playoff, oaspetii pentru iesirea din zona retrogradarii. [Clasament/program EFL](https://www.efl.com/competitions/efl-championship/), [context local](https://www.playmakerstats.com/news/preview-qpr-vs-burnley-team-news-lineups-predictions-efl-championship-03-11/1249940). Ordinea de baza a clasamentului foloseste puncte, golaveraj, apoi goluri marcate; nu este o situatie de calificare decisa azi. [Regulament EFL, sectiunea 9](https://images.gc.eflservices.co.uk/6a32d710-5b0b-11f0-ad0f-97502a343283.pdf) este editia 2025/26; intreaga succesiune a departajarii 2026/27 nu a fost verificata, deci nu o presupun neschimbata.

Un egal nu rezolva obiectivele niciuneia, dar poate fi acceptabil pentru Burnley in deplasare: nu transform nevoia de puncte intr-o promisiune de atac din primul minut. Ambele au jucat pe 31 octombrie, deci aproximativ trei zile intre meciuri. QPR revine acasa; Burnley calatoreste in Anglia, fara diferenta de fus orar.

Forma si antrenori: Julien Stephan conduce QPR, care vine dupa 5-0 acasa cu Swansea si 1-4 la Southampton. Kone a marcat de patru ori in aceste doua meciuri. Celestini a preluat Burnley pe 28 octombrie; debutul din 31 octombrie a fost 0-0 cu Portsmouth. [Preview oficial Burnley](https://www.burnleyfootballclub.com/media-article/match-preview-qpr-v-burnley). Contradictie relevanta: afirmatia din acest preview privind numai doua infrangeri QPR in ultimele noua nu concorda cu istoricul salvat, care include Birmingham, Southampton si Middlesbrough; nu o folosesc in evaluare.

Tactic, ultimele formule verificate au fost QPR 4-4-2 si Burnley 4-2-3-1. Sunt formule din ultimul meci, nu echipe confirmate azi. Smyth-Kone, sprijiniti de Poku/Vale, ofera o cale prin joc lateral si prezenta in careu; cu doi atacanti, QPR poate ocupa fundasii centrali. Burnley are variante de tranzitie prin Ramazani/Edwards si Vardy. Acestea sunt scenarii tactice, nu bonusuri numerice. Un bloc mediu/jos Burnley, protejat de doi mijlocasi, ar putea reduce ritmul si accesul central. Nu exista suficiente meciuri cu Celestini la Burnley pentru a atribui echipei un stil nou deja stabilizat. Ultimele formule sunt verificate in istoricul FotMob; proiectiile de titulari sunt [neconfirmate](https://www.sportsmole.co.uk/football/qpr/preview/qpr-vs-burnley-prediction-team-news-lineups_608199.html). Nu preiau predictia acelui site.

Loturi: Burrell, Lamptey si Madsen sunt raportati indisponibili la QPR; Amdouni, accidentat la glezna, este raportat absent la Burnley. [Burnley Express, 2 noiembrie](https://www.burnleyexpress.net/sport/football/burnley-fc/qpr-v-burnley-early-team-news-9323189). Madsen/Burrell reduc optiunile creative/ofensive. Celestini confirma revenirea lui Cullen, Banel si Ramsey, dar si restrictii de minute; Broja nu este inca aproape de revenire, Satpayev se recupereaza. Antrenorul spune ca programul permite mai ales recuperare, putin antrenament. [Conferinta oficiala](https://www.burnleyfootballclub.com/media-article/celestinis-press-points-qpr). Nicio formula de start nu este confirmata la ora analizei. Lista secundara care afirma ca QPR nu are indisponibili este in conflict cu relatarea locala; nu o consider dovada de lot complet.

Conditii: [Met Office, actualizare 3 noiembrie 04:00 UTC](https://weather.metoffice.gov.uk/forecast/gcpvj0u07) indica nori, averse izolate seara si tendinta de uscare, vant mai accentuat spre coasta. Nu apare de aici un motiv verificat pentru a invalida meciul; este prognoza regionala, nu inspectie a gazonului. Starea exacta a suprafetei si planul efectiv de calatorie nu sunt confirmate. Nu am folosit xG: nu am un set contemporan cu aceeasi definitie si acoperire pentru toate cele 20 partide.

### Numaratori si formula neschimbata

Golurile de mai jos sunt numai cele marcate/primite pana la minutul afisat 70 inclusiv, nu goluri la pauza sau goluri totale la 90. Se includ 45+, se exclud golurile anulate, prelungirile eliminatorii si departajarea. Nu inmultesc din nou cu 70/90.

#### Queens Park Rangers - Burnley

| Esantion | N | GF70 | GA70 | Gol <=70 | Deschidere dupa 70 | 0-0 |
|---|---:|---:|---:|---:|---:|---:|
| Queens Park Rangers sezon | 13 | 18 | 12 | 11 | 0 | 2 |
| Burnley sezon | 13 | 14 | 19 | 12 | 0 | 1 |
| Queens Park Rangers acasa | 6 | 11 | 3 | 5 | 0 | 1 |
| Burnley deplasare | 6 | 8 | 15 | 6 | 0 | 0 |

GF70/meci pe teren: gazda 1,83; oaspete 1,33.

lambda_sezon = ((18/13 + 19/13) + (14/13 + 12/13)) / 2 = 2,423077.

lambda_teren = ((11/6 + 15/6) + (8/6 + 3/6)) / 2 = 3,083333.

lambda_combinat = 0,5 x lambda_sezon + 0,5 x lambda_teren = 2,753205; p = 1-exp(-lambda_combinat) = 93,63%. Numai sezon: 91,14%; numai teren: 95,42%; intensitate -20%: 88,95%; cota echitabila = 1/p = 1,0681.

#### Birmingham City - Millwall

| Esantion | N | GF70 | GA70 | Gol <=70 | Deschidere dupa 70 | 0-0 |
|---|---:|---:|---:|---:|---:|---:|
| Birmingham City sezon | 13 | 19 | 11 | 12 | 0 | 1 |
| Millwall sezon | 13 | 16 | 14 | 11 | 2 | 0 |
| Birmingham City acasa | 6 | 11 | 5 | 6 | 0 | 0 |
| Millwall deplasare | 6 | 7 | 9 | 5 | 1 | 0 |

GF70/meci pe teren: gazda 1,83; oaspete 1,17.

lambda_sezon = ((19/13 + 14/13) + (16/13 + 11/13)) / 2 = 2,307692.

lambda_teren = ((11/6 + 9/6) + (7/6 + 5/6)) / 2 = 2,666667.

lambda_combinat = 0,5 x lambda_sezon + 0,5 x lambda_teren = 2,487179; p = 1-exp(-lambda_combinat) = 91,69%. Numai sezon: 90,05%; numai teren: 93,05%; intensitate -20%: 86,33%; cota echitabila = 1/p = 1,0907.

#### Derby County - Norwich City

| Esantion | N | GF70 | GA70 | Gol <=70 | Deschidere dupa 70 | 0-0 |
|---|---:|---:|---:|---:|---:|---:|
| Derby County sezon | 13 | 8 | 18 | 13 | 0 | 0 |
| Norwich City sezon | 13 | 19 | 19 | 12 | 1 | 0 |
| Derby County acasa | 6 | 4 | 8 | 6 | 0 | 0 |
| Norwich City deplasare | 6 | 8 | 10 | 6 | 0 | 0 |

GF70/meci pe teren: gazda 0,67; oaspete 1,33.

lambda_sezon = ((8/13 + 19/13) + (19/13 + 18/13)) / 2 = 2,461538.

lambda_teren = ((4/6 + 10/6) + (8/6 + 8/6)) / 2 = 2,500000.

lambda_combinat = 0,5 x lambda_sezon + 0,5 x lambda_teren = 2,480769; p = 1-exp(-lambda_combinat) = 91,63%. Numai sezon: 91,47%; numai teren: 91,79%; intensitate -20%: 86,26%; cota echitabila = 1/p = 1,0913.

#### IK Sirius - Halmstads BK

| Esantion | N | GF70 | GA70 | Gol <=70 | Deschidere dupa 70 | 0-0 |
|---|---:|---:|---:|---:|---:|---:|
| IK Sirius sezon | 26 | 42 | 25 | 24 | 2 | 0 |
| Halmstads BK sezon | 26 | 11 | 39 | 23 | 3 | 0 |
| IK Sirius acasa | 13 | 22 | 15 | 13 | 0 | 0 |
| Halmstads BK deplasare | 13 | 5 | 22 | 12 | 1 | 0 |

GF70/meci pe teren: gazda 1,69; oaspete 0,38.

lambda_sezon = ((42/26 + 39/26) + (11/26 + 25/26)) / 2 = 2,250000.

lambda_teren = ((22/13 + 22/13) + (5/13 + 15/13)) / 2 = 2,461538.

lambda_combinat = 0,5 x lambda_sezon + 0,5 x lambda_teren = 2,355769; p = 1-exp(-lambda_combinat) = 90,52%. Numai sezon: 89,46%; numai teren: 91,47%; intensitate -20%: 84,81%; cota echitabila = 1/p = 1,1048.

Ponderarea 50/50 este euristica; esantioanele sezon/teren se suprapun si nu sunt dovezi independente. Modelul Poisson nu este calibrat pentru timpul primului gol. Golurile suplimentare dupa deschiderea scorului, inclusiv in meciuri devenite foarte deschise, pot umfla intensitatea si supraestima sansa de a evita 0-0 pana la 70.

La QPR-Burnley, modelul 93,63% depaseste frecventa istorica distincta 23/26 = 88,46%; pe teren frecventa este 11/12 = 91,67%. Ultimele 10 oficiale sunt 9/10 pentru fiecare, 18/20 distincte. Cele 3 esecuri de sezon sunt QPR-Bolton 0-0 (22 august), Charlton-QPR 0-0 (9 septembrie), Burnley-Portsmouth 0-0 (31 octombrie); nicio deschidere tarzie in acest esantion. Nu inlocuiesc aceste esecuri cu succes si nu prezint valoarea modelului drept rata istorica.

### Sensibilitate la eliminari si rezultate extreme

| Partida | Diagnostic | p | N sezon A/B | N teren A/B |
|---|---|---:|---:|---:|
| QPR-Burnley | original | 93,63% | 13/13 | 6/6 |
| QPR-Burnley | exclude_red_card_matches | 92,54% | 12/12 | 5/5 |
| QPR-Burnley | exclude_qpr_5_0_swansea | 92,83% | 12/13 | 5/6 |
| Sirius-Halmstad | original | 90,52% | 26/26 | 13/13 |
| Sirius-Halmstad | exclude_red_card_matches | 90,29% | 24/24 | 12/13 |

La QPR, eliminarea din QPR-Preston a venit la 15, dupa primul gol din minutul 1; in Norwich-Burnley eliminarea a venit la 77, dupa golul din minutul 2. Aceste eliminari nu au cauzat cronologic primul gol. Excluderea integrala este doar un test conservator de sensibilitate, nu o corectie cauzala; eliminarea de la 77 nici nu poate schimba golurile deja inscrise pana la 70. Rezultatul fara QPR-Swansea 5-0 este 92,83%, deci acel rezultat ridica valoarea, dar nu explica singur semnalul. Nu rescriu predictia initiala cu cea mai convenabila varianta.

## Alternative si contraargumente

**Feyenoord-Inter, 22:00:** 94,28% brut / 89,86% stres, maximul numeric. Dar intrarile sunt Eredivisie/Serie A, adversari neuniformi, numai 5+5 meciuri pe teren; in UCL au impreuna doar 6 observatii, cu 5 succese. Nu il aleg peste QPR pe baza acestui transfer nevalidat. Van Bronckhorst confirma revenirea lui Shaqueel van Persie si Reiss Nelson, St Juste ramane absent. [Feyenoord, 2 noiembrie](https://www.feyenoord.com/nl/news/van-persie-en-nelson-bij-wedstrijdselectie-tegen-inter-021126). Chivu confirma Provedel recuperat si Pavard revenit la grup. [Inter, conferinta](https://www.inter.it/it/notizie/chivu-conferenza-stampa-vigilia-feyenoord-inter-champions-league-2026-27). Listele secundare care ii dau pe cei reveniti drept absenti sunt depasite. Barcelona-Villa si Galatasaray-Stuttgart au aceeasi limita a transferului intre ligi; nu pretind ca le-am verificat contextual la aceeasi adancime.

**Birmingham-Millwall, 22:00:** al doilea model din aceeasi competitie, 91,69% / 86,33%; 23/26 istoric si 11/12 pe teren. Profil ofensiv bilateral: 11/6 si 7/6 GF70 pe teren. Ingrijorarile privind Beadle/Priske si suspendarea Metcalfe apar in [preview-ul secundar](https://www.sportsmole.co.uk/football/birmingham-city/preview/birmingham-vs-millwall-prediction-team-news-lineups_608183.html); listele de absenti publicate de agregatoare nu concorda. Nu tratez ca verificate toate numele si nici nu pretind verificare contextuala completa pentru acest candidat. QPR are scorul brut si stresul superior; istoricul succeselor nu le departajeaza.

**Derby-Norwich, 21:45:** 91,63% / 86,26%; istoric 25/26, pe teren 12/12, mai bun descriptiv decat QPR. Totusi Derby are numai 4 GF70 in 6 acasa (0,67/meci), fata de 1,83 la QPR; nu aleg mecanic 12/12 dintr-un esantion mic peste model. Clement descrie Derby ca direct, puternic in dueluri, cere control cu mingea, recunoaste vulnerabilitatea propriei aparari si posibila rotatie dupa Watford. [Conferinta Norwich, 2 noiembrie](https://www.canaries.co.uk/news/press-conference-clement-im-looking-forward-game-its-going-be-hard-battle). Ozoh/Ward/Meister sunt incerti, Clarke/Agyemang au reluat antrenamentul; sursele nu concorda asupra absentei certe Ozoh, deci o las incertitudine. [Relatarea actualizata](https://staging-aws.sports.yahoo.com/articles/derby-county-duo-closing-returns-170400754.html). Formatiile anterioare, nu confirmate azi, sunt in ultimele 10 de mai jos.

**Sirius-Halmstad, 20:00:** 90,52% / 84,81%; 46/51 istoric distinct si 25/26 pe teren, esantion mai mare. Sirius poate reveni pe primul loc, iar Halmstad este ultima, cu 17 puncte dupa 26 jocuri. [Clasament contemporan](https://en.wikipedia.org/wiki/2026_Allsvenskan), [preview Sirius](https://www.siriusfotboll.se/nyheter/infor-ik-sirius-halmstads-bk-5/). Antrenorii sunt Andreas Engelmark/Stuart Baxter; Soumah este suspendat, Halmstad nu are suspendati. Ultimele dispozitive publicate sunt 4-2-3-1/4-4-2, nu formule confirmate. Ultimul duel a avut primul gol la 14, dar cel din august 2025 abia la 76; H2H nu este garantie. [Ghid oficial HBK](https://www.hbk.se/nyheter/2026/november/matchguide-ik-sirius-hbk). Liimatta/Uneken au situatii medicale insuficient clarificate; nu afirm ca sunt recuperati. [Relatare, 24 octombrie](https://bollsvenskan.se/engelmark-oppnar-for-rutingenomlysning-i-sirius-uneken-missar-aik-och-hammarby-siktar-pa-halmstad-3-november/). Sirius are 5 zile de la ultimul joc, Halmstad 6; teren artificial, [confirmare AIK](https://www.aikfotboll.se/artiklar-och-nyheter/261024-bortareseinformation-ik-sirius-fk-aik-herr). [Prognoza publicata pe 2 noiembrie](https://www.weather2travel.com/sweden/uppsala/forecast/) nu indica ploaie semnificativa. Principalul motiv pentru a ramane alternativa este atacul Halmstad: 5 GF70/13 deplasari = 0,38, cu dependenta crescuta de Sirius. Cota echitabila 1,1048 nu este dovada ca piata ofera 1,10 sau ca exista avantaj la acel pret.

Alte limite: Lyngby-Randers este proiectie de campionat intr-o cupa si revine dupa duelul direct din 1 noiembrie, deci rotatia poate fi substantiala; nu am validat contextual formatul exact al turului, nu il recomand. Shakhtar-Sporting se joaca la Stamford Bridge, Londra, nu in mediul obisnuit al gazdelor: statisticile domestice acasa se transfera si mai slab. [Confirmare Shakhtar, 2 noiembrie](https://shakhtar.com/en/news/2026/november/2_news/2_fc-shakhtar-training-session/).

La Sirius-Halmstad exista si un contraargument recent: ultimele 10 oficiale dau 9/10 pentru Sirius, dar numai 7/10 pentru Halmstad. Cele trei esecuri recente Halmstad sunt deschideri la 76, 90+1 si 90+3, nu partide 0-0. Sunt incluse integral mai jos; frecventa pe tot sezonul ascunde aceasta concentrare recenta. Nu adaug o penalizare procentuala improvizata, dar este un motiv concret pentru a nu promova aceasta alternativa in fata QPR.

## Ultimele 10 partide oficiale verificate ale finalistelor

Am folosit inventarul tuturor competitiilor din API-ul echipelor FotMob, am exclus amicalele, apoi am verificat evenimentele si scorul fiecarei partide. Sezonul de model ramane numai campionatul intern. Competitia Cup din sursa este cupa nationala suedeza. Toate partidele de mai jos sunt anterioare zilei analizei. Un 0-0 are minut NULL, nu minut 0.

### qpr

| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |
|---|---|---|---:|---|
| 2026-10-31 | Championship | [Southampton - QPR 4 - 1](https://www.fotmob.com/matches/southampton-vs-queens-park-rangers/2vf3vf#5836905) | 11' | DA |
| 2026-10-24 | Championship | [QPR - Swansea 5 - 0](https://www.fotmob.com/matches/swansea-city-vs-queens-park-rangers/3d6h1w#5836893) | 2' | DA |
| 2026-10-17 | Championship | [Birmingham - QPR 3 - 0](https://www.fotmob.com/matches/birmingham-city-vs-queens-park-rangers/2xk7i3#5836874) | 27' | DA |
| 2026-10-14 | Championship | [QPR - Derby 4 - 1](https://www.fotmob.com/matches/derby-county-vs-queens-park-rangers/3f6zl5#5836873) | 18' | DA |
| 2026-10-09 | Championship | [West Ham - QPR 1 - 1](https://www.fotmob.com/matches/west-ham-united-vs-queens-park-rangers/2xildx#5836861) | 18' | DA |
| 2026-09-19 | Championship | [QPR - Preston 2 - 2](https://www.fotmob.com/matches/preston-north-end-vs-queens-park-rangers/2ut60f#5836846) | 1' | DA |
| 2026-09-12 | Championship | [West Brom - QPR 1 - 1](https://www.fotmob.com/matches/west-bromwich-albion-vs-queens-park-rangers/2xkm17#5836836) | 38' | DA |
| 2026-09-09 | Championship | [Charlton - QPR 0 - 0](https://www.fotmob.com/matches/charlton-athletic-vs-queens-park-rangers/2v9483#5836822) | fara gol | NU |
| 2026-09-05 | Championship | [QPR - Middlesbrough 0 - 1](https://www.fotmob.com/matches/middlesbrough-vs-queens-park-rangers/2wcc9i#5836808) | 20' | DA |
| 2026-09-02 | Championship | [QPR - Cardiff 2 - 1](https://www.fotmob.com/matches/cardiff-city-vs-queens-park-rangers/2u2iyy#5836800) | 36' | DA |

Bilant: 9/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/qpr_last10.json.

### burnley

| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |
|---|---|---|---:|---|
| 2026-10-31 | Championship | [Burnley - Portsmouth 0 - 0](https://www.fotmob.com/matches/burnley-vs-portsmouth/2akcmh#5836899) | fara gol | NU |
| 2026-10-24 | Championship | [Lincoln - Burnley 4 - 1](https://www.fotmob.com/matches/burnley-vs-lincoln-city/2a8xsp#5836890) | 39' | DA |
| 2026-10-16 | Championship | [Burnley - Wolves 1 - 2](https://www.fotmob.com/matches/burnley-vs-wolverhampton-wanderers/2byja7#5836876) | 20' | DA |
| 2026-10-13 | Championship | [Burnley - Charlton 2 - 2](https://www.fotmob.com/matches/burnley-vs-charlton-athletic/2agfba#5836864) | 36' | DA |
| 2026-10-10 | Championship | [Watford - Burnley 0 - 4](https://www.fotmob.com/matches/burnley-vs-watford/2ojpnp#5836859) | 13' | DA |
| 2026-09-19 | Championship | [Burnley - Derby 1 - 1](https://www.fotmob.com/matches/burnley-vs-derby-county/2sd9kk#5836840) | 34' | DA |
| 2026-09-12 | Championship | [Swansea - Burnley 3 - 1](https://www.fotmob.com/matches/burnley-vs-swansea-city/2qjuay#5836834) | 14' | DA |
| 2026-09-08 | Championship | [Wrexham - Burnley 1 - 1](https://www.fotmob.com/matches/burnley-vs-wrexham/2osze1#5836825) | 26' | DA |
| 2026-09-05 | Championship | [Burnley - Bristol City 1 - 2](https://www.fotmob.com/matches/burnley-vs-bristol-city/2a7vbm#5836803) | 56' | DA |
| 2026-09-02 | Championship | [Burnley - Middlesbrough 1 - 1](https://www.fotmob.com/matches/burnley-vs-middlesbrough/2bfhjz#5836798) | 23' | DA |

Bilant: 9/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/burnley_last10.json.

### derby

| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |
|---|---|---|---:|---|
| 2026-10-31 | Championship | [Millwall - Derby 4 - 1](https://www.fotmob.com/matches/millwall-vs-derby-county/3d61hf#5836902) | 3' | DA |
| 2026-10-24 | Championship | [Sheff Utd - Derby 1 - 1](https://www.fotmob.com/matches/sheffield-united-vs-derby-county/2xizwz#5836894) | 37' | DA |
| 2026-10-17 | Championship | [Derby - Stoke 0 - 1](https://www.fotmob.com/matches/derby-county-vs-stoke-city/3fgl40#5836877) | 49' | DA |
| 2026-10-14 | Championship | [QPR - Derby 4 - 1](https://www.fotmob.com/matches/derby-county-vs-queens-park-rangers/3f6zl5#5836873) | 18' | DA |
| 2026-10-10 | Championship | [Derby - Wrexham 1 - 1](https://www.fotmob.com/matches/wrexham-vs-derby-county/3b7u3n#5836853) | 9' | DA |
| 2026-09-19 | Championship | [Burnley - Derby 1 - 1](https://www.fotmob.com/matches/burnley-vs-derby-county/2sd9kk#5836840) | 34' | DA |
| 2026-09-12 | Championship | [Derby - Birmingham 1 - 2](https://www.fotmob.com/matches/birmingham-city-vs-derby-county/2xjeg0#5836829) | 40' | DA |
| 2026-09-09 | Championship | [Derby - West Brom 0 - 1](https://www.fotmob.com/matches/west-bromwich-albion-vs-derby-county/2xjsz2#5836823) | 16' | DA |
| 2026-09-05 | Championship | [West Ham - Derby 3 - 0](https://www.fotmob.com/matches/west-ham-united-vs-derby-county/2xhsc2#5836813) | 6' | DA |
| 2026-09-01 | Championship | [Portsmouth - Derby 0 - 2](https://www.fotmob.com/matches/portsmouth-vs-derby-county/2vcple#5836792) | 16' | DA |

Bilant: 10/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/derby_last10.json.

### norwich

| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |
|---|---|---|---:|---|
| 2026-10-31 | Championship | [Norwich - Watford 3 - 2](https://www.fotmob.com/matches/watford-vs-norwich-city/375k2g#5836903) | 10' | DA |
| 2026-10-24 | Championship | [Charlton - Norwich 1 - 3](https://www.fotmob.com/matches/charlton-athletic-vs-norwich-city/2rpq65#5836889) | 10' | DA |
| 2026-10-17 | Championship | [Norwich - Portsmouth 0 - 1](https://www.fotmob.com/matches/portsmouth-vs-norwich-city/2ru1jy#5836880) | 51' | DA |
| 2026-10-14 | Championship | [Norwich - Southampton 0 - 1](https://www.fotmob.com/matches/southampton-vs-norwich-city/2rvm2w#5836871) | 54' | DA |
| 2026-10-10 | Championship | [Swansea - Norwich 2 - 1](https://www.fotmob.com/matches/norwich-city-vs-swansea-city/39cc2d#5836858) | 28' | DA |
| 2026-09-20 | Championship | [Norwich - Bolton 3 - 4](https://www.fotmob.com/matches/bolton-wanderers-vs-norwich-city/2sw7sn#5836844) | 5' | DA |
| 2026-09-17 | EFL Cup | [Man City - Norwich 5 - 0](https://www.fotmob.com/matches/manchester-city-vs-norwich-city/2rrosd#6099330) | 29' | DA |
| 2026-09-12 | Championship | [Middlesbrough - Norwich 4 - 3](https://www.fotmob.com/matches/middlesbrough-vs-norwich-city/2ss9sa#5836830) | 1' | DA |
| 2026-09-09 | Championship | [Norwich - Birmingham 2 - 1](https://www.fotmob.com/matches/birmingham-city-vs-norwich-city/2tzduw#5836824) | 28' | DA |
| 2026-09-05 | Championship | [Sheff Utd - Norwich 1 - 3](https://www.fotmob.com/matches/sheffield-united-vs-norwich-city/2tyzks#5836809) | 8' | DA |

Bilant: 10/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/norwich_last10.json.

### sirius

| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |
|---|---|---|---:|---|
| 2026-10-29 | Allsvenskan | [Hammarby - Sirius 2 - 0](https://www.fotmob.com/matches/sirius-vs-hammarby/1ugzop#5107638) | 21' | DA |
| 2026-10-24 | Allsvenskan | [Sirius - AIK 1 - 2](https://www.fotmob.com/matches/sirius-vs-aik/1vdg77#5107634) | 30' | DA |
| 2026-10-18 | Allsvenskan | [Sirius - Elfsborg 2 - 2](https://www.fotmob.com/matches/sirius-vs-elfsborg/1semog#5107626) | 20' | DA |
| 2026-10-10 | Allsvenskan | [GAIS - Sirius 0 - 1](https://www.fotmob.com/matches/sirius-vs-gais/1uwplt#5107616) | 78' | NU |
| 2026-09-19 | Allsvenskan | [Örgryte - Sirius 1 - 2](https://www.fotmob.com/matches/sirius-vs-orgryte/2azoiq#5107606) | 7' | DA |
| 2026-09-14 | Allsvenskan | [Sirius - Degerfors 2 - 0](https://www.fotmob.com/matches/degerfors-vs-sirius/1g6c37#5107604) | 2' | DA |
| 2026-09-05 | Allsvenskan | [Västerås SK - Sirius 0 - 1](https://www.fotmob.com/matches/vasteras-sk-vs-sirius/1dgc4y#5107595) | 51' | DA |
| 2026-08-31 | Allsvenskan | [Sirius - Malmö FF 0 - 1](https://www.fotmob.com/matches/sirius-vs-malmo-ff/2dcdcs#5107587) | 44' | DA |
| 2026-08-26 | Cup | [Friska Viljor - Sirius 0 - 2](https://www.fotmob.com/matches/friska-viljor-vs-sirius/man8h#5820452) | 54' | DA |
| 2026-08-21 | Allsvenskan | [Sirius - Häcken 0 - 3](https://www.fotmob.com/matches/sirius-vs-hacken/1w2zo7#5107579) | 28' | DA |

Bilant: 9/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/sirius_last10.json.

### halmstad

| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |
|---|---|---|---:|---|
| 2026-10-28 | Allsvenskan | [Halmstads BK - Malmö FF 0 - 3](https://www.fotmob.com/matches/halmstads-bk-vs-malmo-ff/2ueu84#5107644) | 4' | DA |
| 2026-10-23 | Allsvenskan | [Västerås SK - Halmstads BK 1 - 2](https://www.fotmob.com/matches/vasteras-sk-vs-halmstads-bk/1qmrqi#5107635) | 45' | DA |
| 2026-10-18 | Allsvenskan | [Halmstads BK - Hammarby 0 - 1](https://www.fotmob.com/matches/hammarby-vs-halmstads-bk/29mj8v#5107628) | 90+1' | NU |
| 2026-10-12 | Allsvenskan | [Elfsborg - Halmstads BK 4 - 0](https://www.fotmob.com/matches/elfsborg-vs-halmstads-bk/27c2n4#5107620) | 3' | DA |
| 2026-09-20 | Allsvenskan | [Halmstads BK - AIK 2 - 1](https://www.fotmob.com/matches/halmstads-bk-vs-aik/2amhnf#5107612) | 48' | DA |
| 2026-09-12 | Allsvenskan | [IFK Göteborg - Halmstads BK 2 - 1](https://www.fotmob.com/matches/halmstads-bk-vs-ifk-goteborg/2qndzz#5107600) | 37' | DA |
| 2026-09-05 | Allsvenskan | [Degerfors - Halmstads BK 2 - 0](https://www.fotmob.com/matches/degerfors-vs-halmstads-bk/1tow3v#5107594) | 76' | NU |
| 2026-08-29 | Allsvenskan | [Halmstads BK - Kalmar FF 1 - 0](https://www.fotmob.com/matches/halmstads-bk-vs-kalmar-ff/2qmzyb#5107588) | 90+3' | NU |
| 2026-08-26 | Cup | [Skövde AIK - Halmstads BK 1 - 2](https://www.fotmob.com/matches/skovde-aik-vs-halmstads-bk/1qa1iu#5820449) | 23' | DA |
| 2026-08-22 | Allsvenskan | [Örgryte - Halmstads BK 1 - 1](https://www.fotmob.com/matches/halmstads-bk-vs-orgryte/2ru0d6#5107576) | 24' | DA |

Bilant: 7/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/halmstad_last10.json.

## Verdict, risc si evidenta prospectiva

Aleg o singura selectie: **QPR-Burnley, GOAL_BY_70**. Argumentul decisiv este maximul modelului intre partidele cu istoric din aceeasi competitie, sustinut de productia ofensiva pe teren a ambelor echipe si de sensibilitatile fara eliminari/fara 5-0. Principalul risc este ca Burnley sub Celestini sa fie structural mai prudenta decat esantionul vechi, concomitent cu absentele creative QPR. 0-0 cu Portsmouth arata deja ca acest scenariu exista. Nu scad sau adaug procente arbitrare pentru el.

Modelul brut nu demonstreaza ca QPR-Burnley are o probabilitate reala mai mare decat Derby-Norwich; frecventele favorizeaza Derby si sunt o contradictie de raportat. Selectia urmeaza regula de clasare declarata, nu un scor contextual inventat. Daca s-ar cere o probabilitate validata de 90% sau o oferta de peste 1,10 confirmata, datele de aici nu o demonstreaza. Este o predictie prospectiva pentru validare, fara recomandare de miza.

Dovezi: `all_models.csv`, `comparison_verified.csv`, `universe_coverage.csv`, `exclusions.csv`, `Data/historical_validated.csv`, `Data/historical_rejected.csv`, `independent_provider_checks.csv`, `sensitivity.csv`, `live_status.json`, `verification.json`. Scripturi locale: `rank_today.ps1`, `verify_today.ps1`, `build_report.ps1`; colectoarele acceseaza doar endpointuri publice si salveaza campurile necesare, fara cote. Modelele de productie, calibrarile si selection_engine nu sunt modificate.
