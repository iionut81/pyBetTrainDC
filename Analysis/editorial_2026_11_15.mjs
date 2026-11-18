import fs from 'node:fs';
import {read,write,csvread,stamp} from './daily_2026_11_15.mjs';
const gp='Analysis/Goal70_2026-11-15',cp='Analysis/Corners1H_2026-11-15',rp='Analysis/Review_2026-11-14',at=stamp();
for(let p of [gp+'/CONTEXT.md',gp+'/COVERAGE.md',gp+'/EUROPE_ALL_MATCHES.md',cp+'/ANALYSIS.md',rp+'/REVIEW.md'])write(gp+'/Revisions/editorial_before_'+(p.startsWith(cp)?'corner_':p.startsWith(rp)?'review_':'')+p.split('/').at(-1),fs.readFileSync(p,'utf8'));
let lessons=`## Lecții aplicate din review

Predarea review-ului și lecțiilor: **${read(rp+'/handoff.json').at_ro}**. Începutul scanării: **${read(gp+'/research_start.json').at_ro}**. Addendumul despre Oviedo este etichetat ulterior predării, fără schimbarea acestor momente.

| Lecție | Verificare și dovadă nouă | Stare / efect |
|---|---|---|
| Model, AR20 și frecvență | Granada Gol70: 90,35/84,60%, 22/25 general și 11/11 teren; cornere: 95,50/89,86%, 25/25 și 11/11 | Aplicată: preferință individuală pentru cornere, fără bonus numeric |
| GF70 propriu și primele goluri tardive/0–0 | Granada 7/5 și Las Palmas 8/6 pe teren; Oviedo–Granada și Las Palmas–Leganés 0–0, Las Palmas–Eldense primul 90+5 | Aplicată: o intensitate mare nu validează timpul primului gol |
| Dispersie, exact 7 și ≥8 | Granada/Las Palmas: niciun 7 sau ≥8 în 25 de meciuri unice; varianțe pe teren 1,30 și 1,77; Tenerife/Ceuta au două totaluri de 9 | Aplicată: coada rămâne explicită; distribuția Poisson nu este validată de aceste eșantioane |
| Ediție, categorii și minime | Cluburi: 12/13 general, 5/6 teren; Nations League: 5/2, sub minimul 3 pe teren; amicale 2026 separat | Aplicată: fără transfer de date între ediții, genuri sau categorii |
| Antrenori și loturi actuale | Albert Sánchez confirmat de club; convocările impun schimbări; Granada are trei accidentați și Owen convocat | Aplicată: baseline-ul întregului sezon include și vechiul antrenor, risc păstrat |
| Program, teren, vreme și status | Granada a jucat ultima dată 1 nov, Las Palmas 9 nov; 15,9°C și vânt redus prognozat; 17:15 RO neînceput la 09:16:36 | Aplicată; XI, încărcarea individuală și gazonul efectiv rămân neverificate |
| Cutoff România | Timestampuri comparate numeric cu 15 nov 00:00 RO; Panama–Curaçao este 16 nov RO, Haiti–USA rămâne 14 nov RO | Aplicată: fără includerea observațiilor de după cutoff |
| Eliminări și cauze tactice | Cartonașele roșii și explicațiile cauzale nu au fost auditate exhaustiv | Neverificată: nu pretindem un diagnostic complet fără eliminări sau eliminarea eșecurilor |
`;
write(gp+'/CONTEXT.md',`# Contextul finalistelor — 15 noiembrie 2026

Actualizare editorială ${at}. Cercetare **parțială**. Datele prospective rămân intacte. [Calcule complete](CALCULATIONS.md), [ultimele zece oficiale](LAST10_OFFICIAL.md), [acoperire](COVERAGE.md); sursele brute sunt în Context/, iar URL-urile și momentele în sources.json.

## Granada–Las Palmas, 17:15 RO

**Candidat Gol70: 90,35% brut / 84,60% stres, fair matematic 1,1068. Candidat cornere și preferința individuală principală: 95,50% / 89,86%, fair 1,0472.** Frecvențele sunt 22/25 general și 11/11 pe teren la Gol70, respectiv 25/25 și 11/11 la cornere. Acestea sunt două piețe analizate separat; nu le combinăm și nu creăm o alegere a utilizatorului.

Granada are 12 meciuri, cinci acasă; Las Palmas are 13, șase în deplasare. GF70 propriu pe teren este 7/5 = 1,40 și 8/6 = 1,33. λ la goluri este 2,1603 sezon / 2,5167 teren / 2,3385 combinat; numai-sezon produce 88,47%, numai-teren 91,93%. Eliminarea diagnostică a diferențelor finale de cel puțin patru goluri nu schimbă rezultatul. Contraexemplele sunt Oviedo–Granada 0–0, Las Palmas–Leganés 0–0 și Las Palmas–Eldense, primul gol 90+5. Ultimul meci al oaspeților arată concret că forma bună și victoria finală pot coexista cu un eșec Gol70.

La cornere, λ combinat este 3,8933. Numai-sezon dă 97,31%, numai-teren 93,02%. Seriile complete nu conțin exact 7 sau ≥8. Totalurile pe teren au medie/varianță 4,40/1,30 pentru Granada și 4,17/1,77 pentru Las Palmas. Aceste eșantioane mici nu demonstrează distribuția sau calibrarea estimării.

[Pagina oficială a partidei](https://www.granadacf.es/partidos/temporada-2026-2027-laliga-hypermotion-14-granada-cf-vs-ud-las-palmas-3448) indică 23 de puncte în 12 jocuri pentru Granada și 20 în 13 pentru Las Palmas; pozițiile din unele articole diferă, deci păstrăm momentul și sursa clasamentului. Ambele urmăresc victoria în apropierea zonei superioare. Ora 15:15 din extrasul paginii nu are fus explicit; [RFEF](https://rfef.es/sites/default/files/designaciones_2a_division_masculina_-_temp_2026-27_-_jornada_14_domingo.pdf) confirmă 16:15 CET, iar FotMob 15:15 UTC: **17:15 RO**. O pagină de presă care indică altă oră nu înlocuiește aceste confirmări.

[Albert Sánchez, sursa oficială din 13 noiembrie](https://www.udlaspalmas.es/noticias/albert-sanchez-el-reto-es-darle-continuidad-a-lo-que-venimos-haciendo-e-ir-a-los-carmenes-a-ganar), confirmă pregătirea ieșirii din pressing și schimbările impuse de convocări. Granada este descrisă ca o echipă care presează sus, joacă direct și produce centrări și faze fixe. Aceste profiluri pot susține golurile, dar pot produce și presiune/cornere sau blocaj. Nu presupunem că un gol timpuriu reduce cornerele. Las Palmas a schimbat recent antrenorul: modelul păstrează sezonul complet, inclusiv vechiul regim, fără a inventa un procent pentru efectul schimbării.

[Pacheta, 14 noiembrie](https://www.granadaenjuego.com/granada-cf/pacheta-no-debe-romperse-el-momento-dulce), confirmă accidentările Diallo, Arnaiz și Casadesús și plecarea lui Owen la U17. Revenirea parțială la antrenament nu înseamnă disponibilitate pentru partidă. Insistența lui asupra solidității și adversarul care păstrează mingea sunt contraargumente pentru Gol70. [Lista deplasării Las Palmas](https://www.udlaspalmas.net/2026/11/14/ale-garcia-se-cae-de-la-convocatoria-a-granada-vuelven-marvin-e-inaki/) consemnează Ale García și Sandro absenți, Marvin Park și Iñaki reveniți. Mai menționează Clemente, Sergio Ruiz, Viti, Cedeño accidentați și Pezzolesi, Herzog, Rafa Cruz convocați. Motivul absenței lui Ale García nu este confirmat în această sursă. Apărarea schimbată și revenirile sunt riscuri; XI complet și minutele individuale nu sunt confirmate.

Ultimul oficial jucat din feed este 1 noiembrie la Granada, după Cupa din 29 octombrie, respectiv 9 noiembrie la Las Palmas, cu Cupa în 27 octombrie. Sunt circa 14 și șase zile calendaristice, fără măsurarea încărcării individuale. Amânarea cu Tenerife a întrerupt ritmul Granadei; deplasarea din Canare și revenirile medicale adaugă incertitudine, fără ajustare numerică. Prognoza Open-Meteo la 15 UTC: 15,9°C, 0 mm, vânt 2,3 km/h, rafale 10,1. Los Cármenes este teren propriu real, cu gazon; starea efectivă a suprafeței și orarul zborurilor rămân necunoscute.

## Alternative comparate

**Eibar–Cádiz, 19:30 RO:** Gol70 88,53/82,32%, 23/25 general și 11/11 teren; GF70 propriu 8/6 și 3/5. Cádiz are 0–0 cu Celta Fortuna și primul gol 87 cu Sporting. Cornere 90,21/80,21%, cu totaluri de 8, 9, 11 și două exact 7. Modelul și stresul sunt inferioare, iar dispersia cornerelor este mai mare. Eibar a jucat Cupa în 10 noiembrie; nu atribuim pauză din 31 octombrie doar din ultimul meci de campionat.

**Tenerife–Ceuta, 15:00 RO:** Gol70 86,16/79,44%, 25/25 și 12/12, dar GF70 propriu 6/6 și 3/6 este mai mic. Diagnosticul fără scoruri ample scade la 84,86%. Cornere 94,45/87,82%, 23/25 și 11/12; ambele echipe au avut nouă cornere cu Córdoba, Ceuta în deplasare. Revenirea Tenerife după tragedie este documentată în presa actuală; efectul nu este cuantificat. Noul antrenor Ceuta este semnalat în presă, fără reconciliere oficială completă în această rulare. Loturile și XI ale alternativelor nu au fost verificate la profunzimea finalistei.

**Eldense–Sabadell, 19:30 RO:** Gol70 77,92/70,13%, 18/26 și 10/13; GF70 propriu 5/6 și 2/7, cu mai multe 0–0 și deschideri tardive. Cornere 87,60/75,89%, 25/26, dar varianța Eldense acasă 7,37 depășește media 5,83; cinci exact 7 și un nouă în cohorta unică. Nu este alternativa preferată.

${lessons}

Decizia compară toate cele patru partide eligibile din inventarul identificat. Preferința individuală este **cornere Granada–Las Palmas**. Gol70 rămâne candidatul separat al celeilalte piețe. Pentru datele insuficiente: NO BET. Nu afirmăm top global, probabilitate calibrată sau profit.
`);
write(gp+'/LESSONS_APPLIED.md',lessons);
const cov=csvread(gp+'/universe_coverage.csv');let old=fs.readFileSync(gp+'/COVERAGE.md','utf8'),leagueTable=old.slice(old.indexOf('|Liga|'),old.indexOf('## Internaționale'));
write(gp+'/COVERAGE.md',`# Acoperire — 15 noiembrie 2026

Actualizare ${at}. **Cercetare parțială**, fără pretenția unui top global. 29 de ligi: primele ligi masculine ale primelor 20 de asociații în clasamentul final UEFA 2026, România, Serbia și șapte divizii secunde urmărite. [Clasamentul 2026](https://mail.kassiesa.net/uefa/data/method5/crank2026.html). În 28 de ligi nu au fost identificate partide azi în programul sezonului și fluxul global; patru partide în Segunda. ESPN confirmă separat 25 de programe; pentru Polonia, Croația, Serbia și Belgia 2 confirmarea independentă rămâne insuficientă. Un răspuns gol sau 400 nu dovedește absența meciurilor.

${leagueTable}
## Inventar extins și limite

18 identități FotMob: patru cluburi, 12 naționale seniori masculine și două meciuri de juniori masculini. Suplimentul conține 34 de partide Coupe de France și șase naționale. Separat există o semnalare de finală feminină U19 cu identitate, oră și dată RO neconfirmate: **59 de rânduri canonice = 58 de partide identificate plus o semnalare incertă**, nu 59 de meciuri independent confirmate. Alte trei meciuri feminine de clasament UNCAF au adversarii TBC.

[Federația Panama](https://www.fepafut.com/blog/2026/11/02/panama-conoce-su-convocatoria-para-el-torneo-uncaf-sub-19/) confirmă finalele pe 15 noiembrie local; fără ora exactă nu presupunem automat 15 noiembrie în România. Relatarea Honduras–Costa Rica intră în conflict cu un tabel al rezultatelor grupelor. Conflictul rămâne în Supplemental/women_scope.json. Nu am identificat o partidă între naționale feminine de seniori în fluxurile consultate; absența mondială nu este demonstrată.

Toate cele opt meciuri Nations League au câte cinci observații generale și două în configurația terenului, sub minimul de trei. Nu folosim ediții vechi pentru eligibilitate. Amicalele din anul 2026 sunt separate de oficiale; U18, U19, U20, U21, U23 și feminin rămân categorii distincte. Belarus este gazdă desemnată, fără locație/neutralitate validată pentru selecție. Kyrgyzstan–Vietnam are surse contradictorii de teren: Bishkek, Nam Dinh sau Ninh Binh. Zambia și Congo U23 folosesc ediția calificărilor 2027, fără istoric exact bilateral suficient; Congo nu este RDC.

USA U20–Colombia U19, Noua Zeelandă–India și meciurile CONCACAF din noaptea RO erau începute/terminate, fără predicție retrospectivă. Panama–Curaçao pe 15 nov la 23 UTC este 16 nov la 01 RO, exclus din ziua 15. Haiti–USA pe 14 nov la 21:30 UTC este 14 nov la 23:30 RO. Montenegro U17–Ukraine U17 și Austria U19–Slovenia U19 apar pe 15 doar în surse cu fus asiatic; rămân 14 noiembrie RO.

Cele 34 de partide Coupe de France sunt inventariate separat. Lipsesc istoricele exacte bilaterale și mostrele 5/3; eventualele proiecții de campionat ar trebui etichetate între competiții. UCL, UEL, UECL, Copa și FA Cup au fost consultate prin competițiile urmărite; fără alte modele eligibile identificate. Suplimentele primesc NO BET formal și probabilitate lipsă.

162 de istorice din edițiile curente; două liste de goluri și 19 observații FirstHalf lipsesc. Lipsurile blochează modelarea pieței, nu devin zero. Au fost colectate separat ultimele zece oficiale pentru fiecare dintre cele opt cluburi: 80 de apariții, cu observațiile de cupă incomplete explicit. Auditul eliminărilor nu este exhaustiv.

Surse și dovezi: Global/14–16, Schedules/, ESPN/, Competitions/, MatchData/, HistoryRaw/, Context/, Supplemental/ și sources.json. În Revisions/ sunt păstrate inventarul inițial, modelele și momentele inițiale, precum și subsetul de club derivat, etichetat corespunzător. Extinderea nu a creat alte modele eligibile. Fără modificarea modelelor de producție.
`);
for(let [p,title,lead]of [[gp+'/EUROPE_ALL_MATCHES.md','Gol70','Granada–Las Palmas, 17:15 RO: **90,35% brut / 84,60% stres**, fair matematic 1,1068.'],[cp+'/ANALYSIS.md','Cornere 1H ≤7','Granada–Las Palmas, 17:15 RO: **95,50% brut / 89,86% stres**, fair matematic 1,0472. Preferința individuală principală a analistului.']]){let s=fs.readFileSync(p,'utf8'),table=s.slice(s.indexOf('|Rang|'),s.indexOf('## Lecții'));write(p,`# Cercetare 15 noiembrie 2026 — ${title}\n\n${lead}\n\nCercetare **parțială**. Toate cele patru partide eligibile au fost comparate; inventarul extins are lipsuri explicite. Nu este top global și nu este alegere a utilizatorului.\n\n${table}\n\n${lessons}\n\nDocumentație: [context și alternative](../Goal70_2026-11-15/CONTEXT.md), [calcule și numărători](../Goal70_2026-11-15/CALCULATIONS.md), [acoperire](../Goal70_2026-11-15/COVERAGE.md), [inventar](../Goal70_2026-11-15/INVENTORY.md), [ultimele zece oficiale](../Goal70_2026-11-15/LAST10_OFFICIAL.md). Rezultatele de azi rămân necunoscute. Fair nu este preț disponibil.\n`);}
const reviewTable=(rs,corner)=>'| Meci / competiție | Rang; brut / AR20 | Frecvență generală; teren | Observație cerută | Rezultat / sursă |\n|---|---|---|---|---|\n'+rs.map(r=>{let a=corner?r.unique_under:r.first_goal_success,b=corner?r.unique_n:r.first_goal_n,x=corner?r.venue_under:r.venue_success,y=corner?r.venue_unique_n:r.venue_n;return `|${r.home}–${r.away} / ${r.competition}|${r.rank||'—'}; ${r.p_raw?(100*r.p_raw).toFixed(2)+'% / '+(100*r.p_stress).toFixed(2)+'%':'nemodelat'}|${b?`${a}/${b}`:'necunoscut'}; ${y?`${x}/${y}`:'necunoscut'}|${corner?(r.actual_corners_1h!==''?r.actual_corners_1h:'necunoscut'):(r.first_goal_minute||'necunoscut')}|[${r.result}](${r.result_source})|`;}).join('\n');
let review=fs.readFileSync(rp+'/REVIEW.md','utf8'),a=review.indexOf('## Lista completă Gol70'),b=review.indexOf('## Restanțe');review=review.slice(0,a)+'## Lista completă Gol70\n\n'+reviewTable(csvread(rp+'/goal70_outcomes.csv'),false)+'\n\n## Lista completă cornere\n\n'+reviewTable(csvread(rp+'/corners1h_outcomes.csv'),true)+'\n\n'+review.slice(b);write(rp+'/REVIEW.md',review);
let j='Analysis/Selection_History/2026-11-15.md';write(j,`# Jurnal PRED — 15 noiembrie 2026

Cerere: rularea zilnică autorizată pentru ambele piețe, inclusiv toate naționalele, fără agenți paraleli. Instrucțiunile și datele canonice au fost citite. Ora inițială: 09:10 RO.

Alegerile explicite din dashboard pentru 14 noiembrie sunt ambele Succes: Valladolid–Castellón Gol70, primul gol 67; Almería–Andorra, FirstHalf 2. Rezultatele au fost actualizate atomic sub lock, păstrând probabilitățile, rangurile și timestampurile. Nicio alegere nouă creată.

Review 14: Gol70 29 succese / 7 fail / 128 pending, modele 6/6; cornere 29 succese / 2 fail / 133 pending, modele 4/5. Oviedo–Girona 9 este eșecul modelat, rang 3, 95,10/89,08%, 9/11 teren, varianță 7,47 peste media 5,33: avertismente cunoscute prospectiv, fără cauză tactică demonstrată. Reîncercate 37 de identități restante; o observație nouă în review 13, cu dovezi și copie anterioară. Review și lecții predate ${read(rp+'/handoff.json').at_ro}; scanare ${read(gp+'/research_start.json').at_ro}. Dashboard reconstruit după rezultate.

29 de ligi, clasament final UEFA 2026 plus România/Serbia și șapte secunde. Inventar: 58 de partide identificate și o semnalare feminină incertă; alte trei meciuri de clasament UNCAF au adversari TBC. 162 de istorice, două liste de goluri și 19 FirstHalf lipsă. Cercetare parțială; patru modele în fiecare piață, toate comparate. Opt Nations League au 5/2, sub minimul de trei pe teren. Edițiile, amicalele, categoriile, genul și neutralitatea sunt separate.

Granada–Las Palmas 17:15 RO este candidatul ambelor piețe: Gol70 90,35/84,60%, 22/25 și 11/11; cornere 95,50/89,86%, 25/25 și 11/11, preferința individuală principală. Antrenori, loturi, convocări, pressing/centrări, program/amânare, vreme și status neînceput verificate; XI, încărcarea individuală, gazonul și eliminările istorice sunt incomplete. Nicio ajustare contextuală de procent. Inventarul și snapshoturile inițiale sunt arhivate în Revisions/.

Limite: istorice exacte insuficiente la suplimente; UNCAF feminin are identități/ore/date RO neconfirmate și conflict între surse; patru ligi fără confirmare independentă. Goluri validate prin reconcilierea scorului și cornere prin FirstHalf+SecondHalf=All. Cutoff numeric Date.parse, fără lookahead. Cauzele efective și calibrarea nu sunt demonstrate de această zi.

Urmează verificarea finală a dashboardului și exportul mesajelor vizibile disponibile local. La 16 noiembrie: alegeri explicite actuale → rezultate 15/restanțe → review/lecții → scanare. Modelele de producție, selection_engine, programarea și Git rămân nemodificate; fără pariuri, mize, mesaje externe sau commit/push.
`);console.log('Readable final reports saved',at);
