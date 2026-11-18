import fs from 'node:fs';
import {read,write,csvread,csvwrite,stamp} from './daily_2026_11_14.mjs';
const gp='Analysis/Goal70_2026-11-14',cp='Analysis/Corners1H_2026-11-14',rp='Analysis/Review_2026-11-13',at=stamp();
const g=csvread(`${gp}/all_models.csv`),c=csvread(`${cp}/all_models.csv`),cov=csvread(`${gp}/universe_coverage.csv`),gm=g.filter(x=>x.p_raw),cm=c.filter(x=>x.p_raw);
for(let p of [`${gp}/EUROPE_ALL_MATCHES.md`,`${gp}/CONTEXT.md`,`${cp}/ANALYSIS.md`,'Analysis/Selection_History/2026-11-14.md'])write(`${gp}/Revisions/editorial_${p.includes('Corners1H')?'corners_':''}${p.split('/').at(-1)}`,fs.readFileSync(p,'utf8'));
const pct=x=>(100*Number(x)).toFixed(2)+'%',ranking=(rows,corner=false)=>'| Rang | Meci | Ora RO | Brut | Stres | Fair matematic | Istoric / teren |\n|---|---|---|---|---|---|---|\n'+rows.map(r=>`| ${r.rank} | ${r.home}–${r.away} | ${r.kickoff_ro.slice(11,16)} | ${pct(r.p_raw)} | ${pct(r.p_stress)} | ${Number(r.fair_odds).toFixed(4)} | ${corner?r.unique_under+'/'+r.unique_n:r.first_goal_success+'/'+r.first_goal_n}; ${corner?r.venue_under+'/'+r.venue_unique_n:r.venue_success+'/'+r.venue_n} |`).join('\n');
const coverage=`## Acoperire și limite

Inventarul are **164 de înregistrări**, dintre care 38 cu identitate FotMob și 126 de suplimente: 51 de partide din Coupe de France, 59 de calificări U17/U19 și 16 alte semnalări. Acesta nu este un total de 164 de partide confirmate independent: există un conflict England U18–Norway U19 versus England U18–Sweden U18. Istoricul exact și minimele 5/3 nu au fost validate pentru suplimente. Nu substituim alte generații, cluburi sau ediții vechi.

Sunt verificate 29 de ligi: primele 20 de asociații din [clasamentul final UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), calculat pe 2021/22–2025/26, plus România, Serbia și cele șapte secunde urmărite. În 28 de ligi nu au fost identificate partide; în Segunda sunt cinci. Am confruntat programul global FotMob, calendarele edițiilor actuale și ESPN. Patru ligi nu au confirmare ESPN: Polonia, Croația, Serbia și secunda belgiană. Un răspuns nesuportat nu dovedește absența meciurilor.

Separat sunt trei partide de cupă din Slovacia/Croația și cele 51 din Coupe de France. Nu au istoric bilateral validat pentru proiecția de campionat. Calendarele UCL/UEL/UECL și Copa del Rey consultate nu indică partide azi. Lista completă, sursele și excluderile sunt în [INVENTORY.md](../Goal70_2026-11-14/INVENTORY.md), universe_coverage.csv și candidate_fixtures.json.

Am colectat **304 istorice unice** în edițiile actuale, cu zece liste de goluri și 36 de observații FirstHalf lipsă sau nereconciliate. Suplimentele fără istoric nu intră în acest numitor. Sunt calculate și comparate toate cele **șase modele Gol70 și cinci modele de cornere eligibile în datele validate**. Modelarea întregului inventar și confirmarea independentă mondială rămân parțiale; nu prezentăm un top global complet.

Cele opt partide UEFA Nations League au numai 4 generale/2 pe teren pentru fiecare echipă, sub 5/3. Edițiile AFCON/CONCACAF sunt și ele limitate. Amicalele folosesc numai anul 2026, separat de competițiile oficiale. Identitățile seniori/juniori/masculin/feminin sunt distincte. Meciurile începute sau terminate în dimineața RO sunt inventariate fără predicții retrospective și fără rezultate fabricate azi.

Două listări duble Friendly/King’s Cup, Oman–Tajikistan și Thailand–Brunei, sunt deduplicate și excluse până la clarificarea competiției. Basque Country–Venezuela este o selecție regională/non-FIFA, distinctă și fără model. Haiti–USA necesită confirmarea gazdei desemnate și a terenului; nu este finalist.

Japan–Brazil este **neutru, la Singapore**, cu Japonia gazdă desemnată. Baseline-ul 87,13/80,60% folosește părțile desemnate ale amicalelor 2026, fără a pretinde avantaj propriu real sau calibrare pe neutru. Frecvențele sunt 11/14 general și 6/8 pe părțile desemnate. Nu este candidat preferat.

[AUF confirmă Uruguay U17 feminin–Peru U17 feminin](https://www.auf.org.uy/la-sub-17-femenina-disputara-dos-amistosos-ante-peru/) la 10:00 local = 15:00 RO, pe teren propriu în Uruguay; istoricul exact al amicalelor 2026 nu este validat. Nu a fost identificată o partidă de naționale feminine de seniori, fără a demonstra absența mondială. Panama–El Salvador și Costa Rica–Guatemala U19 feminin, afișate 14 nov 04:00 în UTC+8, aparțin zilei 13 nov, 22:00 RO, conform surselor secundare UTC.

Snapshotul numeric inițial și reviziile sunt arhivate în Revisions/. Procentele, rangurile originale și momentele inițiale ale predicțiilor numerice sunt păstrate. Nicio alegere nouă a utilizatorului nu a fost creată.
`;
const lessons=`## Lecții aplicate din review

[Lecțiile pentru azi](../Review_2026-11-13/LESSONS_FOR_NEXT_SCAN.md) au fost predate la **${read(`${rp}/handoff.json`).at_ro}**, înaintea scanării începute la **${read(`${gp}/research_start.json`).at_ro}**.

| Lecție | Verificare și dovadă nouă | Stare / efect |
|---|---|---|
| Model, frecvență și stres separate | Valladolid Gol70 91,42/85,98%, cornere 95,72/90,29%; 24/25 general și 13/13 teren | Aplicată: preferință individuală pentru cornere, fără bonus procentual |
| GF70 propriu, 0–0 și concentrarea golurilor | Valladolid 13/6, Castellón 9/7 pe teren; Castellón–Sabadell 0–0; diagnostic fără diferențe FT≥4: 89,50% | Aplicată: intensitatea nu este o rată binară validată |
| FirstHalf exact 7/≥8 și dispersie | Eibar–Valladolid 9; Valladolid–Albacete 7; Sociedad B–Castellón 7; Castellón afară varianță 5,33 versus medie 4 | Aplicată: coada rămâne în istoric și scenariul +20% |
| Campionat versus cupă/Europa | Cluburile folosesc Segunda 2026/27; ultimele zece oficiale sunt diagnostic separat; naționalele au ediții și categorii separate | Aplicată: fără pooling pentru minime |
| Lot actual versus XI vechi | Conferințele Escribá/Hernández; buletinul Rêgo din 13 nov contrazice simpla etichetă „convocat” | Aplicată: API lastStarting11 nu este XI confirmat |
| Odihnă, vreme, teren și status | Trofeul din cursul săptămânii la Valladolid; amânarea Eibar la Castellón; rafale prognozate 40,7 km/h; toate cele șase modele neîncepute la 08:58 RO | Aplicată; minutele individuale și gazonul efectiv rămân neverificate |
| Neutralitate și identitate | Japonia–Brazilia la Singapore; U17/U19/U21/U23 și feminin distincte; conflict King’s Cup | Aplicată: baseline-ul neutru nu este preferat; lipsurile rămân explicite |
| Eliminări și cauze tactice | Extragerile cardurilor nu au fost auditate exhaustiv cronologic | Neverificată parțial: nu pretindem o sensibilitate completă fără roșu sau cauze demonstrate |

Cele trei succese explicite de ieri nu autorizează modificarea formulei, ponderilor, pragurilor sau modelelor. Metoda și scenariile rămân cele salvate.
`;
const context=`# Contextul finalistelor — 14 noiembrie 2026

Revizie editorială ${at}. Datele și predicțiile originale sunt păstrate. [Calcule complete](CALCULATIONS.md), [ultimele zece oficiale](LAST10_OFFICIAL.md), surse și extrageri în sources.json și Context/. Cercetare parțială.

## Real Valladolid–Castellón, 15:00 RO

**Candidat Gol70: 91,42% brut / 85,98% stres; fair matematic 1,0938. Candidat cornere și preferința individuală principală: 95,72% / 90,29%; fair 1,0448.** Istoricul are 24/25 succese generale și 13/13 pe teren în fiecare piață. Valladolid are 13 meciuri, șase acasă; Castellón are 12, șapte în deplasare.

La Gol70, Valladolid a marcat el însuși 13/6 = 2,17 goluri pe teren, Castellón 9/7 = 1,29. Singurul contraexemplu binar din ediție este Castellón–Sabadell 0–0. Fără meciuri cu diferență finală de cel puțin patru goluri, diagnosticul scade la 89,50%; numai-sezon este 90,02%, numai-teren 92,63%. Niciun diagnostic nu înlocuiește probabilitatea inițială.

La cornere, λ sezon/teren/combinat = 3,9551/3,7500/3,8526. Numai-sezon dă 95,15%, numai-teren 96,24%. Media/varianța totalului 1H sunt 3,50/3,50 pentru Valladolid acasă și 4,00/5,33 pentru Castellón afară. Păstrăm Eibar–Valladolid 9 și cele două totaluri exact 7: Valladolid–Albacete și Sociedad B–Castellón. Din acestea, ultimul este în configurația terenului de azi. 13/13 nu garantează următorul rezultat.

[Gazdele confirmă locurile 12/2, cu 18/27 puncte](https://www.realvalladolid.es/noticias/cara-a-cara-or-real-valladolid-vs-cd-castellon-26-27). Castellón are doar cinci goluri primite în campionat și posesie medie 60%. Apărarea bună este un contraargument pentru Gol70; atacurile repetate și pressingul pot produce mai multe cornere, fără relație numerică demonstrată.

[Escribá, 13 noiembrie](https://www.realvalladolid.es/noticias/escriba-previa-j14-real-valladolid-castellon), confirmă intenția ofensivă, dorința de îmbunătățire a apărării și titularizarea anunțată a lui Trilli. Latasa/Guille Bueno sunt accidentați, Alejo suspendat, Homam/Balboa convocați. De Rooij revine. Trofeul din mijlocul săptămânii este relevant pentru încărcare: nu presupunem șapte zile de odihnă din ultimul meci de campionat. O pagină agregată care afirmă zero absențe este contrazisă de club. XI complet rămâne neconfirmat.

[Pablo Hernández](https://www.cdcastellon.com/pablo-hernandez-la-mejor-manera-de-rendirle-homenaje-a-demi-es-luchar-por-nuestro-objetivo/) confirmă revenirea lui Camara, recuperarea lui Fran și intenția de pressing. Castellón revine după amânarea cu Eibar în contextul doliului. Pauza nu este automat un avantaj. [Convocările](https://www.cdcastellon.com/saipi-maty-y-rodri-convocados-con-sus-selecciones/) exclud Saipi și Maty; [buletinul ulterior Rêgo](https://www.cdcastellon.com/parte-medico-rodrigo-rego/) confirmă accidentarea la ischiogambieri. Nu îl mai descriem doar ca plecat la națională. Dorința victoriei la ambele echipe nu garantează goluri timpurii, iar efectul doliului nu este cuantificat.

Zorrilla este teren propriu real, cu gazon. 14:00 CET = 15:00 RO, confirmat de cluburi și FotMob 13:00 UTC. Open-Meteo pentru 13 UTC: 13,6°C, precipitații 0,3 mm, vânt 15,3 km/h și rafale 40,7 km/h. Ploaia și vântul pot afecta circulația și centrările; direcția efectului asupra cornerelor este necunoscută. Gazonul efectiv, minutele individuale și călătoria exactă nu sunt confirmate. API arată lastStarting11, fără XI de azi.

Preferința principală rămâne la cornere: stres 90,29% versus 85,98% la Gol70, cu riscurile tactice și de lot păstrate. Nu combinăm cele două evenimente.

## Almería–Andorra, 19:30 RO — alternativa apropiată

Gol70: 89,24/83,19%, 24/25 general și 11/12 teren. GF70 propriu pe teren este 13/6 = 2,17 și 5/6 = 0,83; contraexemplu Sabadell–Andorra 0–0. Diagnosticul fără scoruri ample este 89,09%. Cornere: 95,24/89,36%, 25/25 și 12/12; un exact 7 la Almería–Eldense și niciun ≥8 în istoricul validat. Acesta este un argument serios față de Valladolid, dar stresul este inferior și eșantionul nu garantează continuitatea.

[Andorra, sursă oficială din 13 noiembrie](https://www.fcandorra.com/es/noticias/previa-jornada-14-almeria), confirmă antrenorii García Pimienta/Carles Manso, Almería lider și profilul de pressing și dueluri individuale al oaspeților. Efe Akman și Lautaro sunt suspendați; Martí Vilà, Owono, Le Normand și Gael Alonso nu au călătorit. Feedul inițial omite unele absențe, deci actualizarea clubului are prioritate. La Almería sunt semnalate Cipenga/Dzodic convocați și Luna suspendat; restul etichetelor medicale din API rămân incomplet reconciliate. Absențele nu înseamnă automat puține cornere.

Prognoza la 17 UTC: 20,1°C, fără ploaie, vânt 12,5 km/h, rafale 35,3. Teren propriu real. Contextul dificil după accidentul de la Tenerife, încărcarea individuală și XI rămân surse de incertitudine. Nu aplicăm procente psihologice sau tactice. Avantajul brut/stres al Valladolidului îl păstrează primul în comparația calculabilă.

## Celelalte alternative calculate

Oviedo–Girona: 95,10/89,08% la cornere, dar numai 9/11 pe teren și eșecuri cu 8/9; media/varianța Oviedo acasă 5,33/7,47 arată mai multă dispersie. Mallorca–Córdoba: 94,49/87,91%, cu 21/25 general versus 13/13 teren. Sociedad B–Sporting: 93,23/85,56%, 25/26 și 12/12, dar stres inferior. Loturile acestora nu au fost verificate la profunzimea primelor două meciuri. Japonia–Brazilia, neutră la Singapore, rămâne nepreferată din cauza frecvențelor mai slabe și transferului părților desemnate.

${lessons}`;
write(`${gp}/CONTEXT.md`,context);
write(`${gp}/EUROPE_ALL_MATCHES.md`,`# Gol70 — 14 noiembrie 2026

**Candidat al pieței: Real Valladolid–Castellón, 15:00 RO — 91,42% brut / 85,98% stres; fair matematic 1,0938.** Preferința individuală principală este același meci la cornere 1H. Evenimentele nu se combină. Toate cele șase modele Gol70 și cinci de cornere au fost comparate; status neînceput reconfirmat la 08:58 RO.

${ranking(gm)}

[Context și contraargumente](CONTEXT.md) · [Calcule și numărători](CALCULATIONS.md) · [Ultimele zece oficiale](LAST10_OFFICIAL.md) · [Inventar și excluderi](INVENTORY.md). Cercetarea este **parțială**.

${coverage}
${lessons}
## Ligile și programele verificate

| Liga | Rang UEFA 2026 / supliment | Global | Ediție | ESPN | Acoperire |
|---|---|---|---|---|---|
${cov.map(r=>`| ${r.country} / ${r.league} | ${r.uefa_2026_rank} | ${r.identified_fixtures} | ${r.season_schedule_daily} | ${r.espn_fixtures===''?'indisponibil':r.espn_fixtures} | ${r.coverage} |`).join('\n')}
`);
write(`${cp}/ANALYSIS.md`,`# Cornere 1H sub 7,5 — 14 noiembrie 2026

**Candidat și preferința individuală principală: Real Valladolid–Castellón, 15:00 RO — 95,72% brut / 90,29% stres; fair matematic 1,0448.** Istoric 24/25 general și 13/13 pe teren. Riscuri: totalurile 7/9 din istoric, dispersia Castellónului în deplasare, absențele și rafalele prognozate. Este o estimare Poisson necalibrată, fără confirmarea vreunui preț disponibil.

${ranking(cm,true)}

[Context și alternative](../Goal70_2026-11-14/CONTEXT.md) · [Calcule complete](../Goal70_2026-11-14/CALCULATIONS.md) · [Inventar](../Goal70_2026-11-14/INVENTORY.md). Datele folosite au FirstHalf explicit și reconciliere FirstHalf+SecondHalf=All. FT nu substituie prima repriză.

${coverage}
${lessons}`);
write(`${gp}/COVERAGE.md`,'# Acoperire — 14 noiembrie 2026\n\n'+coverage);
for(let [p,rows]of [[`${gp}/exclusions.csv`,g],[`${cp}/exclusions.csv`,c]])csvwrite(p,rows.filter(r=>r.eligible!=='true'));
write('Analysis/Selection_History/2026-11-14.md',`# Jurnal PRED — 14 noiembrie 2026

Cerere: rularea zilnică autorizată pentru ambele piețe și toate naționalele, fără agenți paraleli. Ora inițială verificată 08:46 RO. Citite instrucțiunile, SESSION_RESUME, jurnalul și datele canonice actuale. Acestea arată trei alegeri explicite din 13 noiembrie, ulterioare notei vechi de reluare.

Rezultate: Finlanda U21–Irlanda U21 Gol70, primul gol 45+8, Succes; Slovacia–Feroe Gol70, primul 17, Succes; Leganés–Burgos cornere 1H 4, Succes. Actualizate numai câmpurile de rezultat prin lock, recitire și scriere atomică. Procentele, rangurile, momentele predicțiilor și evenimentele alegerilor sunt păstrate; zero alegeri noi create.

Review 13: Gol70 39 succese/7 fail, 9 pending, 1 void; modele separat 5/5. Cornere 29 succese/4 fail, 22 pending, 1 void; modele 1/1. Reîncercate 23 de identități vechi, fără rezultat nou compatibil pentru piața restantă. Review și lecții predate ${read(`${rp}/handoff.json`).at_ro}; scanare începută ${read(`${gp}/research_start.json`).at_ro}. Dashboard reconstruit după review.

Au fost aplicate verificările brut/frecvență/stres, GF70 propriu, deschideri tardive și 0–0, FirstHalf exact 7/≥8, dispersie, competiții/categorii, antrenori și loturi actuale, odihnă, teren, vreme și status. Actualizarea oficială Rêgo corectează eticheta inițială de convocat. Cauzele tactice și sensibilitatea completă fără eliminări rămân neverificate. Fără tuning.

Inventar: 164 de înregistrări, 38 FotMob și 126 suplimente, inclusiv 51 Coupe de France și 59 U17/U19. Există un conflict England U18, deci nu sunt 164 partide independent confirmate. Cele 29 de ligi folosesc top 20 UEFA final 2026 plus RO/RS și șapte secunde; 28 fără partide identificate, cinci meciuri Segunda. 304 istorice, zece liste de goluri și 36 FirstHalf lipsă/neconciliate. Șase modele Gol70 și cinci de cornere calculate. Naționalele și amicalele 2026 sunt separate; Japonia–Brazilia este neutră la Singapore. Uruguay U17 feminin–Peru U17 feminin, 15:00 RO, confirmată AUF, fără istoric exact validat.

Candidat Gol70: Valladolid–Castellón, 15:00 RO, 91,42/85,98%, fair 1,0938. Candidat cornere și preferință individuală principală: același meci, 95,72/90,29%, fair 1,0448. Ambele au 24/25 general și 13/13 teren. Alternativă Almería–Andorra: Gol70 89,24/83,19%, cornere 95,24/89,36%, cu 25/25 la cornere și riscuri de lot. Toate cele șase modele neîncepute reconfirmate la 08:58 RO. Nu combinăm evenimentele și nu atribuim alegeri utilizatorului.

Snapshotul inițial și reviziile sunt păstrate, inclusiv momentele originale. Cercetarea și review-ul rămân parțiale; nicio pretenție de top global complet. Fără pariuri, mize, cote de bookmaker, mesaje externe, modificări în modele/selection_engine, programare, commit sau push.

Urmează verificarea dashboardului și exportul privat al mesajelor vizibile disponibile. La următoarea rulare: citește eventualele alegeri noi, verifică rezultatele din 14 și restanțele, predă review/lecții, apoi scanează noua zi. Nu repeta research-ul de azi la unlock.
`);
console.log({editorial:'saved',at_ro:at});
const P='Previews/Pred_Research_Terminal';
let runtime=fs.readFileSync(P+'/.runtime/daily-2026-11-13-runtime.mjs','utf8');
let start=runtime.indexOf(" await check('Today56"),end=runtime.indexOf(' const screenshot=');
runtime=runtime.slice(0,start)+`
 await check('Today164records6goal5corner','DATA.date==="2026-11-14"&&DATA.matches.length===164&&DATA.coverage.goal===6&&DATA.coverage.corner===5');
 await check('No invented user choice or result','DATA.userSelections.length===0&&DATA.metrics.goal.played===0&&DATA.metrics.corner.played===0&&DATA.metrics.goal.hit_rate===null');
 await check('Today review is dated pending','document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-14"&&!document.querySelector("#outcomeStrip")');
 await evaluate('setMarket("corner")');
 await check('Valladolid corner candidate and original probability','DATA.matches.find(m=>m.id==="5868546").corner.p===0.9571638022943603&&DATA.matches.find(m=>m.id==="5868546").analystCorner&&!document.querySelector("#outcomeStrip")');
 await evaluate('selectDay("2026-11-13");setMarket("goal")');
 await check('Three real choices and39success7fail','DATA.userSelections.length===3&&DATA.userSelections.every(u=>u.result==="success")&&DATA.metrics.goal.success===39&&DATA.metrics.goal.fail===7');
 await check('Explicit goal choices pinned','["6416833","5181942"].includes(filteredMatches()[0].id)&&document.querySelector(".goal-card .choice-result").textContent==="-Succes"');
 await evaluate('setMarket("corner")');
 await check('Leganes choice pinned with four corners and29success4fail','filteredMatches()[0].id==="5868541"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"&&DATA.metrics.corner.success===29&&DATA.metrics.corner.fail===4');
 await evaluate('selectDay("2026-11-14");setMarket("corner")');
 await check('Returning today clears yesterday results','!document.querySelector("#outcomeStrip")&&DATA.metrics.corner.played===0');
`+runtime.slice(end);
runtime=runtime.replaceAll('preview-daily-2026-11-13.png','preview-daily-2026-11-14.png').replaceAll("date:'2026-11-13'","date:'2026-11-14'").replaceAll('daily-verification-2026-11-13.json','daily-verification-2026-11-14.json');
write(P+'/.runtime/daily-2026-11-14-runtime.mjs',runtime);
let inv=fs.readFileSync(`${gp}/INVENTORY.md`,'utf8'),table=inv.indexOf('| Meci |');write(`${gp}/INVENTORY.md`,'# Inventar complet al înregistrărilor cercetate\n\n'+coverage+'\n'+inv.slice(table));
