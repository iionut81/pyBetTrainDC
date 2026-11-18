import fs from 'node:fs';import path from 'node:path';
import {read,csv,save,ro} from './research_2026_11_13.mjs';
const B='Analysis/Goal70_2026-11-13',C='Analysis/Corners1H_2026-11-13',R='Analysis/Review_2026-11-12',H='Analysis/Selection_History',at=ro(),run=read('Analysis/Automation/run_report_2026-11-13.json');
const table=(headers,rows)=>'| '+headers.join(' | ')+' |\n| '+headers.map(()=> '---').join(' | ')+' |\n'+rows.map(r=>'| '+r.map(v=>String(v??'—').replaceAll('|',' / ')).join(' | ')+' |').join('\n')+'\n';
const percent=v=>v===''||v==null?'indisponibil':(100*Number(v)).toFixed(2)+'%';
const files=[B+'/EUROPE_ALL_MATCHES.md',B+'/CONTEXT.md',B+'/International/REPORT.md',C+'/ANALYSIS.md',R+'/REVIEW.md',R+'/LESSONS_FOR_NEXT_SCAN.md',H+'/2026-11-13.md'];
const rev=B+'/Revisions/editorial_'+Date.now();for(const file of files){const dest=rev+'/'+file;fs.mkdirSync(path.dirname(dest),{recursive:true});fs.copyFileSync(file,dest);}save(rev+'/reason.json',{at_ro:at,reason:'Readable editorial revision; no prediction values, ranks or snapshot timestamps changed; initial review handoff preserved'});
const goals=csv(B+'/all_models.csv'),corners=csv(C+'/all_models.csv'),modeled=goals.filter(r=>r.p_raw),corner=corners.find(r=>r.p_raw),sources=read(B+'/sources.json').sources;
const sourceList=sources.map(s=>`- [${s.key}](${s.url}): ${s.note}.`).join('\n');
const comparison=table(['Rang','Meci','Ora RO','Brut','AR20','Fair matematic','Istoric / teren'],modeled.map(r=>[r.rank,r.home+'–'+r.away,r.kickoff_ro.slice(11,16),percent(r.p_raw),percent(r.p_stress),Number(r.fair_odds).toFixed(4),r.first_goal_success+'/'+r.first_goal_n+' ; '+r.venue_success+'/'+r.venue_n]));
const inventory=table(['Meci','Categorie / competiție','Ora RO','Gol70: estimare sau excludere','Cornere: estimare sau excludere'],goals.map(r=>{const c=corners.find(x=>x.id===r.id);return [r.home+'–'+r.away,(r.category||'club masculin')+' / '+r.competition,r.kickoff_ro?.slice(11,16)||'TBD',r.p_raw?percent(r.p_raw):r.reason,c?.p_raw?percent(c.p_raw):c?.reason];}));
const cov=csv(B+'/universe_coverage.csv');const coverage=table(['Liga','Rang UEFA 2026 / supliment','Meciuri global','ESPN','Meciuri sezon','Limită'],cov.map(r=>[r.country,r.uefa_2026_rank,r.identified_fixtures,r.espn_supported==='true'?r.espn_fixtures:'indisponibil',r.season_schedule_daily,r.limit||'—']));
const weather=id=>{const r=read(B+'/Context/weather_'+id+'.json');if(!r.ok)return 'prognoză independentă indisponibilă';const h=r.data.hourly,indices=id==='6416833'?[16,17,18]:id==='6416831'?[17,18,19]:[19,20,21];return indices.map(i=>`${h.time[i].slice(11)} UTC: ${h.temperature_2m[i]}°C, precipitații ${h.precipitation[i]} mm, ninsoare ${h.snowfall[i]} cm, vânt ${h.wind_speed_10m[i]} km/h, rafale ${h.wind_gusts_10m[i]} km/h`).join('; ');};
const context=`# Contextul finalistelor — 13 noiembrie 2026

Actualizare editorială ${at}. [Calcule, numărători și toate deschiderile tardive](CALCULATIONS.md), [ultimele oficiale verificabile](LAST10_OFFICIAL.md). Sursele și extragerile au propriile timestampuri. Probabilitățile prospective rămân neschimbate.

## Finlanda U21–Irlanda U21, 18:15 RO — candidat Gol70

**89,19% brut / 83,14% AR20; fair matematic 1,1212.** Sunt 18/20 succese istorice distincte și 8/10 în configurația terenului, cu zece meciuri generale și cinci pe teren pentru fiecare echipă. Finlanda are GF70 propriu acasă 10/5 = 2,00; Irlanda în deplasare 3/5 = 0,60. Argumentul ofensiv depinde mai ales de gazde.

Contraexemplele din ediție sunt Finlanda–Kosovo 0–0 și Kazahstan–Irlanda, primul gol 89. Ambele apar în configurația terenului de azi. Eliminând simetric meciurile cu diferență finală de cel puțin patru goluri, diagnosticul scade la **81,07%**, cu eșantioane 7/9 generale și 4/4 pe teren. Fără eliminări până la70, rezultatul rămâne 89,19%. Sunt sensibilități, fără înlocuirea modelului sau intervale de încredere.

Este tur de baraj, cu retur pe17 noiembrie la Cork. Un egal lasă calificarea deschisă. Mika Lehkosuo și Jim Crawford sunt antrenorii actuali. [Actualizarea finlandeză din12 noiembrie](https://www.veikkausliiga.com/uutiset/2026/11/12/pikkuhuuhkajat-valmiina-em-jatkokarsinnan-avaukseen-lauri-laine-korvaa-juho-talvitien-joukkueessa) confirmă absența lui Juho Talvitie în ambele partide și înlocuirea cu Lauri Laine. Finlanda s-a pregătit patru zile la Tampere, iar antrenorul menține intenția ofensivă; declarația nu garantează ritmul jocului.

[FAI](https://www.fai.ie/latest/preview-mu21-play-finland-v-ireland/) confirmă Orazi indisponibil, înlocuit de Cummins, și antrenamentul lotului rămas. Crawford amintește că după tur calificarea rămâne deschisă. Irlanda vine după patru victorii în calificări; adversarul actual nu este echivalentul celor mai slabe echipe întâlnite anterior. Lista inițială Palloliitto din4 noiembrie este completată de actualizările ulterioare. Etichetele medicale generice FotMob pentru alte persoane/cohorte nu sunt tratate automat drept absențe ale lotului convocat. XI este neconfirmat.

Tammelan Stadion, Tampere, teren artificial: Finlanda este gazda reală, fără neutralitate presupusă. Prognoza FotMob indică2°C și ploaie/ninsoare. Open-Meteo: ${weather('6416833')}. Sunt prognoze, fără verificarea stării efective a gazonului. Frigul și umezeala sunt riscuri relevante după review-ul de ieri, dar efectul numeric rămâne necunoscut. Irlanda are deplasare; încărcarea individuală de la cluburi nu a fost auditată complet. Patru zile de cantonament nu înseamnă patru zile de odihnă.

Rămâne candidatul Gol70 al comparației calculabile: frecvența generală este mai bună decât la Danemarca–Țările de Jos, iar contribuția proprie a Finlandei acasă este mai mare decât a Leganésului acasă. Riscurile de baraj, vreme și concentrare a golurilor fac însă ca preferința individuală principală de azi să fie candidatul de cornere Leganés–Burgos.

## Danemarca U21–Țările de Jos U21, 19:30 RO — alternativă Gol70

88,42% brut / 82,18% stres, dar numai12/16 succese generale și6/8 pe teren: 75% observat versus88,42% modelat. GF70 propriu pe teren:6/4 = 1,50 și7/4 = 1,75. Contraexemple: Belgia–Danemarca gol80, Danemarca–Belgia0–0, Bosnia–Țările de Jos0–0, Țările de Jos–Norvegia0–0. Fără scoruri cu diferență≥4:85,20%; fără eliminări≤70:88,83%. Diagnosticele nu schimbă clasamentul brut.

[DBU](https://dbu.dk/nyheder/2026/november/u21-truppen-er-klar-til-playoff-kampene-mod-holland/) confirmă Lars Knudsen, lotul și reveniri după convocări la seniori/accidentări. [KNVB](https://www.onsoranje.nl/nieuws/jong-oranje-mannen/82819/kees-smit-mist-uitwedstrijd-jong-oranje-m-tegen-denemarken) confirmă că Kees Smit ratează turul. Michael Reiziger conduce oaspeții. Se joacă la Vejle, gazdă reală, cu retur la Almere pe17 noiembrie. XI, încărcarea individuală și planul tactic final rămân neconfirmate. Prognoză: ${weather('6416831')}. Condițiile mai blânde decât la Tampere sunt un contraargument în comparație, dar frecvența binară mai slabă limitează preferința.

## Leganés–Burgos, 21:30 RO — candidat cornere și preferință principală

**91,25% brut / 82,01% stres; fair matematic1,0959.** Sunt25/26 succese istorice și10/11 pe teren. Leganés acasă obține20/6 = 3,33 cornere1H și cedează11/6 = 1,83; Burgos în deplasare obține7/5 = 1,40 și cedează17/5 = 3,40. Intensitățile sunt4,03846 sezon,4,98333 teren și4,51090 combinate. Numai-sezon94,65%, numai-teren86,84%: configurația terenului este mai puțin favorabilă decât media generală. Varianța Burgos în deplasare6,20 depășește media4,80.

Istoricul păstrează Leganés–Mallorca exact7 și Real Oviedo–Burgos9. Nu eliminăm succesul la limită sau eșecul pentru a îmbunătăți frecvența. Raportul de cornere prezintă întreaga serie.

Luis Carrión este actualul antrenor al Leganésului după schimbarea din octombrie; Sergio Francisco este confirmat de Burgos și conferința din12 noiembrie. Istoricul sezonului include și perioada vechiului antrenor al gazdelor. Gharbi și Diawara sunt la naționale; API adaugă Luca Zidane și suspendarea lui Montero, fără reconciliere independentă completă. La Burgos, Appin/Iribarne apar convocați, iar Del Cerro/Karrikaburu incerți în API. Nu prezentăm această listă drept buletin medical complet. Francisco afirmă că echipa păstrează resurse ofensive în ciuda absențelor; lipsurile nu înseamnă automat puține cornere.

Butarque este teren propriu, ora20:30 CET =21:30 RO confirmată de club. Leganés a disputat Cupa cu Trival pe11 noiembrie, deci numai două zile calendaristice înaintea partidei de azi; planul vechi indica10 noiembrie și este depășit de data efectivă din API. Burgos a jucat ultima dată pe7 noiembrie. Nu deducem minutele individuale de odihnă. Pagina secundară care pretinde XI confirmat îi include pe Gharbi/Diawara și contrazice actualitatea lotului; este respinsă ca dovadă de titularizare. FotMob marchează explicit lastStarting11. Prognoză: ${weather('5868541')}; gazonul efectiv este neverificat.

Meci de campionat, jornada14, fără calificare eliminatorie. Pozițiile exacte actuale din clasament nu au fost reverificate independent integral. Schimbarea antrenorului, absențele și calendarul aglomerat sunt riscurile principale; nu sunt transformate în procente arbitrare. Golul timpuriu poate crește sau reduce presiunea, fără relație automată cu numărul cornerelor.

Pentru Gol70, același meci are86,04% /79,31%,22/26 general și **8/11** pe teren. GF70 propriu: Leganés4/6 =0,67, Burgos5/5 =1,00. Există deschideri90/83 ale Leganésului acasă și0–0 la Real Oviedo–Burgos. Îl preferăm în piața cornerelor, fără combinarea celor două piețe.

## Celelalte modele calculate

Turcia U21–Cehia U21,18:00 RO:73,59% /65,53%,14/18 general și6/9 teren. GF70 propriu pe teren0,50/0,60; deschiderile73 și trei0–0 sunt contraargumente. TFF confirmă Rize și programul, dar loturile nu sunt verificate la profunzimea finalistelor. Nu este candidat preferat.

Salamanca–Celta,22:00 RO:69,63% /61,45%,14/22 general și5/10 teren. Celta are GF70 propriu în deplasare1/5 și cinci0–0 în ediția de campionat. Proiecția între divizii și în cupă nu este validată ca transfer. Programul Helmántico este confirmat; loturile/XI și minutele90+ din feedul inferior nu sunt confirmate independent exhaustiv. Nu este recomandat.

## Lecții aplicate din review

| Lecție | Verificare nouă și dovadă | Stare și efect |
| --- | --- | --- |
| Brut, frecvență și AR20 separate | Finlanda89,19/83,14 versus18/20 și8/10; Danemarca88,42 versus12/16 | Aplicată: preferință Finlanda în Gol70, cu incertitudine explicită |
| GF70 propriu și toate tardivele | Finlanda/Irlanda2,00/0,60; Leganés/Burgos0,67/1,00; liste complete în CALCULATIONS | Aplicată: nu atribuim ambelor echipe aceeași forță ofensivă |
| Concentrarea golurilor și baraj | Finlanda fără diferențeFT≥4:81,07%; tur și retur17 noiembrie | Aplicată: avertisment contextual, brut neschimbat |
| Competiție și identitate | Segundă internă; U21 ediția actuală; Salamanca proiecție în cupă | Aplicată: fără pooling de seniori/cluburi/ediții vechi |
| Cornere exact7/≥8 și dispersie |25/26, un exact7, un9; varianță teren2,17/6,20 | Aplicată: păstrăm coada distribuției și stresul |
| Lot actual versus XI vechi | Talvitie/Orazi/Smit indisponibili, surse actuale; conflict XI Leganés | Aplicată; XI și încărcarea individuală neverificate |
| Meteo și teren | Tampere artificial/rece/umed; Vejle și Butarque separat | Aplicată; starea efectivă a gazonului neverificată |
| Observații exacte și reprogramări | FirstHalf2–1 recuperat și reconciliat6–4; Trival jucat11 noiembrie | Aplicată; celelalte lipsuri rămân NULL |

Nu sunt demonstrate cauze tactice din rezultatele de ieri. Nu se schimbă praguri, ponderi, modele sau probabilități.

## Surse

${sourceList}
`;
fs.writeFileSync(B+'/CONTEXT.md',context);
const report=`# GOAL_BY_70 — 13 noiembrie 2026

Actualizare ${at}. **Candidat al pieței: Finlanda U21–Irlanda U21,18:15 RO,89,19% brut /83,14% stres; fair matematic1,1212.** Preferința individuală principală a zilei este candidatul de cornere Leganés–Burgos. Nu combinăm piețele și nu creăm alegeri în numele utilizatorului.

[Context și contraargumente](CONTEXT.md), [calcule complete](CALCULATIONS.md), [ultimele oficiale](LAST10_OFFICIAL.md). Cercetarea este **parțială**, fără pretenția unui top global complet.

Review-ul și lecțiile au fost predate la **${run.review_handoff_at_ro}**, înaintea scanării începute la **${run.research_started_at_ro}**. [Lecțiile citite](../Review_2026-11-12/LESSONS_FOR_NEXT_SCAN.md).

## Acoperire

Sunt29 campionate: primele20 asociații UEFA din clasamentul final2026, România/Serbia și șapte secunde urmărite. Clasamentul folosește intervalul2021/22–2025/26; nu este FIFA, clasamentul live2027 sau vechea listă de20 ligi. Programul global,29 calendare ale edițiilor actuale și25 programe ESPN valide au fost confruntate. Sunt28 ligi fără partide identificate și un meci de Segunda. Pentru patru ligi lipsește confirmarea ESPN independentă; răspunsul nesuportat nu dovedește absența meciurilor.

Separat: Salamanca–Celta în Copa del Rey; nu am identificat partide UCL/UEL/UECL în programele consultate. Totalul zilei este **56 înregistrări: două cluburi,46 internaționale cu identitate FotMob și opt suplimente**. Suplimentele sunt două partide de juniori masculin și șase de junioare. Naționalele de seniori/U21 și cele feminine sunt identități distincte; nu s-a confirmat o partidă feminină de seniori în sursele consultate, fără a demonstra absența mondială.

Au fost colectate289 istorice unice:48 de club și241 ale naționalelor. Lipsesc12 liste exacte de goluri și103 observații FirstHalf în această uniune. Pentru cele opt suplimente nu există istoric numeric complet. Sunt cinci modele Gol70 și unul de cornere; toți eligibilii calculați au fost comparați înaintea preferinței.

Nations League UEFA are numai4/2 per echipă sau2–3 generale în LigaD, AFCON în general2–4, iar CONCACAF sub minime. Amicalele folosesc separat anul2026. Arabia Saudită–Coreea de Sud are5/8 generale, dar observațiile și configurația terenului nu sunt suficiente bilateral. Cele trei U21 oficiale eligibile folosesc calificarea pentru EURO2027, etichetată2025/26 de furnizor. Nu am completat minimele5/3 cu ediții vechi, cluburi sau alte categorii.

Aruba–Martinica, Antigua–Trinidad și Curaçao–Panama erau deja terminate în dimineața RO: inventariate, excluse de la selecție, fără predicții retrospective. Rezultatele rămân goale în lista prospectivă; acestea nu sunt declarate succese. Argentina–Uruguay23UTC13 noiembrie și Saint Martin–El Salvador23UTC aparțin zilei14 în România. ANFP confirmă Colombia–Chile19:00 local Cali, adică14 noiembrie02:00 RO; calendarul secundar cu21UTC este contradictoriu. Programele UTC adiacente12/13/14 și calendarele competițiilor au fost consultate.

Suplimentele feminine U19 au conflict de clasificare Friendly versus UNCAF și unele fusuri neconfirmate. Neutralitatea din Panama și gazda desemnată sunt explicite; nu li se atribuie avantaj propriu sau istoric masculin. Conflictele Kuwait14UTC/16:30UTC, UAE14:10/14:15UTC și Chile U2017:30/18/19UTC sunt păstrate; niciunul nu este finalist. Identitatea numerică și datele exacte ale suplimentelor rămân incomplete.

Snapshotul de club este arhivat în Revisions/before_international_extension. Recuperarea FirstHalf pentru Leganés are o revizie separată; probabilitățile Gol70 și timestampurile originale sunt intacte. CSV-ul alegerilor și evenimentele utilizatorului nu au fost modificate.

## Clasament brut calculabil

${comparison}
## Lecții aplicate din review

Am verificat GF70 propriu pe teren, toate0–0/deschiderile după70, concentrarea golurilor în scoruri ample, model versus frecvență/AR20, tur/retur, loturi actuale versus XI vechi, meteo/teren, FirstHalf exact și reprogramări. Matricea dovezilor și stările aplicată/neverificată sunt în CONTEXT.md.

Finlanda are8/10 pe teren și diagnosticul81,07% fără diferențe mari de scor. Danemarca are12/16 =75% observat versus88,42% modelat. Aceste avertismente păstrează candidatul Finlanda în Gol70, dar înclină preferința individuală spre cornerele Leganés–Burgos. Nu sunt aplicate ajustări procentuale.

## Ligile verificate

${coverage}
## Inventar și motivele lipsei modelelor

${inventory}
Celelalte cluburi inferioare, amicale de club și competiții din afara domeniului sunt păstrate în exclusions.csv. Nicio absență nu este dedusă dintr-un403.

## Surse

${sourceList}
`;
fs.writeFileSync(B+'/EUROPE_ALL_MATCHES.md',report);
fs.writeFileSync(B+'/International/REPORT.md',`# Inventar internațional — 13 noiembrie 2026

${at}. Sunt46 identități FotMob și opt suplimente, cu seniori/juniori și masculin/feminin distincte. Trei U21 sunt eligibile Gol70, niciuna dintre naționale la cornere. Folosim ediția actuală și separat amicalele2026, minimum5 generale/3 pe teren, fără pooling. Acoperirea rămâne parțială din lipsa istoricului exact și a unor identități/ore.

[Raport și acoperire](../EUROPE_ALL_MATCHES.md), [context](../CONTEXT.md). Datele UTC adiacente și excluderile pentru data RO sunt explicate în raportul principal. Gazda desemnată nu demonstrează avantaj propriu; UNCAF feminin în Panama are neutralitatea sau gazda reală marcate separat.

${inventory}
## Surse

${sourceList}
`);
const series=csv(C+'/historical_corners_1h.csv');
fs.writeFileSync(C+'/ANALYSIS.md',`# CORNERS_1H_UNDER_7_5 — 13 noiembrie 2026

Actualizare ${at}. **Candidat și preferință individuală principală: Leganés–Burgos,21:30 RO;91,25% brut /82,01% stres; fair matematic1,0959.** Sunt25/26 succese istorice distincte și10/11 în configurația terenului. Nu reprezintă probabilitate calibrată, preț disponibil sau alegere atribuită utilizatorului. Finlanda U21–Irlanda U21 este candidatul separat Gol70; nu combinăm evenimentele.

Același inventar de56 înregistrări,29 campionate și extensiile cupe/naționale: un model bilateral eligibil,55 fără model. **Cercetare parțială**, fără top global complet. [Inventar](../Goal70_2026-11-13/EUROPE_ALL_MATCHES.md), [context și lecții aplicate](../Goal70_2026-11-13/CONTEXT.md). Review predat${run.review_handoff_at_ro}; scanare${run.research_started_at_ro}.

## Numărători și calcul

Succesul înseamnă maximum7 cornere cumulate FirstHalf, inclusiv45+;8 este fail. Intensitatea este50% general +50% teren. Folosim PoissonCDF(7,λ), iar stresul multiplicăλ cu1,2.

${table(['Eșantion','n','Obținute','Cedate','≤7','exact7','≥8','Media','Varianța'],['home','away','home_venue','away_venue'].map(k=>[k,corner[k+'_n'],corner[k+'_gf'],corner[k+'_ga'],corner[k+'_wins'],corner[k+'_seven'],corner[k+'_eight'],Number(corner[k+'_mean']).toFixed(4),Number(corner[k+'_variance']).toFixed(4)]))}
λ sezon${corner.lambda_season}; λ teren${corner.lambda_venue}; λ combinat${corner.lambda_combined}. Numai-sezon94,65%, numai-teren86,84%, brut91,25%, stres82,01%. Diferența dintre sezon și teren este un avertisment, nu un motiv de schimbare a ponderilor. Varianța Burgos în deplasare6,20 depășește media4,80; păstrăm Poisson ca baseline.

Istoricul include **Leganés–Mallorca exact7** și **Real Oviedo–Burgos9**. Sunt păstrate atât reușita la limită, cât și eșecul. Seriile complete apar mai jos.

Observația Valladolid–Leganés2–1 FirstHalf a fost recuperată din arhiva cu perioade explicite, apoi confruntată cu două prezentări web. SecondHalf4–3 se reconciliază cu All6–4. Nu este o deducție din FT. Versiunea anterioară și motivul sunt în Goal70/Revisions/corner_period_recovery_*. Aceasta este prima probabilitate eligibilă a pieței pentru meci, fără rescrierea unei estimări vechi. Independența furnizorilor din amonte ai paginilor web nu este demonstrată.

## Context, riscuri și alternative

Campionat intern, Butarque teren propriu, antrenorii Carrión/Francisco actuali. Istoricul include vechiul antrenor Leganés. Convocările, loturile incomplete și numai două zile calendaristice după Cupa disputată pe11 noiembrie sunt riscuri. XI nu este confirmat; o pagină secundară îi include contradictoriu pe jucători plecați la naționale. Nu presupunem că absențele sau un gol timpuriu reduc automat cornerele.

Față de candidatul Gol70 din Tampere, această preferință are istoric bilateral complet în aceeași ligă și25/26 succese observate. Totuși, terenul86,84% și stresul82,01% nu permit prezentarea91,25% drept rată demonstrată. Vremea, odihna, absențele și sursele contrare sunt documentate în CONTEXT.md.

Salamanca are FirstHalf numai pentru Celta, fără bilateralitate. U21 oficiale nu furnizează FirstHalf în feedul consultat; celelalte naționale sunt sub minime ori au observații lipsă. Pentru opt suplimente de juniori/junioare istoricul exact nu este complet colectat. Frecvența unei singure echipe nu devine model bilateral.

## Lecții aplicate din review

Aplicate: perioade și identitate reconciliate; lipsa rămâne NULL; exact7/≥8 și varianța; aceiași numitori pentru frecvență/model; campionat versus cupă; lot actual versus XI vechi; schimbarea antrenorului; data efectivă Trival11 noiembrie; prognoză separată. Neverificate: XI definitiv, încărcare individuală, stare observată a gazonului și buletine medicale complete. Nicio ajustare arbitrară sau schimbare de model.

## Istoric complet pe echipă

${table(['Echipa','Data','Meci','Teren','Total FirstHalf','Obținute / cedate','Sursă'],series.map(h=>[h.team,h.date,h.home+'–'+h.away,h.venue,h.total1h,h.team===h.home?h.home_cf+'/'+h.away_cf:h.away_cf+'/'+h.home_cf,'[date]('+h.source+')']))}
## Surse

${sourceList}
`);
let review=`# Review — 12 noiembrie 2026

Review-ul și lecțiile au fost salvate inițial la${run.review_handoff_at_ro}, înaintea scanării din13 noiembrie${run.research_started_at_ro}. Această revizie editorială${at} păstrează observațiile și predarea inițială; copia anterioară este arhivată.

**Nu există alegeri explicite ale utilizatorului pentru12 noiembrie și nici alegeri active mai vechi fără rezultat.** CSV-ul și evenimentele dashboard au fost citite; revocările nu sunt recreate. Candidatul analistului Islanda U21–Grecia U21 a ratat Gol70, primul gol89. Cornere: NO BET. Cele trei naționale analizate ulterior erau o comparație contextuală, fără model oficial și fără alegere utilizator.

Surse prospective: [inventarul extins](../Goal70_2026-11-12/International/REPORT.md), [Islanda–Grecia](../Goal70_2026-11-12/International/ICELAND_GREECE_CONTEXT.md), [context cluburi](../Goal70_2026-11-12/CONTEXT.md), [comparația celor trei](../Goal70_2026-11-12/Three_Internationals_Comparison/COMPARISON.md) și CSV-urile canonice. Surse finale: FotMob/*.json și independent_sources.json, cu sursă/timestamp în fiecare rând de rezultat.
`;
for(const [market,key]of [['goal70','goal'],['corners1h','corner']]){const rows=csv(R+'/'+market+'_outcomes.csv'),v=rows.filter(r=>['success','fail'].includes(r.result)),m=v.filter(r=>r.p_raw);review+=`\n## ${market}\n\n${v.filter(r=>r.result==='success').length} succese /${v.length} rezultate verificate, ${v.filter(r=>r.result==='fail').length} fail; ${rows.filter(r=>r.result==='pending').length} pending din${rows.length} înregistrări. Separat modele prospective:${m.filter(r=>r.result==='success').length}/${m.length} succese verificate. Numitorul exclude lipsurile.\n\n`+table(['Meci / competiție','Brut / AR20 / rang','Istoric / teren','Observație','Rezultat','Sursă'],rows.map(r=>[r.home+'–'+r.away+' / '+r.competition,percent(r.p_raw)+' / '+percent(r.p_stress)+' / '+(r.rank||'fără model'),key==='goal'?r.first_goal_success+'/'+r.first_goal_n+' ; '+r.venue_success+'/'+r.venue_n:r.unique_under+'/'+r.unique_n+' ; '+r.venue_under+'/'+r.venue_unique_n,key==='goal'?r.first_goal_minute||(r.result==='fail'?'0–0 verificat':'indisponibil'):r.actual_corners_1h===''?'indisponibil':r.actual_corners_1h,r.result,'[date]('+(r.result_source||r.source)+')']));}
review+=`
## Eșecuri și avertismente cunoscute înainte de joc

**Islanda U21–Grecia U21:** rang1,93,92% brut /89,35% AR20;18/20 general,9/10 teren, zece meciuri per echipă și cinci pe teren. GF70 propriu11/5 și10/5. Erau cunoscute turul de baraj, returul16 noiembrie,0–0 Elveția–Islanda, primul gol83 Letonia–Grecia, concentrarea golurilor contra adversarilor slabi și diagnosticul89,98% fără diferențeFT≥4. Frigul și rafalele erau documentate; XI și încărcarea de la cluburi nu erau confirmate. Rezultatul a fost1–1, primul gol89.

[UEFA](https://www.uefa.com/under21/news/02aa-21c4db5accc1-d56c5d87162f-1000--2027-under-21-euro-play-offs-greece-draw-in-iceland/) relatează ulterior ninsoare și un penalty ratat înaintea deschiderii scorului. Sunt fapte post-meci, fără a demonstra că vremea sau o tactică anume au cauzat eșecul. Nu devin predictori retrospectivi și nu justifică o probabilitate nouă.

Celelalte patru failuri Gol70 nu aveau model, AR20 sau rang: Lituania–Suriname0–0, amicale2026, istoric disponibil unilateral4/5 și3/3 pe teren; Guineea Ecuatorială–Zimbabwe0–0, AFCON,4/4 și2/2 sub minime; Kazahstan U21–Belarus U21 gol72, amicale U21, numai1/1 general; Germania U19–Suedia U19 gol74, calificări UEFA, fără istoric numeric colectat. Cauzele tactice sunt necunoscute; contextul lor nu fusese cercetat ca pentru finaliste. Sursa DFB confirmă golurile74/85/89 ale ultimului meci.

Failurile cornerelor sunt Montenegro–Latvia8 și Côte d’Ivoire–Gambia9. Ambele erau în inventar fără model; probabilitate, AR20 și rang indisponibile. Eșantioanele incomplete/sub minime erau salvate prospectiv în tabel, fără excluderi inventate după rezultat. Totalurile nu demonstrează cauza presiunii sau tactica.

## Succese și reușite aproape de prag

Navalcarnero–Betis gol5 și Puertollano–Albacete gol16 sunt cele două succese modelate Gol70, compatibile cu88,25/81,97% și82,36/75,04%. Rămân riscurile cunoscute: proiecție ligă→cupă, stadion diferit, rotație/convocări și frecvențe pe teren7/10, respectiv9/12. Succesul nu validează calibrarea sau transferul între competiții.

Navalcarnero a necesitat prelungiri: lista completă se reconciliază cu2–3, apoi golul96 este exclus din scorul90=2–2. Tarancón–Levante are1–1 la90 și primul gol53; fără model. Helperul inițial respingea listele care conțineau prelungiri; am recuperat observațiile în review fără schimbare de producție. La cornerele acestor meciuri, All include prelungirile și nu se reconciliază cu primele două reprize; rămân pending pentru validare suplimentară.

Anglia–Croația gol24, Turcia–Belgia gol20 și Italia–Franța gol10 sunt succese ale comparației contextuale. Aveau numai4/2 în actuala Nations League, fără probabilități oficiale. Nu sunt alegeri utilizator și nu dovedesc superioritatea ordinii contextuale.

Montenegro–Latvia primul gol70 este succes exact la prag. Armenia–Cipru și Cayman–Guyana au câte7 cornere1H, succese la limită. Candidatul cornere nu existase; Puertollano avea numai istoricul Albacete13/13, fără bilateralitate.

## Restanțe și limite

Toate reîncercările sunt salvate în older_retries.csv. Artis–Jablonec11 noiembrie rămâne fără FirstHalf verificabil; pagina secundară consultată arată numai6–6FT. Persistă lipsuri FirstHalf și conflictele de reprogramare Serbia/Badajoz. Las Palmas–Eldense a fost jucat pe9 noiembrie; observațiile nu sunt mutate implicit în cohorta8 noiembrie. FSS a recuperat Serbia U21–Macedonia de Nord U21 gol19, sursă oficială, fără a modifica probabilități inexistente.

Rămân14 rezultate Gol70 și25 FirstHalf fără verificare suficientă în inventarul12 noiembrie, în special suplimente fără identitate numerică completă. Două semnalări fuseseră neconfirmate. Nu sunt0–0, zero cornere sau succese presupuse. Confirmarea independentă exhaustivă lipsește; inventarul internațional de ieri nu demonstra acoperirea tuturor categoriilor feminine. Acesta este un review parțial, fără concluzii de profit, calibrare sau cauzalitate.
`;
fs.writeFileSync(R+'/REVIEW.md',review);
fs.writeFileSync(R+'/LESSONS_FOR_NEXT_SCAN.md',`# Lecții pentru scanarea din13 noiembrie 2026

Predare inițială **${run.review_handoff_at_ro}**, înaintea scanării${run.research_started_at_ro}. Reformulare editorială${at}; constatările și predarea originală sunt păstrate în arhivă. Surse: REVIEW.md, CSV rezultate, FotMob/, independent_sources.json; rapoartele prospective12 noiembrie și lecțiile cumulative11 noiembrie.

**Constatare:** Islanda U21–Grecia U21 a ratat Gol70 cu primul gol89, în ciuda93,92/89,35%,18/20 general și9/10 teren. Nu existau alegeri explicite. Cele trei succese din comparația naționalelor nu validează probabilități necalculate. Inventarul și modelele au numitori diferiți.

Verificări concrete cerute noilor finaliști:

- Compară brut, frecvență binară cu numitori și AR20; nu trata intensitatea mare drept rată validată.
- Arată GF70 propriu pe teren și toate0–0/deschiderile după70.
- Verifică concentrarea golurilor în scoruri ample și sensibilități separate, fără înlocuirea prognozei.
- Documentează tur/retur, egal acceptabil, ediția actuală și amicalele2026 separat; minime5/3 fără pooling.
- Separă seniori/juniori și masculin/feminin; gazdă desemnată versus teren neutru/real.
- La cornere, păstrează exact7/≥8, dispersia și perioadele explicite; All poate include prelungiri.
- Verifică antrenori, convocări versus XI, încărcare la cluburi, meteo actuală și status neînceput.

**Constatare:** două rezultate de cupă au fost inițial respinse de validarea strictă90; lista completă cu prelungiri trebuie reconciliată înaintea excluderii golurilor>90. Scorul120 nu înlocuiește scorul90. Șapte cornere sunt succes la limită, opt fail.

**Ipoteze de testat cronologic:** concentrarea golurilor după primul, prudența de baraj și vremea pot limita transferul; cauzalitatea rămâne necunoscută. Penaltyul ratat și ninsoarea aflate post-meci nu sunt predictori retrospectivi.

**Lipsă date:** rezultate ale suplimentelor de juniori, FirstHalf, validare independentă exhaustivă, restanțe și conflicte vechi. Rapoartele noi trebuie să arate aplicată/neaplicabilă/neverificată pentru fiecare lecție, cu dovezi și limite. Nu se creează praguri, ponderi, filtre sau calibrări noi după această zi.

## Lecții cumulative anterioare

`+fs.readFileSync('Analysis/Review_2026-11-11/LESSONS_FOR_NEXT_SCAN.md','utf8'));
fs.writeFileSync(H+'/2026-11-13.md',`# Jurnal PRED — 13 noiembrie 2026

Cererea este rularea zilnică autorizată pentru ambele piețe, cu toate naționalele și fără agenți paraleli. Ora inițială verificată09:31 RO. Au fost citite instrucțiunile, nota de reluare, jurnalul12, CSV-ul și evenimentele actuale. Zero alegeri explicite12 noiembrie, zero alegeri active vechi fără rezultat. CSV-ul alegerilor nu a fost modificat.

Review12: Gol7027 succese/5 fail/14 pending; separat modele2/3. Cornere19 succese/2 fail/25 pending, fără modele prospective. Islanda U21–Grecia U21, candidatul analistului, fail cu primul gol89; probabilitățile93,92/89,35% păstrate. Anglia/Turcia/Italia, comparație contextuală: prime24/20/10, succes fără probabilități oficiale sau alegeri. Restanțele au fost reîncercate; Artis FirstHalf rămâne pending. Prelungirile Navalcarnero/Tarancón au fost reconciliate, primele goluri5/53 recuperate; DFB confirmă Germania U19 gol74 fail, FSS Serbia U21 gol19 succes.

Predare review/lecții${run.review_handoff_at_ro}; scanare${run.research_started_at_ro}. Ordinea este păstrată. Revizia editorială ulterioară este arhivată separat și nu mută predarea înaintea scanării.

Acoperire29 ligi după clasamentul final UEFA2026 plus România/Serbia/secundele urmărite:28 fără partide, un meci Segunda; separat un Copa.46 naționale cu identitate FotMob și opt suplimente, inclusiv șase de junioare;56 înregistrări total.289 istorice unice,95 oficiale recente pentru10 echipe. Ediții/genuri/categorii separate; date UTC adiacente filtrate RO. Cinci modele Gol70, unul de cornere. Snapshotul inițial de club este arhivat. FirstHalf Valladolid–Leganés2–1 a fost recuperat din arhiva period-array și reconciliat6–4, apoi coroborat web; revizia păstrează vechiul NULL și nu schimbă modele de producție.

Candidat Gol70: Finlanda U21–Irlanda U21,18:15 RO,89,19/83,14%,18/20 general și8/10 teren. GF70 propriu2,00/0,60; diagnosticul fără scoruri ample81,07%. Riscuri: tur de baraj, vreme rece/umedă, Talvitie/Orazi indisponibili, XI neconfirmat. Alternativă Danemarca–Țările de Jos88,42/82,18%,12/16 și6/8, Smit indisponibil.

Candidat cornere și preferință individuală principală: Leganés–Burgos,21:30 RO,91,25/82,01%,25/26 și10/11, fair1,0959. Istoricul păstrează Leganés–Mallorca7 și Real Oviedo–Burgos9. Numai-teren86,84%; riscuri de convocări, antrenor schimbat și Cupa disputată11 noiembrie. Nu este bilet combinat și nu s-a creat alegere utilizator.

Lecții aplicate: GF propriu, tardive/0–0, scoruri ample, competiții distincte, FirstHalf/dispersie/exact7/8, lot actual versus XI vechi, vreme/teren/program. Limite:12 liste de goluri și103 FirstHalf lipsă în uniunea istorică; suplimente incomplete; patru confirmări ESPN indisponibile; XI, încărcare individuală și unele fusuri/clasificări neconfirmate. Cercetare parțială, fără top global complet sau calibrare demonstrată.

Actualizate Review12, Goal70/Corners1H13, CSV canonice, jurnal, run_report și dashboard. Verificarea finală/exportul privat se adaugă la predare. Modelele, selection_engine, programarea și Git sunt păstrate; fără pariuri, mize sau mesaje externe. Următorul pas14 noiembrie: alegeri noi explicite → rezultate13/restanțe → review/lecții → scanare. Nu repeta researchul la unlock.
`);
console.log(JSON.stringify({status:'readable_reports_saved',at_ro:at,archive:rev}));
