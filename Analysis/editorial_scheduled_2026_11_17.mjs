import fs from 'node:fs';import {read,write,csvread,stamp} from './daily_2026_11_17.mjs';const B='Analysis/Goal70_2026-11-17',C='Analysis/Corners1H_2026-11-17',at=stamp(),rev=B+'/Revisions/editorial_scheduled_'+Date.now();for(let p of [B+'/CONTEXT.md',B+'/EUROPE_ALL_MATCHES.md',B+'/COVERAGE.md',C+'/ANALYSIS.md','Analysis/Selection_History/2026-11-17.md'])write(rev+'/'+p.replaceAll('/','_'),fs.readFileSync(p,'utf8'));
const g=csvread(B+'/all_models.csv').filter(r=>r.p_raw),status=read(B+'/final_status.json'),table='| Rang brut | Meci | Ora RO | Model | AR20% | Cotă matematică | Istoric / teren |\n|---|---|---|---|---|---|---|\n'+g.map(r=>`| ${r.rank} | ${r.home}–${r.away} | ${r.kickoff_ro.slice(11,16)} | ${(100*r.p_raw).toFixed(2)}% | ${(100*r.p_stress).toFixed(2)}% | ${Number(r.fair_odds).toFixed(4)} | ${r.first_goal_success}/${r.first_goal_n} / ${r.venue_success}/${r.venue_n} |`).join('\n');
const coverage=`Cercetare **parțială**: 133 de înregistrări, dintre care 58 au identitate numerică FotMob și 75 sunt suplimentare. Zece suplimente sunt semnalări cu dată/oră/status neconfirmate. Numărul de rânduri nu este un număr de partide independent confirmate. Toate cele cinci partide eligibile Gol70 au fost calculate; la cornere nu există model eligibil. Au fost colectate 361 de partide istorice: 16 liste de goluri și 126 observații FirstHalf lipsesc sau nu se reconciliază. Lipsa nu este zero.

Au fost verificate 29 de ligi: primele ligi din top 20 al [clasamentului final UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), România, Serbia și diviziile secunde urmărite din Anglia, Germania, Spania, Italia, Franța, Belgia și Turcia. Feedul global și calendarele de sezon nu identifică partide azi în aceste ligi. Pentru 25 există și răspuns ESPN utilizabil; Polonia, Croația, Serbia și Belgia 2 păstrează limita confirmării independente. Cupele europene, FA Cup, Coupe de France, Copa del Rey și League Cup urmărite nu au partide identificate azi. EFL Trophy și celelalte competiții din feedul mondial sunt păstrate în fișierele brute, în afara setului de cupe urmărite de acest baseline.

Naționalele de seniori și juniori, masculin și feminin, rămân distincte. Amicalele folosesc numai anul 2026, separat de oficiale; calificările folosesc ediția curentă. Cele opt meciuri UEFA Nations League au 5 observații generale, dar numai 2 în configurația terenului pentru fiecare echipă: sub minimul 3. Nu se adaugă ediții vechi. Pentru suplimentele U17/U19 nu este validat istoricul exact bilateral; cele două meciuri ale mini-turneului nu sunt declarate automat întreaga ediție.

La feminin au fost făcute căutări separate și consultate paginile UEFA WU17/WU19. Paginile dinamice fără rezultate nu demonstrează absența partidelor. Nicio partidă feminină de naționale nu este confirmată independent în această etapă; acoperirea mondială rămâne incompletă. Conflictele de oră și gazdă desemnată sunt păstrate în [crosswalk](Supplemental/crosswalk.json). Japonia–Paraguay este pe teren neutru la Singapore; alte neutralități neconfirmate sunt declarate ca atare.

Inventarul și istoricul suplimentelor sunt parțiale. Nu prezentăm un top global complet. [Inventar și excluderi](INVENTORY.md), [calendare și limite pe ligă](universe_coverage.csv), [lipsuri suplimentare](Supplemental/coverage_gaps.json), [istoric validat](historical_observations.csv).`;
const applied=`## Lectii aplicate din review

- **Aplicată — model față de frecvență.** Grecia U21 a ratat ieri cu primul gol la 72, după un model de 91,60% și istoric 18/21. Pentru Irlanda afișăm separat 95,02% brut, 90,93% stres, 19/21 general, 10/10 teren și 9/10 în ultimele oficiale pentru fiecare echipă. Diferența nu este ascunsă și nu produce o corecție arbitrară.
- **Aplicată — goluri proprii și deschideri tardive.** Irlanda are GF70 propriu 10/5 acasă, Finlanda 15/5 în deplasare. Kazakhstan–Irlanda, primul 89, și Finlanda–Kosovo 0–0 rămân în istoric. Fără diferențe finale de cel puțin patru goluri, diagnosticul scade la 91,06%; baseline-ul rămâne 95,02%.
- **Aplicată — contextul barajului și lotul actual.** Turul 0–1 permite Finlandei să se califice cu egal. Sunt verificate antrenorii, înlocuirile Orazi/Cummins și Talvitie/Laine, programul și ploaia prognozată la Cork. XI, minutele individuale și gazonul efectiv rămân neverificate.
- **Aplicată — cornere exacte și dispersie.** Succesul Albacete cu 4 cornere ieri nu validează distribuția Poisson și nu anulează riscul prospectiv. Azi nu există eșantion bilateral complet 5/3; alegerea este NO BET. Totalurile exact 7 și >=8 disponibile rămân în istoricul și diagnosticele salvate.
- **Aplicată — ediții și cohorte.** Nations League 5/2 rămâne neeligibilă. Amicalele 2026 și calificările curente nu sunt amestecate; terenul neutru este declarat. **Neverificate integral:** acoperirea mondială la feminin, istoricul suplimentelor și auditul eliminărilor. Nu inventăm rezultate, cauze tactice sau probabilități.

[Lecțiile predate înaintea scanării](../Review_2026-11-16/LESSONS_FOR_NEXT_SCAN.md). Timestampurile canonice din handoff și research_start păstrează ordinea. Nu s-au schimbat formulele, ponderile, pragurile sau modelele de producție.`;
write(B+'/CONTEXT.md',`# Context și comparație — 17 noiembrie 2026

**Candidatul analistului și preferința individuală: Irlanda U21–Finlanda U21, GOAL_BY_70, 19:45 RO. Cornere 1H: NO BET.** Verificarea statusului: ${status.find(r=>r.id==='6416837').at_ro}, meci neînceput. Nicio alegere nouă a utilizatorului nu a fost creată.

## Irlanda U21–Finlanda U21

Model brut **95,02%**, scenariu nefavorabil **90,93%**, cotă echitabilă matematică **1,0524**. Acestea sunt estimări exploratorii, fără rată validată sau preț de piață. Istoric deduplicat: **19/21 = 90,48% general**, **10/10 pe teren**. Fiecare echipă are 11 meciuri în actualele calificări EURO U21 2027 și cinci în configurația terenului. Ultimele zece oficiale verificabile au câte 9/10 reușite. Turul comun se numără o singură dată în frecvența comună.

Irlanda are GF70/GA70 de 13/13 în 11 meciuri și 10/7 în cinci acasă. Finlanda are 26/3 în 11 și 15/3 în cinci deplasări. Atacul propriu pe teren este astfel 2,00 și 3,00 goluri până la 70 pe meci. Lambda: 2,50 sezon, 3,50 teren, 3,00 combinat. Numai sezon: 91,79%; numai teren: 96,98%. Kazakhstan U21–Irlanda U21 a avut primul gol la 89, iar Finlanda U21–Kosovo U21 s-a terminat 0–0. Ambele rămân în serie.

Diagnosticul fără partide cu diferență finală de cel puțin patru goluri scade la **91,06%**, pe eșantioane 10/7 general și 5/3 teren. Acesta nu înlocuiește probabilitatea salvată. Golurile numeroase împotriva adversarilor slabi pot ridica lambda; modelul numără toate golurile <=70, iar evenimentul urmărit depinde de primul gol. Diferența dintre cele două probleme rămâne o limită.

[FAI, 16 noiembrie](https://www.fai.ie/latest/preview-mu21-play-ireland-v-finland/) confirmă returul la Turner's Cross, Cork, la 17:45 GMT, adică 19:45 RO, după 0–1 în tur. Irlanda trebuie să recupereze; Finlandei îi ajunge egalul. Inferență contextuală: gazdele pot căuta atacul, iar oaspeții pot controla riscul. Miza nu garantează gol până la 70. Primul gol din tur, 45+8, este succes conform definiției. Zăpada și pauza prelungită din Tampere nu sunt transferate automat la retur.

Antrenorii Jim Crawford și Mika Lehkosuo sunt confirmați de FAI și [Palloliitto](https://api.sttinfo.fi/tiedote/72475543/pikkuhuuhkajat-nimetty-em-jatkokarsintoihin-irlantia-vastaan?publisherId=69819199). FAI actualizat îl arată pe Orazi înlocuit de Cummins. [Actualizarea finlandeză din 12 noiembrie](https://www.veikkausliiga.com/uutiset/2026/11/12/pikkuhuuhkajat-valmiina-em-jatkokarsinnan-avaukseen-lauri-laine-korvaa-juho-talvitien-joukkueessa) îl declară pe Talvitie indisponibil pentru ambele manșe, înlocuit de Lauri Laine; lista inițială din 4 noiembrie este depășită. Intenția ofensivă declarată a antrenorului nu este o intensitate măsurată. Etichetele medicale generice FotMob nu înlocuiesc actualizările lotului.

[Preview-ul finlandez din 16 noiembrie](https://www.veikkausliiga.com/uutiset/2026/11/16/em-kisapaikka-pelissa-tiistaina-pikkuhuuhkajilla-ratkaisuottelu-irlannin-vieraana), publicat de ligă pe baza Palloliitto, confirmă antrenamentele de duminică la Mayfield și luni pe Turner's Cross. Sunt patru zile calendaristice între partide și deplasarea Tampere–Cork; odihna în ore și minutele individuale nu sunt auditate. [Open-Meteo](Context/weather_6416837.json), la 17:00 UTC, cu 45 de minute înainte: 9,8°C, 1,7 mm ploaie, vânt 10,4 km/h, rafale 22,3. Gazonul efectiv, XI și starea completă a revenirilor/suspendărilor rămân neconfirmate.

Prefer Irlanda după compararea tuturor celor cinci eligibile datorită brutului/stresului superior, contribuției ofensive proprii și frecvenței pe teren. Contraargumentele sunt eșantionul mic, modelul peste frecvență, sensibilitatea la scoruri ample, avantajul Finlandei, absențele și ploaia. Eșecul Greciei U21 de ieri nu autorizează tuning și arată de ce barajul nu este o garanție.

## Alternative

${table}

- **Rusia–Venezuela:** 9/7 amicale în 2026 și 7/6 în configurația terenului; GF70 propriu 15/7 și 4/6. Rusia–Mali, Coreea de Sud–Venezuela, Uzbekistan–Venezuela și Iordania–Venezuela au fost 0–0. Diagnosticul fără scoruri ample este 83,38%. [RFS, arhiva din 16 noiembrie](https://www.rfs.ru/archive?category_id=1), confirmă pregătirea la Krasnodar și antrenorii Karpin/Viscarrondo. Lotul și XI sunt incomplet verificate; feedul recent nu a furnizat oficiale, ceea ce nu înseamnă că nu există în istorie. Modelul și frecvența sunt inferioare Irlandei.
- **Țările de Jos U21–Danemarca U21:** 9/9 general și 4/4 teren; GF70 propriu 3/4 și 5/4. Există două 0–0 olandeze, Danemarca–Belgia 0–0 și Belgia–Danemarca cu primul gol la 80. Numai teren: 77,69%; diagnosticul fără scoruri ample: 75,82%. Olandezii au avantaj 3–0 din tur. [KNVB](https://www.onsoranje.nl/nieuws/jong-oranje-mannen/82819/kees-smit-mist-uitwedstrijd-jong-oranje-m-tegen-denemarken) anunță revenirea planificată a lui Smit pentru retur, fără XI confirmat. Rafalele prognozate la Almere ajung la 50,4 km/h. Doar nouă oficiale per echipă au fost recuperate.
- **Japonia–Paraguay:** teren neutru Singapore; [JFA confirmă 21:10 local](https://jfa.jp/samuraiblue/20261117/news/00036974/), adică 15:10 RO. Baseline: 8/6 amicale, 6/5 ca părți desemnate; GF70 propriu 5/6 și 7/5. Nu pretindem avantaj real acasă. Scoția–Japonia a deschis la 84, Japonia–Islanda la 87, Japonia–Ecuador a fost 0–0. Diagnosticul este 81,50%. JFA confirmă plecarea lui Kamada pe 14 noiembrie din motive de club. Japonia a jucat pe 14, Paraguay pe 13: trei față de patru zile calendaristice. Prognoză: 26,3°C, 0,6 mm. Lotul Paraguay și XI sunt incomplet verificate. Cele 4/5 oficiale recuperate nu sunt adăugate amicalelor.
- **Cehia U21–Turcia U21:** 11/9 general și 5/4 teren; GF70 propriu 8/5 și 4/4. Cehia–Portugalia, Scoția–Cehia și Turcia–Ungaria au fost 0–0; Turcia–Ucraina a deschis la 73. Diagnosticul este 72,90%. Cehia conduce cu 1–0 după tur, ceea ce permite prudență. Modelul este ultimul dintre eligibile; lotul actual nu a fost verificat la profunzimea finalistei. Sunt disponibile 10/9 oficiale, prognoză 7,6°C și 1,4 mm.

## Cornere 1H

**NO BET:** niciun eșantion bilateral complet 5/3 în inventarul validat. Pentru Irlanda–Finlanda lipsesc observațiile FirstHalf din calificări; scorul pauzei nu furnizează cornere. Datele exacte 7/>=8 și dispersia disponibile sunt păstrate în [istoric](historical_observations.csv) și [diagnostice](diagnostics.json). Nu alegem un model neeligibil ca alternativă.

${applied}

## Acoperire și limite

${coverage}
`);
write(B+'/EUROPE_ALL_MATCHES.md',`# GOAL_BY_70 — 17 noiembrie 2026

**Candidat al analistului: Irlanda U21–Finlanda U21, 19:45 RO.** Model 95,02%, stres 90,93%, cotă matematică 1,0524. Istoric 19/21 general și 10/10 teren, cu 11 meciuri per echipă și cinci per configurație de teren. GF70 propriu: 10/5 și 15/5.

Diagnosticul fără scoruri ample scade la 91,06%. Turul 0–1, absențele, ploaia și posibilitatea prudenței Finlandei sunt contraargumente. Probabilitatea nu este o rată validată și cota matematică nu este ofertă disponibilă.

${table}

[Context, riscuri și alternative](CONTEXT.md), [calcule și numărători](CALCULATIONS.md), [ultimele oficiale](LAST10_OFFICIAL.md), [inventar și excluderi](INVENTORY.md), [surse](sources.json).

${coverage}

${applied}
`);
const cov=csvread(B+'/universe_coverage.csv');write(B+'/COVERAGE.md','# Acoperire — 17 noiembrie 2026\n\n'+coverage+'\n\n| Liga | Rang UEFA 2026 / supliment | Global | Sezon | ESPN | Acoperire |\n|---|---|---|---|---|---|\n'+cov.map(r=>`| ${r.country}/${r.league} | ${r.uefa_2026_rank} | ${r.identified_fixtures} | ${r.season_schedule_daily} | ${r.espn_fixtures===''?'indisponibil':r.espn_fixtures} | ${r.coverage} |`).join('\n'));
write(B+'/LESSONS_APPLIED.md','# Verificări aplicate\n\n'+applied);
write(C+'/ANALYSIS.md',`# CORNERS_1H_UNDER_7_5 — 17 noiembrie 2026

**NO BET.** Niciun model eligibil în cele 133 de înregistrări evaluate. Lipsesc eșantioane FirstHalf complete bilateral, minimum 5 general și 3 în configurația terenului. Sunt 126 de observații FirstHalf lipsă/neconciliate în cele 361 de istorice, iar 75 de suplimente nu au istoric exact bilateral validat.

Scorul la pauză și cornerele FT nu substituie FirstHalf. Nu publicăm probabilitate sau cotă matematică fără date. Candidatul Gol70 Irlanda–Finlanda nu devine candidat la cornere.

[Calcule și lipsuri](CALCULATIONS.md), [toate excluderile](../Goal70_2026-11-17/INVENTORY.md), [context](../Goal70_2026-11-17/CONTEXT.md).

${coverage}

${applied}
`);
for(let file of [B+'/coverage_summary.json','Analysis/Automation/run_report_2026-11-17.json']){let d=read(file);if(file.endsWith('coverage_summary.json'))d.total=133;else {if(d.counts)d.counts.total=133;d.review_handoff_at_ro=read('Analysis/Review_2026-11-16/handoff.json').at_ro;d.research_started_at_ro=read(B+'/research_start.json').at_ro;}write(file,d);}
console.log({editorial_at_ro:at,archive:rev,rows:csvread(B+'/all_models.csv').length});
