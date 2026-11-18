import fs from 'node:fs';import {read,write,stamp} from '../daily_2026_11_18.mjs';
const B='Analysis/Goal70_2026-11-18',R='Analysis/Review_2026-11-17',at=stamp(),file='Analysis/Automation/run_report_2026-11-18.json';
let report=read(file),data=read(B+'/data-verification.json'),browser=read(B+'/dashboard-verification.json');
if(data.status!=='PASS'||browser.status!=='PASS')throw Error('Verification incomplete');
Object.assign(report,{dashboard_status:'complete',reported_at_ro:at,process_status:'complete',research_complete:false,verification:{data_checks_passed:data.checks.length,browser_checks_passed:browser.checks.length,browser_exceptions:browser.exceptions,models_recomputed:data.models_recomputed,desktop_screenshot_inspected:true,mobile_overflow:false,data_path:B+'/data-verification.json',browser_path:B+'/dashboard-verification.json'},conversation_export:read('Analysis/Conversation_History/transcripts/current-metadata.json')});
// Keep private export metadata out of the public handoff; only coverage counts belong here.
report.conversation_export={message_count:report.conversation_export.message_count,source_count:report.conversation_export.source_files?.length,scope:'Visible messages available locally only; private export, not a complete transcript.'};
report.missing_data=['Review 17: 77 Gol70 / 89 cornere pending; 53 identități restante reîncercate fără rezultate noi pentru piața restantă.','273 istorice: patru liste de goluri și 138 observații FirstHalf lipsă sau nereconciliate.','Patru ligi fără confirmare ESPN independentă; acoperirea mondială a naționalelor feminine și de juniori rămâne incompletă.','XI, loturi bilaterale, tactică recentă, minute individuale, gazon și eliminări incomplete; transferul campionat–cupă nu este validat.'];
write(file,report);
const lessons=`# Lecții pentru 18 noiembrie 2026

Predare **${read(R+'/handoff.json').at_ro}**, înaintea scanării. Corectură editorială ${at}, fără schimbarea concluziilor sau a ordinii. [Review și surse](REVIEW.md), [lecții cumulative](../Review_2026-11-16/LESSONS_FOR_NEXT_SCAN.md), [auditul sistemelor](../System_Audit_2026-11-17/REPORT.md).

|Statut|Dovadă|Verificarea cerută în scanarea nouă|
|---|---|---|
|Constatare|Irlanda a reușit la62; model95,02% versus19/21 istoric și91,06% în sensibilitate|Compară brut, AR20 și frecvențe; arată GF70 propriu, toate0–0/>70 și ultimele10oficiale. Succesul nu validează rangul.|
|Constatare|Cinci modele reușite; șapte fail-uri nemodelate|Separă inventarul, modelele, analistul și utilizatorul. Nu transforma o preferință în alegere explicită.|
|Constatare|Kenya68; Danemarca, Burundi și Liberia exact7; patru fail-uri la8/10cornere|Păstrează exact7/≥8 și dispersia. Golul timpuriu nu garantează puține cornere.|
|Constatare|Irlanda2–1la90,2–2AET; conflict108/110pentru golul din prelungiri|Reconciliere90 separat de ET/shootout;45+ inclus, observațiile lipsăNULL.|
|Constatare|NationsLeague avea5general/2teren; fără modelecornere eligibile|Ediție curentă, minime5/3, categorii și genuri distincte, amicale2026 separat, teren neutru explicit; fără pooling.|
|Ipoteză|Concentrarea golurilor și variația cornerelor pot slăbi clasamentul|Diagnostic separat și validare cronologică; fără challengeri în producție sau praguri noi.|
|Lipsă date|Loturi, XI, încărcare, gazon și suplimente incomplete|Verifică antrenori, lot, miză, tur/retur, odihnă și vreme; păstrează contraargumentele și cauzele necunoscute.|

Rapoartele zilei noi trebuie să precizeze aplicată/neaplicabilă/neverificată, dovada și efectul asupra preferinței. Ponderile50/50 și stresul−20%gol/+20%cornere rămân neschimbate. Auditul și protocolul din17 sunt cercetare izolată, fără autorizarea modificării modelului zilnic.
`;
write(B+'/Revisions/lessons_before_final_editorial.md',fs.readFileSync(R+'/LESSONS_FOR_NEXT_SCAN.md','utf8'));write(R+'/LESSONS_FOR_NEXT_SCAN.md',lessons);
fs.appendFileSync('Analysis/Selection_History/2026-11-18.md',`\nPredare finală ${at}: run_report are review/research partial și dashboard complete; procesul încheiat nu înseamnă cercetare integrală. Export privat sincronizat: două mesaje vizibile disponibile dintr-o sursă locală; nu transcript complet. Nicio modificare de producție sau programare în această rulare.\n`);
console.log({at,status:'partial',dashboard:report.dashboard_status,data_checks:data.checks.length,browser_checks:browser.checks.length});
