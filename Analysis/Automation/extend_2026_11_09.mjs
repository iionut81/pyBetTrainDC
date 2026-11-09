import fs from 'node:fs';
import {read,csv,save,writeCsv,fetch,api,ro} from './research_2026_11_09.mjs';
const b='Analysis/Goal70_2026-11-09', stamp=ro();
const rev=b+'/Revisions/initial_three_fixture_scan';
fs.mkdirSync(rev,{recursive:true});
for(const [src,dst] of [[b+'/candidate_fixtures.json','candidate_fixtures.json'],[b+'/all_models.csv','goal_models.csv'],['Analysis/Corners1H_2026-11-09/all_models.csv','corner_models.csv']])fs.copyFileSync(src,rev+'/'+dst);
const f=read(b+'/candidate_fixtures.json');
const d=read('Analysis/Review_2026-11-08/FotMob/5868533.json');
f.push({id:'5868533',competition:'esp.2',date:'2026-11-09T19:30:00.000Z',home:d.teams[0].name,away:d.teams[1].name,home_id:d.teams[0].id,away_id:d.teams[1].id,status:{started:false,finished:false,cancelled:false,utcTime:'2026-11-09T19:30:00.000Z'},source:d.source,history_basis:'same_league_current_season',league_id:140,schedule_source:'https://www.udlaspalmas.es/noticias/el-partido-j13-laspalmaseldense-se-disputara-manana-lunes-9-de-noviembre-a-las-1930-horas-hora-canaria-en-el-estadio-gran-canaria',schedule_conflict:'FotMob still cancelled on8Nov; ESPN and official club rescheduled9Nov19:30UTC; calculate but do not select without fresh reconciled status',schedule_verified_at_ro:stamp});
save(b+'/candidate_fixtures.json',f);
save(b+'/inventory_revision.json',{at_ro:stamp,reason:'Independent schedule check discovered Las Palmas–Eldense omitted from FotMob global. Official club confirms9Nov19:30Canary/21:30RO. Preserve initial three snapshots and calculate fourth before final ranking.',initial_fixture_count:3,revised_fixture_count:4});
// Retry older pending observations; preserve the original probabilities.
for(const market of ['goal70','corners1h']){
 const file='Analysis/Review_2026-11-07/'+market+'_outcomes.csv',rows=csv(file);
 const r=rows.find(x=>(x.id||x.fotmob_id)==='5887676');
 const m=read('Analysis/Review_2026-11-08/OlderRetry/5887676.json');
 const {observation}=await import('./research_2026_11_09.mjs');const o=observation(m),k=market==='goal70'?'goal':'corner';
 if(r&&['success','fail'].includes(o[k+'_result'])){
  fs.copyFileSync(file,'Analysis/Review_2026-11-08/'+market+'_nov07_before_retry.csv');
  Object.assign(r,{result:o[k+'_result'],reason:o[k+'_reason'],first_goal_minute:k==='goal'?o.first_goal_minute:'',actual_corners_1h:k==='corner'?o.actual_corners_1h:'',score_ft:o.score_ft,result_source:o.result_source,result_verified_at_ro:o.result_verified_at_ro});writeCsv(file,rows);
 }
}
fs.appendFileSync('Analysis/Review_2026-11-07/REVIEW.md',`\n## Restanță verificată la ${stamp}\n\nNacional–Académico Viseu, jucat8noiembrie: vezi observațiile din CSV-urile actualizate și dovada ../Review_2026-11-08/OlderRetry/5887676.json. Predicțiile originale sunt păstrate; rezultatul nou nu era cunoscut la review-ul inițial.\n`);
for(const market of ['goal70','corners1h']){
 const file='Analysis/Review_2026-11-08/'+market+'_outcomes.csv',rows=csv(file),r=rows.find(x=>x.id==='5868533');
 Object.assign(r,{result:'postponed_not_scored',reason:'Official club rescheduled from8Nov to9Nov; provider cancelled flag retained in raw evidence',result_source:f.at(-1).schedule_source,result_verified_at_ro:stamp});writeCsv(file,rows);
}
// Correct copied editorial references while retaining the initial handoff evidence.
const rp='Analysis/Review_2026-11-08/REVIEW.md';let text=fs.readFileSync(rp,'utf8');
text=text.replace('[comparație Blackburn/Málaga](../Goal70_2026-11-08/Cardiff_Malaga_Comparison/COMPARISON.md), [alternative cornere](../Corners1H_2026-11-08/Alternative_Comparison/COMPARISON.md)','[comparație Brighton/Ipswich](../Goal70_2026-11-08/Brighton_Ipswich_Comparison/COMPARISON.md), [audit descriptiv](../Pattern_Review_2026-11-08/REVIEW.md)');
text=text.slice(0,text.indexOf('## Analize originale'))+`## Analize originale și limite

Brighton: 91,62% brut /86,24% stres, rang20,18/18 sezon și8/8 teren; GF70 propriu11/4 și5/4. Riscuri prospective: 8 din11goluri acasă în2victorii ample,0–0 laJablonec și inversarea ordinii în sensibilitatea fără roșii. Primul gol15 precede roșul23: eliminarea ulterioară nu explică primul gol. [Raport oficial Brentford](https://www.brentfordfc.com/en/news/match-reports-brighton-and-hove-albion-2-brentford-0-premier-league-08-11-2026).

Girona:96,97%/92,84%,rang3,24/24 și13/13;exact7 o dată,zero≥8,varianțe2,48/0,67. Riscuri: presing agresiv, suspendareaLópez, loturi probabile. FirstHalf4+1=5,SecondHalf3+1=4,All7+2=9. Succes compatibil cu seria, fără validare a probabilității sau cauzei tactice. Confirmarea independentă exactă FirstHalf rămâne limitată.

Cambuur–PSV, finalistul originalGol70, este separat de alegereaBrighton. Riscurile sale prospective erau progresul defensivCambuur,75Leipzig–PSV și83PEC–Cambuur, loturi și recuperarePSV dupăEuropa; nu se șterg după succes. Toate observațiile și semnalele inițiale sunt în tabel.

Auditul descriptiv8noiembrie avea6alegeri pe piață și nicio ratare proprie: nu transferăm eșecurile universului către utilizator și nu tratăm seria mică drept validare. Cauzele tactice ale failurilor rămân necunoscute în lipsa analizei video/contextuale complete.

Corecție de program la ${stamp}: Las Palmas–Eldense este reprogramat9noiembrie, conform clubului, deșiFotMob îl marchează cancelled pe8noiembrie. În CSV este postponed_not_scored, exclus din rată; FCAndorra–Mallorca șiEstoril–Marítimo păstrează statusul furnizorului cancelled. TreiFirstHalfCipru rămân pending. Toate cele102 observațiiGol70 și99cornere se păstrează:87succese/15fail și87succese/12fail. Nacional–Viseu, restant7noiembrie, este actualizat separat în arhiva7noiembrie din observația verificată8noiembrie. Nicio probabilitate rescrisă.
`;
fs.writeFileSync(rp,text);
const lp='Analysis/Review_2026-11-08/LESSONS_FOR_NEXT_SCAN.md';let lessons=fs.readFileSync(lp,'utf8'),i=lessons.indexOf('## Lecții cumulative păstrate');
fs.writeFileSync(lp,`# Lecții pentru scanarea din9noiembrie2026

Predarea inițială:2026-11-09T08:59:22.313+02:00; înaintea scanării08:59:38RO. Corecție editorială a numelor/surselor la${stamp}, fără modificarea listei de verificări. Fișierul inițial rămâne înLESSONS.initial_handoff.md. Surse: [review](REVIEW.md), CSV-uri rezultate, [Gol70 prospectiv](../Goal70_2026-11-08/EUROPE_ALL_MATCHES.md), [cornere prospectiv](../Corners1H_2026-11-08/ANALYSIS.md), comparațiaBrighton/Ipswich și auditul descriptiv8noiembrie.

Constatare: Brighton primul15 șiGirona5cornere au succes. Golul15 precede roșul23; nu atribuim primul gol eliminării ulterioare. Univers:87/102Gol70 și87/99cornere; cauzele tuturor eșecurilor nu sunt demonstrate. Reușitele aproape de prag și fiecare fail sunt în tabelul review-ului.

Verificări pentru noua cercetare: brut/AR20 versus frecvențe și numitori; GF70 propriu pe teren, deschideri>70 și0–0; ultimele10oficiale, intern separat deEuropa/cupă; FirstHalf exact7/≥8, dispersie și reconciliere; loturi/antrenori actuali, odihnă, vreme, kickoffRO și status neînceput. Anulări/reprogramări trebuie confruntate cu surse oficiale, nu tratate drept0–0.

Ipoteze de testat separat: golurile multiple după deschidere pot umflaλ; schimbările de lot/antrenor pot limita transferabilitatea. Lipsă: cauzele tactice exhaustive, confirmarea independentă a fiecăruiFirstHalf, restanțeleCipru/Serbia. Un succes nu validează procentele. Nu schimbăm ponderi, distribuție, praguri sau modele și nu promitem eliminarea ratărilor.

`+(i>=0?lessons.slice(i):''));
console.log({extended:4,at_ro:stamp});
