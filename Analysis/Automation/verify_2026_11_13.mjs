import fs from 'node:fs';import assert from 'node:assert/strict';
import {read,csv,save,ro} from './research_2026_11_13.mjs';
const B='Analysis/Goal70_2026-11-13',C='Analysis/Corners1H_2026-11-13',H='Analysis/Selection_History',P='Previews/Pred_Research_Terminal',run=read('Analysis/Automation/run_report_2026-11-13.json');
assert(Date.parse(run.review_handoff_at_ro)<Date.parse(run.research_started_at_ro));
const hist=read(B+'/historical_observations.json'),g=csv(B+'/all_models.csv'),c=csv(C+'/all_models.csv');
assert.equal(hist.length,289);assert.equal(new Set(hist.map(h=>h.id)).size,hist.length);assert(hist.every(h=>Date.parse(h.date)<Date.parse('2026-11-13T00:00:00+02:00')));
assert.equal(g.length,56);assert.equal(c.length,56);assert.equal(g.filter(x=>x.p_raw).length,5);assert.equal(c.filter(x=>x.p_raw).length,1);
const cdf=l=>{let t=Math.exp(-l),s=t;for(let k=1;k<=7;k++){t*=l/k;s+=t;}return s;};let checked=0;
for(const [market,rows]of [['goal',g],['corner',c]])for(const r of rows.filter(r=>r.p_raw)){
 const samples=[];for(const [side,venue]of [['home',false],['away',false],['home',true],['away',true]]){const id=r[side+'_id'],rows=hist.filter(h=>h.league===r[side+'_league']&&(venue?h[side+'_id']===id:h.home_id===id||h.away_id===id));assert(rows.every(h=>h[market+'_ok']));assert(rows.length>=(venue?3:5));let gf=0,ga=0;for(const h of rows){const home=h.home_id===id;gf+=Number(h[market==='goal'?(home?'home70':'away70'):(home?'home_cf':'away_cf')]);ga+=Number(h[market==='goal'?(home?'away70':'home70'):(home?'away_cf':'home_cf')]);}samples.push({n:rows.length,gf,ga});}
 const [a,b,av,bv]=samples,ls=(a.gf/a.n+b.ga/b.n+b.gf/b.n+a.ga/a.n)/2,lv=(av.gf/av.n+bv.ga/bv.n+bv.gf/bv.n+av.ga/av.n)/2,l=(ls+lv)/2,prob=market==='goal'?l=>1-Math.exp(-l):cdf;
 for(const [field,value]of [['lambda_season',ls],['lambda_venue',lv],['lambda_combined',l],['p_raw',prob(l)],['p_stress',prob(l*(market==='goal'?.8:1.2))],['fair_odds',1/prob(l)]])assert(Math.abs(Number(r[field])-value)<1e-12,r.id+' '+field);checked++;
}
const oldB=B+'/Revisions/before_international_extension/'+B+'/all_models.csv';for(const o of csv(oldB)){const r=g.find(r=>r.id===o.id);for(const k of ['p_raw','p_stress','fair_odds','recorded_at_ro','lambda_combined'])assert.equal(r[k],o[k]);assert.equal(r.original_screening_rank,o.rank);}
for(const id of ['6416833','6416831','5868541'])assert(read(B+'/FinalStatus/'+id+'.json').unstarted);
const dateRO=s=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Bucharest'}).format(new Date(s));
for(const row of csv(B+'/universe_coverage.csv')){const s=read(B+'/Schedules/'+row.league+'.json');assert(s.ok);assert.equal(s.fixtures.filter(f=>dateRO(f.date)==='2026-11-13').length,Number(row.identified_fixtures));}
assert.equal(csv(B+'/universe_coverage.csv').length,29);
assert.deepEqual(csv(H+'/user_selections.csv'),read('Analysis/Review_2026-11-12/user_selections_before_review.json'));
const today=read(P+'/dist/data/2026-11-13.json'),yesterday=read(P+'/dist/data/2026-11-12.json');assert.equal(today.matches.length,56);assert.equal(today.coverage.goal,5);assert.equal(today.coverage.corner,1);assert.equal(today.userSelections.length,0);assert.equal(today.matches.filter(m=>m.analystGoal).length,1);assert.equal(today.matches.filter(m=>m.analystCorner).length,1);
for(const market of ['goal','corner']){assert.equal(today.metrics[market].played,0);assert.equal(today.metrics[market].hit_rate,null);assert(today.matches.every(m=>m[market].outcome.result==='pending'));}
assert.equal(yesterday.metrics.goal.success,27);assert.equal(yesterday.metrics.goal.fail,5);assert.equal(yesterday.metrics.goal.scored,3);assert.equal(yesterday.metrics.corner.success,19);assert.equal(yesterday.metrics.corner.fail,2);
const ice=yesterday.matches.find(m=>m.id==='6416832');assert(ice.analystGoal);assert.equal(ice.goal.outcome.result,'fail');assert.equal(ice.goal.p,0.9391899373747821);
save('Analysis/Automation/verification_2026-11-13.json',{status:'PASS',at_ro:ro(),independent_history_recalculations:checked,checks:['review handoff before scan','56 unique records,5 goal/1 corner models','289 unique histories before prediction-day cutoff','6 raw/stress/fair formulas recalculated from event histories','original club goal probabilities/timestamps/ranks preserved','29 actual-season schedules checked in Romanian date','3 finalists unstarted at final check','canonical choices unchanged','today pending/null success rates','yesterday27/32 goal and19/21 corners,model cohort distinct','Iceland analyst failure preserved original estimate']});
let script=fs.readFileSync(P+'/.runtime/daily-2026-11-12-runtime.mjs','utf8');const start=script.indexOf(" await check('Today3fixtures"),end=script.indexOf(' const screenshot=');assert(start>0&&end>start);
script=script.slice(0,start)+`
 await check('Today56records5goal1corner','DATA.date==="2026-11-13"&&DATA.matches.length===56&&DATA.coverage.goal===5&&DATA.coverage.corner===1');
 await check('Candidates distinct from choices','DATA.userSelections.length===0&&DATA.matches.filter(m=>m.analystGoal).length===1&&DATA.matches.filter(m=>m.analystCorner).length===1');
 await check('Today review pending','document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-13"&&!document.querySelector("#outcomeStrip")');
 await evaluate('setMarket("corner")');
 await check('Corner candidate correct','DATA.matches.find(m=>m.id==="5868541").corner.p===0.9125132671314271&&!document.querySelector("#outcomeStrip")');
 await evaluate('selectDay("2026-11-12");setMarket("goal")');
 await check('Yesterday analyst failed and all outcomes shown','DATA.metrics.goal.success===27&&DATA.metrics.goal.fail===5&&DATA.matches.find(m=>m.id==="6416832").goal.outcome.result==="fail"');
 await evaluate('setMarket("corner")');
 await check('Yesterday corner21verified','DATA.metrics.corner.success===19&&DATA.metrics.corner.fail===2');
 await evaluate('selectDay("2026-11-09");setMarket("corner")');
 await check('Historical actual user choice pinned','filteredMatches()[0].id==="5868534"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"');
 await evaluate('selectDay("2026-11-13");setMarket("corner")');
 await check('Return today clears old results','!document.querySelector("#outcomeStrip")');
`+script.slice(end);script=script.replaceAll('preview-daily-2026-11-12.png','preview-daily-2026-11-13.png').replaceAll("date:'2026-11-12'","date:'2026-11-13'").replaceAll('daily-verification-2026-11-12.json','daily-verification-2026-11-13.json');fs.writeFileSync(P+'/.runtime/daily-2026-11-13-runtime.mjs',script);console.log('PASS static coherence and6 independent history model recalculations; browser verification prepared');
