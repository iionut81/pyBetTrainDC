import fs from 'node:fs';import assert from 'node:assert/strict';
import {csv,read,save,ro} from './research_2026_11_12.mjs';
const p='Previews/Pred_Research_Terminal',b='Analysis/Goal70_2026-11-12',c='Analysis/Corners1H_2026-11-12',R='Analysis/Review_2026-11-11',H='Analysis/Selection_History';
const run=read('Analysis/Automation/run_report_2026-11-12.json'),today=read(p+'/dist/data/2026-11-12.json'),yesterday=read(p+'/dist/data/2026-11-11.json');
assert(Date.parse(run.review_handoff_at_ro)<Date.parse(run.research_started_at_ro));
assert.equal(today.matches.length,3);assert.equal(today.coverage.goal,2);assert.equal(today.coverage.corner,0);
assert.equal(today.userSelections.length,0);assert(today.matches.every(m=>!m.analystGoal&&!m.analystCorner));
for(const market of ['goal','corner']){assert.equal(today.metrics[market].hit_rate,null);assert.equal(today.metrics[market].played,0);assert(today.matches.every(m=>m[market].outcome.result==='pending'));}
assert.equal(yesterday.metrics.goal.success,5);assert.equal(yesterday.metrics.goal.scored,3);
assert.equal(yesterday.metrics.corner.success,2);assert.equal(yesterday.metrics.corner.pending,3);
const artis=yesterday.matches.find(m=>m.id==='6420192');assert(artis.analystCorner);assert.equal(artis.corner.outcome.result,'pending');assert.equal(artis.corner.outcome.corners,'');assert.equal(artis.corner.p,0.9308691149082812);
assert.deepEqual(csv(H+'/user_selections.csv'),read(R+'/user_selections_before_review.json'));
let models=0;
for(const [market,dir]of [['goal',b],['corner',c]])for(const r of csv(dir+'/all_models.csv')){
 if(!r.p_raw){assert.equal(r.rank,'');assert.equal(r.p_stress,'');continue;}
 const f=p=>Number(r[p+'_gf'])/Number(r[p+'_n']),a=p=>Number(r[p+'_ga'])/Number(r[p+'_n']);
 const ls=(f('home')+a('away')+f('away')+a('home'))/2,lv=(f('home_venue')+a('away_venue')+f('away_venue')+a('home_venue'))/2,l=(ls+lv)/2;
 assert(Math.abs(ls-Number(r.lambda_season))<1e-12);assert(Math.abs(lv-Number(r.lambda_venue))<1e-12);
 assert(Math.abs(1-Math.exp(-l)-Number(r.p_raw))<1e-12);assert(Math.abs(1-Math.exp(-.8*l)-Number(r.p_stress))<1e-12);assert(Math.abs(1/Number(r.p_raw)-Number(r.fair_odds))<1e-12);models++;
}
assert.equal(models,2);
const hist=read(b+'/historical_observations.json');assert.equal(hist.length,56);assert.equal(hist.filter(x=>!x.corner_ok).length,20);assert(hist.every(x=>Date.parse(x.date)<Date.parse('2026-11-12T00:00:00+02:00')));
for(const r of csv(b+'/universe_coverage.csv')){assert.equal(r.season_schedule_ok,'true');const schedule=read(b+'/Schedules/'+r.league+'.json');assert.equal(schedule.fixtures.filter(x=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Bucharest'}).format(new Date(x.date))==='2026-11-12').length,0);}
assert(csv(b+'/status_at_selection.csv').every(x=>x.identity_ok==='true'&&x.unstarted_verified==='true'));
const trival=read(p+'/dist/data/2026-11-10.json').matches.find(m=>m.id==='6386086');assert.equal(trival.corner.outcome.corners,'7');assert.equal(trival.corner.outcome.result,'success');
save('Analysis/Automation/verification_2026-11-12.json',{status:'PASS',at_ro:ro(),models_checked:models,checks:['review before scan','3fixtures 2goal 0corner models','2probabilities/stress/fair recomputed','29season schedules checked in Romanian timezone','today pending/null rates','yesterday5goal and2corner successes/3pending','Artis pending original probability preserved','user choices unchanged','56historical dates pre-cutoff/20FirstHalf missing','Trival7 recovered under original prediction cohort']});
let script=fs.readFileSync(p+'/.runtime/daily-2026-11-11-runtime.mjs','utf8');
const begin=script.indexOf(" await check('Today5fixtures"),end=script.indexOf(' const screenshot=');assert(begin>0&&end>begin);
script=script.slice(0,begin)+`
 await check('Today3fixtures2goal0corner','DATA.date==="2026-11-12"&&DATA.matches.length===3&&DATA.coverage.goal===2&&DATA.coverage.corner===0');
 await check('No choices or analyst selection','DATA.userSelections.length===0&&DATA.matches.every(m=>!m.analystGoal&&!m.analystCorner)');
 await check('Today goal pending','document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-12"&&!document.querySelector("#outcomeStrip")');
 await evaluate('setMarket("corner")');
 await check('Today corner pending','!document.querySelector("#outcomeStrip")');
 await evaluate('selectDay("2026-11-11");setMarket("goal")');
 await check('Yesterday five successes','DATA.metrics.goal.success===5&&DATA.metrics.goal.scored===3');
 await evaluate('setMarket("corner")');
 await check('Yesterday Artis still pending','DATA.metrics.corner.success===2&&DATA.metrics.corner.pending===3&&DATA.matches.find(m=>m.id==="6420192").corner.outcome.result==="pending"');
 await evaluate('selectDay("2026-11-09");setMarket("corner")');
 await check('Existing actual user choice stays pinned','filteredMatches()[0].id==="5868534"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"');
 await evaluate('selectDay("2026-11-12");setMarket("corner")');
 await check('Return today clears prior results','!document.querySelector("#outcomeStrip")');
`+script.slice(end);
script=script.replaceAll('preview-daily-2026-11-11.png','preview-daily-2026-11-12.png').replaceAll("date:'2026-11-11'","date:'2026-11-12'").replaceAll('daily-verification-2026-11-11.json','daily-verification-2026-11-12.json');
fs.writeFileSync(p+'/.runtime/daily-2026-11-12-runtime.mjs',script);
console.log('PASS data/model checks; browser verification prepared.');
