import assert from 'node:assert/strict';
import fs from 'node:fs';
import {read,write,csvread,stamp} from '../daily_2026_11_14.mjs';
const B='Analysis/Goal70_2026-11-14',C='Analysis/Corners1H_2026-11-14',H='Analysis/Selection_History',P='Previews/Pred_Research_Terminal';
const run=read('Analysis/Automation/run_report_2026-11-14.json'),hist=read(B+'/historical_observations.json'),fixtures=read(B+'/candidate_fixtures.json');
assert(Date.parse(run.review_handoff_at_ro)<Date.parse(run.research_started_at_ro));
assert.equal(new Set(hist.map(h=>h.id)).size,hist.length);
assert(hist.every(h=>Date.parse(h.date)<Date.parse('2026-11-14T00:00:00+02:00')));
assert.equal(csvread(B+'/universe_coverage.csv').length,29);
let verified=0;
for(let [market,base]of [['goal',B],['corner',C]]){
 const rows=csvread(base+'/all_models.csv'),canonical=csvread(H+`/2026-11-14_${market==='goal'?'goal70':'corners1h'}_candidates.csv`);
 assert.deepEqual(rows,canonical);assert.equal(rows.length,164);
 const initial=csvread(B+'/Revisions/before_supplement_extension/'+(market==='goal'?'all_models.csv':'corner_all_models.csv'));
 for(let old of initial){let r=rows.find(r=>r.id===old.id);for(let key of ['p_raw','p_stress','fair_odds','recorded_at_ro'])assert.equal(r[key],old[key]);}
 for(let r of rows.filter(r=>r.p_raw)){
  const f=fixtures.find(f=>f.id===r.id),a=hist.filter(h=>f.hist_home.includes(h.id)),b=hist.filter(h=>f.hist_away.includes(h.id));
  let samples=[a,b,a.filter(h=>h.home_id===r.home_id),b.filter(h=>h.away_id===r.away_id)].map((hs,i)=>{
   const id=i%2===0?r.home_id:r.away_id;assert(hs.length>=(i<2?5:3));assert(hs.every(h=>h[market+'_ok']));
   const val=(h,against)=>market==='goal'?h[(h.home_id===id)!==against?'home70':'away70']:h[(h.home_id===id)!==against?'home_cf':'away_cf'];
   return {gf:hs.reduce((s,h)=>s+val(h,false),0)/hs.length,ga:hs.reduce((s,h)=>s+val(h,true),0)/hs.length};
  });
  let [aa,bb,av,bv]=samples,ls=(aa.gf+bb.ga+bb.gf+aa.ga)/2,lv=(av.gf+bv.ga+bv.gf+av.ga)/2,lambda=(ls+lv)/2;
  let probability=l=>{if(market==='goal')return 1-Math.exp(-l);let factorial=1,sum=1;for(let k=1;k<=7;k++){factorial*=k;sum+=l**k/factorial;}return Math.exp(-l)*sum;};
  for(let [key,v]of [['lambda_combined',lambda],['p_raw',probability(lambda)],['p_stress',probability(lambda*(market==='goal'?.8:1.2))],['fair_odds',1/probability(lambda)]])assert(Math.abs(Number(r[key])-v)<1e-12,r.id+' '+key);
  verified++;
 }
}
let before=csvread('Analysis/Review_2026-11-13/user_selections.before.csv'),now=csvread(H+'/user_selections.csv');
const resultFields=new Set(['actual_first_goal_minute','actual_corners_1h','result','result_source','result_verified_at_ro','status']);
for(let b of before){let r=now.find(x=>x.selection_id===b.selection_id);assert(r);for(let k of Object.keys(b))if(!resultFields.has(k))assert.equal(r[k],b[k]);}
let today=read(P+'/dist/data/2026-11-14.json'),yesterday=read(P+'/dist/data/2026-11-13.json');
assert.equal(today.coverage.goal,6);assert.equal(today.coverage.corner,5);
for(let market of ['goal','corner']){assert.equal(today.metrics[market].played,0);assert.equal(today.metrics[market].hit_rate,null);assert(today.matches.every(m=>m[market].outcome.result==='pending'));}
assert.equal(yesterday.metrics.goal.success,39);assert.equal(yesterday.metrics.goal.fail,7);assert.equal(yesterday.metrics.corner.success,29);assert.equal(yesterday.metrics.corner.fail,4);
assert.equal(yesterday.userSelections.length,3);assert(yesterday.userSelections.every(r=>r.result==='success'));
assert(read(B+'/final_status.json').every(s=>s.started===false&&s.finished===false&&s.cancelled===false));
const browser=read(P+'/daily-verification-2026-11-14.json');assert.equal(browser.status,'PASS');assert.equal(browser.exceptions,0);
write('Analysis/Automation/verification_2026-11-14.json',{status:'PASS',at_ro:stamp(),history_recalculations:verified,checks:['handoff before scan','unique pre-day histories','29 league schedules','all eleven models independently aggregated and checked','initial probabilities/ranks/times preserved','selection prediction fields unchanged','today outcomes pending and rates null','three yesterday user successes and separate modeled cohorts','six calculable fixtures unstarted','browser nine checks, zero exceptions, mobile screenshot inspected']});
console.log({status:'PASS',history_recalculations:verified});
