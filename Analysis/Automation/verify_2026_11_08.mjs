import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {csv,read,save} from './research_2026_11_08.mjs';
import {loadArchive} from '../../Previews/Pred_Research_Terminal/data.mjs';
const base='Analysis/Goal70_2026-11-08',corner='Analysis/Corners1H_2026-11-08';
const h=read(base+'/historical_observations.json');
assert.equal(h.length,1974);assert.equal(new Set(h.map(m=>m.id)).size,1974);
assert(h.every(m=>Date.parse(m.date)<Date.parse('2026-11-08T00:00:00+02:00')));
const revisions=fs.readdirSync(base+'/Revisions').filter(p=>p.startsWith('before_context_')).sort();
const rev=base+'/Revisions/'+revisions[0];
function probability(l,market){if(market==='goal')return 1-Math.exp(-l);let term=Math.exp(-l),p=term;for(let k=1;k<=7;k++){term*=l/k;p+=term;}return p;}
let checked=0;
for(const market of ['goal','corner']){
 const rows=csv((market==='goal'?base:corner)+'/all_models.csv'),original=csv(rev+'/'+market+'_models.csv');
 assert.equal(rows.length,105);
 for(const r of rows){
  const prior=original.find(o=>o.id===r.id);
  for(const key of ['p_raw','p_stress','lambda_season','lambda_venue','lambda_combined','rank','recorded_at_ro'])assert.equal(r[key],prior[key],r.id+': original '+key);
  if(!r.p_raw)continue;
  for(const p of ['home','away','home_venue','away_venue']){
   const side=p.startsWith('home')?'home':'away',team=r[side+'_id'];
   const sample=h.filter(m=>m.league===r[side+'_league']&&(p.endsWith('venue')?m[side+'_id']===team:(m.home_id===team||m.away_id===team))).filter(m=>m[market+'_ok']);
   assert.equal(sample.length,Number(r[p+'_n']));
   const hg=market==='goal'?'home70':'home_cf',ag=market==='goal'?'away70':'away_cf';
   const gf=sample.reduce((s,m)=>s+(m.home_id===team?m[hg]:m[ag]),0),ga=sample.reduce((s,m)=>s+(m.home_id===team?m[ag]:m[hg]),0);
   assert.equal(gf,Number(r[p+'_gf']));assert.equal(ga,Number(r[p+'_ga']));
  }
  const ls=(+r.home_gf/r.home_n + +r.away_ga/r.away_n + +r.away_gf/r.away_n + +r.home_ga/r.home_n)/2;
  const lv=(+r.home_venue_gf/r.home_venue_n + +r.away_venue_ga/r.away_venue_n + +r.away_venue_gf/r.away_venue_n + +r.home_venue_ga/r.home_venue_n)/2;
  const l=(ls+lv)/2;
  assert(Math.abs(Number(r.p_raw)-probability(l,market))<1e-12);
  assert(Math.abs(Number(r.p_stress)-probability(l*(market==='goal'?.8:1.2),market))<1e-12);
  checked++;
 }
}

const report=read('Analysis/Automation/run_report_2026-11-08.json');assert(Date.parse(report.review_handoff_at_ro)<Date.parse(report.research_started_at_ro));
const archive=loadArchive(process.cwd()),today=archive.days.find(d=>d.date==='2026-11-08'),prev=archive.days.find(d=>d.date==='2026-11-07');
assert.equal(archive.latestDate,'2026-11-08');assert.deepEqual(today.coverage,{goal:103,corner:81,total:105});
assert.equal(today.userSelections.length,0);assert.equal(prev.userSelections.length,2);assert(prev.userSelections.every(u=>u.result==='success'));
assert.deepEqual(prev.matches.filter(m=>m.userCorner).map(m=>m.home),['Excelsior']);
assert(today.matches.every(m=>m.goal.outcome.result==='pending'&&m.corner.outcome.result==='pending'));
assert.equal(prev.reviews.goal.filter(r=>r.result==='fail').length,14);assert.equal(prev.reviews.corner.filter(r=>r.result==='fail').length,14);
assert.equal(prev.reviews.corner.filter(r=>r.result==='fail'&&r.p!=null).length,13);
const choiceBefore=read('Analysis/Review_2026-11-07/user_selections_before_review.json'),current=csv('Analysis/Selection_History/user_selections.csv');
for(const old of choiceBefore){const now=current.find(r=>r.selection_id===old.selection_id);assert(now);for(const k of ['p_raw','p_stress','raw_rank','model_snapshot_at_ro','selected_at_ro'])assert.equal(now[k],old[k]);}
for(const day of archive.days)for(const m of day.matches)for(const k of ['goal','corner'])if(m[k].report)assert(fs.existsSync('Previews/Pred_Research_Terminal/dist/'+m[k].report));
const result={status:'PASS',models_recomputed_from_raw_counts:checked,history_unique_and_before_prediction:true,original_probabilities_ranks_timestamps_preserved:true,review_before_scan:true,choices_preserved:true,unmodeled_failure_preserved:true,today_results_pending:true,all_report_links_exist:true,verified_at_utc:new Date().toISOString()};
save('Analysis/Automation/verification_2026-11-08.json',result);console.log(JSON.stringify(result));
