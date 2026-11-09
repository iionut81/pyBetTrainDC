import fs from 'node:fs';
import assert from 'node:assert/strict';
import {csv,read,save,writeCsv,ro} from './research_2026_11_09.mjs';
import {loadArchive} from '../../Previews/Pred_Research_Terminal/data.mjs';
const b='Analysis/Goal70_2026-11-09',c='Analysis/Corners1H_2026-11-09',R='Analysis/Review_2026-11-08';
const history=read(b+'/historical_observations.json');
assert.equal(new Set(history.map(r=>r.id)).size,history.length);
assert(history.every(r=>Date.parse(r.date)<Date.parse('2026-11-09T00:00:00+02:00')));
function p(l,k){if(k==='goal')return 1-Math.exp(-l);let t=Math.exp(-l),s=t;for(let i=1;i<=7;i++){t*=l/i;s+=t;}return s;}
let models=0;
for(const k of ['goal','corner']){
 const rows=csv((k==='goal'?b:c)+'/all_models.csv'),original=csv(b+'/Revisions/initial_three_fixture_scan/'+k+'_models.csv');
 assert.equal(rows.length,4);
 for(const r of rows){
  const old=original.find(x=>x.id===r.id);if(old)for(const key of ['p_raw','p_stress','lambda_season','lambda_venue','lambda_combined','recorded_at_ro','rank'])assert.equal(r[key],old[key],key+':'+r.id);
  if(!r.p_raw)continue;
  for(const pref of ['home','away','home_venue','away_venue']){
   const side=pref.startsWith('home')?'home':'away',id=r[side+'_id'];
   const rs=history.filter(m=>m.league===r[side+'_league']&&(pref.endsWith('venue')?m[side+'_id']===id:m.home_id===id||m.away_id===id)).filter(m=>m[k+'_ok']);
   const a=k==='goal'?'home70':'home_cf',z=k==='goal'?'away70':'away_cf';
   assert.equal(rs.length,+r[pref+'_n']);
   assert.equal(rs.reduce((s,m)=>s+(m.home_id===id?m[a]:m[z]),0),+r[pref+'_gf']);
   assert.equal(rs.reduce((s,m)=>s+(m.home_id===id?m[z]:m[a]),0),+r[pref+'_ga']);
  }
  const ls=(+r.home_gf/r.home_n + +r.away_ga/r.away_n + +r.away_gf/r.away_n + +r.home_ga/r.home_n)/2;
  const lv=(+r.home_venue_gf/r.home_venue_n + +r.away_venue_ga/r.away_venue_n + +r.away_venue_gf/r.away_venue_n + +r.home_venue_ga/r.home_venue_n)/2,l=(ls+lv)/2;
  assert(Math.abs(p(l,k)-Number(r.p_raw))<1e-12);assert(Math.abs(p(l*(k==='goal'?.8:1.2),k)-Number(r.p_stress))<1e-12);models++;
 }
}
const current=csv('Analysis/Selection_History/user_selections.csv'),before=read(R+'/user_selections_before_review.json');
for(const prior of before){const now=current.find(x=>x.selection_id===prior.selection_id);assert(now);for(const key of ['p_raw','p_stress','raw_rank','model_snapshot_at_ro','selected_at_ro'])assert.equal(now[key],prior[key]);}
assert(current.filter(r=>r.analysis_date==='2026-11-08').every(r=>r.result==='success'));assert.equal(current.filter(r=>r.analysis_date==='2026-11-09').length,0);
const report=read('Analysis/Automation/run_report_2026-11-09.json');assert(Date.parse(report.review_handoff_at_ro)<Date.parse(report.research_started_at_ro));
const archive=loadArchive(process.cwd()),today=archive.days.find(d=>d.date==='2026-11-09'),yesterday=archive.days.find(d=>d.date==='2026-11-08');
assert.equal(archive.latestDate,'2026-11-09');assert.deepEqual(today.coverage,{goal:4,corner:3,total:4});assert.equal(today.matches.filter(m=>m.analystGoal).length,0);assert.equal(today.matches.filter(m=>m.analystCorner).length,1);
assert(today.matches.every(m=>m.goal.outcome.result==='pending'&&m.corner.outcome.result==='pending'));assert.equal(today.userSelections.length,0);
assert.equal(yesterday.reviews.goal.filter(r=>r.result==='success').length,87);assert.equal(yesterday.reviews.goal.filter(r=>r.result==='fail').length,15);assert.equal(yesterday.reviews.corner.filter(r=>r.result==='success').length,87);assert.equal(yesterday.reviews.corner.filter(r=>r.result==='fail').length,12);
assert.equal(today.matches.find(m=>m.id==='5868533').corner.p,null);
for(const d of archive.days)for(const m of d.matches)for(const k of ['goal','corner'])if(m[k].report)assert(fs.existsSync('Previews/Pred_Research_Terminal/dist/'+m[k].report));
const out={status:'PASS',models_recomputed_from_raw_counts:models,historical_matches:history.length,review_before_scan:true,original_predictions_preserved:true,user_choices_preserved:true,today_results_pending:true,missing_is_null:true,all_report_links_exist:true,verified_at_ro:ro()};save('Analysis/Automation/verification_2026-11-09.json',out);console.log(JSON.stringify(out));
