import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {csv,read,save} from './research_2026_11_06.mjs';
import {loadArchive} from '../../Previews/Pred_Research_Terminal/data.mjs';
const base='Analysis/Goal70_2026-11-06',corner='Analysis/Corners1H_2026-11-06';
const h=read(base+'/historical_observations.json');
assert.equal(h.length,705);assert.equal(new Set(h.map(m=>m.id)).size,705);
assert(h.every(m=>Date.parse(m.date)<Date.parse('2026-11-06T00:00:00+02:00')));
const revisions=fs.readdirSync(base+'/Revisions').filter(p=>p.startsWith('before_context_')).sort();
const rev=base+'/Revisions/'+revisions[0];
function probability(l,market){if(market==='goal')return 1-Math.exp(-l);let term=Math.exp(-l),p=term;for(let k=1;k<=7;k++){term*=l/k;p+=term;}return p;}
let checked=0;
for(const market of ['goal','corner']){
 const rows=csv((market==='goal'?base:corner)+'/all_models.csv'),original=csv(rev+'/'+market+'_models.csv');
 assert.equal(rows.length,34);
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
const recent=csv(base+'/last10_official.csv');
assert.equal(recent.length,80);
assert(recent.some(r=>r.team_id==='9907'&&r.date.startsWith('2026-11-03')&&r.first_goal_minute==='90+3'&&r.goal_result==='fail'));
const report=read('Analysis/Automation/run_report_2026-11-06.json');
assert(Date.parse(report.review_handoff_at_ro)<Date.parse(report.research_started_at_ro));
const archive=loadArchive(process.cwd()),today=archive.days.find(d=>d.date==='2026-11-06'),yesterday=archive.days.find(d=>d.date==='2026-11-05');
assert.equal(today.matches.filter(m=>m.userGoal||m.userCorner).length,0);
assert.equal(yesterday.userSelections.length,2);
assert(yesterday.userSelections.every(s=>s.result==='success'));
const result={status:'PASS',verified_at_utc:new Date().toISOString(),models_recomputed_from_raw_counts:checked,original_probabilities_ranks_timestamps_preserved:true,history_unique_and_before_prediction:true,last10_official_rows:80,late_cup_goal_retained:true,review_before_scan:true,user_choices_preserved:true,choice_file_sha256:crypto.createHash('sha256').update(fs.readFileSync('Analysis/Selection_History/user_selections.csv')).digest('hex')};
save('Analysis/Automation/recovery_verification_2026-11-06.json',result);
console.log(JSON.stringify(result));
