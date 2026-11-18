import fs from 'node:fs';
import assert from 'node:assert/strict';
import {csv,read,save,ro} from './research_2026_11_10.mjs';
const b='Analysis/Goal70_2026-11-10',H='Analysis/Selection_History',R='Analysis/Review_2026-11-09',p='Previews/Pred_Research_Terminal';
const today=read(p+'/dist/data/2026-11-10.json'),yesterday=read(p+'/dist/data/2026-11-09.json'),run=read('Analysis/Automation/run_report_2026-11-10.json');
assert(Date.parse(run.review_handoff_at_ro)<Date.parse(run.research_started_at_ro));
assert.equal(today.matches.length,24);assert.equal(today.coverage.goal,0);assert.equal(today.coverage.corner,0);assert.equal(today.userSelections.length,0);
assert(today.matches.every(m=>m.goal.p===null&&m.corner.p===null&&!['success','fail'].includes(m.goal.outcome.result)&&!['success','fail'].includes(m.corner.outcome.result)));
assert.equal(yesterday.metrics.goal.success,2);assert.equal(yesterday.metrics.goal.fail,2);assert.equal(yesterday.metrics.corner.success,4);assert.equal(yesterday.metrics.corner.fail,0);
const user=csv(H+'/user_selections.csv').find(r=>r.analysis_date==='2026-11-09');assert.equal(user.result,'success');assert.equal(user.actual_corners_1h,'6');assert.equal(user.p_raw,'0.9483686632747383');
const before=read(R+'/user_selections_before_review.json'),after=csv(H+'/user_selections.csv'),resultFields=new Set(['result','actual_first_goal_minute','actual_corners_1h','result_source','result_verified_at_ro','status']);
for(const a of before){const z=after.find(r=>r.selection_id===a.selection_id);assert(z);for(const k of Object.keys(a))if(!resultFields.has(k))assert.equal(String(z[k]??''),String(a[k]??''));}
assert.equal(csv(b+'/universe_coverage.csv').length,29);assert(csv(b+'/universe_coverage.csv').every(r=>r.season_schedule_ok==='true'&&r.season_schedule_daily==='0'));
for(const k of ['goal70','corners1h']){const rows=csv(`${H}/2026-11-10_${k}_candidates.csv`);assert.equal(rows.length,24);assert(rows.every(r=>r.p_raw===''&&r.rank===''&&r.analyst_selected==='false'&&r.user_selected==='false'));}
const result={status:'PASS',checked_at_ro:ro(),checks:['review before scan','24cup records/no invented model or outcome','yesterday metrics2/4and4/4','user6corners success with frozen probability','all saved selection prediction fields preserved','29season schedules','no invented ranks']};save('Analysis/Automation/verification_2026-11-10.json',result);console.log(JSON.stringify(result));
let script=fs.readFileSync(p+'/.runtime/daily-2026-11-09-runtime.mjs','utf8');
const begin=script.indexOf(" await check('Today four"),end=script.indexOf(' const screenshot=');
assert(begin>0&&end>begin);
script=script.slice(0,begin)+`
 await check('Today24cup records with zero models','DATA.date==="2026-11-10"&&DATA.matches.length===24&&DATA.coverage.goal===0&&DATA.coverage.corner===0');
 await check('Today no invented results or choices','DATA.userSelections.length===0&&DATA.matches.every(m=>m.goal.p===null&&m.corner.p===null)&&!document.querySelector("#outcomeStrip")');
 await evaluate('setMarket("corner")');
 await check('Selected date pending corner','document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-10"&&!document.querySelector("#outcomeStrip")');
 await evaluate('selectDay("2026-11-09");setMarket("corner")');
 await check('Sabadell success pinned','filteredMatches()[0].id==="5868534"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"');
 await check('Yesterday4corner outcomes','document.querySelectorAll("#outcomeStrip span").length===4&&document.querySelectorAll("#outcomeStrip .failed").length===0');
 await evaluate('setMarket("goal")');
 await check('Yesterday2goal failures','document.querySelectorAll("#outcomeStrip .failed").length===2');
 await evaluate('selectDay("2026-11-10");setMarket("corner")');
 await check('Returning today clears previous results','!document.querySelector("#outcomeStrip")');
`+script.slice(end);
script=script.replaceAll('2026-11-09.png','2026-11-10.png').replaceAll("date:'2026-11-09'","date:'2026-11-10'").replaceAll('daily-verification-2026-11-09.json','daily-verification-2026-11-10.json');
fs.writeFileSync(p+'/.runtime/daily-2026-11-10-runtime.mjs',script);
