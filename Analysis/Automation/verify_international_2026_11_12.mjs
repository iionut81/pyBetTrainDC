import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {csv,read,save,ro} from './research_2026_11_12.mjs';
const p='Previews/Pred_Research_Terminal',b='Analysis/Goal70_2026-11-12',c='Analysis/Corners1H_2026-11-12',ib=b+'/International',rev=b+'/Revisions/before_international_extension',H='Analysis/Selection_History';
const today=read(p+'/dist/data/2026-11-12.json'),run=read('Analysis/Automation/run_report_2026-11-12.json');
assert(Date.parse(run.review_handoff_at_ro)<Date.parse(read(ib+'/scope.json').requested_at_ro));
assert.equal(today.matches.length,46);assert.equal(today.coverage.goal,3);assert.equal(today.coverage.corner,0);
assert.equal(today.matches.filter(m=>m.analystGoal).length,1);assert.equal(today.matches.find(m=>m.analystGoal).id,'6416832');assert(today.matches.every(m=>!m.analystCorner));
assert(today.matches.every(m=>m.goal.outcome.result==='pending'&&m.corner.outcome.result==='pending'));assert.equal(today.metrics.goal.hit_rate,null);assert.equal(today.metrics.corner.hit_rate,null);
const nat=read(ib+'/historical_observations.json');assert.equal(nat.length,153);assert(nat.every(r=>Date.parse(r.date)<Date.parse('2026-11-12T00:00:00+02:00')));
const rows=csv(b+'/all_models.csv');assert.equal(rows.filter(r=>r.history_basis==='national_current_competition_edition').length,43);
for(const r of rows.filter(r=>r.p_raw!=='')){const f=p=>Number(r[p+'_gf'])/Number(r[p+'_n']),a=p=>Number(r[p+'_ga'])/Number(r[p+'_n']);const ls=(f('home')+a('away')+f('away')+a('home'))/2,lv=(f('home_venue')+a('away_venue')+f('away_venue')+a('home_venue'))/2,l=(ls+lv)/2;assert(Math.abs(Number(r.p_raw)-(1-Math.exp(-l)))<1e-12);assert(Math.abs(Number(r.p_stress)-(1-Math.exp(-l*.8)))<1e-12);assert(Math.abs(Number(r.fair_odds)-1/Number(r.p_raw))<1e-12);}
for(const [market,dir]of [['goal',b],['corner',c]]){const before=csv(rev+'/'+dir+'/all_models.csv'),after=csv(dir+'/all_models.csv');for(const old of before){const now=after.find(r=>r.id===old.id);for(const key of ['p_raw','p_stress','fair_odds','recorded_at_ro','lambda_combined'])assert.equal(now[key],old[key]);assert.equal(now.original_screening_rank,old.rank);}}
const nlf=rows.filter(r=>r.competition.startsWith('UEFA Nations League'));assert.equal(nlf.length,8);assert(nlf.every(r=>r.p_raw===''&&Number(r.home_expected)===4&&Number(r.away_expected)===4));
assert.equal(rows.find(r=>r.id==='5924778').p_raw,'');assert(rows.filter(r=>r.id.startsWith('intl_')).every(r=>r.p_raw===''));
assert.deepEqual(csv(H+'/user_selections.csv'),read('Analysis/Review_2026-11-11/user_selections_before_review.json'));
save(ib+'/verification.json',{status:'PASS',at_ro:ro(),checks:['review precedes extension','46records/43national/3goal/0corner','all3probabilities/stress/fair recomputed','153national histories pre-cutoff','all original club probability/timestamp/rank snapshots retained','8NationsLeague rejected below unchanged5/3 minimum','started and unconfirmed fixtures have no prediction','no user choice inferred and canonical selections unchanged','today pending/null rates'],national_missing_goal:nat.filter(r=>!r.goal_ok).length,national_missing_corners:nat.filter(r=>!r.corner_ok).length});
let script=fs.readFileSync(p+'/check-preview.mjs','utf8');script=script.slice(0,script.indexOf(" await check('Dashboard opens"));script=script.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(path.resolve(p))+';');
script+=`
 await check('Latest day contains extended scope', 'DATA.date==="2026-11-12"&&DATA.matches.length===46&&DATA.coverage.goal===3&&DATA.coverage.corner===0');
 await check('Only U21 is analyst candidate and no user choice is inferred', 'DATA.matches.filter(m=>m.analystGoal).length===1&&DATA.matches.find(m=>m.analystGoal).id==="6416832"&&DATA.userSelections.length===0');
 await check('Current date has no invented result strip', '!document.querySelector("#outcomeStrip")&&DATA.metrics.goal.hit_rate===null');
 await evaluate('setMarket("corner")');
 await check('Corners retain missing probability', 'DATA.matches.every(m=>m.corner.p===null)&&!document.querySelector("#outcomeStrip")');
 await page('Page.navigate',{url:pathToFileURL(path.join(here,'dist/reports/2026-11-12/goal/6416832.html')).href});
 let reportLoaded=false;for(let i=0;i<60;i++){if(await evaluate('document.body?.dataset.matchId==="6416832"')){reportLoaded=true;break;}await sleep(100);}if(!reportLoaded)throw Error('U21 report did not load');
 await evaluate('window.historySection=[...document.querySelectorAll("section")].find(s=>s.querySelector("h2")?.textContent==="Istoricul disponibil pentru acest meci")');
 await check('Report labels national competition correctly', 'document.body.textContent.includes("ediția curentă a competiției naționalelor")&&!document.body.textContent.includes("Bază: proiecție din campionatele interne")');
 await check('Both national team histories are displayed with18success2fail', 'historySection.querySelectorAll("tbody tr").length===20&&[...historySection.querySelectorAll("tbody tr")].filter(r=>r.children[3].textContent==="Fail").length===2');
 await check('Original U21 model and stress appear', 'document.body.textContent.includes("93,92%")&&document.body.textContent.includes("89,35%")');
 await evaluate('historySection.scrollIntoView()');const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-international-2026-11-12.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await check('International report fits mobile width', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));const result={status:'PASS',at_utc:new Date().toISOString(),checks,exceptions:errors.length};fs.writeFileSync(path.join(here,'international-verification-2026-11-12.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
fs.writeFileSync(p+'/.runtime/international-2026-11-12-check.mjs',script);console.log('PASS international data checks; browser script prepared.');
