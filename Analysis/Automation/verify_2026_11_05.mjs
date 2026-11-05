import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {parseCsv,loadArchive} from '../../Previews/Pred_Research_Terminal/data.mjs';
const root=process.cwd(),read=p=>JSON.parse(fs.readFileSync(p,'utf8')),csv=p=>parseCsv(fs.readFileSync(p,'utf8'));
const archive=loadArchive(root),day=archive.days.find(d=>d.date==='2026-11-05'),prev=archive.days.find(d=>d.date==='2026-11-04');
assert.equal(archive.latestDate,'2026-11-05');assert.deepEqual(day.coverage,{goal:33,corner:21,total:35});
assert.equal(day.userSelections.length,2);assert.ok(day.userSelections.every(u=>!u.result));
assert.deepEqual(day.matches.filter(m=>m.userGoal).map(m=>m.id),['6112548']);
assert.deepEqual(day.matches.filter(m=>m.userCorner).map(m=>m.id),['6112347']);
assert.deepEqual(day.matches.filter(m=>m.analystGoal).map(m=>m.id),['6112325']);
assert.deepEqual(day.matches.filter(m=>m.analystCorner).map(m=>m.id),['6112347']);
assert.ok(prev.userSelections.every(u=>u.result==='success'));assert.equal(prev.userSelections.length,2);
assert.equal(prev.reviews.goal.filter(r=>r.result==='fail').length,1);assert.equal(prev.reviews.corner.filter(r=>r.result==='fail'&&r.p!=null).length,1);
for(const market of ['goal','corner']){assert.ok(day.matches.every(m=>m[market].outcome.result==='pending'));}
const report=read('Analysis/Automation/run_report_2026-11-05.json');assert.ok(Date.parse(report.review_handoff_at_ro)<Date.parse(report.research_started_at_ro));
const revision=fs.readdirSync('Analysis/Goal70_2026-11-05/Revisions').find(n=>n.startsWith('context_'));
for(const mkt of ['goal70','corners1h']){
 const original=csv('Analysis/Goal70_2026-11-05/Revisions/'+revision+'/'+mkt+'_canonical_before.csv');const current=csv('Analysis/Selection_History/2026-11-05_'+mkt+'_candidates.csv');
 for(const r of current){const old=original.find(x=>x.id===r.id);for(const k of ['p_raw','p_stress','rank','recorded_at_ro','kickoff_ro'])assert.equal(r[k],old[k]);assert.equal(r.outcome,'');assert.equal(r.actual_corners_1h,'');
  if(r.p_raw){assert.ok(Math.abs((Number(r.lambda_season)+Number(r.lambda_venue))/2-Number(r.lambda_combined))<1e-12);assert.ok(Math.abs(1/Number(r.p_raw)-Number(r.fair_odds))<1e-12);}}
}
// Reuse the existing browser harness, substituting only the checks for today's saved data.
const here=path.join(root,'Previews/Pred_Research_Terminal');let source=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');
source=source.slice(0,source.indexOf(" // The archived reference days"));
source=source.replace("const here=path.dirname(fileURLToPath(import.meta.url));",'const here='+JSON.stringify(here)+';');
source+=`
 await check('Today has35fixtures and both actual choices pending', 'DATA.date==="2026-11-05"&&DATA.matches.length===35&&document.querySelector("#selectionCards").textContent.includes("Mjällby")&&document.querySelector("#selectionCards").textContent.includes("Celta")');
 await check('Goal user choice pinned first with original model rank14', 'filteredMatches()[0].id==="6112548"&&DATA.matches.find(m=>m.id==="6112548").goal.p===0.8986136425527268');
 await evaluate('document.querySelector(".market-switch [data-market=corner]").click()');
 await check('Corner choice pinned first,14 missing estimates stay null', 'filteredMatches()[0].id==="6112347"&&DATA.matches.filter(m=>m.corner.p===null).length===14');
 await check('Today has no fabricated results', 'DATA.matches.every(m=>m.goal.outcome.result==="pending"&&m.corner.outcome.result==="pending")');
 await evaluate('document.querySelector("#dateSelect").value="2026-11-04";document.querySelector("#dateSelect").dispatchEvent(new Event("change"))');
 await check('Yesterday choices retain verified success', 'DATA.userSelections.length===2&&DATA.userSelections.every(u=>u.result==="success")');
 await check('Corner review has its own verified failure', 'document.querySelector("#reviewPanel").dataset.market==="corner"&&document.querySelector("#reviewPanel .failed").title.includes("Fenerbah")');
 await evaluate('document.querySelector("#dateSelect").value="2026-11-05";document.querySelector("#dateSelect").dispatchEvent(new Event("change"))');
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await sleep(200);
 await check('Mobile has no horizontal overflow', 'document.documentElement.scrollWidth<=390');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-daily-2026-11-05.png'),Buffer.from(screenshot.data,'base64'));
 if(errors.length)throw Error(JSON.stringify(errors));
 fs.writeFileSync(path.join(here,'daily-verification-2026-11-05.json'),JSON.stringify({status:'PASS',checks,exceptions:errors.length},null,2));console.log(JSON.stringify({status:'PASS',checks,exceptions:errors.length}));
}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.join(here,'.runtime/daily-verify-2026-11-05.mjs');fs.mkdirSync(path.dirname(runtime),{recursive:true});fs.writeFileSync(runtime,source);
console.log('Archive, user choices, original snapshots, missingness and arithmetic PASS.');
execFileSync(process.execPath,[runtime],{stdio:'inherit',windowsHide:true});
