import fs from 'node:fs';
import path from 'node:path';
const source='Previews/Pred_Research_Terminal/check-preview.mjs';
let s=fs.readFileSync(source,'utf8');
s=s.slice(0,s.indexOf(" await check('Dashboard opens"));
s=s.replace("const here=path.dirname(fileURLToPath(import.meta.url));","const here=path.resolve('Previews/Pred_Research_Terminal');");
s+=`
 await check('Latest date is November4 with17 fixtures and16 corner models', 'window.PRED_PREVIEW.snapshotDate==="2026-11-04"&&window.PRED_PREVIEW.matchCount===17&&window.PRED_PREVIEW.cornerEligible===16');
 await check('All three research dates navigable', '[...document.querySelectorAll("#dateSelect option")].map(o=>o.value).includes("2026-11-02")&&[...document.querySelectorAll("#dateSelect option")].map(o=>o.value).includes("2026-11-03")');
 await check('No new user choice invented today', 'ARCHIVE.days.find(d=>d.date==="2026-11-04").userSelections.length===0');
 await check('Today outcomes pending with missing Sabah corner probability preserved', 'ARCHIVE.days.find(d=>d.date==="2026-11-04").matches.every(m=>m.goal.outcome.result==="pending"&&m.corner.outcome.result==="pending")&&ARCHIVE.days.find(d=>d.date==="2026-11-04").matches.find(m=>m.originalAway==="Sabah FK").corner.p===null');
 await check('Goal tab highlights PSV and identifies the goal analyst candidate', 'document.querySelectorAll(".radar-node[data-highlighted=true]").length===1&&document.querySelector(".radar-node[data-highlighted=true]").dataset.radarMatch==="6106303"&&document.querySelector("#radarFocus").textContent.includes("PSV")&&!document.querySelector("#radarFocus").textContent.includes("Kortrijk")');
 await check('Goal probability is on the vertical axis and PSV is the highest plotted goal probability', '(()=>{const selected=document.querySelector(".radar-node[data-highlighted=true] .node-core");return document.querySelector(".radar-svg").dataset.yMarket==="goal"&&document.querySelector(".radar-axis-title").textContent.includes("GOL")&&Number(selected.getAttribute("cy"))===Math.min(...[...document.querySelectorAll(".node-core")].map(n=>Number(n.getAttribute("cy"))))})()');
 const goalImage=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-radar-goal.png'),Buffer.from(goalImage.data,'base64'));
 await evaluate('document.querySelector(".radar-market-switch [data-market=corner]").click()');
 await check('Corner tab highlights Kortrijk and both market switchers agree', 'document.querySelectorAll(".radar-node[data-highlighted=true]").length===1&&document.querySelector(".radar-node[data-highlighted=true]").dataset.radarMatch==="5811786"&&document.querySelector("#radarFocus").textContent.includes("Kortrijk")&&document.querySelectorAll(".market-switch button.active[data-market=corner]").length===2');
 await check('Corner probability stays vertical and Kortrijk is the highest plotted corner probability', '(()=>{const selected=document.querySelector(".radar-node[data-highlighted=true] .node-core");return document.querySelector(".radar-svg").dataset.yMarket==="corner"&&document.querySelector(".radar-axis-title").textContent.includes("CORNERE")&&Number(selected.getAttribute("cy"))===Math.min(...[...document.querySelectorAll(".node-core")].map(n=>Number(n.getAttribute("cy"))))})()');
 const cornerImage=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-radar-corner.png'),Buffer.from(cornerImage.data,'base64'));
 await evaluate('document.querySelector(".radar-node[data-highlighted=true]").dispatchEvent(new MouseEvent("click",{bubbles:true}))');
 await check('Corner radar click opens corner evidence', 'document.querySelector("#detailDialog").open&&document.querySelector("#dialogEyebrow").textContent.includes("CORNERE")');
 await evaluate('document.querySelector("#closeDialog").click();document.querySelector("#matchesSection .market-switch [data-market=goal]").click()');
 await check('Goal table switch restores PSV highlight and syncs radar controls', 'document.querySelector(".radar-node[data-highlighted=true]").dataset.radarMatch==="6106303"&&document.querySelectorAll(".market-switch button.active[data-market=goal]").length===2');
 await evaluate('document.querySelector(".radar-node[data-highlighted=true]").dispatchEvent(new KeyboardEvent("keydown",{key:"Enter",bubbles:true}))');
 await check('Keyboard activation opens goal evidence in the goal tab', 'document.querySelector("#detailDialog").open&&document.querySelector("#dialogEyebrow").textContent.includes("GOL")&&document.querySelector("#dialogTitle").textContent.includes("PSV")');
 await evaluate('document.querySelector("#closeDialog").click()');
 await evaluate('document.querySelector("#dateSelect").value="2026-11-03";document.querySelector("#dateSelect").dispatchEvent(new Event("change"))');
 await check('Both verified user choices show Success', 'document.querySelector(".goal-card .choice-result").textContent==="-Succes"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"');
 await check('Archived goal market highlights only the user goal choice', 'document.querySelectorAll(".radar-node[data-highlighted=true]").length===1&&document.querySelector(".radar-node[data-highlighted=true]").dataset.radarMatch==="401880179"');
 await check('QPR is pinned first retaining model rank2', 'document.querySelector("#matchesBody tr").dataset.match==="401880179"&&document.querySelector("#matchesBody tr .rank").textContent==="02"');
 await evaluate('document.querySelectorAll(".market-switch button")[1].click()');
 await check('Derby is pinned first in corners', 'document.querySelector("#matchesBody tr").dataset.match==="401880182"');
 await evaluate('document.querySelector("#dateSelect").value="2026-11-04";document.querySelector("#dateSelect").dispatchEvent(new Event("change"))');
 await check('All17 fixtures remain navigable', 'document.querySelectorAll("#matchesBody tr[data-match]").length===17');
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Mobile has no horizontal page overflow', 'document.documentElement.scrollWidth<=390');
 const mobileImage=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-radar-mobile.png'),Buffer.from(mobileImage.data,'base64'));
 if(errors.length)throw Error('Runtime exceptions '+JSON.stringify(errors));
 const result={status:'PASS',checked_at_utc:new Date().toISOString(),checks,exceptionCount:errors.length,scope:'daily archive navigation, actual user badges, ranks, missingness, pending outcomes and radar focus/click/keyboard synchronization with the active market'};
 fs.writeFileSync('Analysis/Automation/dashboard_verification_2026-11-04.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
// The temporary check reuses only the established browser harness. It leaves
// existing interface code and the dated regression expectations untouched.
const target='Analysis/Automation/.dashboard_daily_runtime_2026_11_04.mjs';fs.writeFileSync(target,s);
await import('./.dashboard_daily_runtime_2026_11_04.mjs');
