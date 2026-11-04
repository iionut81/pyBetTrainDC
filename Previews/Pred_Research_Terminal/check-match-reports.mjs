import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { loadArchive } from './data.mjs';
import { matchReportHref } from './match-reports.mjs';

const here=path.resolve('Previews/Pred_Research_Terminal');
const archive=JSON.parse(fs.readFileSync(path.join(here,'dist/snapshot.json'),'utf8'));
const original=loadArchive(process.cwd());
let reports=0;
for(const day of archive.days)for(const m of day.matches)for(const market of ['goal','corner']){
 assert.equal(m[market].report,matchReportHref(day.date,market,m.id));
 const file=path.join(here,'dist',m[market].report),html=fs.readFileSync(file,'utf8');
 assert.ok(html.includes(`data-match-id="${m.id}" data-market="${market}" data-date="${day.date}"`));
 for(const [,href]of html.matchAll(/href="([^"]+)"/g))if(!/^https?:/i.test(href))assert.ok(fs.existsSync(path.resolve(path.dirname(file),href.replaceAll('&amp;','&'))),`Broken local link: ${file} -> ${href}`);
 delete m[market].report;reports++;
}
archive.generatedAt=original.generatedAt;
assert.deepEqual(archive,original,'Reports must not mutate predictions, results or choices');
assert.throws(()=>matchReportHref('2026-11-04','goal','../escape'));
console.log(`${reports} distinct reports and their local links verified; original archive preserved.`);

let harness=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');
harness=harness.slice(0,harness.indexOf(" await check('Dashboard opens"));
harness=harness.replace("const here=path.dirname(fileURLToPath(import.meta.url));","const here=path.resolve('Previews/Pred_Research_Terminal');");
harness=harness.replace('expression,returnByValue:true,awaitPromise:true','expression,returnByValue:true,awaitPromise:true,userGesture:true');
harness+=`
 await evaluate('showDetail("6106288","goal")');
 await check('Porto goal modal links to its own report, with the day report separate', '(()=>{const links=[...document.querySelectorAll("#dialogContent a")];return links.find(a=>a.textContent.includes("Raportul meciului"))?.getAttribute("href")==="reports/2026-11-04/goal/6106288.html"&&links.some(a=>a.textContent.includes("Raportul zilei")&&a.getAttribute("href").includes("EUROPE_ALL_MATCHES.md"))})()');
 const dashboardSession=pageSession;
 await evaluate('[...document.querySelectorAll("#dialogContent a")].find(a=>a.textContent.includes("Raportul meciului")).click()');
 let reportTarget;
 for(let i=0;i<40;i++){reportTarget=(await send('Target.getTargets')).targetInfos.find(t=>t.url.endsWith('/reports/2026-11-04/goal/6106288.html'));if(reportTarget)break;await sleep(100);}
 if(!reportTarget)throw Error('Click did not open the match report');
 pageSession=(await send('Target.attachToTarget',{targetId:reportTarget.targetId,flatten:true})).sessionId;
 await page('Page.enable');await page('Runtime.enable');
 for(let i=0;i<40;i++){if(await evaluate('document.body?.dataset.matchId==="6106288"'))break;await sleep(100);}
 await check('Actual opened page is Porto goal report, without PSV analysis', 'document.body.dataset.market==="goal"&&document.querySelector("h1").textContent.includes("Porto")&&document.querySelector("h1").textContent.includes("Napoli")&&!document.body.textContent.includes("PSV")&&document.body.textContent.includes("89,72%")&&document.body.textContent.includes("83,80%")');
 await check('Porto report contains saved inputs and ten observations for each team', 'document.body.textContent.includes("2,275")&&document.querySelectorAll("h3").length===3&&[...document.querySelectorAll("h3")].filter(h=>h.textContent.includes("10 observații")).length===2');
 await check('Context limits and missingness explicitly disclosed', 'document.body.textContent.includes("nu adaugă o analiză contextuală nouă")&&document.body.textContent.includes("necalibrate")&&document.body.textContent.includes("nu zero")');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-match-report.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Match report is usable on mobile without page overflow', 'document.documentElement.scrollWidth<=390');
 pageSession=dashboardSession;
 await evaluate('document.querySelector("#closeDialog").click();showDetail("6106288","corner")');
 await check('Same fixture corner report has a distinct market-specific link', '[...document.querySelectorAll("#dialogContent a")].some(a=>a.textContent.includes("Raportul meciului")&&a.getAttribute("href")==="reports/2026-11-04/corner/6106288.html")');
 await evaluate('document.querySelector("#closeDialog").click();selectDay("2026-11-03");showDetail("401880182","corner")');
 await check('Archived user choice also opens its own match report', '[...document.querySelectorAll("#dialogContent a")].some(a=>a.textContent.includes("Raportul meciului")&&a.getAttribute("href")==="reports/2026-11-03/corner/401880182.html")');
 await evaluate('document.querySelector("#closeDialog").click();selectDay("2026-11-04");setView("history");document.querySelector("#historyDateSelect").value="2026-11-03";document.querySelector("#historyDateSelect").dispatchEvent(new Event("change"));document.querySelector("[data-history-market=goal]").click();document.querySelector("[data-result=fail]").click()');
 await check('History date differs from main date and both failed goal links use the history date', 'DATA.date==="2026-11-04"&&state.historyDate==="2026-11-03"&&document.querySelectorAll("#historyBody .history-match-report").length===2&&[...document.querySelectorAll("#historyBody .history-match-report")].every(a=>a.getAttribute("href").startsWith("reports/2026-11-03/goal/"))');
 await check('Each failed goal row retains a separate result source', 'document.querySelectorAll("#historyBody .history-report-links a[href^=https]").length===2&&[...document.querySelectorAll("#historyBody .history-report-links a[href^=https]")].every(a=>a.textContent.includes("Sursa rezultatului")&&a.href.startsWith("https://"))');
 async function openHistoryReport(selector,suffix,titleText,observation){
  await evaluate('document.querySelector('+JSON.stringify(selector)+').click()');
  let target;
  for(let i=0;i<40;i++){target=(await send('Target.getTargets')).targetInfos.find(t=>t.url.endsWith(suffix));if(target)break;await sleep(100);}
  if(!target)throw Error('History click did not open '+suffix);
  pageSession=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true})).sessionId;
  await page('Page.enable');await page('Runtime.enable');
  for(let i=0;i<40;i++){if(await evaluate('Boolean(document.body?.dataset.matchId)'))break;await sleep(100);}
  await check(titleText,'(()=>{const result=[...document.querySelectorAll("section")].find(s=>s.querySelector("h2")?.textContent==="Rezultatul verificat, separat de predicție");return document.body.dataset.date==="2026-11-03"&&result.textContent.includes("Fail")&&result.textContent.includes('+JSON.stringify(observation)+')})()');
  await check(titleText+' includes saved day review', '[...document.querySelectorAll("a")].some(a=>a.textContent.includes("Analiza succeselor")&&a.getAttribute("href").includes("Review_2026-11-03/REVIEW.md"))');
  await send('Target.closeTarget',{targetId:target.targetId});pageSession=dashboardSession;
 }
 await openHistoryReport('#historyBody a[href$="/6298366.html"]','/reports/2026-11-03/goal/6298366.html','Lyngby history click opens its own failed Gol70 report','90+3');
 await openHistoryReport('#historyBody a[href$="/401915384.html"]','/reports/2026-11-03/goal/401915384.html','Bodo history click opens its own goalless report','fără gol');
 await check('User choice card exposes its own report and result source', 'document.querySelector("#historyUserCards h3 .history-match-report").getAttribute("href")==="reports/2026-11-03/goal/401880179.html"&&document.querySelectorAll("#historyUserCards .history-report-links a").length===2');
 const historyImage=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-history-reports.png'),Buffer.from(historyImage.data,'base64'));
 await evaluate('document.querySelector("[data-history-market=corner]").click();document.querySelector("[data-result=fail]").click()');
 await check('Corner fail filter routes both matches to their archived corner reports', 'document.querySelectorAll("#historyBody .history-match-report").length===2&&[...document.querySelectorAll("#historyBody .history-match-report")].every(a=>a.getAttribute("href").startsWith("reports/2026-11-03/corner/"))');
 await openHistoryReport('#historyBody a[href$="/401880178.html"]','/reports/2026-11-03/corner/401880178.html','Watford history click opens failed ten-corner report','10 cornere 1H');
 await evaluate('document.querySelector("#historyDateSelect").value="2026-11-02";document.querySelector("#historyDateSelect").dispatchEvent(new Event("change"));document.querySelector("[data-result=all]").click()');
 await check('Changing history date updates the explicit user choice report', '(()=>{const m=ARCHIVE.days.find(d=>d.date==="2026-11-02").matches.find(m=>m.corner.userSelection);return document.querySelector("#historyUserCards h3 .history-match-report").getAttribute("href")===m.corner.report})()');
 await check('Every archived history row in both markets resolves to its own match report', '(()=>{for(const day of ARCHIVE.days)for(const market of ["goal","corner"]){state.historyDate=day.date;state.historyMarket=market;state.result="all";renderHistory();const links=[...document.querySelectorAll("#historyBody .history-match-report")];if(links.length!==day.reviews[market].length||links.some(a=>!a.getAttribute("href").startsWith("reports/"+day.date+"/"+market+"/")))return false;}return true;})()');
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await evaluate('state.historyDate="2026-11-03";state.historyMarket="goal";state.result="fail";renderHistory()');
 await check('History reports and user choice action remain available on mobile without page overflow', 'document.documentElement.scrollWidth<=390&&document.querySelectorAll("#historyBody .history-match-report").length===2&&document.querySelectorAll("#historyUserCards .history-report-links a").length===2');
 if(errors.length)throw Error('Runtime exceptions: '+JSON.stringify(errors));
 const result={status:'PASS',checked_at_utc:new Date().toISOString(),checks,exceptionCount:errors.length,scope:'all128 reports/links and archive invariants; dashboard and history actual clicks; archived failures/choices/date/market/source routing; mobile'};
 fs.writeFileSync('Analysis/Automation/match_report_verification_2026-11-04.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.resolve('Analysis/Automation/.match_reports_runtime.mjs');
fs.writeFileSync(runtime,harness);
await import('file:///'+runtime.split(path.sep).join('/'));
