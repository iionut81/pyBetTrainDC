import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const here=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const site=path.resolve('Previews/Pred_Public_Dashboard'),terminal=path.resolve('Previews/Pred_Research_Terminal');
let s=fs.readFileSync(path.join(terminal,'check-preview.mjs'),'utf8');s=s.slice(0,s.indexOf(" await check('Dashboard opens"));
s=s.replace("const here=path.dirname(fileURLToPath(import.meta.url));",'const here='+JSON.stringify(terminal)+';');
s=s.replace("pathToFileURL(path.join(here,'dist/index.html')).href",JSON.stringify(pathToFileURL(path.join(site,'dist/index.html')).href));
s+=`
 await check('Shared copy has all three days and 17 current fixtures', 'ARCHIVE.days.length===3&&DATA.matches.length===17');
 await check('Shared copy has no choice mutation buttons or local launcher links', 'document.querySelectorAll("[data-choice]").length===0&&[...document.querySelectorAll("a")].every(a=>!a.getAttribute("href")?.startsWith("pred-dashboard:"))&&typeof CHOICES==="undefined"');
 await check('Both explicit existing choices are retained for consultation', 'DATA.userSelections.length===2&&document.querySelector(".goal-card").textContent.includes("Bristol")&&document.querySelector(".corner-card").textContent.includes("Kortrijk")');
 await evaluate('showDetail(DATA.matches[0].id,"goal")');
 await check('Individual match reports remain available', '[...document.querySelectorAll("#dialogContent a")].some(a=>a.textContent.includes("Raportul meciului")&&a.getAttribute("href").includes("reports/2026-11-04/goal/"))');
 await evaluate('document.querySelector("#closeDialog").click();setView("history");document.querySelector("#historyDateSelect").value="2026-11-03";document.querySelector("#historyDateSelect").dispatchEvent(new Event("change"));document.querySelector("[data-history-market=goal]").click();document.querySelector("[data-result=fail]").click()');
 await check('Archived failures and their dedicated report links are available', 'document.querySelectorAll("#historyBody .history-result.fail").length===2&&document.querySelectorAll("#historyBody .history-match-report").length===2');
 await evaluate('setView("overview")');
 const desktop=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(${JSON.stringify(path.join(site,'preview-desktop.png'))},Buffer.from(desktop.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Public view has no horizontal overflow on mobile', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error('Runtime errors: '+JSON.stringify(errors));
 const result={status:'PASS',checks,exceptions:errors.length,scope:'local read-only export, not yet deployed'};fs.writeFileSync(${JSON.stringify(path.join(site,'verification.json'))},JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.join(terminal,'.runtime','public-review-check.mjs');fs.writeFileSync(runtime,s);await import(pathToFileURL(runtime).href);
