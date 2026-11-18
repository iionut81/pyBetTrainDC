import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const here=import.meta.dirname;
let script=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');
script=script.slice(0,script.indexOf(" await check('Dashboard opens"));
script=script.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(here)+';');
script+=`
 await evaluate('selectDay("2026-11-12");setMarket("corner");showDetail("6386074","corner")');
 await check('Dashboard detail names Albacete as the only frequency source', 'document.querySelector("#dialogContent .dialog-pills").textContent.includes("Doar Albacete: 13/13")&&document.querySelector("#dialogContent .dialog-pills").textContent.includes("Doar Albacete · deplasare: 7/7")');
 await check('Dashboard detail explains Puertollano missing first-half counts', 'document.querySelector("#dialogContent .history-coverage").textContent.includes("CS Puertollano: 0 din 10")&&document.querySelector("#dialogContent .history-coverage").textContent.includes("nu din istoricul ambelor echipe")');
 await check('Missing bilateral model remains missing', 'DATA.matches.find(m=>m.id==="6386074").corner.p===null&&DATA.coverage.corner===0');
 await evaluate('document.querySelector("#detailDialog").close();selectDay("2026-11-11");showDetail("6420192","corner")');
 await check('Fully verified Artis history does not receive a partial warning', '!document.querySelector("#dialogContent .history-coverage")&&document.querySelector("#dialogContent .dialog-pills").textContent.includes("Istoric: 26/28")');
 await page('Page.navigate',{url:pathToFileURL(path.join(here,'dist/reports/2026-11-12/corner/6386074.html')).href});
 let reportLoaded=false;for(let i=0;i<60;i++){if(await evaluate('document.body?.dataset.matchId==="6386074"')){reportLoaded=true;break;}await sleep(100);}if(!reportLoaded)throw Error('Puertollano report did not load');
 await check('Report metrics identify unilateral season and venue frequencies', '[...document.querySelectorAll(".metric span")].some(e=>e.textContent==="Doar Albacete")&&[...document.querySelectorAll(".metric span")].some(e=>e.textContent==="Doar Albacete · deplasare")');
 await check('Report explains counts before history tables', 'document.querySelector(".history-coverage").textContent.includes("CS Puertollano: 0 din 10")&&document.querySelector(".history-coverage").textContent.includes("Albacete: 13 din 13")');
 await evaluate('window.historySection=[...document.querySelectorAll("section")].find(s=>s.querySelector("h2")?.textContent==="Istoricul disponibil pentru acest meci")');
 await check('Fixture count no longer implies validated corner observations', 'historySection.querySelector("h3").textContent.includes("10 meciuri în istoric · 0 cu date 1H validate")&&[...historySection.querySelectorAll("tbody")][0].rows.length===10&&[...[...historySection.querySelectorAll("tbody")][0].rows].every(r=>r.children[3].textContent==="Neverificat")');
 await check('Albacete observations stay verified', '[...[...historySection.querySelectorAll("tbody")][1].rows].every(r=>r.children[3].textContent==="Succes")');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-partial-history.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await check('Partial history report fits mobile width', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));const result={status:'PASS',checks,exceptions:errors.length,at_utc:new Date().toISOString()};fs.writeFileSync(path.join(here,'partial-history-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.join(here,'.runtime','partial-history-check.mjs');fs.mkdirSync(path.dirname(runtime),{recursive:true});fs.writeFileSync(runtime,script);await import(pathToFileURL(runtime).href);
