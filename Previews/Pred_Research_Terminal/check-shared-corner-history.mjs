import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const here=import.meta.dirname;
let script=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');
script=script.slice(0,script.indexOf(" await check('Dashboard opens"));
script=script.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(here)+';');
script+=`
 await page('Page.navigate',{url:pathToFileURL(path.join(here,'dist/reports/2026-11-11/corner/6420192.html')).href});
 let reportLoaded=false;for(let i=0;i<60;i++){if(await evaluate('document.body?.dataset.matchId==="6420192"')){reportLoaded=true;break;}await sleep(100);}if(!reportLoaded)throw Error('Artis report did not load');
 await evaluate('window.historySection=[...document.querySelectorAll("section")].find(s=>s.querySelector("h2")?.textContent==="Istoricul disponibil pentru acest meci")');
 await check('Both teams display ten saved observations', 'historySection.querySelectorAll("tbody").length===2&&[...historySection.querySelectorAll("tbody")].every(t=>t.rows.length===10)');
 await check('Historical failures are visible alongside successes', '[...historySection.querySelectorAll("tbody tr")].filter(r=>r.children[3].textContent==="Succes").length===18&&[...historySection.querySelectorAll("tbody tr")].filter(r=>r.children[3].textContent==="Fail").length===2');
 await check('Each row has first-half counts and its public evidence', '[...historySection.querySelectorAll("tbody tr")].every(r=>/^[0-9]+ [+] [0-9]+ = [0-9]+$/.test(r.children[2].textContent)&&[...r.querySelectorAll("a")].some(a=>a.href.includes("matchDetails"))&&[...r.querySelectorAll("a")].some(a=>a.href.includes("fotmob.com/match/")))');
 await check('Shared history source is linked and no false missing state remains', '[...document.querySelectorAll("a")].some(a=>a.href.includes("Goal70_2026-11-11/historical_observations.csv"))&&!/Date 1H indisponibile|Istoricul individual nu este disponibil/.test(historySection.textContent)');
 await check('Original predictions and pending current result are preserved', 'document.body.textContent.includes("93,09%")&&document.body.textContent.includes("85,29%")&&document.body.textContent.includes("Încă neverificat")');
 await evaluate('historySection.scrollIntoView()');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-shared-corner-history.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Report fits mobile width', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));
 const result={status:'PASS',checks,exceptions:errors.length,verified_at_utc:new Date().toISOString()};fs.writeFileSync(path.join(here,'shared-corner-history-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.join(here,'.runtime','shared-corner-history-check.mjs');fs.mkdirSync(path.dirname(runtime),{recursive:true});fs.writeFileSync(runtime,script);await import(pathToFileURL(runtime).href);
