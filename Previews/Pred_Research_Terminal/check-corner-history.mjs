import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const here=path.resolve('Previews/Pred_Research_Terminal');
let script=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');
script=script.slice(0,script.indexOf(" await check('Dashboard opens"));
script=script.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(here)+';');
script+=`
 await check('Dashboard archive still initializes', 'window.PRED_PREVIEW.snapshotDate===ARCHIVE.latestDate');
 await page('Page.navigate',{url:pathToFileURL(path.join(here,'dist/reports/2026-11-09/corner/5868534.html')).href});
 let reportLoaded=false;for(let i=0;i<60;i++){if(await evaluate('document.querySelector("h1")?.textContent.includes("Sabadell")&&location.href.includes("5868534.html")')){reportLoaded=true;break;}await sleep(100);}if(!reportLoaded)throw Error('Sabadell report did not load');
 await evaluate('window.cornerHistory=[...document.querySelectorAll("section")].find(s=>s.querySelector("h2")?.textContent==="Istoricul disponibil pentru acest meci")');
 await check('Both team history tables show20verified outcomes', 'cornerHistory.querySelectorAll("tbody tr").length===20&&[...cornerHistory.querySelectorAll("tbody tr")].every(r=>r.children[3].textContent==="Succes")');
 await check('Explicit first-half home away totals are visible', '[...cornerHistory.querySelectorAll("tbody tr")].every(r=>/^[0-9]+ [+] [0-9]+ = [0-9]+$/.test(r.children[2].textContent))');
 await check('Saved observations no longer incorrectly read unavailable', '!/Date 1H indisponibile|Neverificat/.test(cornerHistory.textContent)');
 await check('Current match still awaits its actual result', 'document.body.textContent.includes("Încă neverificat")&&document.body.textContent.includes("94,84%")');
 await evaluate('cornerHistory.scrollIntoView()');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-corner-history.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Report fits mobile viewport', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));
 const result={status:'PASS',checks,exceptions:errors.length,verified_at_utc:new Date().toISOString()};fs.writeFileSync(path.join(here,'corner-history-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.join(here,'.runtime','corner-history-check.mjs');fs.mkdirSync(path.dirname(runtime),{recursive:true});fs.writeFileSync(runtime,script);await import(pathToFileURL(runtime).href);
