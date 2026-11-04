import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import crypto from 'node:crypto';

const here=path.resolve('Previews/Pred_Research_Terminal');
const protectedFiles=['dist/index.html','dist/snapshot.json','../../Analysis/Selection_History/user_selections.csv'];
const digest=()=>protectedFiles.map(f=>crypto.createHash('sha256').update(fs.readFileSync(path.join(here,f))).digest('hex'));
const before=digest();
let s=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');
s=s.slice(0,s.indexOf(" await check('Dashboard opens"));
s=s.replace("const here=path.dirname(fileURLToPath(import.meta.url));","const here=path.resolve('Previews/Pred_Research_Terminal');");
s=s.replace("path.join(here,'dist/index.html')","path.join(here,'Reviews/choice-button/preview.html')");
s=s.replace("Boolean(window.PRED_PREVIEW)",'document.title.includes("Preview alegere pick")');
s+=`
 await page('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
 await page('Runtime.evaluate',{expression:'document.fonts.ready',awaitPromise:true});
 await check('Preview declares that choices are not saved', 'document.querySelector(".preview").textContent.includes("NU SALVEAZ")');
 await check('Both selection states and a table action are visible', 'document.querySelectorAll(".state-grid .card").length===2&&document.querySelectorAll(".primary").length===1&&document.querySelectorAll(".undo").length===1&&document.querySelectorAll(".table-button").length===1');
 await check('No saving logic or storage exists in the static concept', 'document.scripts.length===0');
 const desktop=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'Reviews/choice-button/preview-desktop.png'),Buffer.from(desktop.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Preview has no horizontal page overflow on mobile', 'document.documentElement.scrollWidth<=390');
 const mobile=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(path.join(here,'Reviews/choice-button/preview-mobile.png'),Buffer.from(mobile.data,'base64'));
 if(errors.length)throw Error('Preview runtime errors: '+JSON.stringify(errors));
 console.log(JSON.stringify({status:'PASS',checks,exceptions:errors.length,scope:'static concept only; no real UI or choices modified'},null,2));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.resolve('Analysis/Automation/.dashboard_daily_runtime_choice_review.mjs');
fs.writeFileSync(runtime,s);
await import(pathToFileURL(runtime).href);
if(JSON.stringify(before)!==JSON.stringify(digest()))throw Error('Real dashboard or choices changed during preview');
console.log('Real dashboard and canonical user selections unchanged (SHA-256 verified).');
