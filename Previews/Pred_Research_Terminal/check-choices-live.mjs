import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
const here=path.resolve('Previews/Pred_Research_Terminal'),file=path.resolve('Analysis/Selection_History/user_selections.csv');
const before=crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
let s=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');s=s.slice(0,s.indexOf(" await check('Dashboard opens"));
s=s.replace("const here=path.dirname(fileURLToPath(import.meta.url));","const here="+JSON.stringify(here)+';');
s=s.replace("pathToFileURL(path.join(here,'dist/index.html')).href","'http://127.0.0.1:18770/Previews/Pred_Research_Terminal/dist/index.html'");
s+=`
 for(let i=0;i<80;i++){if(await evaluate('CHOICES.connected'))break;await sleep(50);}
 await check('Real dashboard is connected to persistent project storage', 'CHOICES.connected&&location.hostname==="127.0.0.1"&&!document.querySelector("#choiceConnection:not([hidden])")');
 await check('Analyst cards expose enabled explicit selection actions', 'document.querySelectorAll("#selectionCards button[data-choice]").length===2&&[...document.querySelectorAll("#selectionCards button[data-choice]")].every(b=>!b.disabled)');
 await check('All goal candidates remain accessible with a choice control', 'document.querySelectorAll("#matchesBody [data-choice]").length===DATA.coverage.total');
 const desktop=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-choices-live.png'),Buffer.from(desktop.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Real interactive dashboard has no mobile page overflow', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error('Runtime exceptions: '+JSON.stringify(errors));
 const result={status:'PASS',checked_at_utc:new Date().toISOString(),checks,exceptions:errors.length,scope:'read-only check of real dashboard; no selection button clicked'};fs.writeFileSync(path.join(here,'choices-live-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
const runtime=path.join(here,'.runtime','choice-live-runtime.mjs');fs.writeFileSync(runtime,s);await import(pathToFileURL(runtime).href);
if(crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')!==before)throw Error('Real choices changed during read-only verification');
console.log('Existing real choices preserved.');
