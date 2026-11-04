import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {makeFixture} from './choice-test-fixture.mjs';
import {startDashboardServer} from './server.mjs';

const here=path.resolve('Previews/Pred_Research_Terminal'),root=path.join(here,'.runtime','choice-qa-'+Date.now()),copy=path.join(root,'Previews/Pred_Research_Terminal');
const actual=path.resolve('Analysis/Selection_History/user_selections.csv'),before=crypto.createHash('sha256').update(fs.readFileSync(actual)).digest('hex');
fs.mkdirSync(copy,{recursive:true});makeFixture(root);
for(const file of ['build.mjs','data.mjs','match-reports.mjs','choices.js','app.js','views.js','style.css','index.template.html'])fs.copyFileSync(path.join(here,file),path.join(copy,file));
const rebuild=()=>execFileSync(process.execPath,[path.join(copy,'build.mjs')],{cwd:root,windowsHide:true,stdio:'pipe'});rebuild();
const {server,url}=await startDashboardServer({root,port:0,now:()=>new Date('2026-11-04T10:00:00Z'),rebuild});
try{
 let s=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');s=s.slice(0,s.indexOf(" await check('Dashboard opens"));
 s=s.replace("const here=path.dirname(fileURLToPath(import.meta.url));","const here="+JSON.stringify(here)+';');
 s=s.replace("pathToFileURL(path.join(here,'dist/index.html')).href",JSON.stringify(url));
 s=s.replace("const [port,endpoint]=fs.readFileSync(active,'utf8').trim().split(/\\r?\\n/);","let portInfo;for(let i=0;i<40;i++){try{const value=fs.readFileSync(active,'utf8').trim();if(value.split(/\\r?\\n/).length>=2){portInfo=value;break;}}catch{}await sleep(50);}if(!portInfo)throw Error('Browser endpoint not readable');const [port,endpoint]=portInfo.split(/\\r?\\n/);");
 s+=`
 async function until(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await sleep(50);}throw Error('Timed out: '+expression);}
 await until('CHOICES.connected');
 await check('Choices initially belong to the analyst, not the user', 'DATA.userSelections.length===0&&document.querySelector(".goal-card [data-choice-action=select]")!==null');
 await evaluate('document.querySelector(".goal-card [data-choice-action=select]").click()');
 await until('!CHOICES.busy&&DATA.userSelections.length===1');
 await check('Card selection shows user badge, pending status, and remove control', 'document.querySelector(".goal-card .choice-result").classList.contains("pending")&&document.querySelector(".goal-card [data-choice-action=remove]")!==null&&document.querySelector("#matchesBody tr").dataset.match==="g1"');
 await evaluate('document.querySelector(".corner-card [data-choice-action=select]").click()');
 await until('!CHOICES.busy&&DATA.userSelections.length===2');
 await check('Two explicit market choices coexist and use the final corner estimate', 'DATA.matches[0].userGoal&&DATA.matches[0].userCorner&&DATA.matches[0].corner.userSelection.p===0.96');
 await page('Page.reload');await until('Boolean(window.PRED_PREVIEW)&&CHOICES.connected');
 await check('Both choices survive reload from project history', 'DATA.userSelections.length===2&&document.querySelector(".goal-card [data-choice-action=remove]")!==null&&document.querySelector(".corner-card [data-choice-action=remove]")!==null');
 await evaluate('document.querySelector(".goal-card [data-choice-action=remove]").click()');
 await until('!CHOICES.busy&&DATA.userSelections.length===1');
 await check('Undo affects only the selected goal choice', '!DATA.matches[0].userGoal&&DATA.matches[0].userCorner');
 await evaluate('document.querySelector("#matchesBody tr[data-match=g2] [data-choice-action=select]").click()');
 await until('!CHOICES.busy&&DATA.userSelections.length===2');
 await check('Table action selects alternative without opening a dialog and pins its original rank', 'DATA.matches.find(m=>m.id==="g2").userGoal&&!document.querySelector("#detailDialog").open&&document.querySelector("#matchesBody tr").dataset.match==="g2"&&document.querySelector("#matchesBody tr .rank").textContent==="02"');
 await evaluate('setView("history");document.querySelector("#historyDateSelect").value="2026-11-04";document.querySelector("#historyDateSelect").dispatchEvent(new Event("change"));document.querySelector("[data-history-market=goal]").click()');
 await check('Direct browser choice also appears in history', 'document.querySelector("#historyUserCards").textContent.includes("Test C")&&document.querySelector("#historyUserCards .choice-result").classList.contains("pending")');
 await evaluate('setView("overview");selectDay("2026-11-03")');
 await check('Verified historical choice is locked', '!document.querySelector(".goal-card [data-choice-action=remove]")&&document.querySelector(".goal-card .pick-locked")!==null');
 await evaluate('selectDay("2026-11-04")');
 await page('Emulation.setDeviceMetricsOverride',{width:1600,height:1200,deviceScaleFactor:1,mobile:false});
 const desktop=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-choices-desktop.png'),Buffer.from(desktop.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Mobile retains the choices without page overflow', 'document.documentElement.scrollWidth<=390&&document.querySelectorAll(".selection-card [data-choice-action=remove]").length===2');
 if(errors.length)throw Error('Runtime errors: '+JSON.stringify(errors));
 const result={status:'PASS',checked_at_utc:new Date().toISOString(),checks,exceptions:errors.length,scope:'isolated test fixture; no real user choice selected or removed'};fs.writeFileSync(path.join(here,'choices-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
 const runtime=path.join(here,'.runtime','choice-browser-runtime.mjs');fs.writeFileSync(runtime,s);await import(pathToFileURL(runtime).href);
}finally{await new Promise(resolve=>server.close(resolve));}
if(crypto.createHash('sha256').update(fs.readFileSync(actual)).digest('hex')!==before)throw Error('Real choice history was modified by test');
console.log('Real user selections unchanged (SHA-256 verified).');
