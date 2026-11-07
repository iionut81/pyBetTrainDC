import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const browser='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile=path.join(here,'.edge-preview-profile-'+Date.now());
fs.mkdirSync(profile,{recursive:true});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const proc=spawn(browser,['--headless=new','--disable-gpu','--disable-extensions','--disable-background-networking','--disable-sync','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:['ignore','ignore','pipe']});
let browserErrors='';proc.stderr.on('data',d=>{browserErrors=(browserErrors+d.toString()).slice(-4000);});
let ws;const pending=new Map();let nextId=0;const errors=[];let pageSession;
function send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++nextId;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},15000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
const page=(method,params={})=>send(method,params,pageSession);
const evaluate=async expression=>{const r=await page('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text+' '+(r.exceptionDetails.exception?.description??''));return r.result.value;};
const checks=[];
async function check(name,expression){const value=await evaluate(expression);if(value!==true)throw Error(name+': '+JSON.stringify(value));checks.push(name);}
try{
 const active=path.join(profile,'DevToolsActivePort');let ready=false;
 for(let i=0;i<100;i++){if(fs.existsSync(active)){ready=true;break;}if(proc.exitCode!==null)throw Error('Browser exited: '+browserErrors);await sleep(100);}
 if(!ready)throw Error('Browser debugging endpoint unavailable: '+browserErrors);
 const [port,endpoint]=fs.readFileSync(active,'utf8').trim().split(/\r?\n/);
 ws=new WebSocket(`ws://127.0.0.1:${port}${endpoint}`);
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);clearTimeout(p.timer);pending.delete(m.id);if(m.error)p.reject(Error(m.error.message));else p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
 const {targetId}=await send('Target.createTarget',{url:'about:blank'});
 pageSession=(await send('Target.attachToTarget',{targetId,flatten:true})).sessionId;
 await page('Page.enable');await page('Runtime.enable');
 await page('Emulation.setDeviceMetricsOverride',{width:1600,height:1200,deviceScaleFactor:1,mobile:false});
 await page('Page.navigate',{url:pathToFileURL(path.join(here,'dist/index.html')).href});
 let loaded=false;for(let i=0;i<60;i++){if(await evaluate('Boolean(window.PRED_PREVIEW)')){loaded=true;break;}await sleep(100);}
 if(!loaded)throw Error('Application did not initialize: '+JSON.stringify(errors));
 await check('Dashboard opens on the latest discovered archive date',`window.PRED_PREVIEW.snapshotDate===ARCHIVE.latestDate`);
 // The archived reference days keep these UI regression checks stable as new days arrive.
 await evaluate(`document.querySelector('#dateSelect').value='2026-11-03';document.querySelector('#dateSelect').dispatchEvent(new Event('change'))`);
 await sleep(300);
 await check('20 fixtures and 19 eligible corner models',`window.PRED_PREVIEW.matchCount===20&&window.PRED_PREVIEW.cornerEligible===19`);
 await check('Archived reference dates are available and Derby is the confirmed user choice',`document.querySelectorAll('#dateSelect option').length>=2&&document.querySelector('.corner-card .badge').textContent==='ALEGEREA TA'&&document.querySelector('.goal-card .badge').textContent==='ALEGEREA TA'`);
 await check('Verified user choices show their confirmed success suffix',`document.querySelector('.goal-card .choice-result').textContent==='-Succes'&&document.querySelector('.corner-card .choice-result').textContent==='-Succes'&&[...document.querySelectorAll('#matchesBody .choice-result')].every(b=>b.textContent==='-Succes')`);
 await check('All result suffixes are distinct, including fail, postponed and missing',`choiceResult('success')==='-Succes'&&choiceResult('fail')==='-Fail'&&choiceResult('postponed')==='-Amânat'&&choiceResult('void')==='-Anulat'&&choiceResult('pending')==='-În așteptare'&&userChoiceBadges({outcome:{result:'fail'}},'goal').includes('choice-result fail')`);
 await check('Initial table includes all 20 rows',`document.querySelectorAll('#matchesBody tr[data-match]').length===20`);
 await check('User goal pick is pinned first while retaining its original model rank',`document.querySelector('#matchesBody tr').dataset.match==='401880179'&&document.querySelector('#matchesBody tr .rank').textContent==='02'`);
 await check('Radar has 19 source-supported data points',`document.querySelectorAll('[data-radar-match]').length===19`);
 await check('Reference date has 20 verified outcomes and two failures',`document.querySelectorAll('#outcomeStrip span').length===20&&document.querySelectorAll('#outcomeStrip .failed').length===2`);
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 fs.writeFileSync(path.join(here,'preview-desktop.png'),Buffer.from(screenshot.data,'base64'));
 await evaluate(`document.querySelector('.market-switch [data-market="corner"]').click()`);
 await check('Corner screening sorts Derby first at 96.35 percent',`document.querySelector('#matchesBody tr').dataset.match==='401880182'&&document.querySelector('#matchesBody tr .table-prob').textContent.includes('96,35')`);
 await check('Derby table badge is the user choice',`document.querySelector('#matchesBody tr .table-badge').textContent==='Alegerea ta'`);
 await evaluate(`document.querySelector('#matchesBody tr').click()`);
 await check('Detail separates final 96.61 percent from screening',`document.querySelector('#detailDialog').open&&document.querySelector('#dialogContent').textContent.includes('96,61%')&&document.querySelector('#dialogContent').textContent.includes('96,35%')`);
 await page('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27,nativeVirtualKeyCode:27});await page('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27,nativeVirtualKeyCode:27});await sleep(100);
 await check('Escape closes analysis dialog',`document.querySelector('#detailDialog').open===false`);
 await evaluate(`document.querySelector('#searchInput').value='shakhtar';document.querySelector('#searchInput').dispatchEvent(new Event('input',{bubbles:true}))`);
 await check('Missing first-half observations remain missing, not zero',`document.querySelectorAll('#matchesBody tr[data-match]').length===1&&document.querySelector('#matchesBody .table-prob').textContent==='—'&&document.querySelector('#matchesBody').textContent.includes('Lipsă date 1H')`);
 await evaluate(`window._originalBlobURL=URL.createObjectURL;URL.createObjectURL=b=>{window._exportedBlob=b;return window._originalBlobURL(b)};document.querySelector('#exportButton').click()`);
 await check('Filtered CSV export keeps missing probability blank',`(async()=>{const csv=await window._exportedBlob.text();return csv.includes('Shakhtar')&&csv.trim().split(/\\r?\\n/).length===2&&csv.includes('"CORNERS_1H_UNDER_7_5","",""')})()`);
 await evaluate(`document.querySelector('#searchInput').value='';document.querySelector('#searchInput').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('[data-league="uefa.champions"]').click()`);
 await check('League filter returns 9 UCL fixtures, 8 radar points',`document.querySelectorAll('#matchesBody tr[data-match]').length===9&&document.querySelectorAll('[data-radar-match]').length===8`);
 await evaluate(`document.querySelector('#dateSelect').value='2026-11-02';document.querySelector('#dateSelect').dispatchEvent(new Event('change'))`);
 await check('Date switch loads all 27 archived fixtures and earlier user choices',`window.PRED_PREVIEW.snapshotDate==='2026-11-02'&&document.querySelectorAll('#matchesBody tr[data-match]').length===27&&document.querySelector('.goal-card').textContent.includes('Mallorca')&&document.querySelector('.corner-card').textContent.includes('Sassuolo')`);
 await check('Verified historical user choices show success in both cards and the comparison table',`document.querySelector('.goal-card .choice-result').textContent==='-Succes'&&document.querySelector('.corner-card .choice-result').textContent==='-Succes'&&document.querySelector('#matchesBody .choice-result').textContent==='-Succes'`);
 await evaluate(`document.querySelector('.market-switch [data-market="goal"]').click()`);
 await check('Archived Mallorca choice is pinned first without changing model rank 21',`document.querySelector('#matchesBody tr .table-match').textContent.includes('Mallorca')&&document.querySelector('#matchesBody tr .rank').textContent==='21'`);
 await evaluate(`document.querySelector('.market-switch [data-market="corner"]').click()`);
 await check('Archived Sassuolo corner choice is pinned first with verified success',`document.querySelector('#matchesBody tr .table-match').textContent.includes('Sassuolo')&&document.querySelector('#matchesBody tr .choice-result').textContent==='-Succes'`);
 await evaluate(`document.querySelector('#dateSelect').value='2026-11-03';document.querySelector('#dateSelect').dispatchEvent(new Event('change'));document.querySelector('[data-view="history"]').click();document.querySelector('#historyDateSelect').value='2026-11-02';document.querySelector('#historyDateSelect').dispatchEvent(new Event('change'));document.querySelector('[data-result="fail"]').click()`);
 await check('Historical failure filter includes four failures',`document.querySelectorAll('#historyBody tr').length===4&&document.querySelectorAll('#historyBody .history-result.fail').length===4`);
 await evaluate(`document.querySelector('[data-result="all"]').click()`);
 await check('Postponed historical fixture is shown separately',`document.querySelectorAll('#historyBody tr').length===27&&document.querySelectorAll('#historyBody .history-result.postponed').length===1`);
 await evaluate(`document.querySelector('[data-history-market="corner"]').click()`);
 await check('Corner history reflects reconciled results and the remaining postponed fixture',`document.querySelectorAll('#historyBody tr').length===27&&document.querySelectorAll('#historyBody .history-result.success').length===22&&document.querySelectorAll('#historyBody .history-result.fail').length===4&&document.querySelectorAll('#historyBody .history-result.pending').length===0&&document.querySelectorAll('#historyBody .history-result.postponed,.history-result.void').length===1`);
 await check('Saved Sassuolo result shows the actual two first-half corners',`document.querySelector('#historyUserCards').textContent.includes('total 1H 2')&&document.querySelector('#historyUserCards').textContent.includes('SUCCES')`);
 await evaluate(`document.querySelector('#historyDateSelect').value='2026-11-03';document.querySelector('#historyDateSelect').dispatchEvent(new Event('change'))`);
 await check('Archived Derby choice has its verified six-corner success',`document.querySelector('#historyUserCards').textContent.includes('Derby')&&document.querySelector('#historyUserCards .choice-result').textContent==='-Succes'&&document.querySelector('#historyUserCards').textContent.includes('total 1H 6')&&document.querySelectorAll('#historyBody .history-result.success').length===18`);
 await evaluate(`document.querySelector('#historyDateSelect').value='2026-11-02';document.querySelector('#historyDateSelect').dispatchEvent(new Event('change'));document.querySelector('[data-history-market="goal"]').click()`);
 await sleep(3300);
 const historyScreenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-history.png'),Buffer.from(historyScreenshot.data,'base64'));
 await evaluate(`document.querySelector('[data-view="validation"]').click()`);
 await check('Evaluation calendar reflects actual result days and remaining pilot days',`document.querySelectorAll('#validationView [data-day]').length===completedDays()&&document.querySelectorAll('#validationView .day.future').length===Math.max(0,30-completedDays())&&window.PRED_EVALUATION.days.length===completedDays()`);
 const validationScreenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-validation.png'),Buffer.from(validationScreenshot.data,'base64'));
 await evaluate(`document.querySelector('[data-view="overview"]').click();document.querySelector('[data-league="all"]').click();document.querySelector('.market-switch [data-market="goal"]').click()`);
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await sleep(200);
 await check('Mobile has no page-level horizontal overflow',`document.documentElement.scrollWidth<=390`);
 await check('Mobile layout keeps all 20 fixtures accessible',`document.querySelectorAll('#matchesBody tr[data-match]').length===20&&getComputedStyle(document.querySelector('.selection-stack')).gridTemplateColumns.split(' ').length===1`);
 await sleep(3300);
 const mobile=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'preview-mobile.png'),Buffer.from(mobile.data,'base64'));
 for(const market of ['goal','corner']){
  await page('Page.navigate',{url:pathToFileURL(path.join(here,`dist/reports/2026-11-07/${market}/5836925.html`)).href});
  for(let i=0;i<60;i++){if(await evaluate(`document.body?.dataset.market===${JSON.stringify(market)}`))break;await sleep(100);}
  await check(`${market} report links the current fixture to its public stats page`, `document.querySelector('header a[href="https://www.fotmob.com/match/5836925"]')?.target==='_blank'`);
  await check(`${market} historical Bolton–Cardiff row links its own match and preserves raw evidence`, `(()=>{const r=[...document.querySelectorAll('tr')].find(r=>r.textContent.includes('Bolton Wanderers')&&r.textContent.includes('Cardiff City'));return !!r&&r.querySelectorAll('a[href="https://www.fotmob.com/match/5836827"]').length===2&&!!r.querySelector('a[href="https://www.fotmob.com/api/data/matchDetails?matchId=5836827"]')&&[...r.querySelectorAll('a')].every(a=>a.target==='_blank'&&a.rel.includes('noopener'))})()`);
  await check(`${market} report stays within mobile width with locally scrolling tables`, `document.documentElement.scrollWidth<=390`);
 }
 await page('Emulation.setDeviceMetricsOverride',{width:1600,height:1200,deviceScaleFactor:1,mobile:false});
 await evaluate(`document.querySelector('a[href="https://www.fotmob.com/match/5836827"]').closest('tr').scrollIntoView({block:'center'})`);
 const statsShot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(here,'report-statistics.png'),Buffer.from(statsShot.data,'base64'));
 if(errors.length)throw Error('Browser runtime exceptions: '+JSON.stringify(errors));
 const result={status:'PASS',browser:'Microsoft Edge headless',desktop:{width:1600,height:1200},mobile:{width:390,height:844},checks,exceptionCount:errors.length,localOnly:true};
 fs.writeFileSync(path.join(here,'verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}catch(error){
 console.error('Browser diagnostics:',browserErrors);
 throw error;
}finally{
 if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}
 if(proc.exitCode===null)proc.kill();
}
