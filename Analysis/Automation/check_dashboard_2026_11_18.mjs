import fs from 'node:fs';import path from 'node:path';import {pathToFileURL} from 'node:url';
const here=path.resolve('Previews/Pred_Research_Terminal');let s=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');s=s.slice(0,s.indexOf(" await check('Dashboard opens"));s=s.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(here)+';');s+=`
 await check('Today22records6goals0corners','DATA.date==="2026-11-18"&&DATA.matches.length===22&&DATA.coverage.goal===6&&DATA.coverage.corner===0');
 await check('No analyst or user choice inferred','DATA.userSelections.length===0&&DATA.matches.every(m=>!m.analystGoal&&!m.analystCorner)&&document.querySelector(".goal-card").textContent.includes("Fără alegere")');
 await check('Today no fabricated results or old review strip','DATA.matches.every(m=>m.goal.outcome.result==="pending"&&m.corner.outcome.result==="pending")&&!document.querySelector("#outcomeStrip")&&DATA.metrics.goal.hit_rate===null');
 await check('Missing corner probabilities null','DATA.matches.every(m=>m.corner.p===null)');
 await evaluate('selectDay("2026-11-17");setMarket("goal")');
 await check('Yesterday56goalverified7fail','document.querySelectorAll("#outcomeStrip span").length===56&&document.querySelectorAll("#outcomeStrip .failed").length===7');
 await check('Ireland62 success, original estimate preserved','DATA.matches.find(m=>m.id==="6416837").goal.outcome.minute==="62"&&DATA.matches.find(m=>m.id==="6416837").goal.p===0.950212931632136');
 await evaluate('setMarket("corner")');
 await check('Yesterday44cornerverified4fail','document.querySelectorAll("#outcomeStrip span").length===44&&document.querySelectorAll("#outcomeStrip .failed").length===4');
 await evaluate('selectDay("2026-11-16");setMarket("goal")');
 await check('Explicit Greece fail still pinned','filteredMatches()[0].id==="6416836"&&document.querySelector(".goal-card .choice-result").textContent==="-Fail"');
 await evaluate('setMarket("corner")');
 await check('Explicit Albacete success still pinned','filteredMatches()[0].id==="5868536"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"');
 await evaluate('selectDay("2026-11-18");setMarket("goal")');
 await check('Returning today clears previous outcomes','!document.querySelector("#outcomeStrip")&&document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-18"');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.resolve('Analysis/Goal70_2026-11-18/dashboard-desktop.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Mobile no overflow','document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));fs.writeFileSync(path.resolve('Analysis/Goal70_2026-11-18/dashboard-verification.json'),JSON.stringify({status:'PASS',checks,exceptions:errors.length,at_utc:new Date().toISOString()},null,2));console.log({status:'PASS',checks,exceptions:errors.length});
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;fs.mkdirSync(path.join(here,'.runtime'),{recursive:true});const runtime=path.join(here,'.runtime','daily-2026-11-18.mjs');fs.writeFileSync(runtime,s);await import(pathToFileURL(runtime).href);
