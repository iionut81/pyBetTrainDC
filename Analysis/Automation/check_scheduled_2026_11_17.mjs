import fs from 'node:fs';import path from 'node:path';import {pathToFileURL} from 'node:url';const here=path.resolve('Previews/Pred_Research_Terminal');let s=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');s=s.slice(0,s.indexOf(" await check('Dashboard opens"));s=s.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(here)+';');s+=`
 await check('Today133records5goals0corners', 'DATA.date==="2026-11-17"&&DATA.matches.length===133&&DATA.coverage.goal===5&&DATA.coverage.corner===0');
 await check('Ireland analyst only; no new user choice', 'DATA.userSelections.length===0&&document.querySelector(".goal-card").textContent.includes("Ireland")');
 await check('Today has no invented outcomes or old strip', 'DATA.matches.every(m=>m.goal.outcome.result==="pending"&&m.corner.outcome.result==="pending")&&!document.querySelector("#outcomeStrip")&&DATA.metrics.goal.hit_rate===null&&DATA.metrics.corner.hit_rate===null');
 await check('Missing corner model remains null', 'DATA.matches.find(m=>m.id==="6416837").corner.p===null&&DATA.matches.find(m=>m.id==="6416837").corner.coverage.season.home.missing===11');
 await evaluate('selectDay("2026-11-16");setMarket("goal")');
 await check('Yesterday27goalverified2fail', 'document.querySelectorAll("#outcomeStrip span").length===27&&document.querySelectorAll("#outcomeStrip .failed").length===2');
 await check('Greece user choice fail pinned; p preserved', 'document.querySelector(".goal-card .choice-result").textContent==="-Fail"&&filteredMatches()[0].id==="6416836"&&DATA.matches.find(m=>m.id==="6416836").goal.p===0.9160280720674825');
 await evaluate('setMarket("corner")');
 await check('Yesterday21cornersverified0fail', 'document.querySelectorAll("#outcomeStrip span").length===21&&document.querySelectorAll("#outcomeStrip .failed").length===0');
 await check('Albacete success pinned; p preserved', 'document.querySelector(".corner-card .choice-result").textContent==="-Succes"&&filteredMatches()[0].id==="5868536"&&DATA.matches.find(m=>m.id==="5868536").corner.p===0.8958553249660637');
 await evaluate('selectDay("2026-11-17");setMarket("goal")');
 await check('Returning today clears results for selected date', '!document.querySelector("#outcomeStrip")&&document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-17"');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.resolve('Analysis/Goal70_2026-11-17/dashboard-scheduled-desktop.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Mobile no horizontal overflow', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));fs.writeFileSync(path.resolve('Analysis/Goal70_2026-11-17/dashboard-scheduled-verification.json'),JSON.stringify({status:'PASS',checks,exceptions:errors.length,at_utc:new Date().toISOString()},null,2));console.log(JSON.stringify({status:'PASS',checks,exceptions:errors.length}));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;fs.mkdirSync(path.join(here,'.runtime'),{recursive:true});let runtime=path.join(here,'.runtime','scheduled-2026-11-17-check.mjs');fs.writeFileSync(runtime,s);await import(pathToFileURL(runtime).href);
