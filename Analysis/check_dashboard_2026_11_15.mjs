import fs from 'node:fs';import path from 'node:path';import {pathToFileURL} from 'node:url';
let here=path.resolve('Previews/Pred_Research_Terminal'),s=fs.readFileSync(path.join(here,'check-preview.mjs'),'utf8');s=s.slice(0,s.indexOf(" await check('Dashboard opens"));s=s.replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(here)+';');
s+=`
 await check('Today59records and4models per market', 'DATA.date==="2026-11-15"&&DATA.matches.length===59&&DATA.coverage.goal===4&&DATA.coverage.corner===4');
 await check('Granada separate analyst candidates; no invented choice', 'DATA.userSelections.length===0&&document.querySelector(".goal-card").textContent.includes("Granada")&&document.querySelector(".corner-card").textContent.includes("Granada")');
 await check('Today outcomes all pending; no older success rate', 'DATA.matches.every(m=>m.goal.outcome.result==="pending"&&m.corner.outcome.result==="pending")&&!document.querySelector("#outcomeStrip")');
 await check('Unavailable national model is null', 'DATA.matches.find(m=>m.id==="5181798").goal.p===null&&DATA.matches.find(m=>m.id==="5181798").corner.p===null');
 await evaluate('selectDay("2026-11-14");setMarket("goal")');
 await check('Yesterday36verified and7fail', 'document.querySelectorAll("#outcomeStrip span").length===36&&document.querySelectorAll("#outcomeStrip .failed").length===7');
 await check('Valladolid user choice success pinned; saved probability preserved', 'document.querySelector(".goal-card .choice-result").textContent==="-Succes"&&filteredMatches()[0].id==="5868546"&&DATA.matches.find(m=>m.id==="5868546").goal.p===0.9142067561259274');
 await evaluate('setMarket("corner")');
 await check('Yesterday31verified and2fail', 'document.querySelectorAll("#outcomeStrip span").length===31&&document.querySelectorAll("#outcomeStrip .failed").length===2');
 await check('Almeria user choice success pinned; saved probability preserved', 'document.querySelector(".corner-card .choice-result").textContent==="-Succes"&&filteredMatches()[0].id==="5868537"&&DATA.matches.find(m=>m.id==="5868537").corner.p===0.9524097294586629');
 await evaluate('selectDay("2026-11-15");setMarket("corner")');
 await check('Return today clears old outcome strip', '!document.querySelector("#outcomeStrip")&&document.querySelector("#reviewPanel").dataset.reviewDate==="2026-11-15"');
 const screenshot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.resolve('Analysis/Goal70_2026-11-15/dashboard-desktop.png'),Buffer.from(screenshot.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Mobile no horizontal overflow', 'document.documentElement.scrollWidth<=390');
 if(errors.length)throw Error(JSON.stringify(errors));
 fs.writeFileSync(path.resolve('Analysis/Goal70_2026-11-15/dashboard-verification.json'),JSON.stringify({status:'PASS',checks,exceptions:errors.length,at_utc:new Date().toISOString()},null,2));console.log(JSON.stringify({status:'PASS',checks,exceptions:errors.length}));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
fs.mkdirSync(path.join(here,'.runtime'),{recursive:true});let runtime=path.join(here,'.runtime','daily-2026-11-15-runtime.mjs');fs.writeFileSync(runtime,s);await import(pathToFileURL(runtime).href);
