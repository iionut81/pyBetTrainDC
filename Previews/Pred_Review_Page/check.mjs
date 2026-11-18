import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url)),dist=path.join(here,'dist'),terminal=path.resolve(here,'../Pred_Research_Terminal');
const targetUrl=process.argv[2]||pathToFileURL(path.join(dist,'index.html')).href;
const html=fs.readFileSync(path.join(dist,'index.html'),'utf8');
const data=JSON.parse(html.match(/<script id="snapshotData" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const manifest=JSON.parse(fs.readFileSync(path.join(dist,'publication.json'),'utf8'));
const files=[];function walk(dir){for(const e of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,e.name);e.isDirectory()?walk(p):files.push(p);}}walk(dist);
const broken=[],unsafe=[];
for(const file of files){
 if(!/\.(?:html|json|csv)$/.test(file))continue;
 const text=fs.readFileSync(file,'utf8');
 if(/(?:pred-dashboard:\/\/|X-Pred-Token|\/api\/(?:choices|state)|CODEX_THREAD_ID|chat_thread_id|[A-Z]:[\\/]Users[\\/]|github_pat_[a-zA-Z0-9_]{20,}|gh[pousr]_[a-zA-Z0-9]{20,})/.test(text))unsafe.push(path.relative(dist,file));
 if(file.endsWith('.html'))for(const match of text.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'').matchAll(/\bhref="([^"]+)"/g)){
  const href=match[1].replaceAll('&amp;','&');if(/^(?:https?:|mailto:|#)/.test(href))continue;
  const target=path.resolve(path.dirname(file),href.split('#')[0].split('?')[0]);
  if(!target.startsWith(dist+path.sep)||!fs.existsSync(target))broken.push({file:path.relative(dist,file),href});
 }
}
if(broken.length||unsafe.length)throw Error(JSON.stringify({broken,unsafe}));
if(data.automation!==null||data.days.some(d=>d.links.journal!==null))throw Error('Internal operational metadata not removed');
const expected={};for(const market of ['goal','corner']){const rows=data.days.flatMap(d=>d.matches.map(m=>m[market])).filter(v=>v.p!=null&&['success','fail'].includes(v.outcome.result));expected[market]={n:rows.length,wins:rows.filter(v=>v.outcome.result==='success').length,mean:rows.reduce((s,v)=>s+v.p,0)/rows.length};}
let runner=fs.readFileSync(path.join(terminal,'check-preview.mjs'),'utf8');
runner=runner.slice(0,runner.indexOf(" await check('Dashboard opens"))
 .replace('const here=path.dirname(fileURLToPath(import.meta.url));','const here='+JSON.stringify(terminal)+';')
 .replace("pathToFileURL(path.join(here,'dist/index.html')).href",JSON.stringify(targetUrl));
runner+=`
 await check('All published dates and the full fixture inventory are available','ARCHIVE.days.length===${manifest.days}&&ARCHIVE.days.reduce((n,d)=>n+d.matches.length,0)===${manifest.matches}&&DATA.date===${JSON.stringify(manifest.latest_date)}');
 await check('Visitors land on the model evaluation for review','state.view==="validation"&&document.querySelector("#validationView").offsetParent!==null');
 await check('Public controls cannot select or revoke project choices','typeof CHOICES==="undefined"&&!document.querySelector("[data-choice]")&&[...document.querySelectorAll("a")].every(a=>!a.href.startsWith("pred-dashboard:"))');
 await check('Latest date keeps missing corner models and unverified results','DATA.matches.length===22&&DATA.coverage.goal===6&&DATA.coverage.corner===0&&DATA.matches.every(m=>m.corner.p===null)&&!document.querySelector("#outcomeStrip")');
 await evaluate('document.querySelector("#searchInput").value="Portugalete";document.querySelector("#searchInput").dispatchEvent(new Event("input",{bubbles:true}))');
 await check('Team search works on the public copy','document.querySelectorAll("#matchesBody tr[data-match]").length===1');
 await evaluate('showDetail("6386073","goal")');
 await check('Match and daily reports use portable public links','document.querySelector("#detailDialog").open&&[...document.querySelectorAll("#dialogContent a")].some(a=>a.getAttribute("href")==="reports/2026-11-18/goal/6386073.html")&&[...document.querySelectorAll("#dialogContent a")].every(a=>!a.getAttribute("href").includes("Selection_History"))');
 await evaluate('document.querySelector("#closeDialog").click();selectDay("2026-11-16");setMarket("goal")');
 await check('Confirmed historical goal failure stays first and separate from recommendations','filteredMatches()[0].id==="6416836"&&document.querySelector(".goal-card .choice-result").textContent==="-Fail"&&!document.querySelector("[data-choice]")');
 await evaluate('setMarket("corner")');
 await check('Confirmed historical corner success and first-half total survive export','filteredMatches()[0].id==="5868536"&&document.querySelector(".corner-card .choice-result").textContent==="-Succes"&&DATA.matches.find(m=>m.id==="5868536").corner.outcome.corners==="4"');
 await evaluate('selectDay("2026-11-17");setMarket("goal")');
 await check('Previous results include every verified goal failure','document.querySelectorAll("#outcomeStrip span").length===56&&document.querySelectorAll("#outcomeStrip .failed").length===7');
 await evaluate('setMarket("corner")');
 await check('Previous results include every verified corner failure','document.querySelectorAll("#outcomeStrip span").length===44&&document.querySelectorAll("#outcomeStrip .failed").length===4');
 await evaluate('setView("validation");document.querySelector("#validationView [data-market=goal]").click()');
 await check('Goal evaluation uses the original verified modeled cohort','window.PRED_EVALUATION.total.n===${expected.goal.n}&&window.PRED_EVALUATION.total.wins===${expected.goal.wins}&&Math.abs(window.PRED_EVALUATION.total.mean-${expected.goal.mean})<1e-12');
 await evaluate('document.querySelector("#validationView [data-market=corner]").click()');
 await check('Corner evaluation uses the original verified modeled cohort','window.PRED_EVALUATION.total.n===${expected.corner.n}&&window.PRED_EVALUATION.total.wins===${expected.corner.wins}&&Math.abs(window.PRED_EVALUATION.total.mean-${expected.corner.mean})<1e-12');
 await evaluate('document.querySelector("#validationView #scope").value="analyst";document.querySelector("#validationView #scope").dispatchEvent(new Event("change"))');
 await check('Evaluation cohort switches without changing overview filters','window.PRED_EVALUATION.state.scope==="analyst"&&state.market==="corner"');
 await evaluate('document.querySelector("#validationView #scope").value="models";document.querySelector("#validationView #scope").dispatchEvent(new Event("change"));setView("overview");selectDay(${JSON.stringify(manifest.latest_date)});setMarket("goal")');
 await check('Returning to the latest date clears old results','!document.querySelector("#outcomeStrip")&&document.querySelector("#reviewPanel").dataset.reviewDate===${JSON.stringify(manifest.latest_date)}');
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Dashboard has no horizontal overflow on mobile','document.documentElement.scrollWidth<=390');
 await page('Emulation.setDeviceMetricsOverride',{width:1600,height:1200,deviceScaleFactor:1,mobile:false});
 await evaluate('setView("validation")');
 const desktop=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(${JSON.stringify(path.join(here,'desktop.png'))},Buffer.from(desktop.data,'base64'));
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await check('Evaluation has no horizontal overflow on mobile','document.documentElement.scrollWidth<=390');
 const mobile=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(${JSON.stringify(path.join(here,'mobile.png'))},Buffer.from(mobile.data,'base64'));
 await page('Page.navigate',{url:${JSON.stringify(new URL('method.html',targetUrl).href)}});
 for(let i=0;i<40;i++){if(await evaluate('document.querySelector("h1")?.textContent==="Ce merită verificat."'))break;await sleep(100);}
 await check('Review guide is readable on mobile and explains limits','document.documentElement.scrollWidth<=390&&document.body.textContent.includes("nu este profit")&&document.body.textContent.includes("necalibrate")');
 if(errors.length)throw Error(JSON.stringify(errors));
 fs.writeFileSync(${JSON.stringify(path.join(here,'verification.json'))},JSON.stringify({status:'PASS',runtime_scope:${JSON.stringify(targetUrl.startsWith('https:')?'public_https':'local_file')},static_files:${files.length},local_links:'PASS',private_or_mutation_content:0,checks,exceptions:errors.length,source_commit:${JSON.stringify(manifest.source_commit)},at_utc:new Date().toISOString()},null,2)+'\\n');console.log(JSON.stringify({status:'PASS',static_files:${files.length},browser_checks:checks.length,exceptions:errors.length}));
}catch(error){console.error('Browser diagnostics:',browserErrors);throw error;}finally{if(ws?.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{}ws.close();}if(proc.exitCode===null)proc.kill();}
`;
fs.mkdirSync(path.join(here,'.runtime'),{recursive:true});const file=path.join(here,'.runtime/check-browser.mjs');fs.writeFileSync(file,runner);await import(pathToFileURL(file).href);
