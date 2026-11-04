import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {loadArchive} from './data.mjs';
import {matchReportHref} from './match-reports.mjs';
import {DEFAULT_PORT,dashboardPath,roClock,readSelections,mutateSelection} from './selection-store.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const mime={'.html':'text/html; charset=utf-8','.json':'application/json; charset=utf-8','.csv':'text/plain; charset=utf-8','.md':'text/plain; charset=utf-8','.png':'image/png','.svg':'image/svg+xml','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8'};
export async function startDashboardServer({root=path.resolve(here,'../..'),port=DEFAULT_PORT,now=()=>new Date(),rebuild=()=>execFileSync(process.execPath,[path.join(here,'build.mjs')],{cwd:root,windowsHide:true,stdio:'pipe'})}={}){
 const token=crypto.randomBytes(32).toString('hex');let queue=Promise.resolve();
 const archive=()=>{const a=loadArchive(root);for(const d of a.days)for(const m of d.matches)for(const k of ['goal','corner'])m[k].report=matchReportHref(d.date,k,m.id);return a;};
 const server=http.createServer(async(req,res)=>{
  res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','same-origin');res.setHeader('X-Frame-Options','DENY');
  const json=(status,body)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8'});res.end(JSON.stringify(body));};
  const origin=`http://127.0.0.1:${server.address().port}`;
  try{
   if(req.headers.host!==new URL(origin).host)return json(403,{error:'Adresă locală invalidă.'});
   if(req.headers.origin&&req.headers.origin!==origin)return json(403,{error:'Origine nepermisă.'});
   const url=new URL(req.url,origin);
   if(req.method==='GET'&&url.pathname==='/api/health')return json(200,{app:'pred-dashboard',root,port:server.address().port});
   if(req.method==='GET'&&url.pathname==='/api/state')return json(200,{token,revision:readSelections(root).revision,clock:roClock(now()),archive:archive()});
   if(req.method==='POST'&&url.pathname==='/api/choices'){
    if(req.headers.origin!==origin||req.headers['x-pred-token']!==token||!/^application\/json(?:;|$)/.test(req.headers['content-type']||''))return json(403,{error:'Cererea trebuie trimisă din dashboardul local.'});
    let raw='';for await(const chunk of req){raw+=chunk;if(raw.length>4096)return json(413,{error:'Cerere prea mare.'});}
    let input;try{input=JSON.parse(raw);}catch{return json(400,{error:'JSON invalid.'});}
    const operation=queue.then(()=>{
     const result=mutateSelection(root,input,{now:now()});let warning=null;
     if(result.changed){try{rebuild();}catch{warning='Alegerea este salvată; regenerarea HTML trebuie reîncercată.';}}
     return {...result,warning,revision:readSelections(root).revision,clock:roClock(now()),archive:archive()};
    });queue=operation.catch(()=>{});
    return json(200,await operation);
   }
   if(req.method!=='GET'&&req.method!=='HEAD')return json(405,{error:'Metodă nepermisă.'});
   let pathname=decodeURIComponent(url.pathname);if(pathname==='/')pathname=dashboardPath;
   const relative=pathname.replace(/^\/+/,''),file=path.resolve(root,relative),rel=path.relative(root,file).split(path.sep).join('/');
   const allowed=rel.startsWith('Previews/Pred_Research_Terminal/dist/')||/^Analysis\/(?:Goal70_\d{4}-\d{2}-\d{2}|Corners1H_\d{4}-\d{2}-\d{2}|Review_\d{4}-\d{2}-\d{2}|Selection_History)\//.test(rel);
   if(!allowed||rel.startsWith('../')||!mime[path.extname(file)]||!fs.existsSync(file)||!fs.statSync(file).isFile())return json(404,{error:'Fișier indisponibil.'});
   const real=fs.realpathSync(file);if(path.relative(fs.realpathSync(root),real).startsWith('..'))return json(404,{error:'Fișier indisponibil.'});
   res.writeHead(200,{'Content-Type':mime[path.extname(file)]});if(req.method==='HEAD')return res.end();fs.createReadStream(file).pipe(res);
  }catch(e){json(e.status||500,{error:e.status?e.message:'Salvarea nu a reușit. Verifică serviciul local.'});}
 });
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});
 return {server,url:`http://127.0.0.1:${server.address().port}${dashboardPath}`};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const {url}=await startDashboardServer();console.log('PRED ready: '+url);
}
