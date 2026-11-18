import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';

const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..'),dist=path.join(here,'dist');
const git=(...args)=>execFileSync('git',['-c','core.packedGitWindowSize=1m',...args],{cwd:root,encoding:'utf8',maxBuffer:64*1024*1024,stdio:['pipe','pipe','pipe']});
const source=git('rev-parse',process.argv[2]||'origin/main').trim();
if(!/^[a-f0-9]{40}$/.test(source))throw Error('Invalid source revision');
const cache=path.join(here,'.runtime','source',source),ready=path.join(cache,'.source-ready');
if(!fs.existsSync(ready)){
 fs.mkdirSync(cache,{recursive:true});const tar=path.join(here,'.runtime',source+'.tar');
 git('archive','--format=tar','--output='+tar,source,'Previews/Pred_Research_Terminal','Analysis/Goal70_*','Analysis/Corners1H_*','Analysis/Review_*','Analysis/System_Audit_*');
 execFileSync('tar',['-xf',path.relative(here,tar),'-C',path.relative(here,cache)],{cwd:here,stdio:['ignore','pipe','pipe']});fs.writeFileSync(ready,source+'\n');
}
const read=file=>fs.readFileSync(path.join(cache,file),'utf8');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const prefix='Previews/Pred_Research_Terminal',sourceIndex=prefix+'/dist/index.html';
const sourceArchive=JSON.parse(read(prefix+'/dist/snapshot.json')),archive=structuredClone(sourceArchive);
const publicationTime=new Date().toISOString(),copied=new Set(),missing=new Set(),blocked=new Set();
const write=(file,text)=>{const target=path.resolve(dist,file);if(!target.startsWith(dist+path.sep))throw Error('Output outside review directory');fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,text);};
const generatedReport=file=>file.startsWith(prefix+'/dist/reports/')&&file.endsWith('.html');
const evidenceFile=file=>/^Analysis\/(?:Goal70_|Corners1H_|Review_|System_Audit_)\d{4}-\d{2}-\d{2}\//.test(file)&&/\.(md|csv|json)$/.test(file)&&!/(?:^|\/)(?:HistoryRaw|MatchData|Context|Backlog|FailedSandbox|transcripts|Conversation_History)(?:\/|$)/.test(file);
function destination(file){
 if(file===sourceIndex)return 'index.html';
 if(generatedReport(file))return file.slice((prefix+'/dist/').length);
 if(evidenceFile(file))return 'evidence/'+file.replace(/\.md$/,'.html');
 return null;
}
function resolveLocal(from,href){
 if(!href||/^(?:https?:|mailto:|#)/i.test(href))return null;
 if(/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href))return false;
 return path.posix.normalize(path.posix.join(path.posix.dirname(from),href.replaceAll('&amp;','&').split('#')[0].split('?')[0]));
}
function mappedLink(from,href){
 const target=resolveLocal(from,href);
 if(target===null)return /^(?:https?:|mailto:|#)/i.test(href)?href:null;
 if(target===false||!destination(target)){blocked.add(String(target));return null;}
 if(target!==sourceIndex&&!copyTarget(target))return null;
 const hash=href.includes('#')?'#'+href.split('#').slice(1).join('#'):'';
 return path.posix.relative(path.posix.dirname(destination(from)),destination(target))+hash;
}
function inline(text,file){
 const links=[];
 const tokens=text.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(_,label,url)=>{
  const href=mappedLink(file,url.replace(/^<|>$/g,''));if(!href)return label;
  links.push(`<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`);return `PUBLICLINK${links.length-1}END`;
 });
 return esc(tokens).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/PUBLICLINK(\d+)END/g,(_,i)=>links[+i]);
}
const documentCss=`body{background:#081214;color:#deebe7;font:15px/1.75 system-ui;margin:0}main{max-width:1100px;margin:auto;padding:30px 24px 80px}a{color:#58e0c0}h1{font-size:clamp(28px,5vw,42px);line-height:1.2}h2{margin-top:34px;font-size:22px}h3{font-size:18px}p,li{overflow-wrap:anywhere}code{font-size:.9em;color:#e1c98c}.scroll,pre{overflow:auto}table{border-collapse:collapse;min-width:100%;font-size:13px}td,th{padding:10px 12px;border:1px solid #29413e;text-align:left}th{background:#10241f}.notice{font-size:12px;color:#9fb5af}.intro{padding:20px;border:1px solid #29413e;border-radius:12px;background:#0b1b18}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.eyebrow{font-size:11px;letter-spacing:.13em;color:#58e0c0}@media(max-width:650px){main{padding:22px 18px 50px}.grid{grid-template-columns:1fr}td,th{padding:8px}}`;
function documentPage(title,body,back='index.html'){
 return `<!doctype html><html lang="ro"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>${esc(title)} · PRED Review</title><style>${documentCss}</style></head><body><main><a href="${esc(back)}">← Dashboard de review</a><p class="eyebrow">PRED / REVIEW PUBLIC</p>${body}<p class="notice">Date salvate până la ${archive.latestDate}. Copie de consultare, publicată ${publicationTime.slice(0,10)}. Estimări necalibrate; nicio promisiune de profit.</p></main></body></html>`;
}
function markdownPage(text,file){
 const lines=text.split(/\r?\n/),blocks=[];let fence=false,code=[];
 for(let i=0;i<lines.length;i++){
  const line=lines[i];
  if(line.startsWith('```')){if(fence){blocks.push('<pre><code>'+esc(code.join('\n'))+'</code></pre>');code=[];}fence=!fence;continue;}
  if(fence){code.push(line);continue;}
  if(/^\s*\|/.test(line)){
   const table=[];while(i<lines.length&&/^\s*\|/.test(lines[i]))table.push(lines[i++]);i--;
   const cells=r=>r.trim().replace(/^\||\|$/g,'').split('|').map(v=>v.trim());
   blocks.push('<div class="scroll"><table>'+table.filter(r=>!/^\s*\|?\s*:?-{2,}/.test(r)).map((r,j)=>'<tr>'+cells(r).map(c=>`<${j?'td':'th'}>${inline(c,file)}</${j?'td':'th'}>`).join('')+'</tr>').join('')+'</table></div>');continue;
  }
  const heading=line.match(/^(#{1,6})\s+(.+)/);
  if(heading){blocks.push(`<h${heading[1].length}>${inline(heading[2],file)}</h${heading[1].length}>`);continue;}
  if(line.trim())blocks.push(`<p>${inline(line,file)}</p>`);
 }
 return documentPage(path.posix.basename(file),blocks.join('\n'),path.posix.relative(path.posix.dirname(destination(file)),'index.html'));
}
function copyTarget(file){
 const out=destination(file);if(!out)return false;if(copied.has(file))return true;
 let text;try{text=read(file);}catch{missing.add(file);return false;}
 copied.add(file);
 if(file.endsWith('.md'))text=markdownPage(text,file);
 else if(generatedReport(file))text=text.replace(/<a\b([^>]*?)href="([^"]+)"([^>]*)>([\s\S]*?)<\/a>/g,(_,before,url,after,label)=>{const href=mappedLink(file,url);return href?`<a${before}href="${esc(href)}"${after}>${label}</a>`:label;});
 write(out,text);return true;
}

archive.automation=null;
const userFields=['selection_id','analysis_date','selected_at_ro','market','home','away','match_id','provider','p_raw','p_stress','fair_odds','raw_rank','result','actual_first_goal_minute','actual_corners_1h','result_source','result_verified_at_ro','status','calibrated','model_snapshot_at_ro','source','p','stress'];
const publicUser=u=>Object.fromEntries(userFields.filter(k=>k in u).map(k=>[k,u[k]]));
for(const day of archive.days){
 day.links.journal=null;day.userSelections=day.userSelections.map(publicUser);
 for(const market of ['goal','corner']){
  day.links[market]=day.links[market]?mappedLink(sourceIndex,day.links[market]):null;
  day.evidence[market]=day.evidence[market]?.historical_validated!=null?{historical_validated:day.evidence[market].historical_validated}:null;
  for(const m of day.matches){if(m[market].userSelection)m[market].userSelection=publicUser(m[market].userSelection);const file=path.posix.join(prefix+'/dist',m[market].report);if(!copyTarget(file))throw Error('Required match report missing: '+file);}
 }
}

const readOnly=`function choiceControl(){return '';}
function initializeChoices(){return Promise.resolve();}
function saveChoice(){toast('Pagina permite numai consultarea.');}
`;
let views=read(prefix+'/views.js').replace(/function automationNote\(\)\{[\s\S]*?\n\}/,`function automationNote(){return 'Copie publică de consultare, cu date salvate până la ${archive.latestDate}. Nu este un flux live. Actualizările necesită o nouă publicare.';}`)
 .replace(/\$\{reportLink\('\.\.\/\.\.\/\.\.\/Analysis\/Selection_History\/DAILY_WORKFLOW\.md','Fluxul salvat'\)\}/g,'')
 .replaceAll('Dashboard local cu','Arhivă publică de review cu').replaceAll('confirmat de tine','alegere confirmată în proiect').replaceAll('ALEGEREA TA','ALEGERE CONFIRMATĂ').replaceAll('Alegerea ta','Alegere confirmată').replaceAll('alegerile tale','alegerile confirmate');
let app=read(prefix+'/app.js').replaceAll('alegerile tale','alegerile confirmate');
const extra=`\nsetView('validation');\ndocument.getElementById('shareReview').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(location.href.split('#')[0]);toast('Linkul paginii a fost copiat.');}catch{toast('Copiază adresa paginii din bara browserului.');}});`;
const evaluation=read(prefix+'/evaluation.js').replaceAll('Alegerile tale','Alegerile confirmate').replaceAll('Alegerile mele','Alegeri confirmate');
const js=readOnly+'\n'+app+'\n'+evaluation+'\n'+views+extra;new Function(js);
const count=archive.days.reduce((n,d)=>n+d.matches.length,0);
const notice=`<details class="public-notice"><summary>Cum verifici această pagină <span>${archive.days.length} zile · ${count} partide inventariate · copie până la ${archive.latestDate}</span></summary><p>Alege o dată și o piață, deschide un meci și verifică estimarea inițială, istoricul și sursele. În Istoric poți vedea separat succesele, eșecurile și rezultatele neverificate. Evaluarea compară observația și estimarea pe aceeași cohortă de rezultate verificate.</p><p>Vizitatorii pot filtra și exporta date. Selecțiile și arhiva nu se modifică din această pagină. Acoperirea diferă de la o zi la alta, iar cercetările parțiale rămân marcate în rapoarte. <a href="method.html">Ghidul de review și limitele sistemelor ↗</a></p></details>`;
let template=read(prefix+'/index.template.html').replace('PRED — Football Research Terminal','PRED — Review public').replace('Preview local al proiectului PRED: selecții de fotbal, istoric verificat și evaluare prospectivă.','Analize de fotbal, istoric verificat și evaluarea transparentă a modelelor Gol70 și cornere 1H. Pagină publică pentru review.')
 .replace('PREVIEW · DATE SALVATE','REVIEW PUBLIC · DOAR CITIRE').replace('Din date, în decizii.','Fiecare analiză, verificabilă.').replace('Gol până la 70′. Cornere la pauză. Fiecare alegere, verificabilă.','Gol până la 70′. Cornere la pauză. Analize, rezultate și limite, la vedere.')
 .replace('<button id="methodButton"','<a class="public-header-link" href="method.html">Ghid de review ↗</a><button class="outline-button public-share" id="shareReview">Copiază linkul</button><button id="methodButton"')
 .replace('<section class="kpi-grid"',notice+'<section class="kpi-grid"');
const css=read(prefix+'/style.css')+'\n'+read(prefix+'/evaluation.css')+'\n'+fs.readFileSync(path.join(here,'public.css'),'utf8');
const payload=JSON.stringify(archive).replace(/</g,'\\u003c').replace(/\u2028/g,'\\u2028').replace(/\u2029/g,'\\u2029');
const html=template.replace('/* STYLES */',()=>css).replace('/* APP_DATA */',()=>payload).replace('/* APP_CODE */',()=>js);
if(/\/api\/(?:choices|state)|X-Pred-Token|pred-dashboard:\/\/|const CHOICES\s*=/.test(html))throw Error('Local choice service or launcher included');
if(/Conversation_History|SESSION_RESUME|CODEX_THREAD_ID|chat_thread_id/.test(payload))throw Error('Private memory included in public payload');
write('index.html',html);
const guide=fs.readFileSync(path.join(here,'method.html.fragment'),'utf8');
write('method.html',documentPage('Ghid de review',guide.replaceAll('{{LATEST_DATE}}',archive.latestDate).replaceAll('{{SOURCE_COMMIT}}',source)));
write('.nojekyll','');
const modelDigest=a=>cryptoDigest(a.days.map(d=>({date:d.date,matches:d.matches.map(m=>({id:m.id,goal:{p:m.goal.p,stress:m.goal.stress,outcome:m.goal.outcome},corner:{p:m.corner.p,stress:m.corner.stress,outcome:m.corner.outcome}})),reviews:d.reviews,metrics:d.metrics})));
function cryptoDigest(value){return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');}
const before=modelDigest(sourceArchive),after=modelDigest(archive);if(before!==after)throw Error('Public export altered predictions or outcomes');
const manifest={source_commit:source,exported_at_utc:publicationTime,latest_date:archive.latestDate,days:archive.days.length,matches:count,match_reports:archive.days.reduce((n,d)=>n+d.matches.length*2,0),read_only:true,public_path:'/pyBetTrainDC/review/',resources:copied.size,model_outcome_digest:after,missing_optional:[...missing].sort(),omitted_private_links:blocked.size};
write('publication.json',JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify(manifest,null,2));
