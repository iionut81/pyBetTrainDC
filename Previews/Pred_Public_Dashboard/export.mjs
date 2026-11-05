import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..'),dist=path.join(here,'dist');
const git=(...args)=>execFileSync('git',['-c','core.packedGitWindowSize=1m',...args],{cwd:root,encoding:'utf8',stdio:['pipe','pipe','pipe']});
const commit=git('rev-parse','origin/main').trim();
const read=file=>git('show',`${commit}:${file}`);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const write=(file,text)=>{const target=path.join(dist,file);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,text);};
const prefix='Previews/Pred_Research_Terminal/dist',copied=new Set(),missing=[];
const archive=JSON.parse(read(`${prefix}/snapshot.json`));archive.automation=null;
for(const day of archive.days){day.links.journal=null;for(const k of ['goal','corner'])if(day.links[k]?.endsWith('.md'))day.links[k]+='.html';}
const allowed=file=>file.startsWith(prefix+'/reports/')||/^Analysis\/(?:Goal70_|Corners1H_|Review_)\d{4}-\d{2}-\d{2}\//.test(file);
function resolveLocal(from,href){
 if(!href||/^(?:https?:|mailto:|#)/i.test(href))return null;
 const clean=href.replaceAll('&amp;','&').split('#')[0].split('?')[0];
 return path.posix.normalize(path.posix.join(path.posix.dirname(from),clean));
}
function copyTarget(file){
 if(!allowed(file))return false;
 if(copied.has(file))return true;
 let text;try{text=read(file);}catch{missing.push(file);return false;}
 copied.add(file);
 if(file.endsWith('.md')){write(file+'.html',markdownPage(text,file));return true;}
 if(!/\.(?:html|csv|json)$/.test(file))return false;
 if(file.endsWith('.html'))text=text.replace(/<a\b([^>]*?)href="([^"]+)"([^>]*)>([\s\S]*?)<\/a>/g,(all,before,href,after,label)=>{
  const target=resolveLocal(file,href);if(!target)return all;
  if(target===prefix+'/index.html')return all;
  if(!copyTarget(target))return '';
  return `<a${before}href="${esc(href.replaceAll('&amp;','&')+(target.endsWith('.md')?'.html':''))}"${after}>${label}</a>`;
 });
 write(file,text);return true;
}
function inline(text,file){
 const links=[];
 const tokens=text.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(all,label,href)=>{
  href=href.replace(/^<|>$/g,'');const target=resolveLocal(file,href);
  if(/^javascript:/i.test(href))return label;
  if(target&&!copyTarget(target))return label;
  links.push(`<a href="${esc(href+(target?.endsWith('.md')?'.html':''))}" target="_blank" rel="noopener">${esc(label)}</a>`);return `LINKTOKEN${links.length-1}END`;
 });
 return esc(tokens).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/LINKTOKEN(\d+)END/g,(_,i)=>links[+i]);
}
function markdownPage(text,file){
 const lines=text.split(/\r?\n/),blocks=[];let fence=false,code=[];
 for(let i=0;i<lines.length;i++){
  const line=lines[i];if(line.startsWith('```')){if(fence){blocks.push('<pre><code>'+esc(code.join('\n'))+'</code></pre>');code=[];}fence=!fence;continue;}if(fence){code.push(line);continue;}
  if(/^\s*\|/.test(line)){
   const table=[];while(i<lines.length&&/^\s*\|/.test(lines[i]))table.push(lines[i++]);i--;
   const cells=row=>row.trim().replace(/^\||\|$/g,'').split('|').map(v=>v.trim());
   blocks.push('<div class="scroll"><table>'+table.filter(r=>!/^\s*\|?\s*:?-{2,}/.test(r)).map((r,j)=>'<tr>'+cells(r).map(c=>`<${j===0?'th':'td'}>${inline(c,file)}</${j===0?'th':'td'}>`).join('')+'</tr>').join('')+'</table></div>');continue;
  }
  const h=line.match(/^(#{1,6})\s+(.+)/);if(h){blocks.push(`<h${h[1].length}>${inline(h[2],file)}</h${h[1].length}>`);continue;}
  if(line.trim())blocks.push(`<p>${inline(line,file)}</p>`);
 }
 return `<!doctype html><html lang="ro"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(path.basename(file))} · PRED</title><style>body{background:#071014;color:#dce9ed;font:15px/1.7 system-ui;margin:0}main{max-width:1100px;margin:auto;padding:28px 22px 70px}a{color:#44e9c8}h1{font-size:28px;line-height:1.3}h2{margin-top:32px;font-size:21px}h3{font-size:17px}code,pre{font-family:monospace}pre,.scroll{overflow:auto}table{border-collapse:collapse;min-width:100%;font-size:13px}td,th{padding:9px 12px;border:1px solid #29424b;text-align:left}th{background:#10242b}p{overflow-wrap:anywhere}.notice{color:#92acb7;font-size:12px}</style></head><body><main><a href="${path.posix.relative(path.posix.dirname(file),prefix+'/index.html')}">← Dashboard</a><p class="notice">Versiune de consultare · date din arhiva publicată · estimări necalibrate</p>${blocks.join('\n')}</main></body></html>`;
}

let html=read(`${prefix}/index.html`),choices=read('Previews/Pred_Research_Terminal/choices.js');
const readOnly=`function choiceControl(){return '';}
function initializeChoices(){const p=document.createElement('p');p.className='choice-connection';p.textContent='Versiune de consultare · Date publicate pentru '+DATA.date+' · Alegerile nu pot fi modificate de vizitatori.';document.querySelector('.page-heading').after(p);}
function saveChoice(){toast('Această versiune permite numai consultarea.');}
`;
if(!html.includes(choices))throw Error('Published choice module not found in committed HTML');
html=html.replace(choices,readOnly).replace(/function automationNote\(\)\{[\s\S]*?\n\}/,"function automationNote(){return 'Copie de consultare a datelor publicate. Actualizările locale nu apar aici până la o nouă publicare.';}");
const payload=JSON.stringify(archive).replace(/</g,'\\u003c').replace(/\u2028/g,'\\u2028').replace(/\u2029/g,'\\u2029');
html=html.replace(/(<script id="snapshotData" type="application\/json">)[\s\S]*?(<\/script>)/,(_,a,b)=>a+payload+b);
html=html.replace(/\$\{reportLink\('\.\.\/\.\.\/\.\.\/Analysis\/Selection_History\/DAILY_WORKFLOW\.md','Fluxul salvat'\)\}/g,'');
if(/\/api\/choices|X-Pred-Token|pred-dashboard:\/\//.test(html))throw Error('Local mutation/launcher code remained in public HTML');
for(const day of archive.days){
 for(const market of ['goal','corner']){
  const href=day.links[market];if(href)copyTarget(path.posix.normalize(prefix+'/'+href.replace(/\.md\.html$/,'.md')));
  for(const m of day.matches)copyTarget(prefix+'/'+m[market].report);
 }
}
write(prefix+'/index.html',html);
write('index.html','<!doctype html><html lang="ro"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0;url=./'+prefix+'/index.html"><title>PRED · Analize și istoric</title></head><body><a href="./'+prefix+'/index.html">Deschide dashboardul PRED</a></body></html>');
const manifest={source_commit:commit,exported_at_utc:new Date().toISOString(),days:archive.days.length,matches:archive.days.reduce((s,d)=>s+d.matches.length,0),read_only:true,resources:copied.size,missing:[...new Set(missing)]};
write('publication.json',JSON.stringify(manifest,null,2));console.log(JSON.stringify(manifest,null,2));
