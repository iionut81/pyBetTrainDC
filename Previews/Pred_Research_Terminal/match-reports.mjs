import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from './data.mjs';

const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => value == null || value === '' || !Number.isFinite(Number(value)) ? null : Number(value);
const fmt = value => number(value) == null ? '—' : Number(value).toLocaleString('ro-RO', {maximumFractionDigits:4});
const pct = value => number(value) == null ? '—' : (Number(value)*100).toLocaleString('ro-RO', {minimumFractionDigits:2,maximumFractionDigits:2})+'%';
const ratio = (wins,n) => number(wins) == null || number(n) == null ? '—' : `${wins}/${n}${Number(n)>0?' · '+pct(wins/n):''}`;
const link = (href,label) => href ? `<a href="${esc(href)}">${esc(label)} ↗</a>` : '';
const source = (href,label='Sursă') => /^https?:\/\//i.test(href || '') ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>` : '—';
const textFile = file => { const b=fs.readFileSync(file);return b.toString(b[0]===255&&b[1]===254?'utf16le':'utf8').replace(/^\uFEFF/,''); };
const firstCsv = (root,files) => {
 const file=files.find(file=>fs.existsSync(path.join(root,file)));
 return {file,rows:file?parseCsv(textFile(path.join(root,file))):[]};
};
const modelRow = (rows,m) => rows.find(r=>[r.id,r.match_id,r.goal70_espn_id].some(id=>id&&m.ids.includes(id)))
 || rows.find(r=>r.home===m.originalHome&&r.away===m.originalAway) || {};
const local = (from,to) => path.relative(from,to).split(path.sep).join('/');

export function matchReportHref(date,market,id) {
 if(!/^\d{4}-\d{2}-\d{2}$/.test(date)||!['goal','corner'].includes(market)||!/^[a-zA-Z0-9_-]+$/.test(id))throw Error('Invalid match report identity');
 return `reports/${date}/${market}/${id}.html`;
}

function historyTable(rows,model,m,side,market,cutoff) {
 const teamId=model[side+'_history_id']||model[side+'_id'];
 const teamName=model[side]||m[side==='home'?'originalHome':'originalAway'];
 const history=rows.filter(r=>{
  const before=Date.parse(r.date)<cutoff;
  return before&&(teamId ? r.home_id===teamId||r.away_id===teamId : r.home===teamName||r.away===teamName);
 }).sort((a,b)=>Date.parse(b.date)-Date.parse(a.date)).filter((r,i,all)=>all.findIndex(x=>x.id===r.id)===i).slice(0,10);
 const body=history.map(r=>{
  let observation,result;
  if(market==='goal'){
   const valid=r.goal_ok==null||String(r.goal_ok).toLowerCase()==='true';
   const success=String(r.success70).toLowerCase();
   const score=r.score||`${r.home_score??'—'}–${r.away_score??'—'}`;
   observation=!valid?'Date nevalidate':r.first_goal?esc(r.first_goal):/^0\s*[-–]\s*0$/.test(score)?'Fără gol':'Minut indisponibil';
   result=valid&&['true','false'].includes(success)?success==='true'?'Succes':'Fail':'Neverificat';
  }else{
   const valid=r.valid==null||String(r.valid).toLowerCase()==='true';
   const total=number(r.total_corners_1h??r.total_1h);
   observation=valid&&total!=null?`${fmt(r.home_corners_1h)} + ${fmt(r.away_corners_1h)} = ${fmt(total)}`:'Date 1H indisponibile';
   result=valid&&total!=null?total<=7?'Succes':'Fail':'Neverificat';
  }
  return `<tr><td>${esc(r.date.slice(0,10))}<small>${esc(r.league)}</small></td><td>${esc(r.home)} – ${esc(r.away)}${market==='goal'?`<small>${esc(r.score||`${r.home_score??'—'}–${r.away_score??'—'}`)}</small>`:''}</td><td>${observation}</td><td>${result}</td><td>${source(r.source||r.api_source)}</td></tr>`;
 }).join('');
 return `<h3>${esc(teamName)} · ${history.length} observații disponibile în eșantion</h3><p class="muted">Cele mai recente rânduri salvate înaintea predicției, maximum 10. Nu reprezintă automat ultimele 10 meciuri oficiale din toate competițiile.</p>${history.length?`<div class="scroll"><table><thead><tr><th>Data / competiția</th><th>Meci / scor final</th><th>${market==='goal'?'Primul gol':'Cornere 1H'}</th><th>Pragul ales</th><th>Dovadă</th></tr></thead><tbody>${body}</tbody></table></div>`:'<p>Istoricul individual nu este disponibil în fișierul sursă identificat. Lipsa nu este zero.</p>'}`;
}

function countsTable(row,market,m) {
 const columns=market==='goal'?['Meciuri','Marcate ≤70′','Primite ≤70′','Gol70 reușit','Primul gol >70′','Fără gol']:['Meciuri','Cornere 1H obținute','Cornere 1H cedate','Total mediu 1H','Exact 7','Minimum 8'];
 const body=['home','away','home_venue','away_venue'].map(prefix=>{
  const values=market==='goal'
   ? [row[prefix+'_n'],row[prefix+'_gf70'],row[prefix+'_ga70'],row[prefix+'_success']??row[prefix+'_wins'],row[prefix+'_late'],row[prefix+'_0_0']??row[prefix+'_nil']]
   : [row[prefix+'_n'],row[prefix+'_cf']??row[prefix+'_gf'],row[prefix+'_ca']??row[prefix+'_ga'],row[prefix+'_mean'],row[prefix+'_7']??row[prefix+'_seven'],row[prefix+'_8plus']??row[prefix+'_eight']];
  const label=(prefix.startsWith('home')?m.originalHome:m.originalAway)+(prefix.endsWith('venue')?(prefix.startsWith('home')?' · acasă':' · deplasare'):' · sezon');
  return `<tr><td>${esc(label)}</td>${values.map(v=>`<td>${fmt(v)}</td>`).join('')}</tr>`;
 }).join('');
 return `<div class="scroll"><table><thead><tr><th>Eșantion</th>${columns.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`;
}

export function writeMatchReports(root,dist,archive) {
 let count=0;
 for(const day of archive.days){
  const goalBase=`Analysis/Goal70_${day.date}`,cornerBase=`Analysis/Corners1H_${day.date}`;
  const goalModels=firstCsv(root,[`${goalBase}/all_models.csv`,`${goalBase}/Europe_Combined/all_models.csv`]);
  const cornerModels=firstCsv(root,[`${cornerBase}/all_models.csv`]);
  const goalHistory=firstCsv(root,[`${goalBase}/historical_observations.csv`,`${goalBase}/Data/historical_validated.csv`,`${goalBase}/Europe_Combined/historical_validated.csv`]);
  const cornerHistory=firstCsv(root,[`${cornerBase}/historical_corners_1h.csv`]);
  for(const m of day.matches)for(const market of ['goal','corner']){
   const sample=m[market],row=modelRow((market==='goal'?goalModels:cornerModels).rows,m);
   const href=matchReportHref(day.date,market,m.id),file=path.join(dist,href),dir=path.dirname(file);
   sample.report=href;
   const dailyTarget=path.resolve(dist,day.links[market]);
   const rawFile=(market==='goal'?goalModels:cornerModels).file;
   const history=market==='goal'?goalHistory:cornerHistory;
   const title=market==='goal'?'Gol până la 70′ inclusiv':'Sub 7,5 cornere în prima repriză';
   const definition=market==='goal'?'Minimum un gol, indiferent de echipă, la minutul afișat ≤70; inclusiv 45+. Fără prelungiri eliminatorii sau lovituri de departajare.':'Maximum 7 cornere cumulate ale ambelor echipe în prima repriză, inclusiv timpul adițional al acesteia.';
   const role=sample.userSelection?'Alegerea ta':(market==='goal'?m.analystGoal:m.analystCorner)?'Candidat analist':'Candidat comparat';
   const cutoff=Date.parse(m.recordedAt||`${day.date}T00:00:00+02:00`);
   const outcome=sample.outcome||{};
   const result={success:'Succes',fail:'Fail',postponed:'Amânat',void:'Anulat',cancelled:'Anulat',pending:'Încă neverificat'}[outcome.result]||'Încă neverificat';
   const observation=outcome.result==='success'||outcome.result==='fail'?(market==='goal'?outcome.minute?` · primul gol ${esc(outcome.minute)}′`:' · fără gol':number(outcome.corners)!=null?` · ${fmt(outcome.corners)} cornere 1H`:''):'';
   const metric=(label,value)=>`<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
   const formula=market==='goal'?'P(≥1 gol ≤70′) = 1 − exp(−λ). Scenariul AR 20% folosește λ × 0,8.':'P(cornere 1H ≤7) = exp(−λ) × Σ(k=0…7) λᵏ/k!. Scenariul AR 20% folosește λ × 1,2.';
   const detailed=sample.userSelection?.analysis_path;
   const reviewFile=path.join(root,`Analysis/Review_${day.date}/REVIEW.md`);
   const reviewLink=fs.existsSync(reviewFile)?link(local(dir,reviewFile),'Analiza succeselor și eșecurilor zilei'):'';
   const html=`<!doctype html><html lang="ro"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(m.home)} – ${esc(m.away)} · ${esc(title)} · PRED</title><style>
    :root{color-scheme:dark;--accent:${market==='goal'?'#3ee6c5':'#b99aff'}}*{box-sizing:border-box}body{margin:0;background:#071014;color:#dce9ed;font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:34px 24px 70px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}nav{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:38px}.eyebrow{color:var(--accent);font:12px/1.6 monospace;letter-spacing:2px}h1{font-size:clamp(26px,4vw,42px);line-height:1.2;margin:12px 0}h2{font-size:22px;margin-top:0}h3{font-size:17px;margin:28px 0 8px}p{margin:12px 0}.muted,small{color:#93acb7}small{display:block;font-size:12px}.badge{display:inline-block;padding:4px 10px;border:1px solid #31505b;border-radius:4px;font-size:12px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:24px}.metric,section{background:#0c1a20;border:1px solid #233e49;border-radius:8px}.metric{padding:18px}.metric span{display:block;color:#93acb7;font-size:12px;min-height:38px}.metric strong{color:var(--accent);font:26px/1.4 monospace}section{padding:24px;margin-top:22px}.notice{border-left:3px solid #deb778;padding:10px 16px;background:#1b211f;color:#dcc9a1}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px;text-align:left;border-bottom:1px solid #243a44;vertical-align:top}th{font-weight:500;color:#93acb7}code{font-family:monospace;color:var(--accent);overflow-wrap:anywhere}.sources{display:flex;gap:20px;flex-wrap:wrap}footer{margin-top:32px;color:#93acb7;font-size:12px}@media(max-width:650px){main{padding:22px 14px}.metrics{grid-template-columns:1fr 1fr}.metric{padding:12px}.metric strong{font-size:23px}section{padding:18px}nav{gap:14px}}
   </style></head><body data-match-id="${esc(m.id)}" data-market="${market}" data-date="${esc(day.date)}"><main>
    <nav>${link(local(dir,path.join(dist,'index.html')),'Înapoi la dashboard')}${link(local(dir,dailyTarget),'Raportul zilei')}</nav>
    <header><span class="eyebrow">PRED / RAPORTUL MECIULUI / ${market==='goal'?'GOL70':'CORNERE 1H'}</span><h1>${esc(m.home)} <span class="muted">vs</span> ${esc(m.away)}</h1><p class="muted">${esc(day.date)} · ${esc(m.kickoff)} RO · ${esc(m.league)}</p><span class="badge">${role}</span><h3>${title}</h3><p>${definition}</p></header>
    <div class="metrics">${metric('Model brut de screening',pct(sample.p))}${metric('Succes Rate AR 20%',pct(sample.stress))}${metric('Frecvență istorică',ratio(sample.wins,sample.n))}${metric('Frecvență acasă/deplasare',ratio(sample.venueWins,sample.venueN))}</div>
    <section><h2>Ce susțin datele salvate</h2><p class="notice">Raport statistic din arhiva zilei. Probabilități necalibrate. Acest document nu adaugă o analiză contextuală nouă și nu confirmă automat tactica, accidentările, loturile sau vremea.</p><p>${m.basis==='same_league_current_season'?'Bază: istoricul din campionatul intern curent.':'Bază: proiecție din campionatele interne într-o altă competiție; transferul comportamentului între competiții este incert.'}</p><p>Momentul snapshotului: <code>${esc(m.recordedAt||'Nespecificat')}</code>. Numărătorile și probabilitățile originale sunt păstrate; rezultatele ulterioare sunt afișate separat.</p><p>Cotă echitabilă matematică din screening: <strong>${sample.p>0?fmt(1/sample.p):'—'}</strong>. Nu este cotă disponibilă în piață.</p>${sample.final?`<p class="notice">Există o estimare finală distinctă: ${pct(sample.final.p)}; AR 20% ${pct(sample.final.stress)}; istoric ${ratio(sample.final.wins,sample.final.n)}; teren ${ratio(sample.final.venueWins,sample.final.venueN)}. Tabelul de mai jos păstrează intrările screeningului inițial. Detaliile excluderilor sunt în raportul zilei.</p>`:''}</section>
    <section><h2>Intrările și calculul screeningului</h2>${countsTable(row,market,m)}<p>λ sezon: <code>${fmt(row.lambda_season)}</code> · λ teren: <code>${fmt(row.lambda_venue)}</code> · λ combinat: <code>${fmt(row.lambda_combined)}</code>.</p><p>Pentru fiecare configurație, λ este jumătate din suma ratelor marcate/obținute și primite/cedate ale celor două echipe. λ combinat = 50% × λ sezon + 50% × λ teren. Ponderarea este euristică; eșantioanele se suprapun.</p><p>${formula}</p><p>Probabilitate doar sezon: ${pct(row.season_only)} · doar teren: ${pct(row.venue_only)}. Liniuța indică o valoare neînregistrată, nu zero.</p>${market==='goal'?'<p class="muted">Golurile ulterioare deschiderii scorului pot supraestima șansa primului gol în modelul Poisson. Frecvența primului gol rămâne o observație separată.</p>':`<p>Screening: exact 7 cornere ${ratio(sample.seven,sample.n)}; minimum 8 cornere ${ratio(sample.eight,sample.n)}. O repriză cu 8 sau mai multe cornere este Fail. Variația și dependența evenimentelor pot limita modelul Poisson.</p>`}</section>
    <section><h2>Istoricul disponibil pentru acest meci</h2>${historyTable(history.rows,row,m,'home',market,cutoff)}${historyTable(history.rows,row,m,'away',market,cutoff)}<p class="muted">O confruntare directă poate apărea în prezentarea ambelor echipe; tabelele nu se însumează pentru a recalcula frecvența agregată. Eșantionul agregat afișat mai sus provine din snapshotul original.</p></section>
    <section><h2>Rezultatul verificat, separat de predicție</h2><p><strong>${result}</strong>${observation}</p>${source(outcome.source,'Dovada rezultatului')}</section>
    <section><h2>Sursele acestui raport</h2><div class="sources">${source(sample.source,'Sursa numerică a partidei')}${rawFile?link(local(dir,path.join(root,rawFile)),'Intrările salvate CSV'):''}${history.file?link(local(dir,path.join(root,history.file)),'Istoricul salvat CSV'):''}${detailed?link(local(dir,path.join(root,detailed)),'Analiza salvată la alegere'):''}${link(local(dir,dailyTarget),'Raportul general al zilei')}${link(local(dir,path.resolve(dist,day.links.journal)),'Jurnalul zilei')}</div><p class="muted">Informațiile contextuale aprofundate există numai unde sunt documentate în analiza salvată. Prezența în comparația zilei nu înseamnă recomandare finală.</p></section>
    ${reviewLink?`<section><h2>Revizuirea după meciuri</h2><p>${reviewLink}</p><p class="muted">Concluziile ulterioare sunt separate de predicția originală. Raportul zilei precizează dovezile, ipotezele și limitele explicațiilor pentru succese și eșecuri.</p></section>`:''}
    <footer>Generat din arhiva locală la ${esc(archive.generatedAt)}. Fără research nou, fără cote de bookmaker și fără modificarea modelelor sau snapshoturilor.</footer>
   </main></body></html>`;
   fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(file,html.replace(/[ \t]+$/gm,''));count++;
  }
 }
 return count;
}
