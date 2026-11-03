'use strict';
const ARCHIVE=JSON.parse(document.getElementById('snapshotData').textContent);
let DATA=ARCHIVE.days.find(d=>d.date===ARCHIVE.latestDate);
const state={view:'overview',market:'goal',league:'all',search:'',result:'all',historyMarket:'goal',historyDate:[...ARCHIVE.days].reverse().find(d=>d.metrics.goal.played||d.metrics.corner.played)?.date||DATA.date};
const $=selector=>document.querySelector(selector);
const $$=selector=>[...document.querySelectorAll(selector)];
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=value=>value==null?'—':(value*100).toLocaleString('ro-RO',{minimumFractionDigits:2,maximumFractionDigits:2})+'%';
const ratio=(wins,n)=>n?`${wins}/${n}`:'—';
const normalize=text=>String(text).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
const marketName=market=>market==='goal'?'Gol până la 70′ inclusiv':'Sub 7,5 cornere în prima repriză';
function filteredMatches(){
 return DATA.matches.filter(m=>(state.league==='all'||m.competition===state.league)&&normalize(m.home+' '+m.away).includes(normalize(state.search))).sort((a,b)=>Number(Boolean(b[state.market].userSelection))-Number(Boolean(a[state.market].userSelection))||(b[state.market].p??-1)-(a[state.market].p??-1));
}
function badge(m){
 if(state.market==='corner'&&!m.corner.eligible)return '<span class="table-badge missing">Lipsă date 1H</span>';
 if(state.market==='goal'&&m.userGoal)return userChoiceBadges(m.goal,'goal');
 if(state.market==='corner'&&m.userCorner)return userChoiceBadges(m.corner,'corner');
 if(state.market==='corner'&&m.analystCorner)return '<span class="table-badge corner">Candidat analist</span>';
 if(m.basis!=='same_league_current_season')return '<span class="table-badge projection">Proiecție internă</span>';
 return '<span class="table-badge">Istoric comparabil</span>';
}
function renderTable(){
 $('#scenarioColumnLabel').title=scenarioHelp(state.market);
 const rows=filteredMatches();
 const modelRanks=new Map([...DATA.matches].sort((a,b)=>(b[state.market].p??-1)-(a[state.market].p??-1)).map((m,i)=>[m.id,i+1]));
 $('#matchesBody').innerHTML=rows.length?rows.map((m,i)=>{
  const v=m[state.market],chosen=state.market==='goal'&&m.userGoal?'picked':state.market==='corner'&&(m.userCorner||m.analystCorner)?'corner-picked':'';
  return `<tr class="${chosen}" data-match="${esc(m.id)}" tabindex="0" aria-label="Deschide analiza ${esc(m.home)} — ${esc(m.away)}"><td class="rank">${String(modelRanks.get(m.id)).padStart(2,'0')}</td><td class="table-match"><strong>${esc(m.home)} <span>vs</span> ${esc(m.away)}</strong><small>${esc(m.league)}</small></td><td class="table-muted">${esc(m.kickoff)}</td><td class="table-prob ${state.market==='corner'?'violet':''}">${percent(v.p)}</td><td class="table-muted">${percent(v.stress)}</td><td class="table-muted">${v.p==null?'—':ratio(v.wins,v.n)}</td><td>${badge(m)}</td><td class="row-arrow">↗</td></tr>`;
 }).join(''):'<tr><td colspan="8" class="empty-cell">Nicio partidă nu corespunde filtrelor. Încearcă alt nume sau altă competiție.</td></tr>';
 $('#tableCount').textContent=`${rows.length} / ${DATA.matches.length} partide · ${state.market==='goal'?'Gol70':'Cornere 1H'} · alegerile tale primele, apoi model ↓`;
 $$('.market-switch button').forEach(b=>{b.classList.toggle('active',b.dataset.market===state.market);b.setAttribute('aria-pressed',String(b.dataset.market===state.market));});
}
function renderRadar(){
 const rows=filteredMatches(),points=rows.filter(m=>m.corner.p!=null&&m.goal.p!=null);
 const W=760,H=390,L=57,R=710,T=32,B=328;
 const supported=DATA.matches.filter(m=>m.goal.p!=null&&m.corner.p!=null);
 const xmin=Math.min(80,Math.floor(Math.min(...supported.map(m=>m.goal.p*100))/5)*5);
 const ymin=Math.min(70,Math.floor(Math.min(...supported.map(m=>m.corner.p*100))/5)*5);
 const x=p=>L+(p*100-xmin)/(100-xmin)*(R-L),y=p=>B-(p*100-ymin)/(100-ymin)*(B-T);
 let chart=`<svg class="radar-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Estimări brute Gol70 pe axa orizontală și cornere 1H pe axa verticală"><defs><radialGradient id="radarGlow"><stop offset="0" stop-color="#44e9c8" stop-opacity=".16"/><stop offset="1" stop-color="#44e9c8" stop-opacity="0"/></radialGradient><filter id="nodeGlow" x="-150%" y="-150%" width="400%" height="400%"><feGaussianBlur stdDeviation="4"/></filter></defs><ellipse cx="570" cy="105" rx="170" ry="130" fill="url(#radarGlow)"/>`;
 for(let value=ymin;value<=100;value+=5)chart+=`<line class="radar-grid" x1="${L}" y1="${y(value/100)}" x2="${R}" y2="${y(value/100)}"/><text class="radar-axis" x="${L-12}" y="${y(value/100)+4}" text-anchor="end">${value}%</text>`;
 for(let value=xmin;value<=100;value+=5)chart+=`<line class="radar-grid" x1="${x(value/100)}" y1="${T}" x2="${x(value/100)}" y2="${B}"/><text class="radar-axis" x="${x(value/100)}" y="${B+23}" text-anchor="middle">${value}%</text>`;
 chart+=`<text class="radar-axis-title" x="${L}" y="15">CORNERE 1H · MODEL BRUT ↑</text><text class="radar-axis-title" x="${R}" y="${B+47}" text-anchor="end">GOL PÂNĂ LA 70′ · MODEL BRUT →</text>`;
 chart+=`<path d="M${L} ${T}v${B-T}h${R-L}" fill="none" stroke="#36535b"/>`;
 for(const m of points){
  const cx=x(m.goal.p),cy=y(m.corner.p),chosen=m.userGoal||m.userCorner||m.analystCorner,color=m.userCorner||m.analystCorner?'#ba9cfa':m.userGoal?'#44e9c8':m.competition==='uefa.champions'?'#7896b0':'#479a91';
  let label='';
  if(chosen){const right=cx<(L+R)/2;label=`<text class="node-label" x="${cx+(right?14:-14)}" y="${cy-15}" text-anchor="${right?'start':'end'}">${esc(m.home)}–${esc(m.away)}</text>`;}
  chart+=`<g class="radar-node" data-radar-match="${esc(m.id)}" tabindex="0" role="button" aria-label="${esc(m.home)} — ${esc(m.away)}, Gol70 ${percent(m.goal.p)}, cornere ${percent(m.corner.p)}"><title>${esc(m.home)} — ${esc(m.away)} | Gol70 ${percent(m.goal.p)} | Cornere screening ${percent(m.corner.p)}</title><circle cx="${cx}" cy="${cy}" r="16" fill="${color}" opacity="${chosen?'.25':'.08'}" filter="url(#nodeGlow)"/>${chosen?`<circle class="outer-ring node-signal" cx="${cx}" cy="${cy}" r="22" stroke="${color}"/><circle class="outer-ring" cx="${cx}" cy="${cy}" r="12" stroke="${color}"/>`:''}<circle class="node-core" cx="${cx}" cy="${cy}" r="${chosen?5.5:4}" fill="${color}" stroke="${chosen?'#c7f5eb':'#0e242b'}" stroke-width="${chosen?1:1.5}"/>${label}</g>`;
 }
 chart+='</svg>';
 $('#radar').innerHTML=chart;
 $('#radarCount').textContent=`${points.length} puncte / ${rows.length} partide${rows.length!==points.length?' · '+(rows.length-points.length)+' fără date cornere':''}`;
}
function setLeague(league){state.league=league;$('#leagueSelect').value=league;$$('.league-pills button').forEach(b=>{b.classList.toggle('active',b.dataset.league===league);b.setAttribute('aria-pressed',String(b.dataset.league===league));});renderTable();renderRadar();}
function setMarket(market){state.market=market;renderTable();renderRadar();}
function setView(view){
 state.view=view;
 $('#overviewView').classList.toggle('hidden',view!=='overview');
 $('#matchesSection').classList.toggle('hidden',!['overview','matches'].includes(view));
 $('#historyView').classList.toggle('hidden',view!=='history');
 $('#validationView').classList.toggle('hidden',view!=='validation');
 $('.kpi-grid').classList.toggle('hidden',['history','validation'].includes(view));
 const copy={overview:['Din date, în decizii.','Gol până la 70′. Cornere la pauză. Fiecare alegere, verificabilă.'],matches:[`Toate cele ${DATA.matches.length} de partide.`,'Filtrează, compară și deschide dovezile din spatele fiecărei estimări.'],history:['Istoricul rămâne întreg.','Alege ziua și piața. Estimările originale sunt păstrate.'],validation:['Mai întâi măsurăm.','Calitatea datelor, calibrarea și performanța selecțiilor se evaluează separat.']}[view];
 $('#pageTitle').textContent=copy[0];$('#pageSubtitle').textContent=copy[1];
 $('#dateSelect').value=DATA.date;
 $('#snapshotTime').textContent=DATA.snapshotAt;
 $$('.nav-icon').forEach(b=>{b.classList.toggle('active',b.dataset.view===view);if(b.dataset.view===view)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
 if(view==='matches')$('.table-scroll').style.maxHeight='65vh';else $('.table-scroll').style.maxHeight='';
 window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
}
function showDialog(content,eyebrow='ANALIZA MECIULUI'){
 $('#dialogEyebrow').textContent=eyebrow;$('#dialogContent').innerHTML=content;
 if(!$('#detailDialog').open)$('#detailDialog').showModal();
}
function toast(message){$('#toast').textContent=message;$('#toast').classList.add('visible');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('#toast').classList.remove('visible'),3200);}
function exportCsv(){
 const rows=[['Data','Gazda','Oaspete','Competitie','Ora_RO','Piata','Model_brut_screening','Succes Rate AR 20%','Istoric_succes','Istoric_n','Tip_date','Rezultat']];
 for(const m of filteredMatches()){const v=m[state.market];rows.push([DATA.date,m.home,m.away,m.league,m.kickoff,state.market==='goal'?'GOAL_BY_70':'CORNERS_1H_UNDER_7_5',v.p??'',v.stress??'',v.p==null?'':v.wins,v.p==null?'':v.n,m.basis,v.outcome.result]);}
 const text='\uFEFF'+rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`PRED_${DATA.date}_${state.market}_screening.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('CSV exportat pentru ziua și piața selectate.');
}
document.addEventListener('click',e=>{
 const historyMarket=e.target.closest('[data-history-market]');if(historyMarket){state.historyMarket=historyMarket.dataset.historyMarket;state.result='all';renderHistory();return;}
 const view=e.target.closest('[data-view]');if(view){setView(view.dataset.view);return;}
 const league=e.target.closest('[data-league]');if(league){setLeague(league.dataset.league);return;}
 const detail=e.target.closest('[data-detail]');if(detail){showDetail(detail.dataset.detail,detail.dataset.market);return;}
 const market=e.target.closest('[data-market]');if(market){setMarket(market.dataset.market);return;}
 const row=e.target.closest('[data-match]');if(row){showDetail(row.dataset.match);return;}
 const node=e.target.closest('[data-radar-match]');if(node){showDetail(node.dataset.radarMatch,DATA.matches.find(m=>m.id===node.dataset.radarMatch)?.userCorner?'corner':state.market);return;}
 const result=e.target.closest('[data-result]');if(result){state.result=result.dataset.result;renderHistory();}
});
document.addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ')return;const row=e.target.closest('[data-match]'),node=e.target.closest('[data-radar-match]');if(row||node){e.preventDefault();showDetail(row?row.dataset.match:node.dataset.radarMatch,DATA.matches.find(m=>m.id===node?.dataset.radarMatch)?.userCorner?'corner':state.market);}});
$('#searchInput').addEventListener('input',e=>{state.search=e.target.value;renderTable();renderRadar();});
$('#leagueSelect').addEventListener('change',e=>setLeague(e.target.value));
$('#dateSelect').addEventListener('change',e=>selectDay(e.target.value));
$('#historyDateSelect').addEventListener('change',e=>{state.historyDate=e.target.value;renderHistory();});
$('#methodButton').addEventListener('click',showMethod);$('#footerMethod').addEventListener('click',showMethod);
$('#closeDialog').addEventListener('click',()=>$('#detailDialog').close());
$('#detailDialog').addEventListener('click',e=>{if(e.target===$('#detailDialog')){const rect=e.target.getBoundingClientRect();if(e.clientX<rect.left||e.clientX>rect.right||e.clientY<rect.top||e.clientY>rect.bottom)e.target.close();}});
$('#exportButton').addEventListener('click',exportCsv);
$('.brand-symbol').addEventListener('click',e=>{e.preventDefault();setView('overview');});
