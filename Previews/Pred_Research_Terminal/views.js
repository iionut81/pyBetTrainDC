const dateLabel=date=>new Date(date+'T12:00:00Z').toLocaleDateString('ro-RO',{day:'2-digit',month:'long',year:'numeric',timeZone:'Europe/Bucharest'});
const timeLabel=value=>value?new Date(value).toLocaleTimeString('ro-RO',{timeZone:'Europe/Bucharest',hour:'2-digit',minute:'2-digit'}):'—';
const resultLabels={success:'SUCCES',fail:'EȘEC',postponed:'AMÂNAT',pending:'NEVERIFICAT',void:'ANULAT',cancelled:'ANULAT'};
function choiceResult(result){return {success:'-Succes',fail:'-Fail',postponed:'-Amânat',void:'-Anulat',cancelled:'-Anulat'}[result]||'-În așteptare';}
function userChoiceBadges(value,market,card=false){
 const result=value.outcome?.result||'pending';
 const role=card?`badge ${market==='goal'?'teal':'violet'}-badge`:`table-badge ${market}`;
 return `<span class="choice-badges"><span class="${role}">${card?'ALEGEREA TA':'Alegerea ta'}</span><span class="table-badge choice-result ${esc(result)}">${choiceResult(result)}</span></span>`;
}
function scenarioHelp(market){return `Estimare necalibrată în scenariul nefavorabil: intensitatea ${market==='goal'?'golurilor −20%':'cornerelor +20%'}. Nu este rata istorică de succes.`;}
function automationNote(){
 return ARCHIVE.automation?'Agentul este programat zilnic la 08:00, ora României: verifică ziua precedentă, cercetează ambele piețe și actualizează dashboardul. Calculatorul trebuie să fie pornit, conectat la internet și utilizatorul autentificat. Pagina se reîncarcă după actualizare. Animația este decorativă.':'Nu există rulare zilnică programată sau flux live. Actualizăm când lucrăm în proiect. Animația este decorativă.';
}
const completedDays=()=>ARCHIVE.days.filter(d=>d.metrics.goal.played||d.metrics.corner.played).length;
const options=selected=>[...ARCHIVE.days].reverse().map(d=>`<option value="${esc(d.date)}" ${d.date===selected?'selected':''}>${esc(dateLabel(d.date))}</option>`).join('');
const reportLink=(href,label)=>href?`<a href="${esc(href)}" target="_blank" rel="noopener">${label} ↗</a>`:'';
function outcomeText(v,market){
 const o=v.outcome;
 if(o.result==='pending')return 'Rezultat încă neverificat';
 if(o.result==='postponed')return 'Amânare · exclusă din rată';
 if(o.result==='void'||o.result==='cancelled')return 'Anulat · exclus din rată';
 return `${resultLabels[o.result]} · ${market==='goal'?(o.minute?`primul gol ${esc(o.minute)}′`:'fără gol'):o.corners!==''?`total 1H ${esc(o.corners)}`:'observație indisponibilă'}`;
}
function choiceValue(m,market){
 const v=m[market];
 return v.final||v;
}
function riskText(m,market){
 if(DATA.date==='2026-11-03'&&m.ids.includes('401880179')&&market==='goal')return 'Burnley are antrenor nou. Debut: 0–0.';
 if(DATA.date==='2026-11-03'&&m.ids.includes('5836912')&&market==='corner')return 'Derby are variație mare. Un total 1H de 10.';
 return 'Model necalibrat. Verifică istoricul și scenariul de stres.';
}
function renderSelections(){
 $('#goalLegendRole').textContent=DATA.matches.some(m=>m.userGoal)?'Alegeri':'Candidat';
 $('#cornerLegendRole').textContent=DATA.matches.some(m=>m.userCorner)?'Alegeri':'Candidat';
 $('#selectionCards').innerHTML=['goal','corner'].map(market=>{
  const selections=DATA.matches.filter(m=>m[market].userSelection);
  const candidates=selections.length?selections:DATA.matches.filter(m=>market==='goal'?m.analystGoal:m.analystCorner).slice(0,1);
  if(!candidates.length)return `<article class="selection-card ${market}-card"><span class="eyebrow">${esc(marketName(market))}</span><h2>Fără alegere salvată</h2><p>Nicio selecție confirmată pentru această zi.</p></article>`;
  return candidates.map(m=>{const v=choiceValue(m,market),user=m[market].userSelection,parts=v.p==null?['—']:percent(v.p).replace('%','').split(',');return `<article class="selection-card ${market}-card"><div class="selection-top"><span class="eyebrow ${market==='goal'?'teal':'violet'}">${market==='goal'?'GOL PÂNĂ LA 70′':'CORNERE LA PAUZĂ'}</span>${user?userChoiceBadges(m[market],market,true):`<span class="badge ${market==='goal'?'teal':'violet'}-badge">CANDIDAT ANALIST</span>`}</div><div class="match-time">${esc(m.league)} <span>${esc(m.kickoff)} RO</span></div><h2>${esc(m.home)} <span>vs</span> ${esc(m.away)}</h2><p class="event-line">${market==='goal'?'Minimum un gol · oricine marchează':'Sub 7,5 cornere cumulate · inclusiv 45+'}</p><div class="probability"><strong>${parts[0]}<span>${parts[1]?','+parts[1]:''}</span><small>%</small></strong><div>${m[market].final?'MODEL FINAL BRUT':'MODEL BRUT'}<span>Încă necalibrat</span></div></div><div class="selection-numbers"><div><span class="scenario-label" title="${esc(scenarioHelp(market))}">Succes Rate <br>AR 20%</span><strong>${percent(v.stress)}</strong></div><div><span>ISTORIC</span><strong>${ratio(v.wins,v.n)}</strong></div><div><span>PE TEREN</span><strong>${ratio(v.venueWins,v.venueN)}</strong></div></div><div class="risk-line"><span>!</span>${riskText(m,market)}</div><p class="small-note">${outcomeText(m[market],market)}</p><div class="pick-actions">${choiceControl(m,market)}<button class="detail-button" data-detail="${esc(m.id)}" data-market="${market}">Deschide analiza <span>↗</span></button></div></article>`;}).join('');
 }).join('');
}
function renderCoverage(){
 const c=DATA.coverage,days=completedDays(),e=DATA.evidence.goal;
 $('.kpi-grid').innerHTML=`<article class="kpi"><div class="kpi-label">UNIVERSUL ZILEI <span>↗</span></div><div class="kpi-main"><strong>${c.total}</strong><span>partide salvate</span></div><div class="micro-bars" aria-hidden="true">${Array.from({length:c.total},()=>'<i></i>').join('')}</div></article><article class="kpi"><div class="kpi-label">GOL PÂNĂ LA 70′ <span class="teal">●</span></div><div class="kpi-main"><strong>${c.goal}<span>/${c.total}</span></strong><span>modele calculate</span></div><div class="kpi-foot">${e?.historical_validated!=null?`<span class="teal">${Number(e.historical_validated).toLocaleString('ro-RO')}</span> meciuri istorice validate`:'Istoric și surse în raportul zilei'}</div></article><article class="kpi"><div class="kpi-label">CORNERE PRIMA REPRIZĂ <span class="violet">●</span></div><div class="kpi-main"><strong>${c.corner}<span>/${c.total}</span></strong><span>modele calculabile</span></div><div class="kpi-foot"><span class="amber">${c.total-c.corner} fără model</span> · lipsă date / excluderi</div></article><article class="kpi"><div class="kpi-label">PILOT DE EVALUARE <span class="amber">◷</span></div><div class="kpi-main"><strong>${String(days).padStart(2,'0')}<span>/30</span></strong><span>zile cu rezultate</span></div><div class="kpi-foot">Primul bilanț <span class="amber">≠ validare completă</span></div></article>`;
 $('#inventoryCount').textContent=c.total+' INVENTARIATE';
 $('#archiveStatus').textContent=`${ARCHIVE.days.length} zile salvate · ${dateLabel(DATA.date)} · regenerat ${new Date(ARCHIVE.generatedAt).toLocaleString('ro-RO',{timeZone:'Europe/Bucharest'})} RO`;
 if(ARCHIVE.automation){const status={scheduled:'Programat',running:'În curs',completed:'Finalizat',partial:'Parțial',failed:'Eroare'}[ARCHIVE.automation.status]||ARCHIVE.automation.status;$('#archiveStatus').textContent+=` · Agent 08:00 RO: ${status}`;}
}
function renderReview(){
 const market=state.market,label=market==='goal'?'Gol70':'Cornere 1H',panel=$('#reviewPanel');
 const day=[...ARCHIVE.days].reverse().find(d=>d.date<=DATA.date&&d.reviews[market].some(r=>r.p!=null&&['success','fail'].includes(r.result)));
 panel.dataset.market=market;panel.dataset.reviewDate=day?.date||'';
 if(!day){panel.innerHTML=`<header class="panel-heading"><h2>Rezultatele urmează</h2></header><p class="panel-disclaimer">Niciun rezultat verificat pentru modelele ${label} din zilele disponibile până aici.</p>`;return;}
 const models=day.reviews[market].filter(r=>r.p!=null),played=models.filter(r=>['success','fail'].includes(r.result));
 const wins=played.filter(r=>r.result==='success').length,fails=played.length-wins,pending=models.filter(r=>r.result==='pending').length,excluded=models.filter(r=>['postponed','void','cancelled'].includes(r.result)).length;
 const all=day.metrics[market],withoutModel=day.reviews[market].filter(r=>r.p==null).length;
 const historyButton=(filter,text)=>`<button class="text-button${filter==='fail'?' review-fail-link':''}" data-review-history="${filter}" data-review-date="${esc(day.date)}" data-review-market="${market}">${text}</button>`;
 panel.innerHTML=`<header class="panel-heading"><div><span class="eyebrow">02 / REZULTATE VERIFICATE</span><h2>Ce a rămas după meciuri</h2></div><span class="small-note">${esc(dateLabel(day.date))}</span></header><div class="review-metric"><strong>${wins} <span>/ ${played.length}</span></strong><div><span class="${market==='corner'?'violet':'teal'}">${percent(wins/played.length)} reușite</span><small>Modele ${label} · ${pending} neverificate</small></div>${historyButton('all','Vezi istoricul ↗')}</div><div class="outcome-strip" id="outcomeStrip" aria-label="${label}: ${wins} succese și ${fails} eșecuri">${played.map(r=>`<span class="${r.result==='fail'?'failed':''}" data-review-match="${esc(r.id)}" title="${esc(r.home)}–${esc(r.away)}: ${resultLabels[r.result]} · ${market==='corner'?esc(r.corners)+' cornere 1H':r.minute?esc(r.minute)+'′':'fără gol'}"></span>`).join('')}</div><div class="review-footer"><span><i class="dot teal-dot"></i>${wins} succese</span><span><i class="dot red-dot"></i>${fails?historyButton('fail',fails+' '+(fails===1?'eșec':'eșecuri')+' ↗'):'0 eșecuri'}</span><span><i class="dot gray-dot"></i>${excluded} amânări/anulări excluse</span></div>${withoutModel?`<p class="review-scope-note">Toate partidele verificate: ${all.success}/${all.played} reușite. ${withoutModel} ${withoutModel===1?'partidă fără model prospectiv, exclusă':'partide fără model prospectiv, excluse'} din bilanțul modelelor.</p>`:''}`;
}
function renderEvidence(){
 const events=[];
 const modelTimes=DATA.matches.map(m=>m.recordedAt).filter(Boolean).sort();
 if(modelTimes.length)events.push({at:modelTimes.at(-1),title:'Comparația Gol70 salvată',detail:`${DATA.coverage.goal} modele · estimări originale`,tag:'SALVAT',market:'goal'});
 for(const m of DATA.matches){if(m.corner.final)events.push({at:m.corner.final.recordedAt,title:`${m.home}–${m.away} · analiza cornerelor`,detail:`${m.corner.final.n} observații în modelul final`,tag:'ANALIZĂ',market:'corner'});}
 for(const u of DATA.userSelections)events.push({at:u.selected_at_ro,title:`${shortName(u.home)}–${shortName(u.away)} confirmat de tine`,detail:u.market==='GOAL_BY_70'?'Alegerea ta · Gol70':'Alegerea ta · cornere 1H',tag:choiceResult(DATA.matches.find(m=>m.ids.includes(u.match_id))?.[u.market==='GOAL_BY_70'?'goal':'corner'].outcome?.result),market:u.market==='GOAL_BY_70'?'goal':'corner'});
 $('#evidencePanel').innerHTML=`<header class="panel-heading"><div><span class="eyebrow">03 / EVIDENCE LOG</span><h2>Decizii cu urmă</h2></div><span class="small-note">${esc(dateLabel(DATA.date))}</span></header>`+events.sort((a,b)=>(a.at||'').localeCompare(b.at||'')).map(e=>`<div class="log-row"><time>${esc(timeLabel(e.at))}</time><i class="${e.market==='goal'?'teal':'violet'}-dot"></i><div>${esc(e.title)}<small>${esc(e.detail)}</small></div><span>${e.tag}</span></div>`).join('');
}
function shortName(name){return name==='Queens Park Rangers'?'QPR':name;}
function historyReportHref(day,market,id){
 return day.matches.find(m=>m.id===id||m.ids.includes(id))?.[market].report;
}
function historyMatchReport(day,market,id,label){
 const href=historyReportHref(day,market,id);
 return href?`<a class="history-match-report" href="${esc(href)}" target="_blank" rel="noopener" aria-label="Raportul meciului ${esc(label)} · ${esc(dateLabel(day.date))} · ${esc(marketName(market))}">${esc(label)}</a>`:esc(label);
}
function renderHistory(){
 const day=ARCHIVE.days.find(d=>d.date===state.historyDate)||DATA,market=state.historyMarket,v=day.metrics[market];
 const userIds=new Set(day.matches.filter(m=>m[market].userSelection).map(m=>m.id));
 const rows=day.reviews[market].filter(m=>state.result==='all'||m.result===state.result).sort((a,b)=>Number(userIds.has(b.id))-Number(userIds.has(a.id))||a.rank-b.rank);
 $('#historyDateSelect').innerHTML=options(day.date);
 $('#historyMarketLabel').textContent=market==='goal'?'UNIVERSUL GOL70':'UNIVERSUL CORNERE 1H';
 $('#historyObservationLabel').textContent=market==='goal'?'PRIMUL GOL':'CORNERE LA PAUZĂ';
 $('#historySummary').innerHTML=`<div><span class="eyebrow">RETROSPECTIVĂ / ${esc(dateLabel(day.date))}</span><h2>Rezultate, inclusiv cele incomode.</h2><p>${v.total} modele · ${v.played} rezultate verificate · ${v.pending} neverificate · ${v.postponed} amânări excluse.</p></div><span class="big-result">${v.success}<span>reușite</span><small>${v.fail} eșecuri</small></span>`;
 $('#historyBody').innerHTML=rows.length?rows.map(m=>{const observation=m.result==='postponed'?'Nu s-a jucat':m.result==='pending'?'—':market==='corner'?(m.corners!==''?esc(m.corners):'—'):m.minute?esc(m.minute)+'′':'Fără gol';return `<tr><td class="rank">${m.rank}</td><td class="table-match"><strong>${historyMatchReport(day,market,m.id,`${m.home} — ${m.away}`)}</strong><small class="history-report-links"><span>Raportul meciului ↗</span>${reportLink(m.source,'Sursa rezultatului')}</small></td><td class="table-muted">${percent(m.p)}</td><td class="table-muted">${esc(m.score||'—')}</td><td class="table-muted">${observation}</td><td class="history-result ${esc(m.result)}">${resultLabels[m.result]||esc(m.result)}</td></tr>`;}).join(''):'<tr><td colspan="6" class="empty-cell">Niciun rezultat în acest filtru.</td></tr>';
 $('#historyUserCards').innerHTML=day.matches.filter(m=>m[market].userSelection).map(m=>`<article class="history-user-card">${userChoiceBadges(m[market],market)}<h3>${historyMatchReport(day,market,m.id,`${m.home} — ${m.away}`)}</h3><p>${outcomeText(m[market],market)}</p><p class="small-note">Estimare la alegere: ${percent(m[market].userSelection.p)}</p><div class="history-report-links">${reportLink(m[market].report,'Raportul meciului')}${reportLink(m[market].outcome.source,'Sursa rezultatului')}</div></article>`).join('');
 $$('[data-result]').forEach(b=>{b.classList.toggle('active',b.dataset.result===state.result);b.setAttribute('aria-pressed',String(b.dataset.result===state.result));});
 $$('[data-history-market]').forEach(b=>{b.classList.toggle('active',b.dataset.historyMarket===market);b.setAttribute('aria-pressed',String(b.dataset.historyMarket===market));});
}
function renderValidation(){
 const days=completedDays();
 const panels=['goal','corner'].map(market=>{
  const rows=ARCHIVE.days.flatMap(d=>d.reviews[market]).filter(r=>r.result==='success'||r.result==='fail');
  const n=rows.length,win=rows.filter(r=>r.result==='success').length,scored=rows.filter(r=>r.p!=null),mean=scored.length?scored.reduce((s,r)=>s+r.p,0)/scored.length:null,brier=scored.length?scored.reduce((s,r)=>s+(r.p-(r.result==='success'?1:0))**2,0)/scored.length:null;
  const chosen=ARCHIVE.days.flatMap(d=>d.matches.filter(m=>m[market].userSelection).map(m=>m[market])).filter(v=>['success','fail'].includes(v.outcome.result)),chosenWins=chosen.filter(v=>v.outcome.result==='success').length;
  return `<article class="panel"><header class="panel-heading"><div><span class="eyebrow">${market==='goal'?'GOL70':'CORNERE 1H'} · TOATE ZILELE SALVATE</span><h2>Model vs. observație</h2></div></header><div class="validation-row"><span>Probabilitate medie · screening</span><strong>${percent(mean)}</strong></div><div class="compare-bar"><i style="width:${(mean||0)*100}%"></i></div><div class="validation-row"><span>Rată observată · ${n} rezultate verificate</span><strong class="teal">${n?percent(win/n):'—'}</strong></div><div class="compare-bar actual"><i style="width:${n?win/n*100:0}%"></i></div><div class="validation-row"><span>Scor Brier · ${scored.length} probabilități</span><strong>${brier==null?'—':brier.toLocaleString('ro-RO',{minimumFractionDigits:4,maximumFractionDigits:4})}</strong></div><div class="validation-row"><span>Alegerile tale · rezultate verificate</span><strong>${chosenWins}/${chosen.length}</strong></div><p class="panel-disclaimer">Doar rezultate verificate; datele lipsă și amânările sunt excluse. Media și Brier folosesc screeningul original. Selecțiile tale sunt raportate separat; un eșantion mic nu validează modelul.</p></article>`;
 }).join('');
 $('#validationView').innerHTML=`<div class="validation-banner"><span class="eyebrow amber">EVALUARE PROSPECTIVĂ</span><h2>Încrederea se măsoară.<br><span>Nu se declară.</span></h2><p>${days===1?'O zi verificată':days+' zile cu rezultate verificate'}. Pot exista zile verificate parțial. Modelul rămâne necalibrat.</p><div class="pilot-grid" id="pilotGrid" aria-label="${days} zile cu rezultate din ținta pilot de 30">${Array.from({length:Math.max(30,days)},(_,i)=>`<span class="${i<days?'verified':''}" title="${i<days?'Zi cu rezultate verificate':'Zi de evaluare necompletată'}"></span>`).join('')}</div><div class="pilot-label"><span>${days} zile cu rezultate</span><span>Primul bilanț: 30 zile</span></div></div><div class="validation-grid">${panels}</div>`;
}
function showDetail(id,market=state.market){
 const m=DATA.matches.find(m=>m.id===id);if(!m)return;
 const sample=m[market],v=choiceValue(m,market),user=sample.userSelection;
 const role=user?'Alegerea utilizatorului':(market==='goal'?m.analystGoal:m.analystCorner)?'Candidat propus de analist':'Candidat comparat';
 const derby=DATA.date==='2026-11-03'&&m.ids.includes('5836912')&&market==='corner';
 const extra=sample.final?`<p>Screeningul inițial: <b>${percent(sample.p)}</b>. Estimarea finală este păstrată separat: ${percent(v.p)}.</p>`:'';
 showDialog(`<h2 id="dialogTitle">${esc(m.home)} <span class="dim">vs</span> ${esc(m.away)}</h2><div class="dialog-match-meta">${esc(dateLabel(DATA.date))} · ${esc(m.kickoff)} RO · ${esc(m.league)}</div>${user?userChoiceBadges(sample,market):`<span class="table-badge ${market}">${esc(role)}</span>`}<p><b>${esc(marketName(market))}</b><br>${market==='goal'?'Oricare echipă marchează la minutul afișat ≤70, inclusiv 45+.':'Cel mult 7 cornere cumulate la pauză, inclusiv timpul adițional al primei reprize.'}</p><div class="dialog-metrics"><div class="dialog-metric"><span>${sample.final?'MODEL FINAL BRUT':'MODEL BRUT DE SCREENING'}</span><strong>${percent(v.p)}</strong></div><div class="dialog-metric"><span class="scenario-label" title="${esc(scenarioHelp(market))}">Succes Rate <br>AR 20%</span><strong>${percent(v.stress)}</strong></div></div><div class="dialog-pills"><span>Istoric: ${ratio(v.wins,v.n)}</span><span>Acasă/deplasare: ${ratio(v.venueWins,v.venueN)}</span>${market==='corner'&&sample.p!=null?`<span>Screening ≥8: ${sample.eight}/${sample.n}</span><span>Exact 7: ${sample.seven}/${sample.n}</span>`:''}</div><p class="dialog-risk">${riskText(m,market)}</p>${extra}${derby?'<p>Derby–Stoke este exclus din modelul final: furnizorii raportează 5 sau 6 cornere 1H. Rămân 24/25 reușite. Frecvența pe teren: 10/11 (90,91%). În Derby–Cardiff pragul de 8 cornere fusese atins în minutul 18, înaintea primului gol.</p>':''}<p>${sample.p==null?'Lipsesc datele necesare calculului; lipsa nu este zero.':m.basis!=='same_league_current_season'?'Proiecție din campionatele interne într-o altă competiție.':'Eșantioanele sezon/teren se suprapun. Estimarea brută și stresul nu sunt probabilități validate.'}</p><p>${outcomeText(sample,market)}</p>${user?`<p>Alegere înregistrată la ${esc(new Date(user.selected_at_ro).toLocaleString('ro-RO',{timeZone:'Europe/Bucharest'}))} RO. Probabilitate salvată la alegere: ${percent(user.p)}.</p>`:''}<div class="pick-actions">${choiceControl(m,market)}</div><div class="dialog-links">${reportLink(sample.report,'Raportul meciului')}${reportLink(DATA.links[market],'Raportul zilei')}${reportLink(sample.source,'Sursa numerică')}${reportLink(sample.outcome.source,'Sursa rezultatului')}${reportLink(DATA.links.journal,'Jurnalul zilei')}</div>`,market==='goal'?'GOL PÂNĂ LA 70′ / DOVEZI':'CORNERE PRIMA REPRIZĂ / DOVEZI');
}
function showMethod(){
 showDialog(`<h2 id="dialogTitle">Analiză → alegere → rezultat → bilanț.</h2><p>Dashboard local cu ${ARCHIVE.days.length} zile salvate. Data afișată: ${esc(dateLabel(DATA.date))}. Datele se regenerează din fișierele proiectului după analize, alegeri sau verificarea rezultatelor.</p><ul class="method-list"><li>Inventarul și modelele fiecărei zile sunt păstrate înaintea rezultatelor. Acoperirea este cea documentată în raport; interfața nu verifică singură programul ligilor.</li><li>Poisson, ponderare 50/50 sezon și acasă/deplasare; stres Gol70 −20%, cornere +20%. Nu recalibrăm retrospectiv.</li><li>Radarul și tabelul arată screeningul. Estimările finale sunt separate în fișe.</li><li>Rezultatele sunt verificate și adăugate separat, cu sursă. Lipsurile nu sunt zero. Alegerile utilizatorului sunt distincte de propunerile analistului.</li><li>30 de zile reprezintă primul bilanț pilot, nu validare garantată.</li></ul><p>${automationNote()}</p><div class="dialog-links">${reportLink(DATA.links.goal,'Analiza Gol70')}${reportLink(DATA.links.corner,'Analiza cornerelor')}${reportLink('../../../Analysis/Selection_History/DAILY_WORKFLOW.md','Fluxul salvat')}</div>`,'METODĂ / ISTORIC / ACTUALIZĂRI');
}
function selectDay(date){
 const day=ARCHIVE.days.find(d=>d.date===date);if(!day)return;
 DATA=day;state.league='all';state.search='';state.result='all';state.historyDate=date;
 $('#searchInput').value='';if($('#detailDialog').open)$('#detailDialog').close();
 renderDay();setView(state.view);
}
function renderDay(){
 $('#dateSelect').innerHTML=options(DATA.date);$('#snapshotTime').textContent=DATA.snapshotAt;
 const leagues=[...new Map(DATA.matches.map(m=>[m.competition,m.league])).entries()];
 $('#leagueSelect').innerHTML='<option value="all">Toate competițiile</option>'+leagues.map(([id,name])=>`<option value="${esc(id)}">${esc(name)}</option>`).join('');
 $('.league-pills').innerHTML='<button data-league="all" class="active">Toate</button>'+leagues.slice(0,3).map(([id,name])=>`<button data-league="${esc(id)}">${esc(name)}</button>`).join('');
 renderCoverage();renderSelections();renderReview();renderEvidence();renderTable();renderRadar();renderHistory();renderValidation();
 window.PRED_PREVIEW={version:2,matchCount:DATA.matches.length,cornerEligible:DATA.coverage.corner,snapshotDate:DATA.date,archiveDays:ARCHIVE.days.length};
}
renderDay();setView('overview');initializeChoices();
