import fs from 'node:fs';
const base='Analysis/System_Audit_2026-11-17';
const rows=JSON.parse(fs.readFileSync(base+'/cohort.json','utf8'));
const numeric=r=>r.p!==null&&r.p!==''&&Number.isFinite(Number(r.p));
const verified=r=>['success','fail'].includes(r.result);
const daily=[];
for(const market of ['goal','corner'])for(const date of [...new Set(rows.map(r=>r.date))].filter(d=>d<'2026-11-17').sort()){
 const all=rows.filter(r=>r.market===market&&r.date===date&&numeric(r)).sort((a,b)=>b.p-a.p||a.id.localeCompare(b.id));
 if(!all.length)continue;
 const known=all.filter(verified),top=all[0],top3=all.slice(0,3);
 daily.push({date,market,eligible:all.length,verified:known.length,pending_or_void:all.length-known.length,selected_id:top.id,home:top.home,away:top.away,p:top.p,result:top.result,y:verified(top)?top.y:null,same_day_all_verified_rate:known.length?known.reduce((s,r)=>s+r.y,0)/known.length:null,top3_ids:top3.map(r=>r.id).join('|'),top3_verified:top3.filter(verified).length,top3_successes:top3.filter(verified).reduce((s,r)=>s+r.y,0),full_day_resolved:known.length===all.length});
}
const summary={};for(const market of ['goal','corner']){const d=daily.filter(r=>r.market===market&&r.y!==null),complete=d.filter(r=>r.full_day_resolved);summary[market]={days_selected:daily.filter(r=>r.market===market).length,days_selected_with_verified_outcome:d.length,top1_successes:d.reduce((s,r)=>s+r.y,0),top1_rate:d.reduce((s,r)=>s+r.y,0)/d.length,equal_day_all_verified_rate:d.reduce((s,r)=>s+r.same_day_all_verified_rate,0)/d.length,fully_resolved_days:complete.length,top1_rate_fully_resolved:complete.reduce((s,r)=>s+r.y,0)/complete.length};}
const headers=Object.keys(daily[0]);fs.writeFileSync(base+'/DAILY_SELECTION_BASELINE.csv',[headers,...daily.map(r=>headers.map(k=>r[k]??''))].map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n'));
fs.writeFileSync(base+'/daily_selection_baseline.json',JSON.stringify({summary,daily,limits:['Descriptive results from already inspected development cohort','Same-day comparator excludes unverified outcomes; fully resolved days shown separately','Raw top ties resolved by fixed id; no pending top replaced by runner-up','No claim of future-data-independent validation']},null,2));console.log(summary);
