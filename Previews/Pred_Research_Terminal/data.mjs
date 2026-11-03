import fs from 'node:fs';
import path from 'node:path';

export function parseCsv(input){
 const rows=[];let row=[],cell='',quoted=false;
 input=input.replace(/^\uFEFF/,'');
 for(let i=0;i<input.length;i++){
  const c=input[i];
  if(c==='"'){if(quoted&&input[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
  else if(c===','&&!quoted){row.push(cell);cell='';}
  else if((c==='\r'||c==='\n')&&!quoted){if(c==='\r'&&input[i+1]==='\n')i++;row.push(cell);if(row.some(Boolean))rows.push(row);row=[];cell='';}
  else cell+=c;
 }
 if(quoted)throw Error('Unterminated CSV quote');
 if(cell||row.length){row.push(cell);rows.push(row);}
 const keys=rows.shift()??[];
 return rows.map(values=>Object.fromEntries(keys.map((k,i)=>[k,values[i]??''])));
}
const num=v=>v==null||v===''?null:Number(v);
const truth=v=>String(v).toLowerCase()==='true';
const labels={'eng.1':'Premier League','eng.2':'Championship','ita.1':'Serie A','ita.2':'Serie B','esp.1':'La Liga','esp.2':'LaLiga 2','ger.1':'Bundesliga','ger.2':'2. Bundesliga','fra.1':'Ligue 1','por.1':'Primeira Liga','ned.1':'Eredivisie','bel.1':'Pro League','swe.1':'Allsvenskan','nor.1':'Eliteserien','den.1':'Superliga','den.cup':'Cupa Danemarcei','gre.1':'Super League Grecia','rou.1':'SuperLiga România','rom.1':'SuperLiga România','srb.1':'SuperLiga Serbia','uefa.champions':'Champions League'};
const short=name=>({'Feyenoord Rotterdam':'Feyenoord','Internazionale':'Inter','Queens Park Rangers':'QPR','Bodo/Glimt':'Bodø/Glimt','AS Roma':'Roma','IK Sirius':'Sirius','Bayern Munich':'Bayern München'}[name]??name);
const normalizeResult=r=>r?.startsWith('postponed')?'postponed':['success','fail','void','cancelled'].includes(r)?r:'pending';
const time=value=>!value?'—':/^\d\d:\d\d$/.test(value)?value:new Date(value).toLocaleTimeString('ro-RO',{timeZone:'Europe/Bucharest',hour:'2-digit',minute:'2-digit'});
export function metrics(rows){
 const evaluated=rows.filter(r=>r.result==='success'||r.result==='fail');
 const scored=evaluated.filter(r=>r.p!=null),success=evaluated.filter(r=>r.result==='success').length;
 return {total:rows.length,played:evaluated.length,success,fail:evaluated.length-success,postponed:rows.filter(r=>r.result==='postponed').length,pending:rows.filter(r=>r.result==='pending').length,hit_rate:evaluated.length?success/evaluated.length:null,mean_predicted_probability:scored.length?scored.reduce((s,r)=>s+r.p,0)/scored.length:null,brier:scored.length?scored.reduce((s,r)=>s+(r.p-(r.result==='success'?1:0))**2,0)/scored.length:null,scored:scored.length};
}
export function loadArchive(root){
 const exists=p=>fs.existsSync(path.join(root,p));
 const text=p=>{const b=fs.readFileSync(path.join(root,p));return b.toString(b[0]===255&&b[1]===254?'utf16le':'utf8').replace(/^\uFEFF/,'');};
 const csv=p=>exists(p)?parseCsv(text(p)):[];
 const json=p=>exists(p)?JSON.parse(text(p)):null;
 const history='Analysis/Selection_History';
 const names=fs.readdirSync(path.join(root,history));
 const dates=new Set(names.map(n=>n.match(/^(\d{4}-\d{2}-\d{2})(?:_(?:goal70|corners1h)_candidates\.csv|\.md)$/)?.[1]).filter(Boolean));
 const users=csv(history+'/user_selections.csv');
 const ids=new Set();for(const u of users){if(ids.has(u.selection_id))throw Error('Duplicate user selection '+u.selection_id);ids.add(u.selection_id);dates.add(u.analysis_date);}
 const days=[...dates].sort().map(date=>{
  const baseGoal=`Analysis/Goal70_${date}`,baseCorner=`Analysis/Corners1H_${date}`,review=`Analysis/Review_${date}`;
  const goals=csv(`${history}/${date}_goal70_candidates.csv`);
  const corners=exists(`${history}/${date}_corners1h_candidates.csv`)?csv(`${history}/${date}_corners1h_candidates.csv`):csv(`${baseCorner}/all_models.csv`);
  const outcomes=csv(`${review}/goal70_outcomes.csv`),cornerOutcomes=csv(`${review}/corners1h_outcomes.csv`);
  const dayUsers=users.filter(u=>u.analysis_date===date);
  const finalCorners=[...csv(`${baseCorner}/prospective_selection.csv`),...csv(`${baseCorner}/prospective_alternative.csv`)];
  const matches=goals.map(g=>({id:g.id,ids:[g.id],home:short(g.home),away:short(g.away),originalHome:g.home,originalAway:g.away,kickoff:time(g.kickoff_ro),competition:g.competition||g.league,league:labels[g.competition||g.league]??g.competition??g.league,basis:g.history_basis||'same_league_current_season',recordedAt:g.recorded_at_ro||g.model_snapshot_at_ro||'',
   goal:{p:num(g.p_raw),stress:num(g.p_stress),wins:num(g.first_goal_success??g.first_goal_unique_success),n:num(g.first_goal_n??g.first_goal_unique_n),venueWins:num(g.venue_success??g.venue_unique_success),venueN:num(g.venue_n??g.venue_unique_n),source:g.source,eligible:num(g.p_raw)!=null},corner:{p:null,stress:null,eligible:false},analystGoal:truth(g.analyst_selected||g.analyst_finalist)}));
  const sameNames=(m,r)=>m.originalHome===r.home&&m.originalAway===r.away;
  const resolve=r=>{
   const keys=[r.id,r.match_id,r.goal70_espn_id,r.prediction_id,r.fotmob_id].filter(Boolean);
   let found=matches.filter(m=>keys.some(k=>m.ids.includes(k)));
   if(!found.length)found=matches.filter(m=>sameNames(m,r));
   if(found.length>1)throw Error(`Ambiguous fixture crosswalk ${date}: ${r.home} ${r.away}`);
   return found[0];
  };
  // An outcome file may supply an explicit cross-provider identity map, never model input.
  for(const o of outcomes){const m=resolve(o);if(m)m.ids=[...new Set([...m.ids,o.prediction_id,o.fotmob_id].filter(Boolean))];}
  for(const c of corners){
   let m=resolve(c);
   if(!m){m={id:c.goal70_espn_id||c.id,ids:[c.id,c.goal70_espn_id].filter(Boolean),home:short(c.home),away:short(c.away),originalHome:c.home,originalAway:c.away,kickoff:time(c.kickoff_ro||c.date),competition:c.competition||c.league,league:labels[c.competition||c.league]??c.competition??c.league,basis:c.history_basis||'same_league_current_season',goal:{p:null,stress:null,eligible:false}};matches.push(m);}
   m.ids=[...new Set([...m.ids,c.id,c.goal70_espn_id].filter(Boolean))];
   m.corner={p:num(c.p_raw),stress:num(c.p_stress),wins:num(c.unique_under),n:num(c.unique_n),venueWins:num(c.venue_under),venueN:num(c.venue_unique_n),eight:num(c.unique_8plus),seven:num(c.unique_7),source:c.source,eligible:truth(c.eligible)&&num(c.p_raw)!=null};
   m.analystCorner=truth(c.analyst_selected);
  }
  for(const f of finalCorners){const m=resolve(f);if(!m)throw Error('Unmatched corner finalist '+date+' '+f.match_id);m.analystCorner=true;m.corner.final={p:num(f.final_p_raw||f.p_raw),stress:num(f.final_p_stress||f.p_stress),wins:num(f.historical_success)??m.corner.wins,n:num(f.historical_n)??m.corner.n,venueWins:num(f.venue_success)??m.corner.venueWins,venueN:num(f.venue_n)??m.corner.venueN,recordedAt:f.recorded_at_ro};}
  for(const market of ['goal','corner']){
   const marketOutcomes=market==='goal'?outcomes:cornerOutcomes;
   for(const m of matches){
    const o=marketOutcomes.find(o=>resolve(o)===m);
    const g=market==='goal'?goals.find(g=>g.id===m.id):corners.find(c=>resolve(c)===m);
    const physicalPostponed=outcomes.find(o=>resolve(o)===m&&normalizeResult(o.result)==='postponed');
    const row=o||g||{};let result=normalizeResult(row.result||row.outcome);
    if(result==='pending'&&physicalPostponed)result='postponed';
    m[market].outcome={result,minute:row.first_goal_minute??row.actual_first_goal_minute??row.first_goal_actual??'',corners:row.actual_corners_1h??'',score:row.score_ft??row.actual_score_ft??'',source:row.result_source||physicalPostponed?.result_source||'',verifiedAt:row.verified_at_utc||row.result_verified_at_ro||''};
   }
  }
  for(const u of dayUsers){
   const m=resolve(u);if(!m)throw Error('Unmatched user selection '+u.selection_id);
   const market=u.market==='GOAL_BY_70'?'goal':u.market==='CORNERS_1H_UNDER_7_5'?'corner':null;
   if(!market)throw Error('Unsupported market '+u.market);
   m[market].userSelection={...u,p:num(u.p_raw),stress:num(u.p_stress)};
   m[market==='goal'?'userGoal':'userCorner']=true;
   if(normalizeResult(u.result)!=='pending'){
    const existing=m[market].outcome.result;
    if(existing!=='pending'&&existing!==normalizeResult(u.result))throw Error('Conflicting verified outcomes '+u.selection_id);
    m[market].outcome={...m[market].outcome,result:normalizeResult(u.result),minute:u.actual_first_goal_minute,corners:u.actual_corners_1h,source:u.result_source,verifiedAt:u.result_verified_at_ro};
   }
  }
  const rows=market=>matches.filter(m=>m[market].source||m[market].p!=null||m[market].userSelection).map(m=>({id:m.id,home:m.home,away:m.away,p:m[market].p,...m[market].outcome})).sort((a,b)=>(b.p??-1)-(a.p??-1)).map((r,i)=>({...r,rank:i+1}));
  const reviews={goal:rows('goal'),corner:rows('corner')};
  const link=p=>exists(p)?'../../../'+p:null;
  const evidence={goal:json(`${baseGoal}/verification.json`),corner:json(`${baseCorner}/verification.json`)};
  const times=[...goals.map(g=>g.recorded_at_ro||g.model_snapshot_at_ro),...finalCorners.map(f=>f.recorded_at_ro),...dayUsers.map(u=>u.selected_at_ro)].filter(Boolean).sort();
  return {date,snapshotAt:time(times.at(-1))+' RO',matches,reviews,metrics:{goal:metrics(reviews.goal),corner:metrics(reviews.corner)},userSelections:dayUsers,evidence,links:{goal:link(`${baseGoal}/EUROPE_ALL_MATCHES.md`),corner:link(`${baseCorner}/ANALYSIS.md`),journal:link(`${history}/${date}.md`)},coverage:{goal:matches.filter(m=>m.goal.p!=null).length,corner:matches.filter(m=>m.corner.p!=null).length,total:matches.length}};
 });
 if(!days.length)throw Error('No dated candidate archives found');
 const localStatus=json('Analysis/Automation/daily_status.json');
 const automation=localStatus?Object.fromEntries(['schedule','timezone','status','run_date','updated_at_utc'].filter(k=>localStatus[k]!=null).map(k=>[k,localStatus[k]])):null;
 return {version:2,generatedAt:new Date().toISOString(),latestDate:days.at(-1).date,days,automation};
}
