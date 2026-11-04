import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {loadArchive,parseCsv} from './data.mjs';

export const DEFAULT_PORT=18770;
export const dashboardPath='/Previews/Pred_Research_Terminal/dist/index.html';
export function roClock(now=new Date()){
 const parts=Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Bucharest',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).formatToParts(now).map(p=>[p.type,p.value]));
 const date=`${parts.year}-${parts.month}-${parts.day}`,time=`${parts.hour}:${parts.minute}:${parts.second}`;
 const localMs=Date.UTC(+parts.year,+parts.month-1,+parts.day,+parts.hour,+parts.minute,+parts.second);
 const offset=Math.round((localMs-now.getTime())/60000),sign=offset<0?'-':'+';
 return {date,time,iso:`${date}T${time}${sign}${String(Math.floor(Math.abs(offset)/60)).padStart(2,'0')}:${String(Math.abs(offset)%60).padStart(2,'0')}`};
}
export const revision=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
export function selectionFile(root){return path.join(root,'Analysis/Selection_History/user_selections.csv');}
export function readSelections(root){const file=selectionFile(root),bytes=fs.readFileSync(file);return {bytes,revision:revision(bytes),rows:parseCsv(bytes.toString(bytes[0]===255?'utf16le':'utf8'))};}
function fail(message,status=409){throw Object.assign(new Error(message),{status});}
export function serializeSelections(rows,headers){return headers.map(v=>'"'+v.replaceAll('"','""')+'"').join(',')+'\r\n'+rows.map(r=>headers.map(k=>'"'+String(r[k]??'').replaceAll('"','""')+'"').join(',')).join('\r\n')+(rows.length?'\r\n':'');}

export function mutateSelection(root,input,{now=new Date()}={}){
 if(!input||!['goal','corner'].includes(input.market)||!['select','remove'].includes(input.action)||!/^\d{4}-\d{2}-\d{2}$/.test(input.date||'')||!/^[\w-]+$/.test(input.matchId||''))fail('Cerere de alegere invalidă.',400);
 const file=selectionFile(root),lock=file+'.lock';let fd;
 try{fd=fs.openSync(lock,'wx');}catch(e){if(e.code==='EEXIST')fail('Istoricul este actualizat acum. Reîncearcă.');throw e;}
 try{
  const current=readSelections(root),archive=loadArchive(root),day=archive.days.find(d=>d.date===input.date),match=day?.matches.find(m=>m.id===input.matchId);
  if(!match)fail('Partida nu există în arhiva zilei.',404);
  const market=input.market,event=market==='goal'?'GOAL_BY_70':'CORNERS_1H_UNDER_7_5',sample=match[market],value=sample.final||sample;
  const existing=current.rows.find(r=>r.analysis_date===day.date&&r.market===event&&(match.ids.includes(r.match_id)||(r.home===match.originalHome&&r.away===match.originalAway)));
  const clock=roClock(now);
  if(day.date!==clock.date||!/^\d\d:\d\d$/.test(match.kickoff)||match.kickoff<=clock.time.slice(0,5)||sample.outcome.result!=='pending')fail('Alegerile arhivate sau meciurile începute nu se modifică din dashboard.');
  if(value.p==null||!sample.eligible)fail('Lipsesc datele necesare pentru această piață.');
  if((input.action==='select'&&existing)||(input.action==='remove'&&!existing))return {changed:false,selection:existing||null};
  if(current.revision!==input.revision)fail('Istoricul s-a schimbat. Am actualizat pagina; repetă alegerea.');
  let selection=existing;
  const next=current.rows.filter(r=>r!==existing);
  if(input.action==='select'){
   const sourceId=sample.source?.match(/(?:matchId=|gameId\/|#)(\d+)/)?.[1],id=match.ids.includes(sourceId)?sourceId:match.id;
   const rank=[...day.matches].sort((a,b)=>(b[market].p??-1)-(a[market].p??-1)).findIndex(m=>m.id===match.id)+1;
   const sourceProvider=/fotmob/i.test(sample.source||'')?'FotMob':/espn/i.test(sample.source||'')?'ESPN':'archived';
   const kickoff=`${day.date}T${match.kickoff}:00${clock.iso.slice(-6)}`;
   selection={selection_id:`${day.date}_${event}_${id}`,analysis_date:day.date,selected_at_ro:clock.iso,selected_by:'user',market:event,event_definition:market==='goal'?'At least one valid goal by either team at displayed minute <=70':'Both teams combined first-half corners <=7, including added time',home:match.originalHome,away:match.originalAway,kickoff_ro:kickoff,match_id:id,provider:sourceProvider,raw_rank:String(rank),model_snapshot_at_ro:sample.final?.recordedAt||match.recordedAt,p_raw:String(value.p),p_stress:value.stress==null?'':String(value.stress),fair_odds:String(1/value.p),calibrated:'False',market_price:'',stake:'',actual_first_goal_minute:'',actual_corners_1h:'',result:'',result_source:'',result_verified_at_ro:'',net_profit:'',status:'user_selection_pending_dashboard',analysis_path:`Previews/Pred_Research_Terminal/dist/reports/${day.date}/${market}/${match.id}.html`,source:sample.source||''};
   next.push(selection);
  }
  const headers=[...new Set([...Object.keys(current.rows[0]||selection||{}),...next.flatMap(Object.keys)])];
  if(!headers.length)fail('Schema istoricului lipsește.');
  if(revision(fs.readFileSync(file))!==current.revision)fail('Istoricul a fost modificat de alt proces. Reîncearcă.');
  const temp=file+'.'+crypto.randomUUID()+'.tmp';
  try{fs.writeFileSync(temp,serializeSelections(next,headers));fs.renameSync(temp,file);}finally{if(fs.existsSync(temp))fs.unlinkSync(temp);}
  fs.appendFileSync(path.join(path.dirname(file),'selection_events.jsonl'),JSON.stringify({event_id:crypto.randomUUID(),at_ro:clock.iso,origin:'dashboard',action:input.action,selection})+'\n');
  return {changed:true,selection};
 }finally{if(fd!=null)fs.closeSync(fd);fs.unlinkSync(lock);}
}
