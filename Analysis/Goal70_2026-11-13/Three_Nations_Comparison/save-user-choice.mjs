import fs from 'node:fs';
import crypto from 'node:crypto';
import {readSelections,serializeSelections,roClock} from '../../../Previews/Pred_Research_Terminal/selection-store.mjs';
import {fetch,api,save} from '../../Automation/research_2026_11_13.mjs';
const root=process.cwd(),dir='Analysis/Goal70_2026-11-13/Three_Nations_Comparison',file='Analysis/Selection_History/user_selections.csv';
const response=await fetch(api('matchDetails?matchId=5181942'));
if(!response.ok)throw Error('Live status unavailable; no selection saved');
save(dir+'/user_choice_status.json',response);
const d=response.data,clock=roClock(),status=d.header?.status;
if(clock.date!=='2026-11-13'||String(d.general?.matchId)!=='5181942'||status?.started||status?.finished||status?.cancelled||Date.parse(status?.utcTime)<=Date.now())throw Error('Fixture not selectable today');
const lock=file+'.lock',fd=fs.openSync(lock,'wx');
try{
 const current=readSelections(root),id='2026-11-13_GOAL_BY_70_5181942';
 if(current.rows.some(r=>r.selection_id===id)){console.log('Already saved; no duplicate');process.exitCode=0;}
 else{
 const selection={selection_id:id,analysis_date:clock.date,selected_at_ro:clock.iso,selected_by:'user',market:'GOAL_BY_70',event_definition:'At least one valid goal by either team at displayed minute <=70',home:'Slovakia',away:'Faroe Islands',kickoff_ro:'2026-11-13T21:45:00+02:00',match_id:'5181942',provider:'FotMob',raw_rank:'',model_snapshot_at_ro:'2026-11-13T15:23:52.183+02:00',p_raw:'',p_stress:'',fair_odds:'',calibrated:'False',market_price:'',stake:'',actual_first_goal_minute:'',actual_corners_1h:'',result:'',result_source:'',result_verified_at_ro:'',net_profit:'',status:'user_selection_pending_chat_contextual',analysis_path:dir+'/COMPARISON.md',source:response.source,selection_basis:'Explicit user choice in chat; contextual preference; current-edition samples 4 overall/2 venue below 5/3; no model probability'};
 const rows=[...current.rows,selection],headers=[...new Set([...Object.keys(current.rows[0]),...rows.flatMap(Object.keys)])];
 if(!fs.readFileSync(file).equals(current.bytes))throw Error('Concurrent update; retry');
 const tmp=file+'.'+crypto.randomUUID()+'.tmp';fs.writeFileSync(tmp,serializeSelections(rows,headers));fs.renameSync(tmp,file);
 fs.appendFileSync('Analysis/Selection_History/selection_events.jsonl',JSON.stringify({event_id:crypto.randomUUID(),at_ro:clock.iso,origin:'chat',action:'select',selection})+'\n');
 const note=`\n\n## Alegere explicită din chat — ${clock.iso}\n\nUtilizatorul cere Slovacia–Insulele Feroe ca pick al zilei Gol70, deoarece butonul este indisponibil. Înregistrat în CSV-ul canonic și selection_events, în așteptare, fără probabilitate/stres/cotă inventate. Modelul rămâne neeligibil4/2 față de5/3; alegerea este contextuală, confirmată explicit. Statutul partidei reconfirmat neînceput înaintea salvării. Alegerea existentă Finlanda U21–Irlanda U21 este păstrată; cererea nu revocă explicit selecțiile anterioare și nu presupune bilet combinat. Urmează rebuild/verificare dashboard; mâine verificarea rezultatului≤70 și review/lecții înaintea scanării. Fără Git sau modificări de model/programare.\n`;
 for(const p of['Analysis/Selection_History/2026-11-13.md','Analysis/SESSION_RESUME.md'])fs.appendFileSync(p,note);
 console.log(JSON.stringify({saved:selection.selection_id,p:null,preserved_prior_choices:current.rows.length,selected_at_ro:clock.iso}));
 }
}finally{fs.closeSync(fd);fs.unlinkSync(lock);}
