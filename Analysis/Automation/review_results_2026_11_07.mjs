import fs from 'node:fs';
import {read,csv,save,writeCsv,pool,match,observation,ro} from './research_2026_11_07.mjs';
const R='Analysis/Review_2026-11-06',H='Analysis/Selection_History';
const ids=[...new Set(['5739547','5803312',...read('Analysis/Goal70_2026-11-06/candidate_fixtures.json').map(f=>f.id)])];
for(const id of ids){const p=R+'/FotMob/'+id+'.json';if(fs.existsSync(p)&&!read(p).ok){save(R+'/FailedSandbox/'+id+'.json',read(p));fs.unlinkSync(p);}}
await pool(ids,async id=>{await match(id,R+'/FotMob');},4);
const output={};
for(const [market,key]of [['goal70','goal'],['corners1h','corner']]){
 const models=csv(`${H}/2026-11-06_${market}_candidates.csv`);
 output[key]=models.map(m=>{const d=read(`${R}/FotMob/${m.id}.json`),o=observation(d);const identity=d.ok&&String(d.general.homeTeam.id)===m.home_id&&String(d.general.awayTeam.id)===m.away_id;return {...m,prediction_id:m.id,fotmob_id:m.id,score_ft:o.score_ft,first_goal_minute:key==='goal'?o.first_goal_minute:'',actual_corners_1h:key==='corner'?o.actual_corners_1h:'',result:identity?(o[key+'_result']??o.result):'pending',reason:identity?(o[key+'_reason']??o.reason):'identity_unverified',result_source:o.result_source,result_verified_at_ro:o.result_verified_at_ro};});
 writeCsv(`${R}/${market}_outcomes.csv`,output[key]);
}
const retries=[];
for(const day of ['2026-11-02','2026-11-03','2026-11-04'])for(const market of ['goal70','corners1h'])for(const old of csv(`Analysis/Review_${day}/${market}_outcomes.csv`).filter(x=>!['success','fail','cancelled','void'].includes(x.result))){const id=old.fotmob_id||old.id;if(!id)continue;const d=await match(id,R+'/OlderRetry'),o=observation(d),key=market==='goal70'?'goal':'corner';retries.push({prediction_date:day,market,id,home:old.home,away:old.away,previous_status:old.result,new_status:o[key+'_result']??o.result,reason:o[key+'_reason']??o.reason,source:o.result_source,verified_at_ro:o.result_verified_at_ro});}
writeCsv(R+'/older_retries.csv',retries);
console.log(JSON.stringify({metrics:Object.fromEntries(Object.entries(output).map(([k,rs])=>[k,{total:rs.length,models:rs.filter(x=>x.p_raw).length,success:rs.filter(x=>x.result==='success').length,fail:rs.filter(x=>x.result==='fail').length,pending:rs.filter(x=>x.result==='pending').length}])),choices:ids.slice(0,2).map(id=>({id,observation:observation(read(`${R}/FotMob/${id}.json`))})),retries},null,2));
