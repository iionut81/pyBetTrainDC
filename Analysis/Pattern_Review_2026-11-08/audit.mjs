import fs from 'node:fs';
import {loadArchive,parseCsv} from '../../Previews/Pred_Research_Terminal/data.mjs';
const dir='Analysis/Pattern_Review_2026-11-08';fs.mkdirSync(dir,{recursive:true});
const archive=loadArchive(process.cwd()),numeric=v=>v==null||v===''||!Number.isFinite(Number(v))?null:Number(v);
const average=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const summary=a=>({n:a.length,success:a.filter(x=>x.result==='success').length,fail:a.filter(x=>x.result==='fail').length,rate:a.length?a.filter(x=>x.result==='success').length/a.length:null,meanP:average(a.map(x=>x.p))});
const find=(r,...keys)=>numeric(keys.map(k=>r[k]).find(v=>v!==undefined&&v!==''));
const rows=[];
for(const d of archive.days.filter(d=>d.date<'2026-11-08'))for(const market of ['goal','corner']){
 let file=`Analysis/Selection_History/${d.date}_${market==='goal'?'goal70':'corners1h'}_candidates.csv`;
 if(!fs.existsSync(file))file=`Analysis/Corners1H_${d.date}/all_models.csv`;
 const models=fs.existsSync(file)?parseCsv(fs.readFileSync(file,'utf8')):[];
 for(const m of d.matches){const s=m[market];if(s.p==null||!['success','fail'].includes(s.outcome.result))continue;
  const r=models.find(r=>[r.id,r.match_id,r.goal70_espn_id].some(id=>id&&m.ids.includes(id)))??{};
  const ownRates=['home','away'].map(side=>{const gf=find(r,side+'_venue_gf70',side+'_venue_gf'),n=find(r,side+'_venue_n');return market==='goal'&&gf!==null&&n>0?gf/n:null;});
  const dispersions=['home','away'].map(side=>{const v=find(r,side+'_venue_variance'),mean=find(r,side+'_venue_mean');return market==='corner'&&v!==null&&mean>0?v/mean:null;});
  rows.push({date:d.date,market,id:m.id,home:m.home,away:m.away,p:s.p,result:s.outcome.result,minute:s.outcome.minute,corners:s.outcome.corners,user:!!m[market==='goal'?'userGoal':'userCorner'],analyst:!!m[market==='goal'?'analystGoal':'analystCorner'],historicalRate:s.n>0?s.wins/s.n:null,venueRate:s.venueN>0?s.venueWins/s.venueN:null,minOwnVenueRate:ownRates.every(v=>v!==null)?Math.min(...ownRates):null,maxVenueDispersion:dispersions.every(v=>v!==null)?Math.max(...dispersions):null,priorEight:s.eight??null,priorSeven:s.seven??null,source:s.outcome.source});
 }
}
const out={asOf:'2026-11-08',verifiedPredictionDays:[...new Set(rows.map(r=>r.date))],latestUserChoices:archive.days.find(d=>d.date==='2026-11-08').userSelections.map(u=>({home:u.home,away:u.away,market:u.market,result:u.result||'pending'})),markets:{}};
for(const market of ['goal','corner']){const all=rows.filter(r=>r.market===market),byResult={};
 for(const result of ['success','fail']){const a=all.filter(r=>r.result===result);byResult[result]={n:a.length};for(const key of ['p','historicalRate','venueRate','minOwnVenueRate','maxVenueDispersion','priorEight','priorSeven']){const v=a.map(r=>r[key]).filter(x=>x!==null);byResult[result][key]={n:v.length,mean:average(v)};}}
 out.markets[market]={models:summary(all),analystFlags:summary(all.filter(r=>r.analyst)),user:summary(all.filter(r=>r.user)),byResult,perfectHistoricalFailures:all.filter(r=>r.result==='fail'&&r.historicalRate===1).map(r=>({date:r.date,match:r.home+'-'+r.away,minute:r.minute,corners:r.corners,p:r.p})),byDay:out.verifiedPredictionDays.map(date=>({date,...summary(all.filter(r=>r.date===date))}))};
}
fs.writeFileSync(dir+'/descriptive_audit.json',JSON.stringify(out,null,2));
fs.writeFileSync(dir+'/rows.json',JSON.stringify(rows,null,2));
console.log(JSON.stringify(out,null,2));
