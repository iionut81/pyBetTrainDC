import {read,save,writeCsv,csv,fetch,api,pool,ro} from './research_2026_11_13.mjs';
const B='Analysis/Goal70_2026-11-13';const d=read(B+'/fixtures_global.json'),f=read(B+'/candidate_fixtures.json');
for(const l of d.leagues.filter(l=>(l.primaryId??l.id)===138))for(const m of l.matches)if(!f.some(x=>x.id===String(m.id)))f.push({id:String(m.id),competition:'esp.copa_del_rey',date:m.status.utcTime,home:m.home.name,away:m.away.name,home_id:String(m.home.id),away_id:String(m.away.id),status:m.status,source:api('matchDetails?matchId='+m.id),history_basis:'domestic_league_projection_cross_competition',league_id:138});
save(B+'/candidate_fixtures.json',f);writeCsv(B+'/exclusions.csv',csv(B+'/exclusions.csv').filter(x=>!f.some(y=>y.id===x.id)));
await pool([138,42,73,10216],async id=>{save(B+'/Cups/'+id+'.json',await fetch(api('leagues?id='+id+'&ccode3=ROU')));});
save(B+'/cups_coverage.json',{at_ro:ro(),fixtures:f,notes:'Competition schedules plus global feed; Copa included as domestic projection, not assumed transferable'});console.log(f.map(x=>x.home+'–'+x.away).join('\n'));
