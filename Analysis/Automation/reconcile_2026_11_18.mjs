import fs from 'node:fs';import {read,write,get,pool,stamp} from '../daily_2026_11_18.mjs';
const b='Analysis/Goal70_2026-11-18';let arr=read(b+'/candidate_fixtures.json');
write(b+'/Revisions/before_identity_reconciliation/candidate_fixtures.json',arr);
let excluded=arr.filter(r=>r.competition==='U19 League');
write(b+'/Supplemental/outside_scope_club_youth.json',excluded.map(r=>({...r,reason:'Ukraine club youth, not national teams; excluded from national scope'})));
arr=arr.filter(r=>r.competition!=='U19 League');
for(let r of arr.filter(r=>r.category==='club_women')){r.history_basis='same_competition_current_season';r.inventory_note='Women Champions League current edition separately; no domestic/mens/national pooling; 4 prior league-phase games below minimum5, venue below3.';}
let cov=read(b+'/scan_checkpoint.json');cov.coverage=cov.coverage.filter(r=>r.league!=='supp_esp_9137');write(b+'/scan_checkpoint.json',cov);
// Confirm domestic identities from each cup team's season fixture list.
let teams=[...new Set(arr.filter(r=>r.category==='club_men').flatMap(r=>[r.home_id,r.away_id]))];
await pool(teams,async id=>get('https://www.fotmob.com/api/data/teams?id='+id,`${b}/Teams/${id}.json`));
let leagues=new Map();for(let id of teams){let d=read(`${b}/Teams/${id}.json`);let f=d?.fixtures?.allFixtures?.fixtures||[];for(let m of f){let t=m.tournament;if(t&&!/cup|copa|friendly/i.test(t.name)&&t.leagueId)leagues.set(String(t.leagueId),t.name);}}
await pool([...leagues],async([id,name])=>{if(cov.coverage.some(x=>x.fotmob_id===id))return;let d=await get(`https://www.fotmob.com/api/data/leagues?id=${id}&ccode3=ROU`,`${b}/Competitions/${id}.json`);let ff=d?.fixtures?.allMatches||d?.overview?.matches?.allMatches||[];if(!ff.length)return;let key='supp_cup_'+id;write(`${b}/Schedules/${key}.json`,d);cov.coverage.push({league:key,fotmob_id:id,country:name,coverage:'supplemental_domestic_current_season',source:`https://www.fotmob.com/api/data/leagues?id=${id}&ccode3=ROU`});});
write(b+'/Supplemental/cup_domestic_leagues.json',{at_ro:stamp(),leagues:[...leagues]});
write(b+'/scan_checkpoint.json',cov);write(b+'/candidate_fixtures.json',arr);console.log({fixtures:arr.length,removed_club_youth:excluded.length,domestic_leagues:[...leagues]});
