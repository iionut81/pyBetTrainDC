import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {cornerHistoryObservation,writeMatchReports} from './match-reports.mjs';
import {loadArchive,parseCsv} from './data.mjs';

test('First-half corner history accepts legacy and current schemas, including zero and the 7/8 boundary',()=>{
 assert.deepEqual(cornerHistoryObservation({valid:'true',home_corners_1h:'3',away_corners_1h:'4',total_corners_1h:'7'}),{observation:'3 + 4 = 7',result:'Succes'});
 assert.deepEqual(cornerHistoryObservation({corner_ok:'true',home_cf:'3',away_cf:'5',total1h:'8'}),{observation:'3 + 5 = 8',result:'Fail'});
 assert.deepEqual(cornerHistoryObservation({corner_ok:'true',home_cf:'0',away_cf:'0',total1h:'0'}),{observation:'0 + 0 = 0',result:'Succes'});
 assert.deepEqual(cornerHistoryObservation({total_1h:'5'}),{observation:'5',result:'Succes'});
});

test('Missing, invalid, conflicting or full-time-only corner data never become a verified result',()=>{
 for(const row of [
  {corner_ok:'false',home_cf:'2',away_cf:'3',total1h:'5'},
  {valid:'false',corner_ok:'true',home_cf:'2',away_cf:'3',total1h:'5'},
  {corner_ok:'true',home_cf:'',away_cf:'',total1h:''},
  {corner_ok:'true',home_cf:'2',away_cf:'3',total1h:'4'},
  {corner_ok:'true',total1h:'5',total_1h:'6'},
  {corner_ok:'true',total1h:'-1'},
  {corner_ok:'true',total1h:'1.5'},
  {home_corners:'4',away_corners:'5',total_corners:'9'}
 ])assert.deepEqual(cornerHistoryObservation(row),{observation:'Date 1H indisponibile',result:'Neverificat'});
});

test('Sabadell report renders all saved recent league observations with their actual first-half counts',t=>{
 const root=path.resolve(import.meta.dirname,'../..'),archive=loadArchive(root);
 const day=archive.days.find(d=>d.date==='2026-11-09');
 const match=day.matches.find(m=>m.ids.includes('5868534'));
 const tempRoot=fs.mkdtempSync(path.join(os.tmpdir(),'pred-corner-report-'));
 t.after(()=>{assert.ok(tempRoot.startsWith(path.join(os.tmpdir(),'pred-corner-report-')));fs.rmSync(tempRoot,{recursive:true,force:true});});
 const frozen=JSON.stringify({goal:match.goal.p,corner:match.corner.p});
 writeMatchReports(root,tempRoot,{...archive,days:[{...day,matches:[match]}]});
 const html=fs.readFileSync(path.join(tempRoot,'reports/2026-11-09/corner',match.id+'.html'),'utf8');
 const history=html.split('<h2>Istoricul disponibil pentru acest meci</h2>')[1].split('<h2>Rezultatul verificat')[0];
 assert.equal((history.match(/<td>Succes<\/td>/g)||[]).length,20);
 assert.equal((history.match(/Neverificat|Date 1H indisponibile/g)||[]).length,0);
 const raw=parseCsv(fs.readFileSync(path.join(root,'Analysis/Corners1H_2026-11-09/historical_corners_1h.csv'),'utf8'));
 const row=raw.find(r=>r.home==='Albacete'&&r.away==='Sabadell'&&r.date.startsWith('2026-10-31'));
 assert.ok(row);assert.ok(history.includes(cornerHistoryObservation(row).observation));
 assert.equal(JSON.stringify({goal:match.goal.p,corner:match.corner.p}),frozen);
});

test('Artis–Jablonec corner report reads the shared observation archive and preserves historical failures',t=>{
 const root=path.resolve(import.meta.dirname,'../..'),archive=loadArchive(root);
 const day=archive.days.find(d=>d.date==='2026-11-11');
 const match=day.matches.find(m=>m.ids.includes('6420192'));
 const tempRoot=fs.mkdtempSync(path.join(os.tmpdir(),'pred-corner-report-'));
 t.after(()=>{assert.equal(path.dirname(tempRoot),os.tmpdir());assert.ok(path.basename(tempRoot).startsWith('pred-corner-report-'));fs.rmSync(tempRoot,{recursive:true,force:true});});
 const files=['Analysis/Selection_History/2026-11-11_corners1h_candidates.csv','Analysis/Selection_History/user_selections.csv','Analysis/Corners1H_2026-11-11/all_models.csv','Analysis/Goal70_2026-11-11/historical_observations.csv'];
 const frozen=files.map(file=>fs.readFileSync(path.join(root,file)));
 writeMatchReports(root,tempRoot,{...archive,days:[{...day,matches:[match]}]});
 const html=fs.readFileSync(path.join(tempRoot,'reports/2026-11-11/corner',match.id+'.html'),'utf8');
 const history=html.split('<h2>Istoricul disponibil pentru acest meci</h2>')[1].split('<h2>Rezultatul verificat')[0];
 assert.ok(history.includes('Artis Brno · 10 observații'));
 assert.ok(history.includes('Jablonec · 10 observații'));
 assert.equal((history.match(/<td>Succes<\/td>/g)||[]).length,18);
 assert.equal((history.match(/<td>Fail<\/td>/g)||[]).length,2);
 assert.doesNotMatch(history,/Date 1H indisponibile|Neverificat|Istoricul individual nu este disponibil/);
 assert.match(history,/<td>2026-10-24[\s\S]*?Teplice – Artis Brno[\s\S]*?<td>4 \+ 5 = 9<\/td><td>Fail<\/td>/);
 assert.match(history,/<td>2026-09-09[\s\S]*?Jablonec – Baník Ostrava[\s\S]*?<td>4 \+ 4 = 8<\/td><td>Fail<\/td>/);
 assert.ok(html.includes('Goal70_2026-11-11/historical_observations.csv'));
 assert.ok(html.includes('93,09%'));assert.ok(html.includes('85,29%'));
 assert.ok(html.includes('Încă neverificat'));
 files.forEach((file,i)=>assert.deepEqual(fs.readFileSync(path.join(root,file)),frozen[i]));
});
