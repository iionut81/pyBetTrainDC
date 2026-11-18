import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {loadArchive,historyCoverage} from './data.mjs';

function fixture(t){
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'pred-dashboard-test-'));
 t.after(()=>{
  const target=path.resolve(root),parent=path.resolve(os.tmpdir());
  assert.equal(path.dirname(target),parent);
  assert.ok(path.basename(target).startsWith('pred-dashboard-test-'));
  fs.rmSync(target,{recursive:true,force:true});
 });
 const put=(name,content)=>{fs.mkdirSync(path.dirname(path.join(root,name)),{recursive:true});fs.writeFileSync(path.join(root,name),content);};
 const csv=(name,rows)=>{const keys=[...new Set(rows.flatMap(Object.keys))];put(name,keys.join(',')+'\n'+rows.map(r=>keys.map(k=>'"'+String(r[k]??'').replace(/"/g,'""')+'"').join(',')).join('\n'));};
 put('Analysis/Selection_History/2026-11-04.md','# NO BET: no usable data');
 return {root,put,csv};
}
test('A new dated journal and a corners-only day are discovered without changing code',t=>{
 const {root,csv}=fixture(t);
 csv('Analysis/Selection_History/2026-11-05_corners1h_candidates.csv',[{id:'c1',home:'A',away:'B',date:'2026-11-05T18:00:00Z',league:'eng.2',p_raw:'.9',p_stress:'.8',eligible:'True',source:'https://example.test/fixture'}]);
 const archive=loadArchive(root);
 assert.deepEqual(archive.days.map(d=>d.date),['2026-11-04','2026-11-05']);
 assert.equal(archive.latestDate,'2026-11-05');
 assert.equal(archive.days[0].coverage.total,0);
 assert.equal(archive.days[1].coverage.goal,0);
 assert.equal(archive.days[1].coverage.corner,1);
 assert.equal(archive.days[1].matches[0].goal.p,null);
 assert.equal(archive.days[1].matches[0].kickoff,'20:00');
});
test('A later verified result changes history, never the prospective probability; missing is distinct from zero',t=>{
 const {root,csv}=fixture(t);
 const file='Analysis/Selection_History/2026-11-05_corners1h_candidates.csv';
 csv(file,[{id:'c1',home:'A',away:'B',date:'2026-11-05T18:00:00Z',league:'eng.2',p_raw:'.9',p_stress:'.8',eligible:'True',source:'https://example.test/fixture'}]);
 const frozen=fs.readFileSync(path.join(root,file));
 assert.equal(loadArchive(root).days.at(-1).matches[0].corner.outcome.corners,'');
 csv('Analysis/Review_2026-11-05/corners1h_outcomes.csv',[{match_id:'c1',home:'A',away:'B',result:'success',actual_corners_1h:'0',result_source:'https://example.test/result',result_verified_at_ro:'2026-11-06T10:00:00+02:00'}]);
 const day=loadArchive(root).days.at(-1);
 assert.equal(day.matches[0].corner.outcome.corners,'0');
 assert.equal(day.metrics.corner.success,1);
 assert.equal(day.metrics.goal.played,0);
 assert.equal(day.matches[0].corner.p,.9);
 assert.deepEqual(fs.readFileSync(path.join(root,file)),frozen);
});
test('Real archive preserves both user choices, screening/final estimates and reconciled corner results',()=>{
 const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
 const archive=loadArchive(root),today=archive.days.find(d=>d.date==='2026-11-03'),past=archive.days.find(d=>d.date==='2026-11-02');
 assert.equal(today.coverage.total,20);assert.equal(today.coverage.corner,19);
 const derby=today.matches.find(m=>m.ids.includes('5836912'));
 assert.equal(derby.userCorner,true);
 assert.equal(derby.corner.p,.963459827387227);
 assert.equal(derby.corner.final.p,.966128975612361);
 assert.equal(derby.corner.userSelection.p,.966128975612361);
 assert.equal(derby.corner.outcome.result,'success');
 assert.equal(derby.corner.outcome.corners,'6');
 assert.equal(today.matches.find(m=>m.id==='401880179').userGoal,true);
 assert.equal(past.metrics.goal.played,26);assert.equal(past.metrics.goal.fail,4);
 assert.equal(past.metrics.corner.played,26);assert.equal(past.metrics.corner.pending,0);
 assert.equal(past.metrics.corner.success,22);assert.equal(past.metrics.corner.fail,4);
 assert.equal(past.metrics.corner.postponed,1);
});
test('Contradictory outcomes stop the build instead of silently choosing a source',t=>{
 const {root,csv}=fixture(t);
 csv('Analysis/Selection_History/2026-11-05_goal70_candidates.csv',[{id:'g1',home:'A',away:'B',kickoff_ro:'20:00',competition:'eng.2',p_raw:'.9',result:'fail',source:'https://example.test/fixture'}]);
 csv('Analysis/Selection_History/user_selections.csv',[{selection_id:'s1',analysis_date:'2026-11-05',market:'GOAL_BY_70',match_id:'g1',home:'A',away:'B',result:'success',actual_first_goal_minute:'20'}]);
 assert.throws(()=>loadArchive(root),/Conflicting verified outcomes/);
});

test('Coverage distinguishes unilateral history, partially verified bilateral history and unknown legacy metadata',()=>{
 const unilateral=historyCoverage({home:'A',away:'B',home_n:'0',home_expected:'10',home_missing:'10',away_n:'13',away_expected:'13',away_missing:'0',home_venue_n:'0',home_venue_expected:'5',away_venue_n:'7',away_venue_expected:'7'},'corner');
 assert.equal(unilateral.season.onlySide,'away');assert.equal(unilateral.season.label,'Doar B');assert.equal(unilateral.venue.label,'Doar B · deplasare');assert.match(unilateral.note,/A: 0 din 10/);assert.match(unilateral.note,/numai din meciurile B/);
 const partial=historyCoverage({home:'A',away:'B',home_n:'4',home_expected:'5',home_missing:'1',away_n:'6',away_expected:'6',away_missing:'0'},'goal');
 assert.equal(partial.season.incomplete,true);assert.equal(partial.season.onlySide,null);assert.match(partial.note,/numai observațiile validate/);
 const legacy=historyCoverage({home:'A',away:'B'},'corner');assert.equal(legacy.season.known,false);assert.equal(legacy.note,'');
});

test('Puertollano corner frequency belongs only to Albacete while the bilateral model remains unavailable',()=>{
 const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
 const m=loadArchive(root).days.find(d=>d.date==='2026-11-12').matches.find(m=>m.id==='6386074');
 assert.equal(m.corner.p,null);assert.equal(m.corner.stress,null);assert.equal(m.corner.wins,13);assert.equal(m.corner.n,13);
 assert.equal(m.corner.coverage.season.home.n,0);assert.equal(m.corner.coverage.season.home.expected,10);
 assert.equal(m.corner.coverage.season.away.n,13);assert.equal(m.corner.coverage.season.onlySide,'away');assert.equal(m.corner.coverage.season.label,'Doar Albacete');
 assert.equal(m.corner.coverage.venue.away.n,7);assert.equal(m.corner.coverage.venue.label,'Doar Albacete · deplasare');
 const full=loadArchive(root).days.find(d=>d.date==='2026-11-11').matches.find(m=>m.id==='6420192');
 assert.equal(full.corner.coverage.season.incomplete,false);assert.equal(full.corner.coverage.note,'');
});
