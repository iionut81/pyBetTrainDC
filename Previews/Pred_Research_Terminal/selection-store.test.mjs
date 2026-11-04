import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {loadArchive} from './data.mjs';
import {mutateSelection,readSelections,serializeSelections,roClock} from './selection-store.mjs';
import {startDashboardServer} from './server.mjs';

import {makeFixture} from './choice-test-fixture.mjs';

function fixture(t){const root=fs.mkdtempSync(path.join(os.tmpdir(),'pred-choices-test-'));makeFixture(root);t.after(()=>{assert.equal(path.dirname(path.resolve(root)),path.resolve(os.tmpdir()));assert.ok(path.basename(root).startsWith('pred-choices-test-'));fs.rmSync(root,{recursive:true,force:true});});return root;}
const now=new Date('2026-11-04T10:00:00Z');
const input=(root,market='goal',action='select')=>({date:'2026-11-04',matchId:'g1',market,action,revision:readSelections(root).revision});

test('Dashboard and chat choices share canonical history; two markets survive rebuild and reload; remove is audited',t=>{
 const root=fixture(t),frozen=fs.readFileSync(path.join(root,'Analysis/Selection_History/2026-11-04_goal70_candidates.csv'));
 const goal=mutateSelection(root,{...input(root),p_raw:1},{now});assert.equal(goal.selection.p_raw,'0.91');
 const corner=mutateSelection(root,input(root,'corner'),{now});assert.equal(corner.selection.p_raw,'0.96');assert.equal(corner.selection.match_id,'1001');
 assert.equal(mutateSelection(root,input(root),{now}).changed,false);
 let day=loadArchive(root).days.at(-1);assert.equal(day.userSelections.length,2);assert.ok(day.matches[0].userGoal);assert.ok(day.matches[0].userCorner);
 assert.equal(day.matches[0].corner.userSelection.p,.96);assert.equal(day.matches[0].goal.outcome.result,'pending');
 mutateSelection(root,input(root,'goal','remove'),{now});day=loadArchive(root).days.at(-1);assert.equal(day.userSelections.length,1);assert.ok(!day.matches[0].userGoal);assert.ok(day.matches[0].userCorner);
 assert.equal(readSelections(root).rows.find(r=>r.selection_id==='old-choice').result,'success');
 assert.deepEqual(fs.readFileSync(path.join(root,'Analysis/Selection_History/2026-11-04_goal70_candidates.csv')),frozen);
 const log=fs.readFileSync(path.join(root,'Analysis/Selection_History/selection_events.jsonl'),'utf8').trim().split('\n').map(JSON.parse);assert.deepEqual(log.map(e=>e.action),['select','select','remove']);
});
test('Stale revisions, old results, started fixtures, missing models and busy locks cannot overwrite choices',t=>{
 const root=fixture(t),stale=input(root);mutateSelection(root,input(root,'corner'),{now});const saved=readSelections(root).bytes;
 assert.throws(()=>mutateSelection(root,stale,{now}),/Istoricul s-a schimbat/);
 assert.throws(()=>mutateSelection(root,{...input(root),date:'2026-11-03',matchId:'old',action:'remove'},{now}),/arhivate/);
 assert.throws(()=>mutateSelection(root,input(root),{now:new Date('2026-11-04T20:00:00Z')}),/începute/);
 assert.throws(()=>mutateSelection(root,{...input(root,'corner'),matchId:'g2'},{now}),/Lipsesc datele/);
 const lock=path.join(root,'Analysis/Selection_History/user_selections.csv.lock');fs.writeFileSync(lock,'test');assert.throws(()=>mutateSelection(root,input(root),{now}),/actualizat acum/);fs.unlinkSync(lock);
 assert.deepEqual(readSelections(root).bytes,saved);
});
test('Romanian timestamps follow winter and summer time',()=>{
 assert.equal(roClock(now).iso,'2026-11-04T12:00:00+02:00');assert.equal(roClock(new Date('2027-07-01T05:00:00Z')).iso,'2027-07-01T08:00:00+03:00');
});
test('HTTP selection API rejects cross-origin and unauthenticated writes; successful writes trigger rebuild',async t=>{
 const root=fixture(t);let rebuilt=0;const {server,url}=await startDashboardServer({root,port:0,now:()=>now,rebuild:()=>rebuilt++});t.after(()=>server.close());const base=new URL(url).origin;
 const state=await (await fetch(base+'/api/state')).json();const post=(origin,token)=>fetch(base+'/api/choices',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json','X-Pred-Token':token},body:JSON.stringify({...input(root),revision:state.revision})});
 assert.equal((await post('https://example.test',state.token)).status,403);assert.equal((await post(base,'wrong')).status,403);
 const r=await post(base,state.token);assert.equal(r.status,200);const data=await r.json();assert.ok(data.archive.days.at(-1).matches[0].userGoal);assert.equal(rebuilt,1);
 assert.equal((await fetch(base+'/automation/daily_research_config.json')).status,404);
 assert.equal((await fetch(base+'/Analysis/Conversation_History/transcripts/current.md')).status,404);
 assert.equal((await fetch(base+'/Analysis/Selection_History/user_selections.csv')).status,200);
});
