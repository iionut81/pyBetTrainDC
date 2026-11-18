import fs from 'node:fs';
const root='Analysis/Goal70_2026-11-04',rev=root+'/Revisions/initial_screening';
fs.mkdirSync(rev,{recursive:true});
for(const f of ['all_models.csv','universe_coverage.csv','calculation_snapshot.json'])if(!fs.existsSync(rev+'/'+f))fs.copyFileSync(root+'/'+f,rev+'/'+f);
for(const slug of ['cyp.1','tur.2','aze.1'])if(!fs.existsSync(rev+'/'+slug+'.json'))fs.renameSync(root+'/Schedules/'+slug+'.json',rev+'/'+slug+'.json');
for(const market of ['goal70','corners1h'])if(!fs.existsSync(rev+'/'+market+'_candidates.csv'))fs.copyFileSync('Analysis/Selection_History/2026-11-04_'+market+'_candidates.csv',rev+'/'+market+'_candidates.csv');
const p='Analysis/Automation/research_2026_11_04.mjs';let s=fs.readFileSync(p,'utf8');
s=s.replaceAll("['cyp.1',171","['cyp.1',136").replaceAll("['tur.2',98","['tur.2',165").replaceAll("['aze.1',114","['aze.1',262");fs.writeFileSync(p,s);
fs.writeFileSync(rev+'/REASON.md','Initial screening archived before publication. Corrected provider league IDs using public catalog: Cyprus 136, Turkey 2 165, Azerbaijan 262. Cyprus 171 was Svenska Cupen and Azerbaijan 114 Friendlies; neither is an accepted domestic league observation. Recompute all eligible fixtures; preserve original snapshots. No production model changes.\n');
