import fs from 'node:fs';
const p='Analysis/Automation/research_2026_11_05.mjs';let s=fs.readFileSync(p,'utf8');
s=s.replaceAll('[42,73,525]','[42,73,10216]');
s=s.replace("const allDefs=[...defs,['aze.1',262,'Azerbaijan']];","const allDefs=[...defs,['aze.1',262,'Azerbaijan'],['hun.1',212,'Hungary'],['bul.1',270,'Bulgaria'],['isr.1',127,'Israel'],['svn.1',173,'Slovenia'],['arm.1',118,'Armenia'],['por.2',185,'Portugal 2'],['fin.1',51,'Finland'],['kaz.1',225,'Kazakhstan'],['lva.1',226,'Latvia'],['ltu.1',228,'Lithuania'],['alb.1',260,'Albania'],['bih.1',267,'Bosnia'],['geo.1',439,'Georgia']];");
s=s.replace("['swe.1','nor.1'].includes(slug)","['swe.1','nor.1','fin.1','kaz.1','lva.1','ltu.1','geo.1'].includes(slug)");
fs.writeFileSync(p,s);
for(const [dir,names] of [['Analysis/Goal70_2026-11-05',['all_models.csv','candidate_fixtures.json','universe_coverage.csv','exclusions.csv','calculation_snapshot.json']],['Analysis/Corners1H_2026-11-05',['all_models.csv']],['Analysis/Selection_History',['2026-11-05_goal70_candidates.csv','2026-11-05_corners1h_candidates.csv']]])for(const n of names){
 const target='Analysis/Goal70_2026-11-05/Revisions/initial_17/'+(dir.includes('Corners')?'corner_':'')+n;fs.mkdirSync(target.slice(0,target.lastIndexOf('/')),{recursive:true});if(!fs.existsSync(target))fs.copyFileSync(dir+'/'+n,target);
}
fs.writeFileSync('Analysis/Goal70_2026-11-05/Revisions/initial_17/REASON.md','Unpublished initial screening preserved. Official UEFA and provider global catalog expose18 Conference fixtures under primaryId10216, not old525. Add all18 and domestic history leagues for European participants. Recalculate complete discovered universe before ranking.');
