import fs from 'node:fs';
let s=fs.readFileSync('Analysis/Automation/research_2026_11_04.mjs','utf8');
s=s.replace("date='2026-11-04',prev='2026-11-03'","date='2026-11-05',prev='2026-11-04'").replaceAll('20261104','20261105');
// Include Conference League, and retain validated domestic identifiers.
s=s.replace("[42,73,525].includes(l.primaryId??l.id)&&[42,73].includes(l.primaryId??l.id)","[42,73,525].includes(l.primaryId??l.id)");
s=s.replace("((l.primaryId??l.id)===42?'uefa.champions':'uefa.europa')","((l.primaryId??l.id)===42?'uefa.champions':(l.primaryId??l.id)===73?'uefa.europa':'uefa.conference')");
s=s.replace("const mode=process.argv[2];", "export {read,csv,save,writeCsv,pool,fetch,api,extract,match,observation,ro,defs};\nconst mode=process.argv[2];");
// Do not expose the older review writer, which predates concurrent dashboard choices.
s=s.replace("if(mode==='review')await doReview();",'');
fs.writeFileSync('Analysis/Automation/research_2026_11_05.mjs',s);
