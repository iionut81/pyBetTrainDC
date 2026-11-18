import fs from 'node:fs';
const file='Previews/Pred_Research_Terminal/views.js';
let text=fs.readFileSync(file,'utf8');
const before='<div class="micro-bars" aria-hidden="true">';
const after='<div class="micro-bars" style="gap:${Math.min(3,50/Math.max(1,c.total))}px" aria-hidden="true">';
if(!text.includes(after)){
 if(text.split(before).length!==2)throw Error('Expected one decorative bar group');
 fs.mkdirSync('Analysis/Goal70_2026-11-14/Revisions',{recursive:true});
 fs.writeFileSync('Analysis/Goal70_2026-11-14/Revisions/views.before-mobile-fix.js',text);
 fs.writeFileSync(file,text.replace(before,after));
}
