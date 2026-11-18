import fs from 'node:fs';
const p='Analysis/Automation/research_2026_11_04.mjs';let s=fs.readFileSync(p,'utf8');
const old="'wolverhamptonwanderers':'wolves','wolverhampton':'wolves'";
const additions="'ajaxamsterdam':'ajax','scheerenveen':'heerenveen','feyenoordrotterdam':'feyenoord','cerclebruggeksv':'cerclebrugge','kaagent':'gent','lommelsk':'lommel','royalantwerp':'antwerp','racinggenk':'genk','sttruiden':'sinttruidense','sportingcharleroi':'royalcharleroisc','skbeveren':'waaslandbeveren','kvcwesterlo':'westerlo'";
s=s.replace(old,old+','+additions).replace('first_half:[nh,na]','first_half:complete?[nh,na]:null');fs.writeFileSync(p,s);
