import fs from 'node:fs';
import {read,save,ro} from '../../Automation/research_2026_11_07.mjs';
const base='Analysis/Goal70_2026-11-07',dir=base+'/Cardiff_Malaga_Comparison';
const data=read(dir+'/comparison.json'),history=read(base+'/historical_observations.json');
const pct=p=>(100*p).toFixed(2)+'%',n=x=>Number(x).toFixed(4);
function sensitivity(f,exclude){
 const sets={};
 for(const [key,team,venue]of[['a',f.home_id,null],['b',f.away_id,null],['av',f.home_id,'home'],['bv',f.away_id,'away']]){
  const h=history.filter(h=>h.league===f.competition&&h.goal_ok&&!exclude(h)&&(venue?h[venue+'_id']===team:h.home_id===team||h.away_id===team));
  sets[key]={n:h.length,gf:h.reduce((s,h)=>s+(h.home_id===team?h.home70:h.away70),0),ga:h.reduce((s,h)=>s+(h.home_id===team?h.away70:h.home70),0)};
 }
 const {a,b,av,bv}=sets,ls=(a.gf/a.n+b.ga/b.n+b.gf/b.n+a.ga/a.n)/2,lv=(av.gf/av.n+bv.ga/bv.n+bv.gf/bv.n+av.ga/av.n)/2;
 return {sets,lambda:(ls+lv)/2,p:1-Math.exp(-(ls+lv)/2)};
}
const sensitivities=data.matches.map(s=>({id:s.row.id,large_margin_4plus:sensitivity(s.row,h=>{const score=h.score.match(/\d+/g)?.map(Number);return score&&Math.abs(score[0]-score[1])>=4;}),any_red_card:sensitivity(s.row,h=>h.cards.some(c=>['Red','YellowRed'].includes(c.card)))}));
save(dir+'/sensitivity.json',{method:'Diagnostics only, symmetric exclusions for every compared match. Neither a calibrated probability nor a replacement for the frozen forecast. Final-score margin exclusion is a historical sensitivity, not a future-known match characteristic.',matches:sensitivities});
const sources=[
 ['Cardiff: avancronica actuală, 6noiembrie','https://www.cardiffcityfc.co.uk/news/match-preview-cardiff-city-vs-burnley-1'],
 ['Celestini despre Cardiff, 5noiembrie','https://www.burnleyfootballclub.com/media-article/celestinis-press-points-cardiff-city'],
 ['Málaga: avancronica etapei12','https://www.malagacf.com/noticias/previa-del-malaga-racing0611'],
 ['Málaga: ultimul antrenament','https://www.malagacf.com/noticias/puesta-a-punto-en-el-escenario-del-malagaracing'],
 ['Racing: deplasare și absențe','https://www.realracingclub.es/noticias/el-racing-busca-en-malaga-sus-primeros-puntos-a-domicilio-del-curso-2026-27'],
 ['José Alberto: intenția de joc','https://www.realracingclub.es/noticias/queremos-trasladar-nuestra-personalidad-como-locales-a-los-partidos-fuera-de-casa'],
 ['West Ham: deplasarea confirmată','https://www.whufc.com/en/news/over-land-and-sea-to-blackburn-rovers-november-2026'],
 ['Veltman: start mai bun și debutul lui Parker, 7noiembrie','https://www.whufc.com/en/news/veltman-or-we-need-to-get-it-right-from-the-start'],
 ['Blackburn: program confirmat','https://www.rovers.co.uk/news/2026/november/03/ticket-news--rovers-v-west-ham-united/']
];
save(dir+'/sources.json',{updated_at_ro:ro(),sources:sources.map(([title,url])=>({title,url})),limits:['Nu s-au folosit cote sau predicții de bookmaker.','Formațiile FotMob sunt lastStarting11, nu titulari confirmați sau un 11 probabil verificat.','Preview-ul Cardiff fără sufix -1 este din2022 și a fost exclus. Preview-ul Málaga fără0611 privește etapa41 și a fost exclus.','Diferență clasament Cardiff–Burnley: preview4puncte, tabel FotMob15vs10=5; nerezolvată, nefolosită în formula probabilității.','Cauzele tactice ale eșecurilor nu sunt deduse din scor.','Datele recente individuale provin din ultimele10oficiale returnate de furnizor; completitudinea tuturor competițiilor nu este verificată independent.']});
const table='| Meci | Ora RO | Rang brut original | Model brut | AR20 | Istoric sezon unic | Teren unic | Cotă echitabilă |\n|---|---|---:|---:|---:|---:|---:|---:|\n'+data.matches.map(s=>{const r=s.row;return `| ${r.home}–${r.away} | ${r.kickoff_ro.slice(11,16)} | ${r.rank} | ${pct(+r.p_raw)} | ${pct(+r.p_stress)} | ${r.first_goal_success}/${r.first_goal_n} | ${r.venue_success}/${r.venue_n} | ${n(r.fair_odds)} |`;}).join('\n');
const context=`
## Verdicturi pentru comparațiile cerute

Cardiff–Burnley este preferat lui Málaga–Racing în prima comparație. În cererea ulterioară Blackburn–West Ham versus Málaga–Racing, prefer Blackburn–West Ham. Acestea sunt preferințe între alternativele cerute, nu două recomandări zilnice de pariat și nu o înlocuire a finalistului global salvat Luzern–Young Boys sau a principalului individual Istra–Rijeka cornere. Utilizatorul nu a declarat o alegere; user_selections.csv și evenimentele rămân neschimbate.

Blackburn–West Ham are27/28observații favorabile și14/14în configurația terenului; fiecare are10/10în seria oficială recentă returnată. Producția proprie≤70este9/7acasă Blackburn și10/7în deplasare WestHam, față de4/5și4/5la Málaga/Racing. Preferința rămâne contextuală/diagnostică, deși modelul brut92,36% este sub93,09%al Málaga. Modelul însumează toate golurile până la70, inclusiv după primul; frecvența binară măsoară direct evenimentul urmărit. Eșantioanele nu permit transformarea27/28într-o garanție pentru azi.

Risc major Blackburn–WestHam: Parker a fost numit pe4noiembrie, chiar înaintea Cardiff3–3. Un singur meci sub noul antrenor nu validează statistica sezonului; o reorganizare defensivă poate încetini jocul. Intenția declarată de Veltman de a începe mai bine este o declarație, nu dovadă că vom avea gol. Ultimele formații returnate sunt4-2-3-1Blackburn(TonyMowbray conformAPI, numele neverificat independent) și4-4-1-1WestHam(Parker confirmat oficial); nu sunt formule confirmate azi. Interpretare: dacă Blackburn avansează pe benzi, spațiile din tranziție pot produce ocazii; dacă ambele păstrează bloc compact, avantajul istoric poate să nu se manifeste. Nu atribuim automat un stil unui antrenor doar din nume.

Málaga–Racing: LaLiga etapa12, gazdele20/5puncte, oaspeții16/11conform avancronicii oficiale. Miza supraviețuirii dă motive pentru victorie ambelor, dar nu impune atac imediat: și teama de a pierde poate produce prudență. Gazda are4goluri proprii≤70în5meciuriacasă, oaspeții4în5deplasări. Toatecele5deplasăriRacingpierduteFT nu înseamnă automat gol timpuriu; Getafe–Racing deschis81estecontraexemplu. Barça7–2Racing/Celta5–0Racing șiRealMadrid4–0Málaga măresc intensitatea statistică împotriva unor adversari ofensivi mai buni decât adversarul de azi. Eliminând simetric din istoric toate partidele cu diferențăFT≥4, diagnosticul scade la89,40%Málaga versus91,76%Blackburn; nu este un nou model sau o metodă validată și nu înlocuiește prognoza originală.

Gazda spaniolă a avut patru antrenamente săptămâna aceasta; ultimul meci31octombrie, Racing1noiembrie:7și6zileîntrepartide. Blackburna jucat3noiembrie, WestHam4noiembrie:4și3zileîntrepartide, deci vizitatorii au recuperare mai scurtă. Málaga: ultimulXI5-4-1, Racing4-2-3-1; schiță documentară, nu titulari. Un5-4-1poate comprima centrul și reduce expunerea gazdei; contraargument pentru gol, dacă este păstrat azi. JoséAlberto a declarat intenția unui joc mai convingător în deplasare; nu este un bonus procentual.

Cardiff–Burnley: gazda are14goluri proprii≤70în7meciuriacasă, Burnley9în7deplasări;26/28 și14/14istoric;Cardifflast10=9/10,Burnley9/10. ContraexempleBolton90șiBurnley–Portsmouth0–0păstrate. AntrenorulBurnleyCelestini,noudefinaloctombrie, a descris pressingul înalt și centrările Cardiff; inferență favorabilă ocaziilor la pierderea mingii sau în spatele presiunii. Riscul: accentul lui pe constanță/organizare și0–0Portsmouthrecent; doar2meciuri subel, nu o validare a sezonului. Cardiffvine după3–3WestHam4noiembrie, BurnleydupăQPR1–1din3noiembrie, recuperare3/4zileîntrepartide. Nicio schimbare de probabilitate din aceste constatări.

## Loturi și condiții: confirmări și limite

Titularii sunt neconfirmați la verificare. Cardiff: APIindicăNgincert,Bielik/Lawlor/Chambersaccidentați. Burnley: Broja/Beyerincerți,Satpaev/Mejbriaccidentați;Amdouni/Humphreys/Barnesrevenițilaantrenamente, nu absențe certe. Nu am găsit confirmare medicală primară completă pentru ambele loturi. Ng apare simultan în vechiulXIșiîntreincerți, dovadă că lastStarting11nu trebuie prezentat ca echipa zilei.

Blackburn: APIindicăAlebiosuincert,Forshaw/Morishita/Millerindisponibili;WestHamLamadridindisponibil. Confirmarea independentă medicală este incompletă; existențaBowen/Mukasa/CastellanosînvechiulXI nu confirmă participareaazi. Racing: MantillașiAndrésabsențe confirmate oficial, revenireaSalinasdupăsuspendare menționată înraportulRacing–RealMadrid trebuie reconfirmată pentruXI. Málaga: oficialJoaquínșiMartínezlaacelașiritm, Pugarevenitparțial, JuanCruzseparat,Lobete/Ochoaaccidentațidelungădurată. APImaiincludeMurillo; neconfirmat independent înultimaștire. Nu convertim lipsa din articol în disponibilitate.

MeteoFotMob: Cardiff10°Cploaieușoară,Blackburn9°Cînnorat,Málaga21°Cparțialnoros. Unitatea windSpeednu esteexplicitată,nuraportămkm/hfărăconfirmare; prognozele provin dinacelașifurnizor, neverificate independent. Nu există dovezi verificate de teren impracticabil sau perturbare a călătoriei; absența unei dovezi nu garantează condiții perfecte.

## Aplicarea lecțiilor precedente

Am consultatLESSONS_FOR_NEXT_SCANdin6noiembrie înaintea comparației: păstrat0–0 și minutele71/81/85/90,pusGF70propriuterenînaintea volumuluigeneral, separatcupadecampionat, expus antrenorii noi,loturile neconfirmate și influența scorurilor mari. Sensibilitățile sunt diagnostice simetrice, nu calibrare sau ajustări arbitrare. Review-ul anterior și cercetarea globală au fost deja realizate azi; această cerere nu le reexecută și nu pretinde un topglobalnou.
`;
let out=`# Comparație Gol70: Cardiff, Málaga și Blackburn–West Ham\n\nActualizare ${ro()}. Eveniment: minimumun golvalidaloricărei echipe laminutulafișat≤70inclusiv45+,excluzândextra-time/departajare. Ambele alternative finale neîncepute la verificareaAPI${data.updated_at_ro}; revalidarea statusului este necesară dacă se citește ulterior. Fără cote bookmaker.\n\n${table}\n\nProbabilități Poisson necalibrate; frecvențele istorice nu sunt probabilități validate pentru meciul de azi. Cotele1/p sunt exclusiv matematice, nu oferte disponibile.\n${context}\n`;
for(const s of data.matches){const r=s.row,ss=sensitivities.find(x=>x.id===r.id);out+=`\n## ${r.home}–${r.away}: intrări și calcul\n\n[Sursă meci](https://www.fotmob.com/match/${r.id}). Campionat${s.current.general?.leagueName},etapa${s.current.general?.matchRound};${r.kickoff_ro}. Statusstarted=${s.current.status?.started},finished=${s.current.status?.finished},cancelled=${s.current.status?.cancelled};verificat${s.current.verified_at_utc}.\n\n| Configurație | n | GF70 | GA70 | Gol70 |\n|---|---:|---:|---:|---:|\n`;
 for(const [key,label]of[['home',r.home+'sezon'],['away',r.away+'sezon'],['home_venue',r.home+'acasă'],['away_venue',r.away+'deplasare']]){const v=s.sets[key];out+=`|${label}|${v.n}|${v.gf}|${v.ga}|${v.wins}/${v.n}|\n`;}
 const {home:a,away:b,home_venue:av,away_venue:bv}=s.sets;
 out+=`\nλsezon=((${a.gf}/${a.n}+${b.ga}/${b.n})+(${b.gf}/${b.n}+${a.ga}/${a.n}))/2=${n(r.lambda_season)}.\n\nλteren=((${av.gf}/${av.n}+${bv.ga}/${bv.n})+(${bv.gf}/${bv.n}+${av.ga}/${av.n}))/2=${n(r.lambda_venue)}.\n\nλcombinat=0,5×λsezon+0,5×λteren=${n(r.lambda_combined)};P=1−exp(−λ)=${pct(+r.p_raw)}. Doarsezon${pct(+r.season_only)},doarteren${pct(+r.venue_only)},intensitate−20%${pct(+r.p_stress)}. Ponderarea50/50este euristică; eșantioanele sezon/teren se suprapun. Numărătorile au fost recalculate din observațiile salvate și coincid cu snapshotulprospectiv, fără mutarea rangurilor.\n\nDiagnosticeliminândtoatepartidelecartonașroșu=${pct(ss.any_red_card.p)},iarcelecuFTdiferență≥4=${pct(ss.large_margin_4plus.p)}. Eliminareaoricaruiroșunuînseamnă că a precedat primul gol; datelemomentelor sunt păstrate separat. Eșantioanele excluderilor sunt însensitivity.json; rezultatele nu sunt probabilități finale.\n`;
 for(const id of [r.home_id,r.away_id]){const recent=s.recent.filter(x=>x.team_id===id);out+=`\n### ${id===r.home_id?r.home:r.away}: ultimele${recent.length}oficiale returnate de furnizor\n\n|Data|Competiție|Meci|Scor90|Primulgol|Gol70|Sursă|\n|---|---|---|---|---|---|---|\n`+recent.map(x=>`|${x.date.slice(0,10)}|${x.competition}|${x.home}–${x.away}|${x.score_ft}|${x.first_goal_minute||'Fărăgol'}|${x.goal_result||'Neverificat'}|[Statistici](https://www.fotmob.com/match/${x.id})|`).join('\n')+'\n';}
}
out+='\n## Surse contextuale consultate\n\n'+sources.map(([title,url])=>`- [${title}](${url})`).join('\n')+'\n\nFișiere: comparison.json,numărători/status/loturi;last10_official.csv;6teamși3matchJSON;Recentpentru meciuri suplimentare; sensitivity.json. Date brute originale rămân în ../historical_observations.csv și ../all_models.csv.\n';
fs.writeFileSync(dir+'/COMPARISON.md',out);
console.log('Saved focused comparisons, 60 recent-match rows, current sources and diagnostics; no selections or forecasts modified.');
