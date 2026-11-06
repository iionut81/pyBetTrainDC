import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {read,csv,save,ro} from '../../Automation/research_2026_11_06.mjs';
const dir='Analysis/Goal70_2026-11-06/Derby_Bolton_Review',id='5836927';
const inputFiles=['Analysis/Selection_History/2026-11-06_goal70_candidates.csv','Analysis/Selection_History/2026-11-06_corners1h_candidates.csv','Analysis/Selection_History/user_selections.csv'];
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const before=inputFiles.map(hash),g=csv(inputFiles[0]).find(r=>r.id===id),c=csv(inputFiles[1]).find(r=>r.id===id);
const history=read(dir+'/season_evidence.json'),recent=csv(dir+'/last10_official.csv'),s=read(dir+'/sensitivity.json');
assert.equal(new Set(history.map(r=>r.id)).size,28);assert.equal(recent.length,20);
assert(Math.abs(s.baseline.p-Number(g.p_raw))<1e-12);
const now=ro(),match=read(dir+'/match_'+id+'.json');
assert.equal(String(match.data.general.matchId),id);assert.equal(match.data.header.status.started,false);
const pct=v=>(Number(v)*100).toFixed(2)+'%',table=(heads,rs)=>'| '+heads.join(' | ')+' |\n| '+heads.map(()=>'---').join(' | ')+' |\n'+rs.map(r=>'| '+r.join(' | ')+' |').join('\n')+'\n';
const firstTable=team=>table(['Data','Competiție','Meci','Scor90','Primul gol','Gol≤70','Dovadă'],recent.filter(r=>r.team_id===team).map(r=>[r.date,r.competition,r.home+'–'+r.away,r.score,r.first_goal==='0-0'?'Fără gol':r.first_goal,r.success70==='true'?'Succes':'Fail',`[sursă](${r.source})`]));
const sources={fixture:'https://www.fotmob.com/api/data/matchDetails?matchId=5836927',derbyInterview:'https://www.dcfc.co.uk/news/2026/11/interview-we-want-to-put-pride-back-in-the-shirt-eustace',derbyLast:'https://www.dcfc.co.uk/news/2026/11/report-derby-county-vs-norwich-city',boltonCoach:'https://www.bwfc.co.uk/news/schumacher-capable-winning-anywhere',boltonLast:'https://www.bwfc.co.uk/news/match-report-charlton-athletic-0-1-wanderers',efl:'https://www.efl.com/news/2026/november/05/derby-county-v-bolton-wanderers--pre-match-stat-pack/',secondaryInjuries:'https://uk.sports.yahoo.com/news/xav-simons-eyeing-another-memorable-050000390.html',secondaryPreview:'https://www.sportsmole.co.uk/football/derby-county/preview/derby-vs-bolton-prediction-team-news-lineups_608406.html'};
const weather=read(dir+'/weather.json'),wh=weather.data.hourly,wi=wh.time.indexOf('2026-11-06T20:00');
save(dir+'/context_sources.json',{updated_at_ro:now,sources,source_limits:['EFL indexed preview statistics extracted by search; full article body unavailable in web open','Sports Mole and The Bolton News/Yahoo are secondary sources, no prediction/odds used','Full2026/27tie-break hierarchy not independently retrieved','No confirmed XI; provider explicitly labels lastStarting11, not predicted/confirmed'],weather_source:weather.source,fixture_verified_at_utc:match.verified_at_utc});
const text=`# Derby County–Bolton Wanderers: analiza completă Gol70

Actualizare: ${now}, Europe/Bucharest. Meci: 6 noiembrie2026,22:00RO/20:00GMT, Pride Park, Championship2026/27,etapa15. Status neînceput verificat${match.verified_at_utc}. Analiză suplimentară solicitată de utilizator după screeningul08:31; nu rescrie acel snapshot.

## Verdict și rol în listă

Gol până la minutul afișat70inclusiv, oricine marchează: model brut${pct(g.p_raw)}, AR20${pct(g.p_stress)}, cotă echitabilă matematică${Number(g.fair_odds).toFixed(4)}. Rangul brut este13din34partide inventariate/33modele eligibile. NUeste candidatul principal al analistului și nu este înregistrat ca alegere a utilizatorului la verificare. Este candidat comparat. Nu îi atribuim retrospectiv o selecție pe care nu am făcut-o.

Partida este în listă pentru că are istoric complet, comparabil din același campionat,25/28deschideri≤70și defensiva Derby vulnerabilă. Este o alternativă explicabilă, însă nu aș promova-o peste candidatulGol70al dimineții doar pentru14/14al Derby. Atacul modest al gazdei, indisponibilitatea confirmatăSzmodics și incertitudinile din mijlocul Bolton pot împiedica deschiderea scorului. Procentele sunt necalibrate;81,99%AR20este scenariu de sensibilitate, nu probabilitate reală validată. Nu s-au folosit cote de bookmaker și nu se garantează prețul1,10.

## Datele și calculele originale

GF70/GA70 înseamnă toate golurile proprii/primite la minutul afișat≤70, inclusiv45+, fără prelungiri eliminatorii/departajare. Frecvența se referă separat la primul gol al partidei.

${table(['Eșantion','n','GF70','GA70','Primul gol≤70','Primul gol>70','0–0'],[['Derby sezon',14,8,22,'14/14',0,0],['Bolton sezon',14,14,16,'11/14',1,2],['Derby acasă',7,4,12,'7/7',0,0],['Bolton deplasare',7,7,7,'5/7',0,2]])}
λsezon=((8/14+16/14)+(14/14+22/14))/2=2,142857.

λteren=((4/7+7/7)+(7/7+12/7))/2=2,142857.

λcombinat=0,5λsezon+0,5λteren=2,142857.

P=1−exp(−2,142857)=88,268083%. Numai sezon=88,27%; numai teren=88,27%. Egalitatea este rezultatul numărătorilor, nu un bug. Intensitate−20%:1−exp(−0,8×2,142857)=81,990769%. Cota matematică1/P=1,132912, nu ofertă disponibilă.

Ponderarea este euristică, eșantioanele se suprapun și nu există calibrare independentă. Nu se multiplică din nou cu70/90. Istoric deduplicat25/28=89,29%; teren12/14=85,71%. Nu transformăm aceste frecvențe în probabilitatea individuală de azi.

Pe teren, contribuțiile modelului sunt: componenta Derby=(4/7+7/7)/2=0,785714; componenta Bolton=(7/7+12/7)/2=1,357143. Nu reprezintă probabilități despre cine deschide scorul. Ele arată că scenariul modelului se sprijină mai mult pe Bolton contra defensivei Derby.

## De ce există argument pentru gol

1. Derby a avut primul gol≤70în14/14meciuri, inclusiv7/7acasă; toate14s-au deschis cel târziu49. Este regularitatea evenimentului în meciurile Derby, NU14meciuri în care Derby a marcat.
2. Acasă, Derby a primit gol≤70în6/7meciuri și a marcat≤70în3/7. Ratele proprii sunt4/7=0,57GF70/meci versus12/7=1,71GA70/meci. Principalul argument este vulnerabilitatea defensivei, nu o ofensivă puternică a gazdei.
3. Bolton a marcat≤70în4/7deplasări și a primit în4/7; cinci meciuri au avut gol≤70. Două sunt0–0:QPR22august șiSouthampton17octombrie. Există contraexemple clare pentru a nu trata adversarul vulnerabil ca gol sigur.
4. Bolton a luat10puncte în ultimele4deplasări și Dalby are6goluri de campionat conform[EFL](${sources.efl});[raportul oficial Charlton](${sources.boltonLast}) confirmă golul18din ultima deplasare. Această formă susține o cale ofensivă, nu un bonus procentual.

## Context, miză și avantajul terenului

Tabelul actual al furnizorului, salvat în team_10170.json șiteam_8559.json: Derby23/7puncte după14meciuri(1V4E9Î),golaveraj11–29; Bolton17/17puncte(5V2E7Î),19–26. Previewul secundar publicat4noiembrie scrie16pentruBolton; păstrăm diferența și folosim tabelul actual17la extragere. [Pagina oficială Derby](${sources.derbyInterview}) confirmă23/7și decalajul de7puncte față de locul21.

Este meci de campionat pentru puncte, nu retur/eliminatoriu: victoria aduce3,egalul1, fără scor agregat sau prelungiri. Nu pretind verificarea integrală a tuturor criteriilor de departajare2026/27; documentul EFL dinamic nu a fost extras integral. Aceste criterii nu produc o obligație de gol sau atac în această partidă.

Derby are presiune pentru puncte și reacție după0–4cuNorwich. Un egal ar opri seria de înfrângeri și poate fi acceptabil în cursul jocului; nu există dovadă că Eustace trebuie să atace cu orice risc din minutul1. Bolton poate beneficia de fragilitatea adversarului, dar poate accepta un început controlat într-o deplasare. Niciun antrenor nu a declarat verificabil un plan«jucăm pentru0–0».

Pride Park este terenul obișnuit Derby, suprafață naturală. Avantajul publicului poate coexista cu presiune/frustrare; nu îi atribuim efect numeric. H2Hvechi favorabil Derby, menționat deEFL, nu este folosit în formulă sau ca dovadă despre loturile de azi.

## Tactică, antrenori și mindset — fapte separate de interpretare

JohnEustace cere o reacție colectivă și responsabilitate după ultimele prestații. [Interviul oficial5noiembrie](${sources.derbyInterview}) confirmă obiectivul și Szmodicsoperat,absent6săptămâni. Acesta nu este un angajament că echipa va ataca agresiv imediat sau o dovadă privind starea mentală individuală.

API etichetează explicit formațiile ca lastStarting11: Derby4-4-2,Bolton4-2-3-1. NUeste un11probabil de azi și, cu atât mai puțin, confirmat. Același câmp includeSzmodics,SheehanșiIto. Previewuri secundare propun Derby4-2-3-1și diferă la Bolton; forma finală rămâne necunoscută. Nu afișăm ultima echipă ca titulari actuali.

Interpretare tactică pentruGol70:

- Dacă Derby împinge fundașii laterali și pierde mingi în construcție, Bolton poate găsi spații pentruDalbyși jucătorii care îl sprijină. Golul Bolton de laCharlton a venit din exploatarea unei pase greșite; este o rută observată, nu garanție că Derby repetă eroarea.
- Dacă gazda joacă mai direct, Morris/Meisterpot oferi prezență pentru centrări și faze fixe. [Raportul oficial contraNorwich](${sources.derbyLast}) descrie și șansa luiMorrisla un corner. Șansele izolate nu demonstrează un volum ofensiv mare.
- Dacă Eustace prioritizează distanțe mici și protecția careului după cele8goluri primite în două meciuri, începutul poate deveni prudent. Un4-2-3-1cu doi mijlocași de protecție poate limita tranzițiile; schimbarea nu este confirmată.
- [Schumacher,2noiembrie](${sources.boltonCoach}) vorbește despre încredere în deplasare și reducerea erorilor. Este o declarație recentă înainteaCharlton, nu o conferință nouă pentruDerby. Absențele din mijloc pot limita progresia și calitatea ultimei pase chiar dacă Derby continuă să se apere slab.

Nu avem xG comparabil pe starea0–0, PPDA sau cronologie de ocazii pentru toate meciurile. Nu inventăm pressing sus, posesie dominantă sau un volum numeric de ocazii pentru această seară.

## Loturi și contradicții

- Derby: Szmodicseste CONFIRMAT absent6săptămâni după operația la genunchi, sursă primară. API îl prezintă doarDoubtfulși îl păstrează în ultimul11; pentru disponibilitate prioritizăm clubul.
- API listează probleme pentruHedges,Ward,Ozoh,Travis,Langås,Cashin,Taylor,Forsyth,Murkin,Clarke,Agyemang. Nu sunt12absențe confirmate independent azi. SportsMole relatează căClarkeșiAgyemangau revenit la antrenament; asta nu confirmă capacitatea de titularizare. Timpul revenirilor rămâne incert.
- Bolton: Bazunu,Toal,Erhahon,Hardie,McAtee,Nelsonapar indisponibili în sursele secundare/API. SheehanșiItonecesită evaluare conform relatării[BoltonNews/Yahoo6noiembrie](${sources.secondaryInjuries}).[Raportul oficial Charlton](${sources.boltonLast}) confirmă accidentarea luiSheehanîn timpul meciului și înlocuirea luiIto, fără să confirme individual disponibilitatea pentru vineri. Nu extrapolăm lista«fără probleme noi»din2noiembrie peste evenimentele din3noiembrie.

Impact: problemele defensive Derby favorizează ruta spre golul adversarului, iar lipsaSzmodicsreduce alternativa ofensivă a gazdei. DacăBoltonpierde creativitate prinSheehan/Ito, ambele efecte trebuie evaluate împreună; nu aplicăm ajustări−5%sau+3%arbitrare. Revenirea portaruluiBazununu este confirmată,Bonhameste portarul ultimului11.

## Odihnă, deplasare și vreme

Ambele au jucat31octombrie și3noiembrie; acesta este al treilea meci în7zile. Intervalul între starturile3și6noiembrie este72h15min. Derby a jucat marți acasă, Bolton laCharlton; vineri Bolton are din nou deplasare. Nu cunoaștem itinerarul, ora plecării sau dacă echipa a revenit acasă între meciuri. Nu presupunem că Bolton este automat epuizată sau că oboseala produce goluri.

Open-Meteoextras${weather.verified_at_utc},pentru20:00Europe/London:temperatură${wh.temperature_2m[wi]}°C,probabilitate precipitații${wh.precipitation_probability[wi]}%,vânt${wh.wind_speed_10m[wi]}km/h. Sunt prognoze, nu măsurători ale terenului. FotMobmai vechi indica9°C/16%ploaie; probabilitatea ploii diferă între surse/momente. Nu apare vânt puternic în prognoza independentă, dar nu există inspecție actuală a gazonului. [Sursa orară](${weather.source}).

## Ultimele10oficiale — Derby

${firstTable('10170')}
10/10Gol70în aceste10, dar fără victorie(3E7Î). Nu confundăm succesul evenimentului cu forma echipei.

## Ultimele10oficiale — Bolton

${firstTable('8559')}
8/10Gol70în ultimele10. În întreg sezonul mai existăQPR–Bolton0–0din22august. Bolton–Cardiff1–0cu primul gol90este pierzător pentruGol70deșiBolton a câștigat. Tabelele și numărătorile includ45+,exclud prelungiri/departajare. Datele sunt din28partide unice, înaintea zilei6noiembrie; fără dublare deH2H.

## Sensibilitate la eliminări

Excluderea mecanică a celor3istorice cu eliminare până la70(Derby–WestBrom,SheffieldUnited–Derby,Bolton–Stoke)produceP=${pct(s.without_pre70_reds.p)},AR20=${pct(s.without_pre70_reds.stress)}. Numitori rămași:Derby12/6pe teren,Bolton13/7pe teren. Nu promovăm acest rezultat la estimare finală și nu schimbăm snapshotul. LaSheffield–DerbyșiDerby–WestBromprimul gol precedase eliminarea; efectul asupra primei deschideri nu poate fi atribuit acelui cartonaș. LaBolton–Stokeoponentul a fost eliminat20,primul gol45+4. Vezi[sensitivity.json](sensitivity.json).

## Și la cornere?

În aceeași comparație este tot rang13, brut${pct(c.p_raw)},AR20${pct(c.p_stress)},λ4,357143. Istoric25/28,teren12/14,3reprizecu≥8și2cu exact7. Derbyacasămedia4,57/varianța9,29;Boltonafaramedia4,14/varianța5,81. Varianțele mai mari decât mediile și12/14teren sunt contraargumente la o interpretare de«92,47%sigur». Nu îl promovăm peste candidatulcornereSevilla–Alavésși nu combinăm piețele. DateleGol70nu sunt substitut pentrucornere1H.

## De ce loc13 și verdictul profesional

Silkeborg–Lyngbyare93,05%brut/88,16%AR20,24/25istoric și7/6GF70propriu pentru fiecare pe terenul relevant, conform screeningului original. Derby–Boltonare88,27%/81,99%,25/28și4/7respectiv7/7GF70. Diferența se explică prin intensitate, nu printr-un veto arbitrar. Această comparație nu este o revalidare completă la ora curentă a loturilor tuturor celorlalți candidați.

Pro decisiv: istoricul constant al deschiderii scorului în meciurile Derby și defensiva care cedează devreme oferă Bolton o cale credibilă de gol. Risc decisiv: Boltonare două0–0în7deplasări și incertitudini în construcție, iar Derbyproduce numai0,57GF70/meci acasă și pierdeSzmodics. O reacție defensivă a gazdei poate diminua ritmul. NO BET rămâne permis; pentru prioritate finală ar trebui clarificateSheehan/Itoși formulaDerby. Nu este nevoie de cote pentru această concluzie.

Nu este o selecție rea doar fiindcă nu ocupă primul loc. Este candidat secundar, susținut de date, cu avertismente concrete. Nu îl numesc prima alegere și nu afirm o probabilitate reală validată de88–90%.

## Lecții aplicate și integritate

Aplicate:GF70propriu separat deGA70;prime goluri și0–0păstrate;același campionat în formulă;context la0–0;lot actual versuslastStarting11;surse contradictorii,prognoză versusgazon;scenariuAR20și sensibilitate la eliminări fără procente contextuale. Modelul și toate clasamentele rămân originale.

Intrări numerice:season_evidence.json/.csv,last10_official.csv,sensitivity.json; sursele actuale încontext_sources.jsonși răspunsurileAPI. Copiile CSVcanonice de azi șiuser_selectionsnu sunt scrise. O analiză cerută nu devine alegere sau pariu plasat.
`;
const replacements={'JohnEustace':'John Eustace','StevenSchumacher':'Steven Schumacher','SportsMole':'Sports Mole','WestBrom':'West Brom','SheffieldUnited':'Sheffield United','BoltonNews':'Bolton News','lastStarting11':'lastStarting11'};
const readable=text.split(/(\[[^\]]*\]\([^)]*\)|\b[\w./-]+\.(?:mjs|csv|json)\b)/).map((x,i)=>{if(i%2)return x;for(const [a,b]of Object.entries(replacements))x=x.replaceAll(a,b);return x.replace(/(\p{L})(\d)/gu,'$1 $2').replace(/(\d)(\p{L})/gu,'$1 $2').replace(/(%)(\p{L})/gu,'$1 $2').replace(/\b(Gol|GF|GA|AR) (70|20)\b/g,'$1$2');}).join('');
fs.writeFileSync(dir+'/ANALYSIS.md',readable);
assert.deepEqual(inputFiles.map(hash),before);
save(dir+'/verification.json',{status:'PASS',updated_at_ro:now,history_unique:28,last10_official:20,original_probability_reproduced:true,input_hashes_unchanged:true,not_a_user_selection:true,fixture_unstarted_at_utc:match.verified_at_utc,snapshot_at_ro:g.recorded_at_ro,model_rank:Number(g.rank),sources:inputFiles.map((file,i)=>({file,sha256:before[i]}))});
console.log(JSON.stringify({report_saved:true,rank:g.rank,p:g.p_raw,stress:g.p_stress,history28:25,last10:20,original_inputs_unchanged:true}));
