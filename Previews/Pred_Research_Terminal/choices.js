const CHOICES={connected:false,busy:false,token:null,revision:null,clock:null,receivedAt:0};
function choiceEditable(m,market){
 if(!CHOICES.connected||!CHOICES.clock)return false;
 const elapsed=Math.max(0,Date.now()-CHOICES.receivedAt),now=new Date(Date.parse(CHOICES.clock.iso)+elapsed);
 const parts=Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Bucharest',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(now).map(p=>[p.type,p.value]));
 return DATA.date===`${parts.year}-${parts.month}-${parts.day}`&&/^\d\d:\d\d$/.test(m.kickoff)&&m.kickoff>`${parts.hour}:${parts.minute}`&&m[market].outcome.result==='pending'&&m[market].eligible&&choiceValue(m,market).p!=null;
}
function choiceControl(m,market,compact=false){
 const selected=Boolean(m[market].userSelection),text=selected?'Anulează alegerea':'＋ Alege acest pick',css=selected?'pick-remove':'pick-select';
 if(location.protocol==='file:')return selected?'<span class="pick-locked">Alegere salvată</span>':`<a class="pick-select" href="pred-dashboard://open" title="Deschide în Chrome pentru salvare">${text}</a>`;
 const editable=choiceEditable(m,market),reason=CHOICES.busy?'Se salvează…':!CHOICES.connected?'Serviciul local nu este conectat. Reîncarcă pagina.':!editable?'Alegere arhivată, meci început sau date insuficiente.':'';
 if(selected&&!editable&&!CHOICES.busy&&CHOICES.connected)return '<span class="pick-locked">Alegere arhivată</span>';
 return `<button class="${css}${market==='corner'?' corner-choice':''}${compact?' compact':''}" data-choice="${esc(m.id)}" data-choice-market="${market}" data-choice-action="${selected?'remove':'select'}" ${!editable||CHOICES.busy?'disabled':''} title="${esc(reason)}">${CHOICES.busy?'Se salvează…':text}</button>`;
}
function updateChoiceConnection(){
 let status=$('#choiceConnection');if(!status){status=document.createElement('p');status.id='choiceConnection';status.className='choice-connection';document.querySelector('.page-heading').after(status);}
 status.hidden=CHOICES.connected;
 status.innerHTML=location.protocol==='file:'?'Pentru a salva alegeri: <a href="pred-dashboard://open">deschide dashboardul în Chrome ↗</a>. Fișierul HTML rămâne disponibil pentru consultare.':CHOICES.connected?'':'Salvarea alegerilor nu este conectată. Deschide din shortcutul PRED Dashboard - Chrome sau reîncarcă pagina.';
}
function applyChoiceState(payload){
 const date=DATA.date;Object.assign(ARCHIVE,payload.archive);DATA=ARCHIVE.days.find(d=>d.date===date)||ARCHIVE.days.at(-1);
 Object.assign(CHOICES,{connected:true,token:payload.token||CHOICES.token,revision:payload.revision,clock:payload.clock,receivedAt:Date.now()});
 renderDay();$('#leagueSelect').value=state.league;updateChoiceConnection();
}
async function initializeChoices(){
 if(location.protocol!=='http:'||location.hostname!=='127.0.0.1'){updateChoiceConnection();return;}
 try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error('Local service unavailable');applyChoiceState(await r.json());}
 catch{CHOICES.connected=false;updateChoiceConnection();renderSelections();renderTable();}
}
async function saveChoice(id,market,action){
 if(CHOICES.busy)return;
 const match=DATA.matches.find(m=>m.id===id);if(!match||!choiceEditable(match,market))return toast('Alegerea nu poate fi modificată pentru acest meci.');
 const open=$('#detailDialog').open;CHOICES.busy=true;renderSelections();renderTable();if(open)showDetail(id,market);
 try{
  const r=await fetch('/api/choices',{method:'POST',headers:{'Content-Type':'application/json','X-Pred-Token':CHOICES.token},body:JSON.stringify({date:DATA.date,matchId:id,market,action,revision:CHOICES.revision})});
  const payload=await r.json();
  if(!r.ok){if(r.status===409)await initializeChoices();throw Error(payload.error||'Salvarea nu a reușit.');}
  applyChoiceState(payload);toast(payload.warning||(action==='select'?'Alegerea ta a fost salvată în istoric.':'Alegerea a fost anulată.'));
 }catch(error){toast(error.message||'Salvarea nu a reușit. Reîncearcă.');}
 finally{CHOICES.busy=false;renderSelections();renderTable();if(open&&$('#detailDialog').open)showDetail(id,market);}
}
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&!CHOICES.busy)initializeChoices();});
window.addEventListener('focus',()=>{if(!CHOICES.busy)initializeChoices();});
