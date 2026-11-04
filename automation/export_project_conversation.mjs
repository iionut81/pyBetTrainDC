import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

// Export only visible conversation messages. Never export reasoning, system or
// developer instructions, tool calls, tool output, session settings or images.
export function visibleMessages(lines){
 const messages=[];
 for(const line of lines){
  let row;try{row=JSON.parse(line);}catch{continue;}
  const p=row.payload;
  if(row.type!=='response_item'||p?.type!=='message')continue;
  if(p.role!=='user'&&!(p.role==='assistant'&&['final','final_answer','commentary'].includes(p.channel)))continue;
  const text=(Array.isArray(p.content)?p.content:[]).map(c=>typeof c.text==='string'?c.text:/(?:image)/.test(c.type||'')?'[Imagine atasata; continutul nu este exportat]':'').filter(Boolean).join('\n');
  if(!text.trim())continue;
  const redacted=text.replace(/\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b/g,'[credential ascuns]').replace(/\b(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{4}-[A-Z0-9]{4}\b/g,'[cod temporar ascuns]');
  messages.push({timestamp:row.timestamp||'',role:p.role,channel:p.channel||'user',text:redacted,redacted:redacted!==text});
 }
 return messages;
}

export function exportConversation(projectDir,threadId,sessionsRoot){
 if(!/^[a-f0-9-]{36}$/i.test(threadId||''))throw Error('A valid conversation UUID is required.');
 const files=[];
 function find(dir){for(const entry of fs.readdirSync(dir,{withFileTypes:true})){const full=path.join(dir,entry.name);if(entry.isDirectory())find(full);else if(entry.name.endsWith('.jsonl')&&entry.name.includes(threadId))files.push(full);}}
 find(sessionsRoot);
 if(!files.length)throw Error('No local transcript source found for this conversation. Existing exports were preserved.');
 const messages=files.flatMap(f=>visibleMessages(fs.readFileSync(f,'utf8').split(/\r?\n/))).sort((a,b)=>a.timestamp.localeCompare(b.timestamp));
 const seen=new Set();const unique=messages.filter(m=>{const key=crypto.createHash('sha256').update(JSON.stringify(m)).digest('hex');if(seen.has(key))return false;seen.add(key);return true;});
 if(!unique.length)throw Error('No visible messages found. Existing exports were preserved.');
 const target=path.join(projectDir,'Analysis/Conversation_History/transcripts'),threadDir=path.join(target,threadId);
 fs.mkdirSync(threadDir,{recursive:true});
 const metadata={exported_at_utc:new Date().toISOString(),thread_id:threadId,source_files:files.map(f=>path.basename(f)),message_count:unique.length,first_message_at:unique[0].timestamp,last_message_at:unique.at(-1).timestamp,redacted_messages:unique.filter(m=>m.redacted).length,scope:'Visible user/assistant messages available locally for this thread; not all chats or hidden reasoning.'};
 const markdown='# Conversatia proiectului — mesaje disponibile local\n\n'+`Export: ${metadata.exported_at_utc}. ${unique.length} mesaje din ${files.length} surse locale.\n\n`+'Sunt excluse instructiunile de sistem/dezvoltator, rationamentele interne, uneltele si continutul imaginilor. Credentialele si codurile temporare sunt redactate. Arhiva nu garanteaza recuperarea mesajelor sterse sau a altor chaturi.\n\n'+unique.map(m=>`## ${m.timestamp} · ${m.role==='user'?'Utilizator':'Asistent'}\n\n${m.text}\n`).join('\n');
 fs.writeFileSync(path.join(threadDir,'conversation.md'),markdown);
 fs.writeFileSync(path.join(threadDir,'messages.jsonl'),unique.map(m=>JSON.stringify(m)).join('\n')+'\n');
 fs.writeFileSync(path.join(threadDir,'metadata.json'),JSON.stringify(metadata,null,2));
 fs.writeFileSync(path.join(target,'current.md'),markdown);
 fs.writeFileSync(path.join(target,'current-metadata.json'),JSON.stringify(metadata,null,2));
 return {message_count:unique.length,source_count:files.length,redacted_messages:metadata.redacted_messages,export_path:'Analysis/Conversation_History/transcripts/current.md'};
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const projectDir=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
 const configFile=path.join(projectDir,'automation/daily_research_config.json');
 const config=fs.existsSync(configFile)?JSON.parse(fs.readFileSync(configFile,'utf8').replace(/^\uFEFF/,'')):{};
 const at=process.argv.indexOf('--thread');
 const threadId=at>=0?process.argv[at+1]:process.env.CODEX_THREAD_ID||config.chat_thread_id;
 const codexRoot=process.env.CODEX_HOME||path.join(process.env.USERPROFILE||process.env.HOME,'.codex');
 console.log(JSON.stringify(exportConversation(projectDir,threadId,path.join(codexRoot,'sessions'))));
}
