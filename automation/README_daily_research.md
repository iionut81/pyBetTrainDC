# PRED — agent zilnic, 08:00 Romania

Autorizare explicita a utilizatorului: 3 noiembrie 2026. Sarcina Windows:
`Pred_Daily_Research_0800`, verificata ca activa. Prima executie programata:
**4 noiembrie 2026, 08:00 Europe/Bucharest**. Sistemul Windows foloseste
GTB Standard Time (Bucuresti), deci calendarul urmeaza si ora de vara.

## Ce face

1. Citeste arhiva analizelor si alegerile explicite ale utilizatorului pentru
   ziua precedenta; reincearca si rezultatele mai vechi inca neverificate.
2. Verifica observatiile exacte din surse publice, salveaza rezultatul separat
   de predictie si actualizeaza bilantul. Badge: Alegerea ta -Succes/-Fail;
   pana la verificare -In asteptare; amanarile/anularile raman distincte.
3. INAINTEA scanarii curente, analizeaza succesele/esecurile/reusitele la limita
   fata de predictiile originale, salveaza REVIEW.md si LESSONS_FOR_NEXT_SCAN.md
   in directorul Review al zilei revizuite. Citeste lectiile relevante acumulate.
   Distinge dovezile, cauzele necunoscute si ipotezele; nu modifica modele sau
   procente. Aceasta ordine este explicita din5 noiembrie2026. Lipsurile se
   documenteaza intr-un review partial inainte de a continua.
4. Cerceteaza programul curent pentru GOAL_BY_70 si CORNERS_1H_UNDER_7_5,
   pe intregul scop convenit in AGENTS.md si promptul master. Salveaza listele,
   dovezile, lipsurile si candidatii analistului. Utilizatorul decide alegerea.
   Explica in rapoarte ce lectii au fost aplicate si cu ce dovezi; timestampurile
   review_handoff_at_ro si research_started_at_ro documenteaza ordinea.
5. Reconstruieste dashboardul; scrie raportul de executie si rezumatul in romana.
6. Continua conversatia utilizatorului prin Codex CLI `exec resume` pentru
   publicarea rezumatului si a linkului absolut catre `dist/index.html`.
7. Sincronizeaza local mesajele vizibile disponibile ale conversatiei, inclusiv
   dupa publicarea raportului. Pastreaza punctul de reluare in
   `Analysis/SESSION_RESUME.md` si arhiva privata in
   `Analysis/Conversation_History/`. Nu publica transcrierile in GitHub.

08:00 este ora de pornire; durata depinde de programul meciurilor si de surse.
Un research partial este marcat partial. Un proces incheiat nu inseamna automat
ca toate datele au fost obtinute sau ca toate cele trei etape au reusit.

## Implementare si conditii

- Runner: `run_daily_research.ps1`; prompt: `daily_research_prompt.md`.
- Executabilul Codex si Node sunt in `daily_research_config.json`, fara chei.
  Codex CLI foloseste autentificarea ChatGPT existenta. Daca extensia VS Code
  este actualizata si calea dispare, runnerul cauta executabilul Codex instalat.
- Lansare directa, fara fereastra vizibila, cu web search si `--approve-for-me`
  (sandbox workspace-write si evaluarea automata a aprobarilor). Nu sunt folosite
  optiuni de dezactivare a sandboxului sau de ocolire a aprobarilor.
- Calculator pornit, internet si utilizator autentificat in Windows. Sarcina
  foloseste InteractiveToken, fara parola stocata si fara privilegii ridicate.
  Nu necesita fereastra Codex deschisa; utilizeaza CLI. Autentificarea/limitele
  contului, retea indisponibila sau un refuz de aprobare pot opri o rulare.
- StartWhenAvailable pentru ora ratata; pornirea necesita retea. Aceeasi sarcina
  are 5 declansatoare: zilnic la 08:00, la logon, la deblocare, la reconectarea
  retelei si la revenire din sleep. Ora efectiva depinde de disponibilitatea
  Windows/retelei; nu este promisa o latenta de zero secunde.
  `daily_research_policy.ps1` blocheaza researchul inainte de 08:00 si inainte
  de prima zi programata. `daily_completion_YYYY-MM-DD.json` previne repetarea
  cercetarii dupa o executie incheiata, inclusiv partiala. Daca doar mesajul
  catre chat a esuat, se reincearca doar livrarea la urmatorul trigger eligibil.
  IgnoreNew previne suprapunerile Task Scheduler, iar un lock protejeaza runnerul.
  Limita de executie este de 4 ore; niciun alt task nu a fost modificat.
- Nu ruleaza automat modelele de productie, nu schimba selection_engine/,
  nu pariaza, nu inventeaza cote, nu trimite mesaje externe si nu face commit/push.
- Dashboardul este un HTML local: reincarca-l dupa actualizare.
- `publish_daily_research.ps1` publica rezumatul in conversatia exacta din
  `chat_thread_id`, verificata read-only in registrul local Codex pentru acest
  proiect. Comanda foloseste aceeasi autentificare; nu scrie direct in baza de
  date sau transcript. Confirma exit 0, raspuns nou si prezenta linkului HTML.
  Pastreaza `chat_delivery.json` sau `chat_delivery_error.txt`. Livrarea reala
  va fi verificata la prima rulare; acum s-au verificat destinatia si configuratia.

Constatare din prima rulare, 4 noiembrie: researchul a pornit la08:00:01 si
dashboardul a fost actualizat; livrarea prin exec resume a esuat cu
`already has an active writer`, deoarece IDE-ul avea conversatia deschisa.
La aceasta eroare se foloseste acum comanda nativa `codex queue`, fara a forta
inchiderea sau deblocarea conversatiei. Acceptarea se salveaza ca `queued`,
distinct de `delivered`, iar receiptul `chat_queued.json` previne mesaje duplicate.
Un mesaj acceptat in coada nu este prezentat ca raspuns deja afisat in chat.

## Dovezi si inspectie

- `Analysis/Automation/daily_status.json`: programare/starea ultimei rulari.
- `Analysis/Automation/scheduled_task.xml`: definitia exportata din Windows.
- `Analysis/Automation/preflight_verification.json`: test valid exit 0/READY
  si test invalid exit 1. Ambele fara lansarea agentului sau crearea unui task.
- `Analysis/Automation/Runs/YYYY-MM-DD_HHMMSS/`: promptul datat, run.json,
  summary.md si logurile locale de diagnostic.
- `Analysis/Automation/run_report_YYYY-MM-DD.json`: starea fiecarei etape.
- `Analysis/Automation/chat_destination_verification.json`: destinatia validata,
  fara mesaje de test sau pornirea unui model.

Verificari fara cercetare noua:

```powershell
powershell -NoProfile -File .\automation\run_daily_research.ps1 -Preflight
Get-ScheduledTask -TaskName Pred_Daily_Research_0800
Get-ScheduledTaskInfo -TaskName Pred_Daily_Research_0800
```

Instalatorul `install_daily_research.ps1` valideaza preflight-ul intr-un proces
separat: cere atat exit 0, cat si JSON READY, directorul si fusul orar corecte.
Refuza inregistrarea la orice esec. Nu inlocuieste un task cu acelasi nume care
ar indica alt script. `-PreviewOnly` prezinta definitia fara inregistrare.
Inregistrarea a fost verificata; prima executie completa a cercetarii este
inca viitoare si nu este prezentata drept testata end-to-end.

Testarea regulilor de catch-up, dubluri si ora de vara, fara agent/programare:

```powershell
powershell -NoProfile -File .\tests\test_daily_research_policy.ps1
```
