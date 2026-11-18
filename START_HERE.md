# PRED — memoria proiectului și reluarea de pe alt calculator

Actualizat: 18 noiembrie 2026. Acest repository păstrează regulile, deciziile, analizele, rezultatele și o copie a conversațiilor vizibile disponibile local. Nu importă automat istoricul în ChatGPT, Codex sau Claude.

## Ce trebuie citit la reluare

1. [AGENTS.md](AGENTS.md): regulile persistente și limitele autorizării.
2. [CLAUDE.md](CLAUDE.md): arhitectura existentă; regulile mai recente din AGENTS și conversație au prioritate.
3. [Starea curentă](Analysis/SESSION_RESUME.md): unde a rămas lucrul și pasul următor.
4. [Jurnalul din 18 noiembrie](Analysis/Selection_History/2026-11-18.md), apoi cel mai recent jurnal disponibil.
5. [Fluxul zilnic](Analysis/Selection_History/DAILY_WORKFLOW.md), [alegerile canonice](Analysis/Selection_History/user_selections.csv) și [evenimentele alegerilor](Analysis/Selection_History/selection_events.jsonl).
6. [Promptul de cercetare](Prompts/Football_Goal70_Corners1H_Master.md), înaintea cercetării celor două piețe.

## Conversații și explicații

- [Pagina publică pentru review](https://iionut81.github.io/pyBetTrainDC/review/): evaluare, meciuri, istoric și rapoarte, numai pentru consultare; copie cu date până la18noiembrie2026. Nu se actualizează automat la modificarea fișierelor locale.

- [Indexul conversațiilor publicate](Analysis/Conversation_History/published/README.md): mesajele vizibile găsite local pentru sesiunile principale ale acestui proiect, inclusiv rulările zilnice. Inventarul exact este în manifestul din același director.
- [Jurnalele zilnice](Analysis/Selection_History/README.md): decizii, alegeri explicite, rezultate și lecții.
- [Auditul sistemelor](Analysis/System_Audit_2026-11-17/REPORT.md) și [protocolul propus pentru îmbunătățirea selecției](Analysis/System_Audit_2026-11-17/SELECTION_IMPROVEMENT_PROTOCOL.md).
- [Cercetarea din 18 noiembrie](Analysis/Goal70_2026-11-18/CONTEXT.md), [cornere](Analysis/Corners1H_2026-11-18/ANALYSIS.md) și [review-ul zilei precedente](Analysis/Review_2026-11-17/REVIEW.md).

Publicarea conversațiilor a fost autorizată explicit pe 18 noiembrie 2026. Copia publicată exclude raționamente interne, instrucțiuni de sistem/dezvoltator, înregistrări de unelte, sesiuni de agenți interni, credentiale și imaginile atașate. Nu garantează recuperarea chaturilor șterse sau a celor de pe alte dispozitive. Afirmațiile istorice se citesc împreună cu corecțiile și starea curentă.

## De pe alt calculator

Poți citi direct fișierele în GitHub sau poți clona proiectul:

```powershell
git clone https://github.com/iionut81/pyBetTrainDC.git
cd pyBetTrainDC
```

Deschide folderul în editorul/asistentul ales și folosește această cerere:

> Citește START_HERE.md, AGENTS.md, CLAUDE.md și Analysis/SESSION_RESUME.md, apoi cel mai recent jurnal și alegerile/evenimentele canonice. Continuă din starea salvată. Răspunde în română și păstrează predicțiile și alegerile originale.

Dashboardul salvat poate fi deschis din [Previews/Pred_Research_Terminal/dist/index.html](Previews/Pred_Research_Terminal/dist/index.html) după clonare sau descărcare. GitHub afișează codul HTML; pentru interfață deschide fișierul local în browser. Pentru regenerare, dacă ai Node.js instalat:

```powershell
node Previews/Pred_Research_Terminal/build.mjs
```

Sarcina Windows de cercetare, autentificările și handlerul `pred-dashboard://open` sunt configurări ale calculatorului actual; clonarea nu le instalează pe alt dispozitiv. Pentru alegeri din dashboard se folosește serviciul local descris în [README-ul dashboardului](Previews/Pred_Research_Terminal/README.md); fișierul HTML deschis direct este pentru consultare.

## Starea la publicare

18 noiembrie: NO BET la GOAL_BY_70 și CORNERS_1H_UNDER_7_5; review și cercetare parțiale, dashboard verificat. Modelul de producție nu a fost schimbat. Protocolul challengerilor este o propunere separată, fără implementare autorizată în producție.

În Git sunt incluse rapoartele, numărătorile, snapshoturile de predicție și sursele citate. Cache-urile voluminoase de răspunsuri brute și logurile private de execuție rămân locale; această publicare a memoriei nu este un backup integral al discului sau al mediului de execuție.
