# Continuitatea conversatiilor PRED

Cerinta utilizatorului, 3 noiembrie 2026: pastram conversatiile proiectului si
deciziile, astfel incat inchiderea IDE-ului sa nu impuna reluarea explicatiilor.

- `../SESSION_RESUME.md`: punctul scurt de reluare; se actualizeaza dupa fiecare
  solicitare care schimba regulile, alegerile, implementarea sau pasii urmatori.
- `../Selection_History/YYYY-MM-DD.md`: jurnal zilnic al cererilor, deciziilor,
  analizelor, alegerilor, rezultatelor si lectiilor.
- `transcripts/current.md`: copia lizibila a mesajelor disponibile pentru
  conversatia exportata cel mai recent. Fiecare conversatie are si propria
  subarhiva, identificata prin ID, pentru a nu pierde exporturile anterioare.

Sincronizare locala, fara retea, fara agent nou si fara modificarea istoricului
nativ Codex:

```powershell
node .\automation\export_project_conversation.mjs
```

Exportul citeste numai sursele conversatiei curente (sau `--thread UUID`
explicit) din istoricul local. Salveaza mesajele user/asistent vizibile, inclusiv
comentariile, dar exclude rationamentele interne, instructiunile de sistem sau
dezvoltator, apelurile/rezultatele uneltelor si continutul imaginilor. Redacteaza
credentialele si codurile temporare. Nu cauta continutul altor conversatii.

Nu este o promisiune ca toate chaturile de pe alte dispozitive sau mesajele
sterse sunt disponibile. Ultimul raspuns aflat inca in generare poate intra
in export la urmatoarea sincronizare; deciziile sale se salveaza imediat in
punctul de reluare si jurnal. Istoricul nativ al chatului si arhiva proiectului
sunt mecanisme separate.

Transcrierile sunt locale si ignorate de Git: repository-ul utilizatorului
este public. Nu le publica sau include in commit fara o cerere explicita noua.
Pe alt calculator sunt necesare fisierele proiectului/arhivele copiate; nu
presupunem memorie automata intre chaturi sau dispozitive.
