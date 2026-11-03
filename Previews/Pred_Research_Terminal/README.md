# PRED Research Terminal — dashboard local cu istoric

Deschidere directa in Chrome: `pred-dashboard://open`. Handler Windows per-user
instalat la cererea utilizatorului, fara modificarea browserului implicit.
Comanda sa are numai executabilul Chrome si calea fixa a acestui index; nu
preia argumente din URI. Test real: titlul ferestrei este
`PRED — Football Research Terminal - Google Chrome`.

Shortcut Explorer: `Deschide PRED in Chrome.url`. Daca aplicatia de chat
blocheaza linkurile cu scheme personalizate, foloseste acest shortcut din
Explorer. Linkul obisnuit spre HTML poate continua sa fie deschis in editor.
La mutarea pe alt calculator/folder ruleaza `install-chrome-link.ps1` acolo;
`-PreviewOnly` arata exact handlerul fara instalare. Instalarea refuza
inlocuirea unui protocol detinut de alta aplicatie. Detalii locale in
`chrome-launcher.json`.

Configuratiile locale (executabile, chat_id, scurtaturi si starea Task Scheduler)
nu sunt versionate. Pe un calculator nou ruleaza instalatoarele locale.
Datele incluse in dashboard publica numai campurile operationale necesare
afisarii starii, fara identificatorul conversatiei sau caile locale ale runnerului.

Concept interactiv inspirat de capturile utilizatorului: fundal inchis,
accente turcoaz/violet, panouri compacte, explorarea vizuala a meciurilor.
Implementarea este HTML/CSS/JavaScript, cu SVG pentru grafic. Frameworkul sau
tehnologia exacta a aplicatiilor din capturi nu poate fi identificata din imagini.

Deschide `dist/index.html` in Edge sau Chrome. Nu necesita instalare, server,
cont, servicii externe ori conexiune la internet pentru interfata. Linkurile
catre sursele sportive necesita internet; rapoartele locale necesita pastrarea
structurii proiectului. Preview-ul nu este publicat si nu modifica modelele.

Date reale din proiect, incarcate pe zile: 2 si 3 noiembrie sunt disponibile
initial. Pentru 3 noiembrie sunt 20 partide Gol70 si 19 modele de cornere din
20 inventariate. QPR-Burnley Gol70 si Derby-Norwich cornere sunt alegerile
utilizatorului, cu rezultate neverificate. Estimarea finala Derby ramane
separata de screening. Pentru 2 noiembrie, Gol70 are 22/26 succese si o amanare;
cornerele au doar Sassuolo verificat, o amanare si 25 rezultate neverificate.
Lipsurile raman NULL. Nu exista flux live sau plasare de pariuri. Utilizatorul
a autorizat ulterior agentul zilnic de la 08:00 RO; vezi detaliile de mai jos.

Controale: navigare Panou/Meciuri/Istoric/Evaluare, selectorul zilei de analiza,
selectie piata, competitie, cautare, detalii prin clic sau tastatura, filtre
de rezultate si export CSV al zilei/pietei filtrate. Istoric are selector
propriu de zi si piata, initial pe cea mai recenta zi cu rezultate. Evaluarea
agrega zilele existente, separat pe piete si separat pentru alegerile tale;
nu include rezultate lipsa. Animatia este decorativa; respecta reduced-motion.

Reconstruire din instantaneele existente, fara a executa modelele:

```powershell
node .\Previews\Pred_Research_Terminal\build.mjs
```

`data.mjs` descopera automat jurnalele `YYYY-MM-DD.md` si CSV-urile datate
`*_goal70_candidates.csv` / `*_corners1h_candidates.csv` din
`Analysis/Selection_History/`, impreuna cu `user_selections.csv`. Citeste
doar sursele analitice aferente acelor zile: finalisti cornerelor, verificari
si `Analysis/Review_YYYY-MM-DD/{goal70,corners1h}_outcomes.csv`. Exista fallback
la `Analysis/Corners1H_YYYY-MM-DD/all_models.csv`. Nu citeste secrete sau conturi.

O zi noua nu necesita schimbarea codului. Dupa fiecare analiza, alegere sau
rezultat salvat, ruleaza constructorul. `dist/index.html` contine CSS, JavaScript
si toata arhiva inclusa pentru functionarea directa de pe disc;
`dist/snapshot.json` este copia structurata completa, iar `dist/data/` contine
cate un JSON pentru fiecare zi. Reincarca pagina dupa reconstruire.
Constructorul nu colecteaza date si nu executa modelele de productie.

Badge-urile **Alegerea ta** sunt urmate de **-Succes/-Fail** exclusiv din
rezultatele verificate; pana atunci afiseaza **-In asteptare**, respectiv
**-Amanat/-Anulat**. Etichetele se actualizeaza in tabel, carduri, istoric si
fisa meciului. Propunerile analistului nu devin automat alegeri ale utilizatorului.
Alegerile tale sunt fixate primele in tabel pentru ziua/piata selectata;
restul raman ordonate dupa model. Coloana # pastreaza rangul original al
modelului, iar filtrele de competitie/cautare continua sa se aplice.

Agent zilnic autorizat la 3 noiembrie: Windows task `Pred_Daily_Research_0800`,
prima executie 4 noiembrie 2026 la 08:00 Europe/Bucharest. Verifica ziua
precedenta si alegerile utilizatorului, salveaza lectiile, cerceteaza ambele
piete si reconstruieste dashboardul. Starea ultimei rulari este citita din
`Analysis/Automation/daily_status.json` si afisata in subsol. Conditii si
verificari: `automation/README_daily_research.md`. Reincarca HTML dupa actualizare.

Eticheta de interfata ceruta de utilizator este `Succes Rate AR 20%`, in loc
de `Stres`, inclusiv in export. Textul se imparte pe randuri pentru a incapea.
Calculul ramane scenariul nefavorabil initial (intensitate goluri -20%,
cornere +20%); explicatia apare la hover. Nu este rata de succes observata.

Procedura zilnica si schema rezultatelor sunt descrise in
`Analysis/Selection_History/DAILY_WORKFLOW.md`. Procentele prospective si
estimarea salvata la alegere nu se suprascriu la aflarea rezultatului.
Un conflict de rezultat intre jurnal si dovezi opreste construirea.

Capturi reale din browser: `preview-desktop.png`, `preview-mobile.png`,
`preview-history.png`, `preview-validation.png`. Verificarile functionale
si dimensiunile desktop/mobil sunt in `verification.json`.
Pentru repetarea lor pe acest calculator, cu Microsoft Edge instalat:

```powershell
node .\Previews\Pred_Research_Terminal\check-preview.mjs
```

Teste offline ale arhivei (detectarea zilelor noi/NO BET, rezultate separate de
predictii, zero distinct de missing, alegeri si conflicte):

```powershell
node --test .\Previews\Pred_Research_Terminal\data.test.mjs
```
