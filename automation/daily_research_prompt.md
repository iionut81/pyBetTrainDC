# PRED — agent zilnic, 08:00 Europe/Bucharest

Aceasta este executia zilnica autorizata explicit de utilizator la 3 noiembrie
2026. Data de lucru calculata in Romania: {{TODAY}}. Ziua precedenta: {{YESTERDAY}}.
Lucreaza direct in proiectul Pred indicat de directorul curent. Nu cere cote,
lista de meciuri, indicii, confirmarea inceperii sau un nou prompt.

Citeste AGENTS.md, CLAUDE.md, Prompts/Football_Goal70_Corners1H_Master.md si
Analysis/Selection_History/DAILY_WORKFLOW.md. Mesajul explicit actual autorizeaza
aceasta rulare programata si cercetarea AMBELOR piete. Nu autorizeaza pariuri,
mize, cote de bookmaker, notificari externe, commit/push, schimbarea modelelor
de productie sau a selection_engine/. Nu porni agenti paraleli.

Pentru continuitate citeste si Analysis/SESSION_RESUME.md, apoi verifica datele
canonice mai noi, care au prioritate fata de rezumat. La final actualizeaza
punctul de reluare si jurnalul cu cereri, decizii, rezultate, limite si pasul
urmator. Sincronizeaza mesajele disponibile cu
`node automation/export_project_conversation.mjs`; exportul ramane local si
nu include rationamente, instructiuni ascunse, unelte sau credentiale.

Ordine obligatorie, clarificata de utilizator la 4 noiembrie, pentru rularile
din 5 noiembrie 2026 inainte: rezultate -> analiza succeselor/esecurilor ->
lectii salvate -> abia apoi scanarea, researchul si pickurile zilei curente.
Nu incepe inventarul sau selectia zilei curente in paralel cu review-ul.

## Pasul 1: verifica ziua precedenta si restantele

- Verifica data/ora reala in Europe/Bucharest. Citeste listele prospective,
  rapoartele si user_selections.csv pentru {{YESTERDAY}}, apoi orice alegere
  mai veche inca fara rezultat. Daca ieri nu exista analiza, documenteaza lipsa;
  nu inventa predictii retrospective.
- Identifica exact meciul/furnizorul, statusul final si sursele publice. Pentru
  GOAL_BY_70 foloseste primul gol valid cu minut afisat <=70, inclusiv 45+.
  Pentru CORNERS_1H_UNDER_7_5 foloseste maximum 7 cornere cumulate FirstHalf,
  inclusiv 45+. FT sau scorul pauzei nu substituie observatiile cerute.
- Un click explicit pe Alege acest pick in dashboard este tot alegere a
  utilizatorului: citeste CSV-ul canonic actualizat de serviciul local si
  selection_events.jsonl, daca exista. Pastreaza alegerile din dashboard si
  chat, inclusiv mai multe alegeri explicite; nu recrea o alegere anulata din
  candidatul analistului sau un snapshot vechi. Nu deduce pariuri combinate.
  La modificari respecta user_selections.csv.lock, reciteste continutul dupa
  conflicte si foloseste scriere atomica; nu suprascrie o alegere noua dintr-o
  copie veche. Serviciul local poate primi alegeri in timpul researchului.
  Prioritizeaza alegerile EXPLICITE ale utilizatorului, apoi verifica si listele
  complete cercetate. Pentru fiecare rezultat salveaza observatia exacta,
  success/fail, sursa si timestampul verificarii in Review_YYYY-MM-DD si in
  campurile de rezultat ale user_selections.csv. O alegere comunicata nu
  inseamna un pariu plasat. Nu modifica probabilitatile, rangul sau timestampul
  predictiei. Nu completa rezultat pe baza unei presupuneri.
- Meciurile amanate/anulate sau fara observatie verificabila raman distincte
  de success/fail si se exclud din rata. Lipsa nu este zero. Pastreaza conflictele.
  Reincearca rezultatele restante la urmatoarele rulări; nu opri tot fluxul
  deoarece un singur rezultat nu se poate verifica.
- Reconstruieste dashboardul dupa acest pas. "Alegerea ta" primeste -Succes
  sau -Fail doar din rezultatele verificate; altfel -In asteptare/-Amanat/-Anulat.

## Pasul 2: analiza si lectiile, inaintea oricarei scanari noi

- Reciteste analizele PROSPECTIVE originale si lectiile relevante din review-urile
  anterioare. Compara pentru ambele piete lista completa, candidatii analistului
  si, separat, alegerile explicite ale utilizatorului. Include succesele,
  fiecare fail verificat si reusitele aproape de prag, fara selectie convenabila.
- Pentru fiecare fail arata ce se stia inainte de meci: model brut, AR20%,
  frecvente si esantioane, rang original, competitie si riscuri documentate.
  Pentru succese arata argumentele compatibile cu rezultatul, riscurile care
  au ramas si ce NU poate fi validat dintr-o reusita. Nu atribui cauze tactice
  numai din scor, primul gol sau numarul cornerelor; marcheaza cauza necunoscuta.
- Scrie Analysis/Review_{{YESTERDAY}}/REVIEW.md cu bilant, comparatia ipotezelor
  cu observatiile, erori de date/informatii omise, limite si surse. Distinge
  constatarile demonstrate de ipoteze si de variatia posibila. Un rezultat
  izolat nu demonstreaza nici cauza, nici calibrarea modelului.
- INAINTE sa incepi scanarea zilei curente, salveaza
  Analysis/Review_{{YESTERDAY}}/LESSONS_FOR_NEXT_SCAN.md. Include timestamp RO,
  sursele prospective si de rezultat, lectiile relevante acumulate, verificarea
  concreta ceruta pentru candidatii noi, statutul constatare/ipoteza/lipsa date
  si ce ramane de testat. Prioritati: model versus frecventa istorica si AR20%,
  intern versus Europa/cupa, GF70 propriu pe teren si primele goluri tardive/0-0,
  dispersia cornerelor, exact7/>=8, schimbari de antrenor si loturi actuale.
- Daca lipsesc rezultate sau ieri nu exista analiza, salveaza explicit review-ul
  partial/not_available si lectiile verificabile deja disponibile. Nu inventa
  explicatii si nu astepta nelimitat un rezultat indisponibil. Predarea scrisa
  a constatarilor si limitelor trebuie totusi sa preceada scanarea noua.
- Lectiile imbunatatesc verificarea si argumentarea. Nu crea automat praguri,
  excluderi, procente, ponderi sau calibrari noi dupa o singura zi. Pastreaza
  snapshoturile si modelele de productie; schimbarile de model necesita evaluare
  separata si autorizare. Nu promite eliminarea esecurilor.

## Pasul 3: research complet pentru ziua curenta si pickuri

- Citeste mai intai LESSONS_FOR_NEXT_SCAN.md salvat la pasul2. In rapoartele
  zilei curente include sectiunea "Lectii aplicate din review": pentru fiecare
  lectie relevanta, verificarea efectuata, dovada noua si daca este aplicabila,
  neaplicabila sau neverificata. Explica efectul asupra preferintei pentru
  finalisti fara a rescrie clasamentul brut ori a aplica procente arbitrare.
- Cerceteaza autonom pe internet programul zilei {{TODAY}} in primele ligi
  masculine ale primelor 20 asociatii UEFA (precizeaza anul clasamentului),
  plus Romania, Serbia si diviziile secunde europene urmarite. Include separat
  competitiile europene/cupele relevante deja urmarite; marcheaza proiectiile
  intre competitii. Nu confunda vechea lista de 20 ligi cu top 20 asociatii UEFA.
- Inventariaza toate partidele si ligile fara meciuri, excluderile, lipsurile
  si statusul acoperirii. Un raspuns gol/403 nu dovedeste ca nu exista meciuri.
  Calculeaza fiecare partida eligibila inainte de clasament. Nu opri cercetarea
  la primul candidat acceptabil si nu numi o cautare partiala top global.
- Aplica EXACT metoda prospectiva salvata pentru GOAL_BY_70 si, separat,
  CORNERS_1H_UNDER_7_5: numaratori pe sezon si acasa/deplasare, lambda 50/50,
  model brut Poisson si scenariu nefavorabil intensitate -20% la goluri,
  +20% la cornere. Distinge frecventele istorice, screeningul, estimarea finala
  si contextul. Nu folosi cote sau procente de tipsteri. Nu inventa date xG.
- Verifica in profunzime finalistele: context/miza, tactica/antrenori, loturi,
  adversari, program, oboseala, vreme relevanta si surse contradictorii.
  Pastreaza seriile cu primul gol dupa 70/0-0 si totalurile 1H exact 7 sau >=8.
  Arata riscurile si de ce sunt preferate fata de alternative. Nu forta 85-90%.
- Publica separat candidatul analistului pentru fiecare piata, daca datele
  sustin alegerea. NO BET este permis. Utilizatorul va comunica alegerea sa;
  NU seta user_selected si NU adauga in user_selections.csv o selectie noua
  pe care nu a declarat-o. Nu combina evenimente pentru o cota tinta.
- Alege doar meciuri neincepute si salveaza timestampul verificarii kickoffului
  RO. Daca rularea a inceput tarziu, exclude live/terminate din selectie si
  noteaza motivul; nu produce retrospectiv predictii pentru ele.
- Salveaza datele, sursele, acoperirea, calculele, finalistele si limitele in
  Analysis/Goal70_{{TODAY}}/ si Analysis/Corners1H_{{TODAY}}/. Arhiveaza listele
  canonice in Analysis/Selection_History/{{TODAY}}_goal70_candidates.csv si
  {{TODAY}}_corners1h_candidates.csv, dupa schema existenta. Scrie jurnalul
  {{TODAY}}.md. Nu suprascrie un snapshot prospectiv deja publicat: pastreaza
  reviziile cu timestamp, diferentele de sursa si motivul actualizarii.

## Pasul 4: dashboard, verificare si predare

- Ruleaza node Previews/Pred_Research_Terminal/build.mjs. Verifica faptul ca
  datele de ieri si azi sunt navigabile, alegerile reale au etichetele corecte,
  lipsurile nu sunt zero si rezultatele de azi nu sunt fabricate.
- Daca lipsesc surse suficiente, salveaza raportul partial, excluderile si
  NO BET, apoi actualizeaza dashboardul cu limitele reale. Nu marca un research
  partial ca integral sau ca finalizat complet.
- Scrie Analysis/Automation/run_report_{{TODAY}}.json cu cheile:
  date, review_status, goal_research_status, corner_research_status,
  dashboard_status, missing_data, review_path, goal_path, corner_path.
  Adauga lessons_path, review_handoff_at_ro si research_started_at_ro pentru
  trasabilitatea ordinii: predarea lectiilor precede inceputul scanarii noi.
  Statusuri: complete, partial, blocked, not_available. Include sursele in
  rapoartele analitice. Nu confunda incheierea procesului cu cercetare completa.
- Raspuns final in romana: rezultatele alegerilor de ieri, linkul review-ului,
  lectiile si verificarile aplicate azi, apoi candidatii
  de azi pentru cele doua piete, limitele si fisierele actualizate. Nu afirma
  profit sau pret disponibil. Include intotdeauna linkul Markdown absolut
  catre Previews/Pred_Research_Terminal/dist/index.html, cu tinta intre < >
  daca include spatii. Runnerul va publica rezumatul si linkul in conversatia
  utilizatorului, conform cererii explicite ulterioare din 3 noiembrie. Nu trimite
  emailuri/mesaje externe si nu schimba programarea. Alegerile utilizatorului
  sunt fixate primele in lista pietei respective, apoi restul dupa model;
  pastreaza rangurile si probabilitatile prospective originale.
- Preferinta browser confirmata la 3 noiembrie: daca exista
  Previews/Pred_Research_Terminal/chrome-launcher.json cu installed=true,
  ofera mai intai [Deschide dashboardul in Chrome](pred-dashboard://open).
  Pastreaza si linkul HTML absolut ca alternativa pentru editor sau daca
  aplicatia de chat nu permite scheme personalizate. Nu modifica browserul
  implicit si nu reinstala handlerul la fiecare analiza.
- Formatarea linkurilor, clarificata de utilizator: nu adauga '/' in fata
  unei tinte. Foloseste exact formatul cerut C:/Users/... pentru HTML,
  fara slash initial si fara file://. Nu transforma calea in C:\Users\... .
  Pastreaza schema pred-dashboard://open exacta, fara '/' inaintea ei.
