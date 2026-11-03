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

## Pasul 1: verifica ziua precedenta si restantele

- Verifica data/ora reala in Europe/Bucharest. Citeste listele prospective,
  rapoartele si user_selections.csv pentru {{YESTERDAY}}, apoi orice alegere
  mai veche inca fara rezultat. Daca ieri nu exista analiza, documenteaza lipsa;
  nu inventa predictii retrospective.
- Identifica exact meciul/furnizorul, statusul final si sursele publice. Pentru
  GOAL_BY_70 foloseste primul gol valid cu minut afisat <=70, inclusiv 45+.
  Pentru CORNERS_1H_UNDER_7_5 foloseste maximum 7 cornere cumulate FirstHalf,
  inclusiv 45+. FT sau scorul pauzei nu substituie observatiile cerute.
- Prioritizeaza alegerile EXPLICITE ale utilizatorului, apoi verifica si listele
  complete cercetate. Pentru fiecare rezultat salveaza observatia exacta,
  success/fail, sursa si timestampul verificarii in Review_YYYY-MM-DD si in
  campurile de rezultat ale user_selections.csv. O alegere comunicata nu
  inseamna un pariu plasat. Nu modifica probabilitatile, rangul sau timestampul
  predictiei. Nu completa rezultat pe baza unei presupuneri.
- Meciurile amanate/anulate sau fara observatie verificabila raman distincte
  de success/fail si se exclud din rata. Lipsa nu este zero. Pastreaza conflictele.
  Reincearca rezultatele restante la urmatoarele rulări; nu opri tot fluxul
  deoarece un singur rezultat nu se poate verifica.
- Scrie REVIEW.md cu succese, esecuri, rezultate la limita si acoperire lipsa.
  Compara ce prevedea analiza cu ce s-a intamplat: ipoteza tactica sustinuta,
  variatie normala, eroare de date sau informatie omisa. Un succes nu dovedeste
  singur ca argumentul era corect, iar un fail nu justifica o ajustare arbitrara.
  Salveaza lectiile ca ipoteze verificabile prospectiv, fara recalibrare ad-hoc.
- Reconstruieste dashboardul dupa acest pas. "Alegerea ta" primeste -Succes
  sau -Fail doar din rezultatele verificate; altfel -In asteptare/-Amanat/-Anulat.

## Pasul 2: research complet pentru ziua curenta

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

## Pasul 3: dashboard, verificare si predare

- Ruleaza node Previews/Pred_Research_Terminal/build.mjs. Verifica faptul ca
  datele de ieri si azi sunt navigabile, alegerile reale au etichetele corecte,
  lipsurile nu sunt zero si rezultatele de azi nu sunt fabricate.
- Daca lipsesc surse suficiente, salveaza raportul partial, excluderile si
  NO BET, apoi actualizeaza dashboardul cu limitele reale. Nu marca un research
  partial ca integral sau ca finalizat complet.
- Scrie Analysis/Automation/run_report_{{TODAY}}.json cu cheile:
  date, review_status, goal_research_status, corner_research_status,
  dashboard_status, missing_data, review_path, goal_path, corner_path.
  Statusuri: complete, partial, blocked, not_available. Include sursele in
  rapoartele analitice. Nu confunda incheierea procesului cu cercetare completa.
- Raspuns final in romana: rezultatele alegerilor de ieri, lectiile, candidatii
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
