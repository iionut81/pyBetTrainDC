# Protocol de lucru: cum îmbunătățim alegerea meciului

17 noiembrie 2026. Cerere explicită: lucrăm la diferența dintre procent mare și alegere bună. Acest protocol organizează cercetarea offline; nu schimbă producția, selection_engine, probabilitățile originale, alegerile sau sarcina zilnică.

## Obiectiv și prioritate

Obiectivul de selecție este alegerea unui singur eveniment individual mai bun din aceeași listă zilnică eligibilă, cu posibilitatea NO BET. Nu fixăm o țintă artificială de 90% și nu confundăm rata de succes cu profitul.

Prioritate propusă: cornere. Gol70 rămâne comparator/cercetare separată; nu transferăm performanța sau regulile între piețe. Nu schimbăm automat cercetarea zilnică pe ambele piețe.

## Ce este deja pregătit

Auditul canonic reproduce dashboardul. DAILY_SELECTION_BASELINE.csv și daily_selection_baseline.json fixează pentru fiecare zi lista eligibilă numeric, primul loc brut, rezultatul lui, top3 și rata aceleiași zile. Dacă primul loc este pending, nu îl înlocuim cu locul2. Zilele cu rezultate incomplete sunt marcate. Departajarea egalităților este fixă după ID.

Aceste rezultate sunt diagnostic/dezvoltare, deoarece le-am văzut deja. Nu sunt validare independentă a unei reguli inventate astăzi.

## Trei variante de comparat

A. Referință: metoda actuală, locul1 după probabilitatea Poisson brută. Formulele, minimele5/3 și definițiile rămân reperul fix.

B. Estimare de număr regularizată: ratele fiecărei echipe sunt trase spre o bază de ligă/ediție învățată exclusiv din trecut. Pentru cornere se compară Poisson și binomial negativ, cu dispersia estimată pe o populație suficientă, nu liber din3meciuri. Incertitudinea parametrilor intră în predicție. Efectul urmărit: o medie extrem de mică obținută din puține meciuri nu devine automat locul1. Ajustarea pentru adversari/teren trebuie predefinită și verificată; nu promitem că oricare variantă va câștiga.

C. Comparator empiric regularizat al evenimentului: succesul definit direct (FirstHalf<=7 sau primul gol<=70), cu priorul și greutatea lui învățate din trecut. Identități deduplicate și tratament explicit al suprapunerii dintre istoric general și teren. Este comparator experimental, nu media ad-hoc a procentelor echipelor. Dacă datele sunt insuficiente pentru o formulare validă, rămâne neimplementat, fără procente fabricate.

O calibrare globală monotonă și AR20 sunt controale de probabilitate/sensibilitate; nu sunt variante noi de clasament când păstrează aceeași ordine. Nu le prezentăm drept soluții pentru mai multe selecții câștigătoare.

## Separarea celor două întrebări

1. Care candidat are cea mai mare probabilitate estimată a evenimentului? Pentru maximum de succese așteptate la o selecție, regula este maximul probabilității predictive. Problema actuală este estimarea acelei probabilități.
2. Este dovada suficient de solidă pentru recomandare? Aici intră lipsa datelor, incertitudinea și NO BET. Un criteriu conservator de selecție are alt obiectiv decât maximizarea probabilității medii și trebuie testat separat; nu îl presupunem superior.

Nu scădem arbitrar puncte pentru baraj, ploaie sau un fail anterior. Nu folosim turul89 drept veto universal construit după Grecia. Confirmarea unui XI sau o lipsă esențială poate schimba statutul datelor; efectul sportiv asupra probabilității se estimează numai dacă există bază validată.

## Fișa concretă a finalistelor

Pentru fiecare dintre primele3 eligibile se salvează înaintea startului:

- piață, identitate, ediție/categorie, kickoff și status;
- probabilitate brută, predicție experimentală distinctă, intervalul/diagnosticul de incertitudine dacă este valid;
- n pe fiecare echipă/teren, observații lipsă, deduplicare;
- cornere: numărul>=8, exact7, medie, dispersie, obținute/cedate și sensibilitate sezon/teren;
- gol: zero/>70, GF70 propriu, adversari și concentrarea golurilor în scoruri ample;
- context cunoscut: grupă/baraj, agregat, lot/XI, rotație, odihnă, teren neutru;
- motiv precis pentru preferință sau NO BET și ce informație ar schimba decizia.

Verificarea contextuală minimă trebuie să fie aceeași pentru cele trei finaliste. Altfel comparăm un meci studiat cu două meciuri despre care știm mai puțin. Nu condiționăm eligibilitatea de existența exact a zece oficiale dacă protocolul permite5/3; lipsurile de context rămân vizibile.

## Testarea care poate schimba sistemul

1. Reconstruim istoric complet înaintea fiecărei zile: ligi/ediții izolate, aceleași minime, minute exacte sau FirstHalf, fără rezultate din ziua prezisă. Dacă disponibilitatea istorică la momentul predicției nu poate fi dovedită, testul este etichetat reconstrucție retrospectivă.
2. Folosim datele deja inspectate pentru dezvoltare, cu evaluare cronologică internă. Ponderile, priorul, dispersia, predictorii, regulile de selecție și departajare se fixează înaintea testului final.
3. Comparăm A/B/C pe aceeași listă și aceleași zile. Salvăm alegerile lor înaintea rezultatelor. Pentru zilele cu NO BET raportăm și acoperirea, nu ascundem zilele excluse.
4. Măsura principală: diferența de reușită a selecțiilor zilnice. Pentru aceleași zile, raportăm explicit Bcâștigă/Apierde, Acâștigă/Bpierde și restul. Nu comparăm doar două procente din universuri diferite.
5. Separat: Brier/log-loss, calibrare pe benzi, performanță pe ligă/categorie, performanța topului față de univers și față de o alegere uniformă din aceeași zi. Intervalele trebuie să respecte dependența temporală; eșantionul mic nu permite certitudine.
6. Pentru varianta cu abstinență: comparăm zile comune și rata de participare; măsurăm separat relația risc–acoperire. Un NO BET în plus nu este o victorie de selecție.
7. Fixăm dinainte durata/volumul evaluării și regula statistică de acceptare, în funcție de efectul minim practic urmărit și variabilitatea din dezvoltare. Nu oprim testul la prima serie bună;30zile reprezintă pilot operațional, nu validare garantată.
8. Promovarea cere dovadă pe un segment neatins și avantaj practic fără ascunderea acoperirii sau a riscului. Rezultatul „neconcludent” este acceptabil. Orice modificare a producției se prezintă separat, cu schimbările și dovezile concrete.

## Pasul tehnic următor

Pentru cornere, verificăm întâi dacă istoricul brut al auditului (10.477partide utilizabile la1nov) poate reconstrui exact413zilele top1 fără lipsuri/contaminare. Apoi construim variantaB ca experiment izolat, comparat cuA. Cele293rezultate prospective actuale nu sunt suficiente singure pentru a selecta multe modele și filtre. ComparatorulC vine după ceB și baza de evaluare sunt reproductibile.

Nu am început antrenare/recalibrare de producție și nu am schimbat programarea. Acest fișier, baza zilnică și auditul constituie rezultatul concret pentru review al pasului curent.
