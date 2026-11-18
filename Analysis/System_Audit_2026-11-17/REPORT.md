# Audit Gol70 și Cornere 1H — 17 noiembrie 2026

Cerere: evaluare directă a calculelor, predicției și selecției; explicația diferenței observat–estimat și alegerea priorității de continuare.

**Verdict: aș prioritiza cornerele pentru dezvoltare și validare. Gol70 l-aș păstra în cercetare, fără recomandări ferme bazate pe procentul actual. Niciunul nu este demonstrat ca sistem profitabil.** Aceasta este o propunere pentru review; nu schimbă automat cercetarea zilnică pe ambele piețe.

## Cifrele exacte din dashboard

Am reprodus proiecția din `evaluation.js` folosind atât arhiva reconstruită din canonice, cât și snapshotul dashboardului. Cele șase combinații piață/cohortă coincid exact. Se compară media estimată și rata observată pe ACELEAȘI rezultate verificate, cu probabilitate prospectivă existentă.

| Cohortă | Gol70 | Cornere 1H |
|---|---:|---:|
| Modele verificate | 359 | 293 |
| Succese | 307 | 261 |
| Rată observată | 85,52% | 89,08% |
| Medie estimată | 87,39% | 89,26% |
| Observat minus estimat | −1,8767 pp | −0,1864 pp |
| Eșecuri efective | 52 | 32 |
| Eșecuri așteptate conform estimărilor | 45,26 | 31,45 |

La cornere, cifra actuală se rotunjește la **−0,19 pp**. Nu pot identifica exact revizia care afișa 0,18 fără snapshotul ei. Diferențele sunt puncte procentuale, nu procente relative și nici eroarea medie a unui meci.

Gol70 a produs aproximativ 6,74 eșecuri în plus față de totalul estimat; cornerele, 0,55. O singură observație schimbă rata cu 0,279 pp, respectiv 0,341 pp. Acest eșantion nu dovedește că diferența de calibrare dintre piețe este structurală. Partidele au echipe/zile comune, iar rezultatele lipsă pot introduce bias.

**Corecție a auditului meu preliminar:** acesta număra doar 248 de cornere și mai puține propuneri ale analistului. Citea numai unele fișiere/câmpuri. Dashboardul unește și listele de candidați, finalistele, alternativele și aliasurile de identitate. Referința acestui audit este acum cohorta exactă a dashboardului, cu 293 de cornere.

## De ce media aproape perfectă la cornere poate induce în eroare

| Banda probabilității la cornere | N | Estimat mediu | Observat |
|---|---:|---:|---:|
| Sub 80% | 24 | 76,61% | 87,50% |
| 80–85% | 35 | 83,25% | 88,57% |
| 85–90% | 86 | 87,84% | 83,72% |
| 90–95% | 105 | 92,45% | 93,33% |
| Cel puțin 95% | 43 | 96,29% | 90,70% |

Erorile de sens opus se compensează. **Un gap agregat mic nu înseamnă probabilități corecte pentru fiecare meci și nu validează clasamentul.** Aceste benzi sunt diagnostice, cu eșantioane limitate; nu devin filtre de selecție după observarea rezultatelor.

Problema apare și în auditul istoric al aceleiași familii de modele Poisson:

| Cohortă istorică de cornere | N | Observat | Estimat |
|---|---:|---:|---:|
| Toate eligibile | 4.563 | 89,37% | 89,42% |
| Primul loc zilnic | 413 | 88,86% | 94,27% |

Aici este avertismentul principal: agregatul arată bine, dar alegerea automată a celui mai mare procent nu a îmbunătățit rata observată și a supraestimat puternic siguranța. Testul istoric este exploratoriu și separat de cohorta prospectivă din noiembrie.

## Formula și limitele ei

Ambele sisteme estimează o intensitate lambda din golurile/cornerele obținute și cedate, combinând 50% sezon și 50% teren.

- Gol70: `P(N70 >= 1) = 1 - exp(-lambda)`.
- Cornere: `P(C1H <= 7) = PoissonCDF(7, lambda)`.

**Formulele sunt corecte dacă distribuția și intensitatea sunt corecte.** Folosirea tuturor golurilor nu este, singură, o eroare matematică. Problema este că media golurilor nu determină probabilitatea de zero fără ipoteze despre distribuție.

Exemplu didactic: 10 meciuri, dintre care opt cu câte trei goluri și două fără gol, au media 2,4 și o frecvență de gol de 80%. Alte zece, șase cu două goluri și patru cu trei, au aceeași medie și frecvență 100%. Poisson dă 90,93% în ambele situații. La cornere există aceeași problemă: media nu descrie singură coada de minimum opt cornere.

În meciuri reale, adversarii diferiți, golurile numeroase după deschidere, schimbarea comportamentului la 1–0 și contextul grupă/baraj pot afecta distribuția. Sunt mecanisme plauzibile, nu explicații cauzale demonstrate pentru fiecare eșec. Variația deterministă a ritmului în timp nu invalidează singură un Poisson cu intensitate integrată corectă.

## Puncte bune și slabe, pe componente

| Componentă | Punct bun | Punct slab | Ce aș face |
|---|---|---|---|
| Gol70 Poisson | Simplu, verificabil, interval exact | Sensibil la scoruri ample și adversari; validare insuficientă pentru zero goluri și top zilnic | Comparator; test separat al unui model direct al primului gol |
| Cornere Poisson | Prag direct, date istorice mai ample | Impune egalitatea mediei cu varianța condițională; poate subestima coada >=8 | Prioritate de cercetare; comparație cu NB și empiric regularizat |
| Ponderi 50/50 | Combină sezonul și terenul | Euristice; eșantioanele se suprapun | Păstrat ca reper; testate ponderi/regularizare numai pe trecut |
| Minime 5/3 | Elimină unele estimări fără bază | Nu demonstrează precizie; trei meciuri sunt foarte puține | Măsurarea incertitudinii și regularizare |
| AR20 | Arată sensibilitatea la intensitate | ±20% nu este interval de încredere sau probabilitate minimă garantată | Diagnostic, fără etichetă de garanție |
| Frecvență istorică | Măsoară direct evenimentul | 10/10 nu înseamnă probabilitate reală 100%; adversari diferiți | Comparator regularizat, evaluat cronologic |
| Context manual | Identifică retur, absențe, neutralitate, rotație | Subiectiv; enumerarea riscurilor nu demonstrează că selecția le folosește bine | Decizii prospective verificabile, comparate cu modelul singur |
| Clasament după probabilitatea brută | Transparent | Favorizează și estimările extreme produse de zgomot | Nu l-aș transforma automat în recomandare |

Dacă jumătate din sezon este deja jucată pe terenul relevant, combinația 50/50 echivalează aproximativ cu 75% pondere totală pentru acel teren și 25% pentru cel opus. Nu sunt două confirmări independente.

Există și codul legacy `train_corners_1h.py`, cu binomial negativ și calibrare Platt. **Acesta nu produce procentele Poisson discutate aici.** NB poate reprezenta variație mai mare, dar nu repară automat medii sau date greșite. În codul legacy, calibrarea este ajustată și evaluată pe aceleași predicții walk-forward; rezultatul calibrat nu este o validare independentă a calibrării. Nu am executat sau modificat acel model.

## Unde s-a greșit în selecție

La Grecia U21–Islanda U21, avertismentele erau prospective: turul deschis la 89, retur după 1–1, frecvență 18/21 și ultime oficiale 8/10 pentru fiecare echipă. Diferența brută de 1,27 pp față de Albacete nu demonstra o superioritate reală. **Am justificat prea ferm preferința, lăsând rangul brut să domine avertismentele.** Nu pot demonstra că o absență sau o tactică anume a cauzat primul gol la 72.

Propunerile analistului din dashboard: Gol70 11/13, 84,62%, față de 93,14% estimat; cornere 14/14, față de 95,31%. Cohorta cornerelor include și alternative. Eșantioanele sunt prea mici pentru rate validate. Nu confundăm propunerile cu primul loc brut sau cu alegerile utilizatorului.

Brier este 0,12284 la gol și 0,09792 la cornere. Comparatorul constant care folosește retrospectiv rata aceleiași cohorte are 0,12387 și 0,09729. Nu este un comparator prospectiv, dar arată că nu avem încă dovadă puternică de diferențiere utilă a partidelor. Brier măsoară mai mult decât calibrarea; valorile absolute dintre două evenimente cu rate de bază diferite nu stabilesc singure piața mai bună.

## Continuarea propusă

1. Cornere: comparație cronologică între Poisson, NB și empiric regularizat; evaluare pe toate meciurile, top zilnic, benzi și acoperire. Nu presupunem că NB câștigă.
2. Gol70: păstrat în cercetare; model binar penalizat sau survival pentru primul gol, cu Poisson drept comparator. Nu construim un model complex pe doar 359 de rezultate.
3. Învățarea ponderilor, dispersiei, calibrării și a regulilor NO BET numai pe trecut; specificații fixate înaintea unui segment final neatins. Rezultatele deja văzute din 2–16 noiembrie sunt diagnostic/dezvoltare.
4. Repararea raportării cohortelor, păstrarea lipsurilor, identităților și probabilităților originale. O recalibrare monotonă poate corecta procentele, dar păstrează ordinea și nu schimbă succesele acelorași selecții.

**Aș continua cu o singură prioritate: cornere. Aș păstra golurile ca experiment separat, nu ca al doilea sistem de recomandări cu aceeași încredere. Motivul este baza de validare mai solidă, nu gapul de 0,19 pp.** Nu am aplicat această schimbare în modele sau în programarea zilnică.

Dovezi: [metrici](metrics.json), [cohorta exactă](cohort.json), [script de reproducere](calculate.mjs), [audit istoric](../Football70_Corners1H_Audit_2026-11-01/FAILURE_ANALYSIS.md).

Metodologie: [scikit-learn — calibrare și Brier](https://scikit-learn.org/stable/modules/calibration), [Cornell — lipsă de potrivire versus dispersie](https://cscu.cornell.edu/wp-content/uploads/poisfit.pdf), [SAS — varianța binomialului negativ](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_genmod_details01.htm).
