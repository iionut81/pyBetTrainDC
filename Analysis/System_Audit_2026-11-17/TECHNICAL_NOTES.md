# Audit concret al sistemelor Gol70 si Cornere1H — 17 noiembrie 2026

Audit solicitat explicit de utilizator: puncte bune/slabe, verdict direct, explicatia diferentelor1,88/0,18. Nu modifica productia, formulele, probabilitatile, alegerile sau programarea zilnica pe ambele piete. Recomandarea de prioritizare de mai jos este pentru review, nu schimbare automata a domeniului.

## Verdict

As prioritiza CORNERE pentru dezvoltare si validare; GOAL70 ramane cercetare in umbra, fara recomandare ferma bazata pe procentul actual. Nu as pune niciuna in exploatare cu bani ca sistem demonstrat. Motiv:cornere are un audit cronologic mult mai larg si o problema de distributie/varianța masurabila; golul are mai putina validare independenta si modelul mediei nu descrie suficient probabilitatea de zero. Nu aleg cornere doar fiindca gapul agregat e mai mic. Nu recomand automat binomial negativ inaintea comparatiei pe viitor.

## Reproducerea exacta a dashboardului

Sursa:loadArchive(data.mjs), aceeasi proiectie ca evaluation.js. Recalcularea din canonice si din snapshot.json a produs egalitate exacta pentru toate sase combinatiile piata/cohorta. Include numai probabilitati existente si rezultate success/fail pe aceleasi randuri. Fara probabilitati pentru nemodelate, fara pending/void in numitor.

|Piata/cohorta|Succes/N|Observat|Estimat|Observat minus estimat|Brier|
|---|---|---|---|---|---|
|goal/models|307/359|85.52%|87.39%|-1.877 pp|0.122837|
|goal/analyst|11/13|84.62%|93.14%|-8.525 pp|0.136610|
|goal/user|10/11|90.91%|89.88%|1.033 pp|0.086978|
|corner/models|261/293|89.08%|89.26%|-0.186 pp|0.097918|
|corner/analyst|14/14|100.00%|95.31%|4.691 pp|0.002693|
|corner/user|11/11|100.00%|95.11%|4.895 pp|0.002952|

Gol:52fail fata de45,26asteptate,6,74in plus. Cornere:32 fata de31,45,0,55in plus. Corneregapexact-0,1864187pp se rotunjeste-0,19pp, nu-0,18; cifra utilizatorului poate proveni din alta revizie/rotunjire. Nu pretindem identificarea acelei revizii fara snapshotul ei. O singura observatie muta rata cu0,279pp la gol si0,341pp la cornere. Diferenta mica nu demonstreaza cauzalitate/calibrare; sub ipoteza independenta SDnumarsuccese6,23gol siaproximativ5,21cornere, iar partidele reale impart echipe/zile. Intervalele Wilson i.i.d. descriptive sunt inmetrics.json; nu reprezinta intervale valide pentru o cohorta dependenta.

Auditul preliminar dinReview16 avea248cornere si flaguri analist11/10. Era incomplet: citea numai CSV-urileoutcomes si campulp_raw, in timp ce constructorul uneste si candidate/all_models/prospective_selection/alternative si analyst_finalist. Dashboardul corect include293cornere,13/14propuneri analist. Valorile utilizatorului sunt din cohorta modele, nu alegeri sau analisti. Cohorta analistcornere include si alternative, nu exact o singura recomandare zilnica.

## Erori care se compenseaza

|Piata/banda bruta|N|Estimat mediu|Observat|Gap|
|---|---|---|---|---|
|goal/0.00%–80%|22|76.33%|81.82%|5.491 pp|
|goal/80.00%–85%|76|82.83%|80.26%|-2.565 pp|
|goal/85.00%–90%|146|87.54%|85.62%|-1.927 pp|
|goal/90.00%–95%|110|92.19%|90.00%|-2.191 pp|
|goal/95.00%–100%|5|95.47%|80.00%|-15.471 pp|
|corner/0.00%–80%|24|76.61%|87.50%|10.892 pp|
|corner/80.00%–85%|35|83.25%|88.57%|5.319 pp|
|corner/85.00%–90%|86|87.84%|83.72%|-4.123 pp|
|corner/90.00%–95%|105|92.45%|93.33%|0.884 pp|
|corner/95.00%–100%|43|96.29%|90.70%|-5.593 pp|

La cornere grupele<85%subestimeaza reusita,85–90%supraestimeaza,90–95%subestimeaza usor,95+supraestimeaza. Grupele sunt diagnostice post-hoc si mici; NU filtre validate. Cohorte diferite:c​ornere293,gol359; nu comparatie controlata intre aceleasi evenimente. Dupa eticheta istorica history_basis,252cornere same_league au88,10% fata de89,55%;40projection_cross_competition au95% fata de87,57%;1projection_to_cup e succes. Aceste diferente se compenseaza; etichetele istorice necesita audit separat si nu sunt clasificare noua curatata.

## Ce calculeaza sistemul

Ambele piete folosesc lambda_sezon=(rataGFgazda+rataGAoaspeti+rataGFoaspeti+rataGAgazda)/2, analog lambda_teren, apoi lambda=0,5sezon+0,5teren. Gol70foloseste toate golurile valide<=70: P(N70>=1)=1-exp(-lambda). Cornere foloseste CDFPoisson(7,lambda1H). GF/GAla gol nu este scorHT/FT; cornerele sunt explicitFirstHalf<=7,inclusiv45+. Formulele sunt matematic corecte SUB ipoteza distributiei; faptul ca se folosesc toate golurile nu este singur o eroare matematica. Eroarea posibila este presupunerea ca media identificata determina corect probabilitatea de zero, in prezenta adversarilor diferiti, starii scorului si incertitudinii intensitatii. Nici variatia determinista a ritmului in timp nu invalideaza singura formula daca numarul total ramane Poisson cu intensitatea integrata corecta.

Gol:10partide cu8totaluri de3si2zero au media2,4,dar rata80%;10partide cu6totaluri de2si4de3 au aceeasi medie2,4si rata100%. Poisson ar da90,93% ambelor. Exemplu didactic, nu date reale. La cornere media4poate proveni din10partide toate4(100%under) sau2partidecu8si8cu3(80%under); acelasiPoisson94,89%. Amandoua au nevoie de distributie, nu doar medie.

## Evaluarea componentelor

|Componenta|Ce e bun|Ce e slab|Decizia propusa|
|---|---|---|---|
|Gol70Poisson|transparent,reproductibil,foloseste intervalul exact,benchmark simplu|medie sensibila la multe goluri dupa deschidere/adversari slabi;probabilitatezero subreprezentata posibil;grupa vsbaraj neexplicita;fara regularizare|pastreaza comparator;testeaza modelbinar/survival direct|
|CornerePoisson|intervalFirstHalfexact,pragdirect,dateistorice mai ample|coada>=8 poatefi subestimata,media/varianta conditionala impuse egale;topuri supraoptimiste|prioritatecercetare,comparaNBsiempiricregularizat|
|50/50sezon/teren|include contextteren faraa folosi exclusiv3meciuri|ponderi euristice,mostre suprapuse;cu50%dinsezonpeacelteren rezulta aproximativ75%ponderetotala terenului;fara ajustarevaloareadversarului|benchmarkfix,nutuning;testregularizareierarhica|
|Minime5/3|oprestelipsurile si unele estimari fragile|suntminime de eligibilitate,nudovada preciziei;topulalegeextremezgomotoase;multiplecomparatii|masoaraincertitudine,nuafisaconvingere95–99% validata|
|StresAR20|sensibilitatetransparenta,neschimbata decontext|−20%lambdaGol/+20%cornerearbitrarcaamplitudine;nueintervaldeincredere,calibrare sau probabilitatemininima|pastreazadiagnostic,eticheteazacorect|
|Frecventaistorica|masoaradirectpiata,arata0–0/tardive/exact7|samplemic,adversari/editiidiferite,10/10nue100%adevarat|comparatorempiricregularizat,testseparat|
|Contextmanual|detecteazabaraj,rotatie,absente,neutru,vreme|subiectiv,acoperireinegala;enumerareavertismentenufixeazaselectia;nuvalidatplusvaloare|documenteazadeciziileprospectiv,compara cuunmodel singur|
|Clasamentdupapbrut|simplu si verificabil|confundagradulestimarii cusigurantaei;alegeextremeeronatedinsamplemic|nuautomatiza top1 ca recomandare|

## Dovezi despre clasament si selectie

Top1brut recalculat pe zi din snapshoturile existente, fara inlocuirea unui pending cu locul2:gol11/14=78,57%fata de92,82%;cornere10/12=83,33%fata de95,65%. Sunt putine zile, diagnostice, NU toate propunerile analistului. Propuneri analistgol11/13fata de93,14%;cornere14/14fata de95,31%,incluzandalternative. Seria14/14 nu demonstreaza rata100% si nu se transfera la top1brut. Alegerile utilizatorului raman distincte.

Auditulistoriccornere1nov,aceeasi familiePoisson:4563eligibile,4078succese89,37%fata de89,42%;413top1,367succese88,86%fata de94,27%. Inbanda95–100%,804cazuri:90,55%real vs96,59%estimat. Arata ca agregatul apropiat poatecoexistacuclasament slab/supraestimareatopului. Media/varianta agregata nu demonstreaza singura supradispersie conditionala; sunt necesare reziduuri conditionale si testare. Vezi[reviewistoric](../Football70_Corners1H_Audit_2026-11-01/FAILURE_ANALYSIS.md).

Brieractualgol0,12284 fata deconstant0,12387;cornere0,09792 fata deconstant0,09729. Constantulfoloseste rataobservata din ACEEASIcohorta;esteparatorretrospectivoptimist,nubenchmarkprospectiv. Rezultat:concordantamedie nu demonstreaza ca procentele diferentiaza bine partidele. Brierinclude atatcalibrare cat sidiscriminare/dificultate;nucomparamdirectvalorileabsolutegol vs​cornerecareautintebazadiferite.

## LegacycornereNBsiPlatt

train_corners_1h.py/run_corners_1h_daily.pyexista, dar NU sunt formulele care au generat aceste procente ale dashboardului. PlusNB:poatereprezentavariantasuperioaram​ediei;minus:dispersia suplimentara este instabila insamplemic si nu repara automatmedii/identitati/adversari gresiti. CalibrareaPlatt este potrivita pentru studiuseparat;codulfitpeps/ys si raporteazacalibrat peaceleasipredictiipoateproduceevaluareoptimistaacalibrarii. Nu folosimacele cifre ca validareindependenta. ExperimentulcronologicseparatdinFAILURE_ANALYSISimbunatateste Brierulglobal,dar pastreaza ordinea;deciNUpoateimbunatatihitrateaceluiasitop1. Faraexecutie/modificarealegacy.

## Unde s-a gresit concret

1. Prezentareaprocentelebrute capreferintapreaferma fara validarecalibrarii/ordinii.GreciaU21tur89siretur72aveausemnaleprospective;diferentade1,27ppfata deAlbacetenudemonstra superioritate.
2. Interpretareagapului globalcapreciziemeci;compensarile si top1arata contrariul.
3. Confuziecohorte:inventar,modele,topbrut,analisti(inclalternative),utilizator. Auditulmeupreliminar aomis surse;corectatprinproiectiacanonica.
4. Intensitateestimata dinputinedate,fara regularizare/masurareincertitudine;50/50suprapuneesantioanele.
5. Contextgrupecupa/barajsideplasariinegalintegrat;manualnevalidat. Cauzeleprecisealefiecaruiesec ramannecunoscute.

## Plan concret de continuare

A.Pastreazadate/snapshoturi/probabilitatioriginale;cohortasicelexclus/pendingexplicit,cutoffsiidentitativerificate.
B.PeCORNERE:comparaPoissonactual,NBsiempiricregularizat;invataintensitate/dispersie/ponderi/calibrared​oarpe trecut,incluzandadversari/teren/competitie. EvaluaretoatecazurileSItop1;logloss/Brier/calibrarepebenzi,acoperiresizileNO BET.
C.PeGOL:comparabinarpenalizat ori survival pentruprimulgol,cuPoissonbenchmark;nu simplainlocuirecu9/10,faramodelecomplexepe359date. Colecteazaminutelesi0–0,adversari/competitie,goluripre/postdeschidere.
D.Fixeazaspecificatiileinainte,evaluarecronologica,fiecareziinaintearezultatelorsale,segmentfinalneatins.2–16novdejavazutsuntdezvoltare,nuvalidarefinala. Invatareguliledeveto siNO BETfara cautarepe test.
E.Nicio modificare automata;prioritatecornere e judecataengineeringbazatapedovezim​aiample,nudovadaprofitului sau ordin saoprimprogramareaGol70.

Sursemetodologiceprimare:[scikit-learncalibrare](https://scikit-learn.org/stable/modules/calibration),[Cornell: lipsadefitvsdispersie](https://cscu.cornell.edu/wp-content/uploads/poisfit.pdf),[SAS: NBvariance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_genmod_details01.htm). Artefacte[metrici](metrics.json),[cohorta](cohort.json),[script](calculate.mjs),hashuri.
