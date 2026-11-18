# Review 16 noiembrie 2026

Predare 2026-11-17T09:03:35.630+02:00, partial. Au fost citite canonicul actual si evenimentele dashboardului. Alegeri explicite: Grecia U21-Islanda U21 Gol70 FAIL, primul72; Albacete-Celta Fortuna cornere SUCCES, FirstHalf4 (1+3). Rezultatele au fost actualizate atomic sub lock; estimarile originale, rangurile si momentele au ramas intacte. [Cronologia Grecia](https://www.skysports.com/football/greece-u21-vs-iceland-u21/7598040611329237232); [LaLiga: cornere15,16,30,39 si autogol10](https://www.laliga.com/es-GB/partido/temporada-2026-2027-laliga-hypermotion-albacete-bp-rc-celta-fortuna-14). Penaltiurile de departajare/prelungirile sunt excluse.

Gol70: {"success":25,"fail":2,"pending":24,"void":1,"postponed_not_scored":0}; cornere: {"success":21,"fail":0,"pending":30,"void":1,"postponed_not_scored":0}. Modele Gol70:1/2; cornere:1/1. Nu confundam inventarul cu pickurile.

## Greseala de selectie care necesita review

Grecia era preferinta analistului si ulterior alegerea explicita a utilizatorului:91,60% brut,86,22% stres. Primul gol72, egalarea90+1. Doua minute peste prag inseamna FAIL integral. Nu mutam pragul si nu numim rezultatul succes aproape.

Avertismente disponibile INAINTE: turul primul89; retur dupa1-1 fara obligatia unui gol rapid; frecventa18/21=85,71% fata de91,60% model, ultimele10oficiale8/10 pe fiecare echipa; Letonia-Grecia83 si Elvetia-Islanda0-0; diagnostic fara diferenteFT>=4 scadea la88,54%; XI actuale neconfirmate, absente Monastirlis/Allen/Smirlis. Ele erau enumerate in [contextul prospectiv](../Goal70_2026-11-16/CONTEXT.md), dar nu au schimbat preferinta. Diferenta bruta de numai1,27pp fata de Albacete nu reprezenta o superioritate validata. Critica procesului: am lasat rangul Poisson si atacul propriu pe teren sa domine semnalele binare si contextul dublei. Aceasta este o justificare insuficienta pentru o preferinta ferma, nu dovada ca Albacete trebuia sigur sa castige.

Mecanism statistic cunoscut: Poisson foloseste numarul tuturor golurilor<=70. Partidele cu multe goluri maresc lambda; interesul utilizatorului este timpul PRIMULUI gol. Eterogenitatea adversarilor si baraj versus grupa nu sunt reprezentate explicit. Acest mecanism poate explica supraestimarea, dar contributia sa cauzala la acest meci nu este demonstrata. Nu afirmam ca absentele sau prudenta au CAUZAT golul72. Cronologia nu furnizeaza xG/ocazii complete in0-70 si nu permite contrafactualul cu alta echipa.

Albacete alternativaGol70:90,33/84,58%,25/26general,11/12teren,10/10oficiale fiecare;GF70 propriu1,00/0,67 si absente atac. Autogol10 a adus succes; un autogol este valid pentru piata, nu dovada atacului propriu superior. La cornere89,59/79,16%,25/26 si11/12,4exact7 si variantaCelta7,50>4,50; rezultatul4 nu elimina riscul prospectiv.

## Succese, esecuri, limita

### Gol70

|Meci|Rezultat|Observatie|p original|Sursa|
|---|---|---|---|---|
|Greece U21-Iceland U21|fail|72|0.9160280720674825|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6416836)|
|Albacete-Celta Fortuna|success|10|0.9033383421675388|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5868536)|
|Panama-Curacao|success|45|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6391960)|
|Germany-Netherlands|success|41|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181835)|
|Greece-Serbia|success|69|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181836)|
|North Macedonia-Slovenia|success|29|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181786)|
|Switzerland-Scotland|success|22|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181787)|
|Faroe Islands-Moldova|success|25|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181943)|
|Kazakhstan-Slovakia|success|29|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181944)|
|Estonia-Bulgaria|success|8|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181959)|
|Iceland-Luxembourg|success|25|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181960)|
|Lithuania-Liechtenstein|success|16|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181885)|
|Gibraltar-Malta|success|19|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181879)|
|Andorra-Azerbaijan|success|2|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6140732)|
|Zimbabwe-Equatorial Guinea|success|4|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040266)|
|Namibia-Cameroon|success|12|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040339)|
|Sudan-Senegal|success|4|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040403)|
|Somalia-Ghana|success|36|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040242)|
|Eritrea-Guinea|success|28|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040253)|
|Central African Republic-Benin|fail|90+3|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040279)|
|Guyana-Puerto Rico|success|14|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6375160)|
|Guadeloupe-Bermuda|success|7|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6375164)|
|Cuba-Grenada|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6375167)|
|Montenegro U21-Bosnia and Herzegovina U21|void||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6629454)|
|Kosovo U21-Luxembourg U21|success|6|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6652383)|
|Italy U21-Belgium U21|success|25|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6652384)|
|Hungary U21-Cyprus U21|success|44|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6652385)|
|England U21-France U21|success|6|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6558200)|
|Poland U18-Czech Republic U18|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_1)|
|Switzerland U18-England U18|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_2)|
|Chile U20-Mexico U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_3)|
|Czech Republic U20-Germany U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_4)|
|Italy U20-Switzerland U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_5)|
|Portugal U20-England U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_6)|
|Netherlands U18-Sweden U18|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_7)|
|Uzbekistan U20-Venezuela U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_8)|
|Kyrgyzstan U23-Russia U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_9)|
|Indonesia U20-USA U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_10)|
|Egypt U20-Tanzania U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_11)|
|Japan U19-Colombia U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_12)|
|Ivory Coast U20-Morocco U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_13)|
|Algeria U23-Eswatini U23|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_14)|
|Tunisia U17-Morocco U17|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_15)|
|Algeria U17-Egypt U17|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_16)|
|Spain U17 Women-Switzerland U17 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_17)|
|Portugal U17 Women-Belgium U17 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_18)|
|Panama U19 Women-Guatemala U19 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_19)|
|Costa Rica U19 Women-Honduras U19 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_20)|
|El Salvador U19 Women-Nicaragua U19 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_21)|
|Bulgaria U19-Latvia U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_23)|
|Lithuania U19-Sweden U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_24)|
|Palestine U17-India U17|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_25)|

### Cornere

|Meci|Rezultat|Observatie|p original|Sursa|
|---|---|---|---|---|
|Albacete-Celta Fortuna|success|4|0.8958553249660637|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5868536)|
|Panama-Curacao|success|6|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6391960)|
|Germany-Netherlands|success|1|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181835)|
|Greece-Serbia|success|7|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181836)|
|North Macedonia-Slovenia|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181786)|
|Switzerland-Scotland|success|6|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181787)|
|Faroe Islands-Moldova|success|4|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181943)|
|Kazakhstan-Slovakia|success|0|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181944)|
|Estonia-Bulgaria|success|2|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181959)|
|Iceland-Luxembourg|success|4|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181960)|
|Lithuania-Liechtenstein|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181885)|
|Gibraltar-Malta|success|0|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=5181879)|
|Andorra-Azerbaijan|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6140732)|
|Zimbabwe-Equatorial Guinea|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040266)|
|Namibia-Cameroon|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040339)|
|Sudan-Senegal|success|7|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040403)|
|Somalia-Ghana|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040242)|
|Eritrea-Guinea|success|4|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040253)|
|Central African Republic-Benin|success|2|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6040279)|
|Guyana-Puerto Rico|success|6|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6375160)|
|Guadeloupe-Bermuda|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6375164)|
|Cuba-Grenada|success|5|fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6375167)|
|Greece U21-Iceland U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6416836)|
|Montenegro U21-Bosnia and Herzegovina U21|void||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6629454)|
|Kosovo U21-Luxembourg U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6652383)|
|Italy U21-Belgium U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6652384)|
|Hungary U21-Cyprus U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6652385)|
|England U21-France U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=6558200)|
|Poland U18-Czech Republic U18|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_1)|
|Switzerland U18-England U18|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_2)|
|Chile U20-Mexico U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_3)|
|Czech Republic U20-Germany U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_4)|
|Italy U20-Switzerland U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_5)|
|Portugal U20-England U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_6)|
|Netherlands U18-Sweden U18|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_7)|
|Uzbekistan U20-Venezuela U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_8)|
|Kyrgyzstan U23-Russia U21|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_9)|
|Indonesia U20-USA U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_10)|
|Egypt U20-Tanzania U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_11)|
|Japan U19-Colombia U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_12)|
|Ivory Coast U20-Morocco U20|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_13)|
|Algeria U23-Eswatini U23|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_14)|
|Tunisia U17-Morocco U17|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_15)|
|Algeria U17-Egypt U17|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_16)|
|Spain U17 Women-Switzerland U17 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_17)|
|Portugal U17 Women-Belgium U17 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_18)|
|Panama U19 Women-Guatemala U19 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_19)|
|Costa Rica U19 Women-Honduras U19 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_20)|
|El Salvador U19 Women-Nicaragua U19 Women|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_21)|
|Bulgaria U19-Latvia U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_23)|
|Lithuania U19-Sweden U19|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_24)|
|Palestine U17-India U17|pending||fara model|[sursa](https://www.fotmob.com/api/data/matchDetails?matchId=supp_20261116_25)|

Restante: 47identitati reinterogate; 42randuri cu cel putin o observatie, ceea ce NU inseamna rezultate noi ale pietei restante. Suplimentele fara identificare verificabila raman pending. Nu atribuim zero lipsurilor.
