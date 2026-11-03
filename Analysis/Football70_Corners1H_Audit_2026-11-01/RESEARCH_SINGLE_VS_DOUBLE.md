# O selectie sau doua: comparatie pentru Gol70 si cornere 1H

Data cercetarii: 2026-11-01. Comparatia priveste bilete la care TOATE
selectiile trebuie sa reuseasca, nu pariuri individuale separate.
Nu au fost consultate cote de bookmaker pentru calcule.

## Rezultatul recomandarii

Pentru obiectivul de a maximiza frecventa zilelor castigatoare, propunerea
este maximum o selectie individuala pe zi, cu posibilitatea de NO BET.
Nu se adauga o selectie doar pentru a urca produsul cotelor la 1,10.
Aceasta este o propunere de protocol; nu o automatizare instalata si nici
o garantie de reusita zilnica sau de profitabilitate.

Momentan exista backtest local pentru cornere, nu pentru Gol70. Nu exista
o comparatie empirica completa intre doua goluri, gol+cornere si doua cornere.

## Comparatia efectiv masurata pe aceleasi zile

Din cele 413 zile cu Top 1 au fost pastrate cele 323 cu minimum doua selectii
eligibile. Prima si a doua selectie provin din acelasi clasament istoric
Poisson, anterior zilei analizate. Nu este nevoie de ipoteza independentei:
s-au numarat direct rezultatele simultane ale celor doua meciuri.

| Strategie | Zile | Reusite | Rata reusitei | Rata esecului |
|---|---:|---:|---:|---:|
| Prima selectie singura | 323 | 287 | 88,85% | 11,15% |
| Primele doua cornere, combinate | 323 | 262 | 81,11% | 18,89% |

Adaugarea celei de-a doua selectii a transformat 25 de zile castigatoare
pentru prima selectie in zile pierdute pentru dubla. Diferenta este de
7,74 puncte procentuale. A doua selectie separat a reusit in 296/323 cazuri
(91,64%); aceasta observatie retrospectiva nu autorizeaza inversarea automata
a clasamentului pentru meciurile viitoare.

Date: `single_vs_double_corners_same_days.csv`.
Verificari trecute: 323 zile, 287 rezultate individuale castigatoare,
262 rezultate comune castigatoare; rezultatul dublu nu poate depasi
niciunul dintre cele doua rezultate individuale pentru aceeasi zi.

Limite: univers istoric partial si invechit, fara filtrul tactic manual,
fara preturi reale, fara masurarea ROI. Nu se compara 323 duble cu toate cele
413 single, fiindca zilele ar fi diferite. Nu este dovada ca o piata este
universal mai sigura decat alta.

## Matematica celor trei combinatii

Pentru doua evenimente independente, P(A si B)=P(A)*P(B).
[OpenStax: Independent Events](https://openstax.org/books/introductory-statistics/pages/3-2-independent-and-mutually-exclusive-events)

Exemplu STRICT IPOTETIC: p_gol=95%, p_cornere=90%, aceleasi valori pentru
fiecare componenta, evenimente independente. Aceste valori NU sunt
probabilitati validate ale modelelor noastre.

| Bilet | Probabilitate ipotetica |
|---|---:|
| Doua goluri pana la 70 | 0,95*0,95 = 90,25% |
| Gol pana la 70 + sub 7,5 cornere | 0,95*0,90 = 85,50% |
| Doua sub 7,5 cornere | 0,90*0,90 = 81,00% |

Daca ambele piete au probabilitatea 90%, toate cele trei duble devin 81%
sub ipoteza independentei. Diferenta vine din calitatea componentelor,
nu din eticheta gol/cornere. Doua selectii de 95% pot depasi un single
diferit de 90%, dar fiecare dintre acele selectii de 95% singura ramane
mai probabila decat dubla lor.

Fara independenta, P(A si B)=P(A)*P(B|A). Pentru gol si cornere in acelasi
meci trebuie masurata probabilitatea comuna. Daca fiecare marginala este
90%, limitele matematice ale probabilitatii comune sunt 80-90%; produsul
81% este valabil numai in ipoteza independentei. Ambele conditii simultan
nu pot avea probabilitate mai mare decat oricare conditie individuala.

Golul poate veni dupa pauza si sa nu influenteze deloc cornerele primei
reprize; un gol timpuriu poate modifica ritmul. Nu s-a demonstrat aici
semnul sau marimea dependentei dintre evenimente.

## Cercetare sportiva relevanta

Studiul Yip et al. descrie sosirea cornerelor in grupuri si foloseste un
model Poisson compus. Acesta sustine testarea unor distributii mai flexibile
decat Poisson simplu, nu o probabilitate concreta pentru selectiile noastre.
Lucrarea foloseste si informatii de piata; acele cote nu au fost preluate
in calculele acestui proiect.
[Lucrarea originala](https://arxiv.org/abs/2112.13001)

Literatura despre goluri trateaza explicit timpul pana la eveniment si
efectele contextului/formatiilor. Pentru Gol70 este necesar un model si un
test propriu al primului gol. Nu se importa rata de succes a cornerelor.
[Studiu despre analiza duratei pana la gol](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1323930/full)

Sursele publice de protectie a consumatorului descriu si riscul suplimentar
al biletelor multiple. Aceste surse nu sunt folosite ca predictori ai meciurilor.
[GambleAware NSW](https://www.gambleaware.nsw.gov.au/learn-about-gambling/how-gambling-works/sports-and-race-betting)
[VGCCC](https://www.vgccc.vic.gov.au/for-community/how-gambling-works/understanding-sports-betting)

## Cota 1,10 nu este profit zilnic de 10%

La cota 1,10, castigul brut in cazul reusitei este 10% din miza. Pragul de
echilibru inainte de costuri este 1/1,10 = 90,91% probabilitate reala.
Doua componente independente cu probabilitati egale trebuie sa aiba fiecare
cel putin sqrt(1/1,10) = 95,35% pentru ca o dubla la cota totala 1,10 sa fie
doar la echilibru. Nu este suficient ca ambele sa para de aproximativ 90%.

Cu ratele observate in cele 323 zile, cotele matematice de echilibru ar fi
1,1254 pentru single si 1,2328 pentru dubla. Sunt calcule retrospective,
nu preturi disponibile si nici dovezi de valoare pentru meciurile viitoare.

## Protocolul propus

1. Completeaza istoricul Gol70 si actualizeaza datele cornerelor. Auditeaza
   separat acoperirea pe ligi, competitia, sezonul si identitatea meciurilor.
2. Pastreaza predictiile cronologice si calibrate numai pe trecut. Pentru
   cornere compara Poisson cu modele de variatie mai flexibile; pentru
   gol compara baseline-ul cu un model dedicat primului gol. Reguli fixe
   inaintea evaluarii, fara ajustari arbitrare ca sa se atinga 90-95%.
3. Cerceteaza intregul program al ligilor cu date eligibile si verifica
   finalistii: lot, stil, miza, odihna si conditii. Maximum o selectie;
   NO BET cand sustinerea este insuficienta.
4. Raporteaza probabilitatea, limitele, scenariul nefavorabil si cota
   matematica de echilibru. Cerinta unei cote disponibile >=1,10 nu poate
   fi confirmata fara o observatie reala de pret; nu se cere utilizatorului
   sa o furnizeze si nu se blocheaza analiza statistica din acest motiv.
5. Evalueaza prospectiv toate strategiile pe aceleasi zile, folosind
   probabilitati si reguli salvate inainte de rezultate. Pentru combinatia
   din acelasi meci, pastreaza si modeleaza rezultatul comun.
6. Pana la demonstrarea unui avantaj, etapa este de validare fara bani
   reali. O miza virtuala de 2 lei dintr-o banca de test de 100 lei poate
   servi ca limita operationala de expunere, nu ca proportie optima validata.
   Nu se aplica all-in sau progresie de recuperare a pierderilor.
