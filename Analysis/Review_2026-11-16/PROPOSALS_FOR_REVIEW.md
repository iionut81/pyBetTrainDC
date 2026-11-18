# Solutii propuse spre review -17noiembrie2026

## 1. Separarea selectiei de rangul brut

Problema demonstrata:Grecia a ramas prima preferinta pentru un avantaj brut1,27pp, desi turul89,18/21 si8/10 contraziceau fermitatea. Propunere: fisa obligatorie pro/contra si verdict recomandare/NO BET independent de rangul matematic; fara procente noi. Azi aplicam documentarea si prudenta deja autorizate; un veto automat nou necesita review si testare.

## 2. Model direct al primului gol, in studiu separat

Compara baseline50/50 cu un model binar/survival care estimeaza direct P(Tprim<=70), folosind numai date disponibile inaintea zilei. Grupeaza separat cluburi/seniori/U21/feminin si grupa/baraj cand exista date suficiente; foloseste shrinkage invatat pe trecut, nu estimari perfecte din esantioane mici. Comparator suplimentar: sensibilitate la scoruri ample, fara eliminari retrospective arbitrare.

## 3. Dispersia cornerelor

Compara Poisson cu binomial negativ/estimare empirica regularizata. Pastreaza toate exact7/>=8; invata dispersia pe train, evalueaza pe viitor. Oviedo-Girona9 anterior este exemplu de risc, Albacete4 de succes cu risc; ambele trebuie pastrate. Nu schimba distributia doar pentru a elimina un fail.

## 4. Protocol concret de acceptare

Ingheata specificatiile si cutoffs inaintea testului; evaluare cronologica expanding-window, toate predictiile zilei calculate inaintea oricarui rezultat din zi. Pastreaza un segment final neatins. Compara aceleasi cohorte verificate: Brier/log-loss/calibrare, hit rate si numar de recomandari/NO BET, top1 zilnic fata de intreg universul, breakdown liga/categorie si incertitudine cu resampling pe zile. Selectiile si pragurile se invata numai in segmentul de dezvoltare. Rezultatele2-16nov sunt deja vazute: sunt diagnostic/dezvoltare, NU test final independent. Acceptarea cere imbunatatire pe segment neatins fara a ascunde scaderea acoperirii; nicio tinta90% sau profit fara date. Criteriile numerice se fixeaza din variabilitatea istorica inaintea testului, nu dupa rezultat.

## 5. Calitatea datelor

Prioritate recuperare FirstHalf si minute exacte, identitate si reconciliere pe echipe; pentru AET separa explicit scor90 de120 si shootout. XI dinlastStarting11 nu este confirmat. Conflictele raman pending. Datele complete reduc incertitudinea de masurare, nu garanteaza reusita.

Status: propuneri pentru review; nu am modificat modele, ponderi, calibrare, filtre de productie, selection_engine sau programare.
