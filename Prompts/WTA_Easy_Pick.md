PASUL 1 – MODEL SIMPLIFICAT (fără overfitting)

Tu probabil faci deja regresii. Problema e că:
👉 sunt prea multe variabile → zgomot → blocaj

Reducem la 3 indicatori cheie (atât):

1. Hold % (serviciu)
   Player A ≥ 78%
   Player B ≥ 78%

👉 dacă unul e slab pe serviciu → OUT

2. First Set Avg Games

(din Tennis Abstract)

medie ≥ 9.2

👉 îți arată tendința reală de over

3. Elo Difference (sau ranking proxy)
   diferență mică → meci echilibrat
   ideal: sub ~50 Elo diferență

👉 evită meciurile dezechilibrate

🔹 PASUL 2 – FILTRU FINAL (foarte important)

Din meciurile care trec mai sus:

👉 ELIMINI:

jucători cunoscuți cu start lent
zgură + jucători defensivi extremi
reveniri după accidentări

👉 Aici intervine „ochiul tău”, dar limitat

🔹 PASUL 3 – SCOR SIMPLU (fără AI complicat)

Fiecare meci primește scor:

Hold % ambii → +1
Avg games > 9.5 → +1
Elo apropiat → +1

👉 alegi doar scor 2 sau 3

🔹 PASUL 4 – SELECȚIA FINALĂ

👉 MAXIM 1–2 pariuri / zi

Cotă ideală:

1.35 – 1.55
🔹 PASUL 5 – TRACKING CORECT (aici se face diferența)

În Excel:

Data
Meci
Hold A / Hold B
Avg games
Elo diff
Cotă
Rezultat