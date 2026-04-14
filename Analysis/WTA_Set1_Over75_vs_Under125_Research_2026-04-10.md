# Research: Over 7.5 vs Under 12.5 — Care e mai productiv?
## Data: 11,940 meciuri WTA backtest (2017-2026)

---

## 1. DISTRIBUTIA JOCURILOR IN SET 1

| Jocuri in Set 1 | Exemple | Count | % | Ce castiga? |
|-----------------|---------|-------|---|-------------|
| 6 (6-0) | Bagel | 551 | 4.6% | Under 12.5 ONLY |
| 7 (6-1) | Breadstick | 1,393 | 11.7% | Under 12.5 ONLY |
| **8 (6-2)** | | **2,116** | **17.7%** | **AMBELE** |
| **9 (6-3)** | | **2,677** | **22.4%** | **AMBELE** (cel mai frecvent!) |
| **10 (6-4)** | | **2,613** | **21.9%** | **AMBELE** |
| **12 (7-5)** | | **1,186** | **9.9%** | **AMBELE** |
| 13 (7-6 TB) | Tiebreak | 1,366 | 11.4% | Over 7.5 ONLY |

### Rezumat:

| Zona | % | Ce se intampla |
|------|---|----------------|
| **Under 7.5** (6-0, 6-1) | **16.6%** | Under 12.5 castiga, Over 7.5 pierde |
| **Sweet spot** (6-2 → 7-5) | **72.0%** | **AMBELE castiga** |
| **Tiebreak** (7-6) | **11.4%** | Over 7.5 castiga, Under 12.5 pierde |

**72% din meciuri sunt in sweet spot-ul unde ambele piete castiga simultan.**

---

## 2. HIT RATE COMPARISON

| Market | Hit Rate Global | Pierde cand? |
|--------|----------------|-------------|
| **Over 7.5** | **83.4%** | Bagel/Breadstick (16.6% din meciuri) |
| **Under 12.5** | **88.6%** | Tiebreak (11.4% din meciuri) |

**Under 12.5 are hit rate MAI MARE (+5.2pp)** pentru ca tiebreak-urile sunt mai rare decat bagel/breadstick-urile.

### Per suprafata:

| Surface | Over 7.5 | Under 12.5 | Mai bun | De ce |
|---------|----------|-----------|---------|-------|
| **Clay** | 82.8% | **89.9%** | **Under 12.5 (+7.1pp)** | Clay = mai multe break-uri = mai putine TB |
| **Hard** | 83.5% | **88.3%** | **Under 12.5 (+4.8pp)** | Hard = moderate TB rate |
| **Grass** | **87.5%** | 84.8% | **Over 7.5 (+2.7pp)** | Grass = servele domina = TB frecvent |

**KEY INSIGHT: Pe CLAY si HARD, Under 12.5 e mai bun. Pe GRASS, Over 7.5 e mai bun.**

---

## 3. CAND MODELUL E CONFIDENT

| Threshold | Over 7.5 Hit | Under 12.5 Hit | Diferenta |
|-----------|-------------|----------------|-----------|
| p ≥ 0.80 | 83.8% | **88.9%** | Under +5.1pp |
| p ≥ 0.85 | 84.2% | **89.3%** | Under +5.1pp |
| p ≥ 0.90 | 85.5% | **89.8%** | Under +4.3pp |

**La TOATE nivelurile de confidence, Under 12.5 castiga mai des.**

---

## 4. CALIBRARE (cat de precis e modelul?)

### Over 7.5:

| Band | Predicted | Actual | Gap |
|------|-----------|--------|-----|
| 0.75-0.80 | 77.8% | 76.3% | +1.5pp (bun) |
| 0.80-0.85 | 82.9% | 80.1% | +2.8pp (over-confident) |
| 0.85-0.90 | 88.2% | 82.8% | **+5.4pp (over-confident)** |
| 0.90-1.00 | 92.0% | 85.5% | **+6.5pp (over-confident)** |

### Under 12.5:

| Band | Predicted | Actual | Gap |
|------|-----------|--------|-----|
| 0.75-0.80 | 77.8% | **83.5%** | **-5.7pp (UNDER-confident!)** |
| 0.80-0.85 | 82.8% | **86.4%** | **-3.6pp (UNDER-confident!)** |
| 0.85-0.90 | 87.8% | **88.3%** | -0.5pp (aproape perfect) |
| 0.90-1.00 | 93.9% | 89.8% | +4.1pp (over-confident) |

**KEY INSIGHT:** 
- Over 7.5 e **over-confident** la toate bandele (+2 la +6pp) → modelul supraevalueaza
- Under 12.5 e **under-confident** la bandele 0.75-0.85 → modelul subevalueaza → **EDGE ASCUNS!**

Asta inseamna: cand modelul zice 80% Under 12.5, realitatea e ~85%. **Modelul ne da un avantaj pe Under 12.5 fara sa stie.**

---

## 5. PROFIL DE RISC

| | Over 7.5 | Under 12.5 |
|---|----------|-----------|
| **Castiga cand** | Set 1 are ≥8 jocuri | Set 1 nu ajunge la 6-6 |
| **Pierde cand** | Bagel (6-0) sau Breadstick (6-1) | Tiebreak (6-6) |
| **Frecventa pierdere** | **16.6%** | **11.4%** |
| **Predictibilitate pierdere** | Greu de prezis (mismatch surpriza) | Mai usor de prezis (ambele hold bine) |
| **Cel mai periculos scenariu** | Favorita domina 6-0/6-1 | Meci echilibrat merge la TB |

**Under 12.5 pierde MAI RAR (11.4% vs 16.6%)** si pierderile sunt mai predictibile (tiebreak-urile depind de hold quality care e measurabila).

---

## 6. SET 1 vs SET 2 — CARE E MAI BUN?

### Set 1 avantaje:
- **Cel mai curat semnal** — nicio influenta de la seturi anterioare
- **Zero fatigue** — ambele jucatoare proaspete
- **Zero momentum shifts** — nu exista frustrare din set pierdut
- **Modelul e antrenat pe Set 1** — predictiile sunt optimizate aici
- **Servele sunt cele mai consistente** in Set 1 (fresh muscles)

### Set 2 dezavantaje:
- **Momentum effect** — jucatoarea care a castigat Set 1 e in control
- **Mental collapse risk** — perdanta poate ceda (6-1 Set 2 dupa pierdere)
- **Fatigue** — servele scad in calitate
- **Model NOT trained on Set 2** — nu avem predictii calibrate

### Concluzie: **SET 1 e superior** pentru pariuri. Set 2 introduce noise nepredictibil (mental, fatigue, momentum).

---

## 7. RECOMANDARI FINALE

### Cand sa pariezi UNDER 12.5:

| Conditie | De ce |
|----------|-------|
| **Clay** | TB rate doar 10.1% (cel mai mic) → Under 12.5 = 89.9% hit |
| **Gap > 0.15** | Favorita break-uieste repede → set scurt |
| **Un hold < 0.50** | Jucatoare slaba pe serva → nu poate tine pana la 6-6 |
| **Favorita agresiva la return** | Break-uri rapide |
| **NU cand:** ambele hold > 0.55, hot streak, grass | TB risk prea mare |

### Cand sa pariezi OVER 7.5:

| Conditie | De ce |
|----------|-------|
| **Grass** | Servele domina → 87.5% Over, TB favorizeaza Over |
| **Ambele hold > 0.60** | Ambele tin serva → minim 8 jocuri |
| **Gap < 0.10** | Foarte echilibrat → meci lung, competitiv |
| **Blowout score < 5** | Niciuna nu domina → nu exista bagel risk |
| **NU cand:** gap > 0.18, un hold < 0.45 | Blowout risk → 6-0/6-1 |

### Strategy optima:

| Surface | Market principal | Market secundar |
|---------|-----------------|----------------|
| **Clay** | **Under 12.5** (89.9% hit, edge ascuns) | Over 7.5 doar cu meciuri echilibrate |
| **Hard** | **Under 12.5** (88.3% hit) | Over 7.5 ca backup |
| **Grass** | **Over 7.5** (87.5% hit) | Under 12.5 doar cu gap mare + hold slab |

---

## 8. BOTTOM LINE

**Under 12.5 este piata MAI PRODUCTIVA overall:**
- Hit rate mai mare: 88.6% vs 83.4% (+5.2pp)
- Pierde mai rar: 11.4% vs 16.6%
- Model under-confident la 0.75-0.85 = **edge ascuns**
- Functioneaza cel mai bine pe **Clay** (89.9% hit)

**Over 7.5 este mai bun DOAR pe Grass** si in meciuri ultra-echilibrate.

**Set 1 este superior Set 2** — zero noise de la mental/fatigue/momentum.

**Strategia optima: Under 12.5 pe Clay Set 1 ca piata principala, Over 7.5 pe Grass si meciuri echilibrate ca piata secundara.**

---

*Research generated from 11,940 WTA backtest matches (2017-2026). Data source: wta_predictions.csv.*
