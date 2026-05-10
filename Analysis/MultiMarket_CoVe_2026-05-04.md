# Multi-Market Accumulator CoVe
**Date:** 2026-05-04 | **Template:** 1.0.4MultiMarket.md v1.1
**Selecție:** Top 3 picks ale zilei (user-specified)

---

## STEP 1 — CHECKLISTS RAPIDE

| # | Meci | Piață | Score individual | Pass? |
|---|------|-------|-----------------|-------|
| 1 | Roma vs Fiorentina (I1) | Goals U4.5 | **10/10** | ✅ |
| 2 | Cremonese vs Lazio (I1) | Goals U4.5 | **9/10** | ✅ |
| 3 | Chelsea vs Nottm Forest (E0) | Corners Over 6.5 | **10/10** | ✅ |

Toate 3 scor 8+ → calificate pentru acumulator per regulă.
Meciuri diferite → nicio combinație pe același meci ✅.

---

## STEP 2 — DATE CHEIE PER LEG

### LEG 1: AS Roma vs Fiorentina — Goals U4.5

| Metric | Value |
|--------|-------|
| p_cal (model) | 88.8% |
| p_research (CoVe) | **~91%** |
| Fair odds model | 1.1265 |
| Fair odds research | **~1.10** |
| Combined goal avg | **2.35** 🔥 PREMIUM |
| λ_total | 2.43 |
| Liga | I1 = under-friendly ✅ |

**Motiv cheie:** Dovbyk OUT (golgheter Roma) + Kean OUT (golgheter Fiorentina) = ambele atacuri principale absente structural. Fiorentina unbeaten 7 dar fără forță atacantă. Roma pentru Europa, joc serios dar fără arme.

**Cum pierd:** Rezerve surprinzătoare + gol devreme deschide meciul → 3-2.

---

### LEG 2: Cremonese vs Lazio — Goals U4.5

| Metric | Value |
|--------|-------|
| p_cal (model) | 90.1% |
| p_research (CoVe) | **~91%** |
| Fair odds model | 1.1102 |
| Fair odds research | **~1.10** |
| Combined goal avg | **2.16** 🔥 PREMIUM |
| λ_total | **1.92** (cel mai mic din model) |
| Liga | I1 = under-friendly ✅ |

**Motiv cheie:** Cremonese NU A MARCAT în ultimele 6 reprize (3 meciuri). λ=1.92 — raritate în model. Lazio câștigă controlat 1-0 sau 2-0.

**Cum pierd:** Cremonese disperată (relegare) marchează haotic în ultimele 20 minute → 3-2 spre final.

---

### LEG 3: Chelsea vs Nottingham Forest — Corners Over 6.5

| Metric | Value |
|--------|-------|
| p_research (CoVe O6.5) | **~90%** |
| Fair odds research | **~1.11** |
| Chelsea FOR/g | **6.15** |
| Forest FOR/g | **5.38** |
| Total expected corners | **10.31** |
| Liga | E0 = cel mai bun pentru Over 6.5 ✅ |
| Blowout_score model | **2** ✅ |

**Motiv cheie:** Ambele echipe cu >5 FOR/game (profilul Anderlecht-Brugge, 16 cornere ieri). Chelsea pierdut 5 consecutive — disperată acasă. Forest 6 neînvins — atacă și ei. E0 avg 10.30 cornere.

**Cum pierd:** Forest parkează autobuzul + Chelsea nu convertește în centrări → sub 7 cornere total. Probabilitate ~10%.

---

## STEP 3 — SELF-VERIFICATION

- [x] Toate 3 legs din meciuri diferite ✅
- [x] Toate 3 scor 8+ ✅
- [x] Nicio combinație inversă pe același meci ✅
- [x] Legs 1+2 în aceeași ligă (I1) → **corelație parțială** — notat explicit
- [x] Leg 3 în piață diferită (Corners Over vs Goals Under) ✅
- [x] Research probability ≥ 82% pentru toate 3 ✅
- [x] Cap +10pp respectat în analizele individuale ✅
- [x] Surse citate în analizele individuale ✅

---

## STEP 4 — CORELAȚIE ȘI RISC

### ⚠️ Atenție: Legs 1 + 2 în Serie A aceeași zi

Roma-Fiorentina și Cremonese-Lazio sunt amândouă Serie A, Matchday 35, aceeași zi.

**Riscul de corelație:** Dacă există o condiție sistematică care afectează toate meciurile Serie A azi (ex: teren umezit, schimbare de arbitri, căldură extremă în mai multe stadii), ambele legs ar putea fi afectate simultan.

**De ce riscul este mic:**
- Cele 2 meciuri se joacă în stadii diferite (Olimpico Roma vs Cremona)
- Motivele structurale sunt independente: Dovbyk/Kean absent ≠ goal drought Cremonese
- Statisticile sunt confirmate din surse separate
- Nu există factor sistemic identificat pentru Serie A azi

**Concluzie:** Corelație de ligă acceptabilă în acumulatoare. Risc real dar manageable.

---

## STEP 5 — CALCUL COMBINATOR

### Fair Odds Acumulator

| Leg | p_research | Fair odds |
|-----|-----------|-----------|
| Roma-Fiorentina U4.5 | 91% | 1.099 |
| Cremonese-Lazio U4.5 | 91% | 1.099 |
| Chelsea-Forest Over 6.5 | 90% | 1.111 |
| **COMBINATE** | **~74.6%** | **~1.340** |

**Probabilitate combinată:** 0.91 × 0.91 × 0.90 = **0.745 (74.5%)**

**Fair odds acumulator: ~1.34**

### Estimare odds piață

Piețele nu sunt încă known (offered_odds=missing), estimare rezonabilă:

| Leg | Estimare odds piață |
|-----|-------------------|
| Roma-Fiorentina U4.5 | ~1.30–1.40 (injuries cunoscute = odds mai mari) |
| Cremonese-Lazio U4.5 | ~1.25–1.35 |
| Chelsea-Forest Over 6.5 | ~1.10–1.15 |
| **COMBINATE estimat** | **~1.79–2.15** |

**EV estimat:** Market 1.79-2.15 vs Fair 1.34 → **pozitiv**, cu marjă bună.

⚠️ **Verifică odds reale înainte de plasare.** Dacă Over 6.5 Chelsea-Forest < 1.08 → înlocuiește cu Everton-Man City Over 6.5 (backup) sau elimină leg 3 și joacă primele 2 în sistem simplu.

---

## STEP 6 — ACUMULATOR FINAL

| Leg | Meci | Piață | Score | Fair odds |
|-----|------|-------|-------|-----------|
| 1 | **AS Roma vs Fiorentina** | Goals U4.5 | 10/10 | 1.10 |
| 2 | **Cremonese vs Lazio** | Goals U4.5 | 9/10 | 1.10 |
| 3 | **Chelsea vs Nottm Forest** | Corners Over 6.5 | 10/10 | 1.11 |
| **COMBO** | | | | **1.34 fair** |

### Probabilitate și risc

| Metric | Value |
|--------|-------|
| Probabilitate combinată | **74.5%** |
| Fair odds | **~1.34** |
| Odds piață estimat | **~1.80–2.10** |
| Bankroll recomandat | **2–3% stake** (3 legs = risc mai mare) |

### Cum pierde acumulatorul (scenariu real):
Fiorentina sau un rezervist al Romei marchează în minutul 15, meciul se deschide, se termină 3-2. Celelalte 2 legs câștigă — dar acumulatorul pierdut. Aceasta este realitatea: oricare din cele 3 meciuri poate eșua independent.

---

## ROI CONTEXT (din template MultiMarket v1.1)

| Piață | ROI/bet |
|-------|---------|
| U4.5 Goals la 1.10-1.15 | break-even (valid DOAR în acumulatoare) |
| Corners Over 6.5 | nelistat (market nou — bazat pe 2 WINs ieri: Anderlecht 16 cornere, Freiburg-Wolfsburg 8 cornere) |

**Concluzie ROI:** Acumulatorul transformă 3 legs break-even / nouă piață într-un combined bet cu EV pozitiv față de fair odds. Exact pentru asta sunt accumulatoarele cu picks high-probability.

---

## BACKUP OPȚIONAL

Dacă oricare leg are odds < 1.08 sau context se schimbă înainte de meci:

| Backup | Piată | p_research | Score |
|--------|-------|-----------|-------|
| Everton vs Man City | Corners Over 6.5 | ~88% | 8.5/10 |
| Slobozia vs Hermannstadt | Goals U4.5 | ~89% | 8/10 |
| Rapid vs CFR Cluj | Goals U4.5 | ~88% | 8/10 |

---

*Analysis: 2026-05-04 | Template MultiMarket v1.1*
