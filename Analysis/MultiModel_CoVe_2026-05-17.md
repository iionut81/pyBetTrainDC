# CoVe Multi-Model — Top 5 Mix
## DC + Goals + Corners + WTA (Flashscore slug fix activ)
## Data: 2026-05-17 | MD37 Serie A + Strasbourg WTA 500 R1

---

## SITUAȚIE MODELE AZI

| Model | Fixtures | Recomandate | Note |
|-------|----------|-------------|------|
| DC | 79 | **0** | — |
| Goals | 79 | 45 (mismatch<0.6) | I1, SP1, SP2, F1, E0, RO1 |
| Corners | 79 | 48 (λ≤10.5) | I1 cluster TRUST MODEL |
| WTA | 14 | 6 U12.5 + 1 O7.5 | Strasbourg R1 + Rabat Q |

**Fix aplicat permanent:** STRASBOURG + RABAT în Flashscore slugs → 5 meciuri noi găsite.

---

## EXCLUSE AUTOMAT

| Pick | Motiv |
|------|-------|
| Elche/Getafe Goals | λ=1.949 dar Elche 17th MUST-WIN → asimetric, risc deschidere joc |
| Oviedo/Alaves Goals | Oviedo 20th RETROGRADAT (29pts) → dead rubber total → CHAOS |
| Cagliari/Torino Goals | Cagliari 17th must-win → joc deschis → risc |
| Roma/Lazio Corners | Derby + Roma must-win CL → emoție ridicată → cornere pot spike |
| Keys/Parry U12.5 | Step F.5: FINAL + min_hold=0.657 ≥ 0.65 → HARD PASS |
| Maya Joint/Ann Li U12.5 | both > 0.55 (min_hold=0.566) → Step F.1 -2pp → 6/10 |

---

## GOALS CoVe

### G1 — Genoa vs AC Milan U4.5 (I1, MD37) ⭐ DOUBLE SIGNAL

**Model:** λ_total=2.208 | mismatch=0.409 | p_cal=**89.3%**

**Context:**
- Genoa: **15th (41pts)** → SAFE, fără miză → apărare compactă
- AC Milan: **5th (67pts)** → luptă pentru CL (egal cu Roma!) → MUST WIN dar disciplinat
- Milan atacă, Genoa apără → meci controlat, nu haotic
- I1 = under-friendly liga (~2.55 avg goals) → +1pp
- Genoa safe + Milan presă → nu se deschide jocul aleatoriu

**Step 4C — forma recentă:** Milan vine după formă bună, Genoa stabilă defensiv (safe)
**Dead rubber:** Genoa SAFE → potențial apatie, DAR Milan atacă ordonat, nu dezordonat → BTTS puțin probabil

**Cum pierd:** Milan marchează rapid 2-0, Genoa răspunde cu 2 goluri pe contraatac. Probabilitate: ~11%

**Verdict: ✅ BET — 8/10** | p_research: ~90% | Fair odds: ~1.11 | BET ≥ 1.09

---

## CORNERS CoVe

### C1 — Como vs Parma U12.5 (I1, MD37) ⭐ TOP PICK

**Model:** λ=7.392 | p_cal=**90.7%** | Fair odds: 1.102

**Step 0:** λ=7.39 → **< 9 → ✅ TRUST MODEL** (cel mai mic din batch I1!)

**Context:**
- Como: **6th (65pts)** → urmăresc Europa → joacă disciplinat Fabregas INVERTED style
- Parma: **13th (42pts)** → safe, fără miză → nu presează
- **Como INVERTED style**: 61% posesie, atac din interior → PUȚIN cornere proprii (confirmat în backtest nostru anterior)
- Parma safe → nu generează presing agresiv → puțin cornere

**C2 Style:** Como INVERTED → +5pp boost U12.5 (din template Corners)

**Cum pierd:** Como atacă disperat (scor 0-0 la pauză), generează 8+ cornere + Parma contraatac 4+. Probabilitate: ~9%

**Verdict: ✅ BET — 9/10 MODERATE** | p_research: ~93% | Fair odds: ~1.08 | BET ≥ 1.06

---

### C2 — Genoa vs AC Milan U12.5 (I1) ⭐ DOUBLE SIGNAL cu G1

**Model:** λ=7.610 | p_cal=**89.98%** | Fair odds: 1.111

**Step 0:** λ=7.61 → **< 9 → ✅ TRUST MODEL**

**Context:** Milan atacă ordonat (disciplinat pentru CL), Genoa apără compact (safe) → puțin pressing disorganizat → puțin cornere.
- Milan: atacant dar de calitate → construiește prin centru, nu prin flanc → mai puțin cornere decât echipele directe
- Genoa: safe → nu deschide jocul → puțin cornere proprii

**Cum pierd:** Milan domină total și generează 9+ cornere prin atacuri repetate pe flancuri. Probabilitate: ~10%

**Verdict: ✅ BET — 8/10 MODERATE** | p_research: ~90% | Fair odds: ~1.11 | BET ≥ 1.09

---

## WTA CoVe

### W1 — Leolia Jeanjean vs Leylah Fernandez U12.5 (Strasbourg WTA 500, R1) ⭐ ELITE

**Model:** p_hold_a=0.390 | p_hold_b=?? | min_hold=**0.390** | gap=0.411 | tb_p_raw=0.000 | blowout=9

**Step A:** min_hold=0.390 → 🔥🔥 **ELITE** (< 0.42, cel mai mic din batch)
**Step C:** Strasbourg WTA 500 Clay → ✅ **GOOD** (fără restricție)
**Step F:** tb_p_raw=0.000 → ✅ perfect, niciun risc tiebreak

**Research:**
- Fernandez: **7th seed Strasbourg**, favorită clară (72% win probability)
- Jeanjean: wildcard, outside top 100, n-a trecut de R2 în niciun turneu WTA 2026
- Fernandez just lost Paris 125 vs Blinkova → motivată să câștige la Strasbourg
- Jeanjean: hold=0.390 → se rupe la aproape fiecare serviciu

**Gap=0.411 → p_markov ≈ 0.91 (Fernandez 91% favorită)** → seturi scurte garantate

**Cum pierd:** Jeanjean intră în meciul vieții, ține servici și setul merge la 6-5/TB. Probabilitate: ~10%

**Verdict: ✅ BET — 9/10 MODERATE** ⭐ | p_research: ~93% | Fair odds: ~1.08 | BET ≥ 1.06

---

### W2 — Maria Sakkari vs Peyton Stearns O7.5 Set 1 (Strasbourg WTA 500, R1)

**Model:** p_hold_a=0.674 | p_hold_b=~0.763 | min_hold=0.674 | gap=0.089 | tb_p_raw=0.094 | blowout=**3**

**Step B Market:** O7.5 → Step A: Hold < 0.45 either player? NO (both > 0.67) ✅
**Step B:** Both > 0.60 → 🔥 Premium signal for O7.5
**Step C:** WTA 500 Clay → ✅ Downgrade (per template) → ⚠️ -1 context

**Research:**
- Sakkari: top 15 WTA, excelentă la serve-hold
- Stearns: WTA 20s, hold rate ridicat
- Gap=0.089 → meci extrem de echilibrat → competitive = BINE pentru O7.5
- blowout=3 → risc blowout minimal → setul va fi competitiv → O7.5 confirmat

**Cum pierd:** Sakkari domină complet (6-2 sau mai mic). Probabilitate: ~20%

**Verdict: ✅ BET — 7/10 MODERATE** | p_research: ~83% | Fair odds: ~1.20 | BET ≥ 1.18

---

## TABEL FINAL — TOP 5 MIX

| # | Pick | Model | p_research | Score | Conf. | Acțiune | Fair Odds |
|---|------|-------|-----------|-------|-------|---------|-----------|
| **1** | **Jeanjean / Fernandez — U12.5 Set 1** ⭐ | WTA Strasbourg R1 | ~93% | **9/10** | MODERATE | ✅ **BET** | ~1.08 |
| **2** | **Como / Parma — Cornere U12.5** ⭐ | I1 MD37 | ~93% | **9/10** | MODERATE | ✅ **BET** | ~1.08 |
| **3** | **Genoa / AC Milan — U4.5** ⭐ | I1 MD37 | ~90% | **8/10** | MODERATE | ✅ **BET** | ~1.11 |
| **4** | **Genoa / AC Milan — Cornere U12.5** ⭐ | I1 MD37 | ~90% | **8/10** | MODERATE | ✅ **BET** | ~1.11 |
| **5** | **Sakkari / Stearns — O7.5 Set 1** | WTA Strasbourg R1 | ~83% | **7/10** | MODERATE | ✅ **BET** | ~1.20 |

---

## BONUS: picks solide excluse din top 5

| Pick | p_cal | Score | Note |
|------|-------|-------|------|
| Roma/Lazio Cornere U12.5 | 90.2% | 7/10 | Derby → risc spike cornere |
| Cagliari/Torino Cornere U12.5 | 87.4% | 7/10 | TRUST MODEL, dar Cagliari must-win |
| Kessler/Selekhmeteva U12.5 | — | 7/10 | Strasbourg Q, Premium hold |
| Putintseva/Oliynykova U12.5 | — | 7/10 | Strasbourg Q, Putintseva în formă |

---

## PARLAY PROPUS (3 legs)

**Jeanjean U12.5** + **Como/Parma Cornere** + **Genoa/Milan U4.5**
- WTA + Corners + Goals → diversitate maximă

---

## SURSE

- [Roma vs Lazio preview — SportsMole](https://www.sportsmole.co.uk/football/roma/rome-derby/preview/roma-vs-lazio-prediction-team-news-lineups_597621.html)
- [Serie A standings — WorldFootball](https://www.worldfootball.net/competition/co111/italy-serie-a/results-and-standings/)
- [Elche/Oviedo context — Futbol24](https://www.futbol24.com/national/Spain/Primera-Division/2025-2026)
- [Jeanjean vs Fernandez — Dimers](https://www.dimers.com/news/leolia-jeanjean-vs-leylah-fernandez-tennis-prediction-wta-strasbourg-open-2026-ac)
- [Strasbourg WTA draw — Tennis365](https://www.tennis365.com/wta-tour/wta-strasbourg-draw-victoria-mboko-emma-raducanu-alex-eala-find-out-paths)
