# CoVe Multi-Model — Top 5 Mix
## Modele: DC + Goals + Corners + WTA
## Data: 2026-05-15 | Goals v2.3 | Corners v1.6 | WTA v3.3

---

## SITUAȚIE MODELE

| Model | Fixtures | Recomandate | Note |
|-------|----------|-------------|------|
| DC | 11 | **0** | Niciun edge |
| Goals | 11 | 14 (U4.5+U3.5) | I2, RO1, SP2, TR1, TR2, SA1, B1, E0 |
| Corners | 10 | 7 | TR2, I2, SP2, TR1, RO1, B1 |
| WTA | 3 | 3 | Paris 125 QF |

---

## EXCLUSE AUTOMAT

| Pick | Motiv |
|------|-------|
| Castellon/Cadiz Goals | mismatch=1.459 → HARD PASS |
| Leuven/Antwerp Corners | λ=10.56 > 10.5 → HARD PASS |
| Leuven/Antwerp Goals | Leuven concedă 2+ goluri în **5 consecutive** → defensivă prăbușită, model subestimează |
| Aston Villa/Liverpool Goals | E0 high-scoring -2pp + ambele echipe need CL (atacă) + Watkins 12 goluri în 13 → prea mult risc ofensiv |
| Charaeva/Sasnovich U12.5 | Charaeva hot streak (bătut Korpatsch ieri 2-1) → Step F.2: hold subestimat |

---

## GOALS CoVe

### G1 — FC Arges vs Rapid Bucuresti U4.5 (RO1, Championship Round 9)

**Model:** λ_total=1.749 | mismatch=0.313 | p_cal=**88.3%**

**Step 0:** Liga RO1 → profil medium (~2.70 avg goals). λ=1.749 = cel mai mic din batch → 🔥 **Premium**
**Step 1A:** Combined avg = (Arges: ~2.0 + Rapid: ~3.0) / 2 → ✅ Good
**Dead rubber:** Championship Group Round 9 → obective de clasament, nu supraviețuire → ⚠️ medium urgency

**Research:**
- Arges: **cea mai slabă ofensivă din playoff** → marcat O SINGURĂ dată în ultimele 4 meciuri acasă
- **10 din ultimele 11 meciuri Arges → Under 2.5 goluri total** → pattern extrem de consistent
- H2H: Arges **0 victorii** în ultimele 7 vs Rapid (4W Rapid, 3D) → Rapid domină
- Pattern H2H: 4 din 5 meciuri directe ≤ 2 goluri total
- Rapid marchează și pleacă → tempo controlat, nu deschis
- mismatch=0.313 → ✅ < 0.6

**Step 4B Unbeaten run:** Rapid are formă bună dar nu 5+ unbeaten → OK

**Tabel ajustări:**
| Ajustare | | pp |
|---------|--|--|
| Liga RO1 medium | | 0 |
| Arges: 1 gol marcat în 4 home recent | +2pp | +2 |
| 10/11 meciuri recente U2.5 | +2pp | +2 |
| H2H ≤2 goluri în 4/5 | +1pp | +1 |
| Motivație medium | | 0 |
| **TOTAL** | | **+5pp** |

**p_research = 88.3% + 5pp = cap 10pp → ~93%**

**Cum pierd:** Rapid deschide larg cu 2-0, Arges marchează 2 pe rând → 2-3. Probabilitate: ~7%

**Verdict: ✅ BET — 8/10 MODERATE** | p_research: ~93% | Fair odds: ~1.08 | BET ≥ 1.07

---

## CORNERS CoVe

### C1 — Corum vs Bodrumspor U12.5 (TR2 = Turkey 1. Lig)

**Model:** λ=8.671 | p_cal=**89.2%** | Fair odds: 1.121

**Step 0:** λ=8.67 → **< 9 → ✅ TRUST MODEL** (skip mismatch)
**Dead rubber:** Corum 4th (56pts), Bodrumspor 6th (51pts) → ambele în playoff zone → MOTIVATE

**Research:**
- Corum: 30 meciuri, GF=49 (1.63/m), GA=33 (1.10/m) → stil defensiv/controlat
- Bodrumspor: 30 meciuri, GF=64 (2.13/m), GA=35 (1.17/m) → ofensivă mare, dar nu neapărat corner-heavy
- Ambele echipe cu miză → joacă disciplinat, nu deschis → BINE pentru U12.5 corners
- λ=8.67 → EXCELLENT (cel mai mic din batch)

**Cum pierd:** Bodrumspor atacă din primele minute, generează 8+ cornere, Corum contraatacă 4+. Total 13+. Probabilitate: ~11%

**Verdict: ✅ BET — 8/10 MODERATE** | p_research: ~89% | Fair odds: 1.121 | BET ≥ 1.07

---

### C2 — Bari vs Sudtirol U12.5 (I2, Serie B Relegation Playoff)

**Model:** λ=8.658 | p_cal=**85.6%** | Fair odds: 1.169

**Step 0:** λ=8.66 → **< 9 → ✅ TRUST MODEL**
**Context:** PLAYOUT Serie B — Bari 17th vs Sudtirol 16th. **Dacă scor egal pe total, Sudtirol supraviețuiește** (poziție mai bună) → Bari TREBUIE să câștige.

**Research:**
- Bari recent: a câștigat 2 consecutive (inclusiv 3-2 la Catanzaro → ⚠️ a marcat 3 recent)
- Sudtirol recent: 4-0 vs Reggiana, dar 1-3 Frosinone, 2-3 Avellino → **7 goluri concedute în ultimele 5**
- Playoff = tensionat, tactical → echipele joacă compact, nu riscă

**Flag:** Bari a marcat 3 recent (3-2). Totuși, playoff = meci de supraviețuire → Sudtirol se va apăra compact.

**Cum pierd:** Bari deschide 0-1 rapid → Sudtirol atacă pentru egalare → meci deschis → 3-4 cornere extra pentru fiecare. Total 13+. Probabilitate: ~14%

**Verdict: ✅ BET — 7/10 MODERATE** | p_research: ~85% | Fair odds: 1.169 | BET ≥ 1.12

---

## WTA CoVe

### W1 — Volynets vs Starodubtseva U12.5 (Paris 125 QF)

**Model:** p_hold_a=0.609 | p_hold_b=0.362 | min_hold=**0.362** 🔥 | tb_p_raw=**0.022** | p_cal_adj=74.1%

**Step A:** min_hold=0.362 → 🔥 **Elite Under signal** (< 0.42)
**Step C:** WTA 125 Clay → ✅, tb_p_raw=0.022 < 0.20

**Step F.2 — Hot Streak Check:**
- Starodubtseva a bătut Putintseva **6-1, 6-1** în QF → hold poate fi +0.05-0.08 mai mare azi
- Putintseva era în formă bună (6-1, 6-1 vs Kessler) → Starodubtseva a dominat o adversară de calitate
- Chiar cu +0.08 boost: 0.362 → 0.442 → încă sub 0.50 → Premium zone, Step F.1 nu se aplică
- **Aplicam -3pp per Step F.2**

**Research:**
- Volynets a bătut Zhang 6-1, 6-4 → dominanță totală, Volynets în formă
- H2H: Volynets 1-0 vs Starodubtseva (Roma 2024, 4-6, 6-4, 6-4 în 3 seturi → competitiv!)

**Tabel ajustări:**
| | pp |
|--|--|
| tb_p_raw=0.022 (excelent) | +0 (deja în model) |
| Step F.2 Hot streak Starodubtseva | -3pp |
| H2H competitiv (3 seturi în 2024) | -1pp |
| **TOTAL** | **-4pp** |

**p_research = backtest baseline 91.6% - 4pp = ~88%**

**Cum pierd:** Starodubtseva intră în formă (ca vs Putintseva) și ține 5-6 servicii → Set 1 merge la 6-5 sau TB. Probabilitate: ~15%

**Verdict: ✅ BET — 7/10 MODERATE** | p_research: ~88% | Fair odds: ~1.14 | BET ≥ 1.10

---

### W2 — Keys vs Zakharova U12.5 (Paris 125 QF)

**Model:** p_hold_a=0.747 | p_hold_b=0.502 | min_hold=0.502 | tb_p_raw=0.060 | p_cal_adj=73.1%

**Step A:** min_hold=0.502 → ✅ Good (< 0.55)
**Step F.2 — Fatigue check:**
- Zakharova a jucat 3 seturi ieri vs Boulter (6-3, 3-6, 6-4, ~2h) → **oboseală reală**
- Fatigue → Zakharova se rupe mai ușor decât normal → BINE pentru U12.5

**Research:**
- Keys a bătut Ferro 6-4, 6-2 pe 12 mai → fresh
- Zakharova: o victorie surpriză în 3 seturi → acum obosită
- Keys hold=0.747 → ține bine; Zakharova 0.502 obosită → se rupe frecvent

**Fatigue bonus:** -pp pentru Zakharova hold rate în normal; azi probabil mai scăzut → favorabil U12.5

**Cum pierd:** Zakharova găsește energie (adrenalina QF) și ține 5-6 servicii → Set 1 competitiv 7-5 sau 7-6. Probabilitate: ~18%

**Verdict: ✅ BET — 7/10 MODERATE** | p_research: ~88% | Fair odds: ~1.14 | BET ≥ 1.10

---

## TABEL FINAL — TOP 5 MIX

| # | Pick | Model | p_research | Score | Conf. | Acțiune | Fair Odds |
|---|------|-------|-----------|-------|-------|---------|-----------|
| **1** | **Corum vs Bodrumspor — Cornere U12.5** | TR2 Corners | ~89% | **8/10** | MODERATE | ✅ **BET** | ~1.12 |
| **2** | **FC Arges vs Rapid — U4.5 goluri** | RO1 Goals | ~93% | **8/10** | MODERATE | ✅ **BET** | ~1.08 |
| **3** | **Bari vs Sudtirol — Cornere U12.5** | I2 Corners | ~85% | **7/10** | MODERATE | ✅ **BET** | ~1.17 |
| **4** | **Volynets vs Starodubtseva — U12.5 Set 1** | WTA Paris 125 | ~88% | **7/10** | MODERATE | ✅ **BET** | ~1.14 |
| **5** | **Keys vs Zakharova — U12.5 Set 1** | WTA Paris 125 | ~88% | **7/10** | MODERATE | ✅ **BET** | ~1.14 |

---

## PICKS EXCLUSE (motivate)

| Pick | p_cal | Motiv excludere |
|------|-------|----------------|
| Aston Villa/Liverpool U4.5 | 82.7% | E0 -2pp + ambele need CL + Watkins 12 goluri/13 |
| Leuven/Antwerp U4.5 | 83.5% | Leuven concede 2+ în 5 consecutive = defensivă prăbușită |
| Bari/Sudtirol Goals U4.5 | 90.6% | I2 high-scoring -2pp + Bari 3-2 recent + playoff agitated |
| Charaeva/Sasnovich U12.5 | 75.5% | Charaeva hot streak (beat Korpatsch ieri) → Step F.2 PASS |

---

## PARLAY PROPUS (3 legs)

**Corum/Bodrumspor Cornere** + **Arges/Rapid U4.5** + **Volynets/Starodubtseva U12.5**
- 3 piețe diferite, 3 ligi diferite

---

## SURSE

- [Bari vs Sudtirol playoff — Il Messaggero](https://www.ilmessaggero.it/en/bari_vs_sudtirol_the_battle_to_stay_in_serie_b-9532110.html)
- [Bari vs Sudtirol form — Forebet](https://www.forebet.com/en/football/matches/as-bari-fc-s%C3%BCdtirol-2462769)
- [Arges vs Rapid — Flashscore RO](https://www.flashscore.ro/meci/fotbal/fc-arges-65I2MLv6/rapid-bucuresti-YFCpigVG/)
- [Leuven vs Antwerp form — SportsGambler](https://www.sportsgambler.com/betting-tips/football/oh-leuven-vs-antwerp-prediction-lineups-odds-2026-05-15/)
- [Corum standings — AiScore](https://m.aiscore.com/team-al-draih/ezk96inj2wtwkn5/standings)
- [Volynets vs Starodubtseva — Flashscore H2H](https://www.flashscore.com/h2h/tennis/volynets-katie-vw1oqIPi/starodubtsewa-yulia-GlqMuuhi/)
- [Aston Villa vs Liverpool preview — ESPN](https://www.espn.com/soccer/story/_/id/48756342/aston-villa-vs-liverpool-premier-league-tv-channel-kick-live-stream-referee-injury-team-news)
- [WTA CoVe → Analysis/WTA_Set1_U125_CoVe_2026-05-13.md]
