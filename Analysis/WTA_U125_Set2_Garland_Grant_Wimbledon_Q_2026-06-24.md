# CoVe Analysis — U12.5 Set 2 | Wimbledon Qualifying 2026
## Joanna Garland vs Tyra Caterina Grant
**Data:** 2026-06-24 | **Ora:** 11:00 BST (13:00 CEST) — ⚡ URGENT
**Turneu:** Wimbledon Qualifying Round 2 — Grand Slam, Roehampton
**Suprafață:** Iarbă (outdoor, Roehampton Community Sports Centre)
**Analist:** AI Sports Analyst | **Surse:** TennisAbstract, TennisStats, Model (Markov+WElo)

---

## PASUL 1 — TRIPLE FILTER

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **8.64%** | ✅ ≤ 10% |
| p_markov | **0.5055** (50.55% Garland) | ⚠️ 50/50 |
| p_elo | **0.5000** (50.00% Garland) | ⚠️ 50/50 exact |
| Elo/Markov gap | **\|50.55 - 50.0\| = 0.55pp** | ✅ minimal |
| p_elo = 0 | Nu | ✅ |
| **UNSTABLE flag** | **✅ YES** — fatigue Garland (4 meciuri în 5 zile) | ⚠️ max 7/10 |
| hold_asym | **0.0pp** — IDENTIC 64.26% ambele | 🔴 |
| blowout_score | 2/9 | Redus |
| data_source | sackmann/sackmann | — |

**PASUL 1: ✅ TRECUT** (dar UNSTABLE = max 7/10, hold identic = incertitudine maximă)

---

## PASUL 2 — TENNISABSTRACT (iarbă) — 🔴 ASYMETRIE CRITICĂ

### Joanna Garland — Iarbă 2023-2026

**Sample: 7 meciuri** ⚠️ (borderline, similar cu Kawa)

| Meci | Turneu | Scor S1 | S1 TB? | Scor S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs Yashina (W) | s'Hertogenbosch 2025 | 6-3 | ❌ | 6-2 | ❌ NO |
| vs Sasnovich (W) | s'Hertogenbosch 2025 | 6-0 | ❌ | 6-3 | ❌ NO |
| vs Carle (W) | Wimbledon Q1 2025 | 6-1 | ❌ | 3-6 | ❌ NO |
| vs Jamrichova (W) | Wimbledon Q2 2025 | 7-5 | ❌ | 6-1 | ❌ NO |
| vs Knutson (L) | Birmingham 125 2026 | **7-6(6)** | ✅ | 6-3 | **❌ NO** |
| vs Du Pree (W) | s'Hertogenbosch 2026 | 6-4 | ❌ | 7-5 | ❌ NO |
| vs Montgomery (L) | s'Hertogenbosch 2026 | **7-6(3)** | ✅ | 6-3 | **❌ NO** |

**Garland S2 TB pe iarbă: 0/7 = 0%** ✅✅✅ EXCEPȚIONAL
**S1 TB → S2: 0/2 = 0% TB în S2 după S1 TB** ✅✅ (Knutson, Montgomery: ambele S2 decisive 6-3)

Pattern remarcabil: chiar când Set 1 merge la TB, Garland nu ajunge la TB în Set 2.

---

### Tyra Caterina Grant — Iarbă 2023-2026

**Sample: 0 MECIURI** 🔴 — ELIMINARE AUTOMATĂ

Grant nu a jucat NICIUN meci pe iarbă în baza TennisAbstract 2023-2026.
Activitate exclusivă pe clay și hard (ITF/WTA 125).

Hold rate 64.26% din model = calculat din clay/hard. Comportament pe iarbă = necunoscut complet.

---

### Verdict Pasul 2

| | Garland | Grant |
|---|---|---|
| Sample iarbă | 7 (borderline) | **0** 🔴 |
| S2 TB rate | **0/7 = 0%** ✅✅✅ | N/A |
| S1 TB → S2 | **0/2 = 0%** ✅✅ | N/A |
| Threshold ≥10 | ❌ (7<10) | ❌ (0<10) |

**PASUL 2: ❌ PASS** — Grant are 0 meciuri pe iarbă. Hold 64.26% = clay/hard estimate.

**PARADOXUL MECIULUI:** Garland are date grass EXCELENTE (0/7 S2 TB) — cel mai bun pattern din toate analizele de azi. Dar Grant este un NECUNOSCUT COMPLET pe iarbă. Nu avem cum să știm dacă va ține 50% sau 80% din servicii pe suprafața nouă.

---

## ⚠️ VERDICT: PASS

Analiza completă urmează pentru înțelegerea meciului.

---

## 1. MATCH CONTEXT

**Wimbledon Q Round 2** — Roehampton.
Garland (#188) vs Grant (#173) — meci extrem de egal între două jucătoare similare ca nivel.

**Q1 paths:**
- Garland: a bătut Celine Naef (#88!) în Q1 — surpriză majoră, #188 vs #88
- Grant: a bătut Taylah Preston în Q1

**Condiții Roehampton:** ~19-21°C, iarbă standard britanică.

---

## 2. PROFILURI JUCĂTOARE

### Joanna Garland (Taiwan)
- **Rang:** #188 | **Vârstă:** 24 ani | **Înălțime:** 177cm | **Elo:** 396
- **Stil:** Jucătoare all-court, servici decent (3.62 aces/meci = înălțime 177cm ajută), forehand agresiv
- **2026 form:** 41.2% (7/17) — an slab ← STRUGGLING
- **Form recent:** LWLLWLL — pierdut 5 din ultimele 7!
- **BUT:** tocmai a bătut Naef #88 în Q1 → moment de formă
- **Grass:** 7 meciuri, 5-2, inclusiv 2 runde Q Wimbledon 2025
- **FATIGUE:** ⚠️ 4 meciuri în 5 zile → UNSTABLE flag

### Tyra Caterina Grant (SUA)
- **Rang:** #173 | **Elo:** 438
- **Stil:** Baseliner agresiv, DFs 4.27/meci (servici streaky), return-game puternic
- **2026 form:** 76.7% (23/30) — formă EXCELENTĂ
- **Form recent:** WLWWWWL — 5 victorii consecutive recent
- **Grass:** **0 meciuri** — necunoscută complet pe suprafață
- **Avantaj:** formă mult mai bună și mai odihnită

---

## 3. STATISTICI MODEL

| Parametru | Garland (A) | Grant (B) |
|---|---|---|
| **Hold % grass (model)** | **64.26%** | **64.26%** ← IDENTICE |
| hold_asym | 0.0pp | |
| p_markov | **50.55%** Garland | |
| p_elo | **50.00%** Garland | |
| gap | **0.55pp** | ✅ consistent |
| expected_games | **24.56** | |
| blowout | 2 | slab |

**Hold identic = model nu vede niciun avantaj structural**. Aceasta e cea mai echilibrată predicție posibilă. Nu există un "favorit" conform modelului.

---

## 4. STATISTICI TENNISABSTRACT

### TennisStats (toate suprafețele, 2026)

| Statistică | Garland | Grant | Combinat |
|---|---|---|---|
| Aces/meci | **3.62** ← ridicat | 2.27 | 5.89 |
| DFs/meci | 3.15 | **4.27** ← ridicat | 7.42 |
| **Over 12.5/set** | **19%** 🔴 | **13%** | **16%** ← CEL MAI RIDICAT din azi |
| TB/meci | **31%** 🔴 | 20% | 26% |
| Avg games/set | **10.12** 🔴 MAXIM | 9.53 | **9.83** |
| Set 1 Win Rate | 35% | **70%** | |
| Set 2 Win Rate | 56% | **77%** | |

**RED FLAGS:**
- Garland 10.12 avg games/set = cel mai LUNG din circuit relativ. Joacă seturi lungi de tip 7-5, 7-5 etc.
- Garland 19% Over 12.5 = semnificativ (1 din 5 seturi merge la TB pe alte suprafețe)
- Garland 31% TB/meci = al treilea cel mai ridicat din toate analizele de azi

---

## 5. CONDIȚIE FIZICĂ & OBOSEALĂ

### Garland — 🔴 OBOSITĂ REAL
- **4 meciuri în 5 zile** → UNSTABLE flag
- A jucat ieri Q1 (probabilit în 2 seturi vs Naef)
- Formă slabă în 2026 (41.2%) → și mentală, nu doar fizică
- PARADOX U12.5: oboseala → hold scade → se rupe mai ușor → seturi mai decisive

### Grant — ✅ Fresh
- days_rest = 1 (a jucat ieri Q1), dar fără 3 seturi recente
- Formă excelentă (76.7%) → mental puternică
- **Avantaj fizic: Grant**

---

## 6. CONTEXT MOTIVAȚIONAL & PSIHOLOGIC

### Garland — ⬆️ Moment de formă (dar epuizată)
- A bătut Naef #88 în Q1 — victorie majoră vs ranking superior
- Wimbledon = turneu important pentru Asia-Pacific circuit
- Fatigue contrabalansează momentul

### Grant — ⬆️ Formă bună, nimic de pierdut
- 76.7% win rate 2026 = în cea mai bună perioadă a carierei
- PRIMELE meciuri pe iarbă → factor "nou" și necunoscut
- Poate fi destabilizată de suprafața nefamiliară OR energizată de noutate

---

## 7. ANALIZA U12.5 SET 2 — TENSIUNEA CENTRALĂ

**Ce spun datele pentru U12.5:**

| Sursă | Valoare | Verdict U12.5 |
|---|---|---|
| Model tb_p_cal | 8.64% | ✅ 91.4% U12.5 |
| TennisStats combined | **16% Over 12.5** | ⚠️ 84% U12.5 |
| Garland S2 TB grass | **0/7 = 0%** | ✅✅ |
| Grant S2 TB grass | **0 meciuri** | 🔴 necunoscut |
| Garland avg games/set | **10.12** all surfaces | 🔴 seturi lungi |
| Garland TB/meci | **31%** all surfaces | 🔴 |
| Grant TB/meci | 20% | ⚠️ |

**Tensiunea:**
- Garland pe IARBĂ = 0/7 S2 TB (excellent)
- Garland pe ALTE SUPRAFEȚE = 19% Over 12.5, 31% TB (risc)
- Grant = 0 date pe iarbă → comportament necunoscut

**Întrebarea critică:** Va juca Grant pe iarbă ca pe clay/hard (unde are 13% Over 12.5) sau diferit? Nu știm.

---

## 8. SCOR FINAL U12.5 SET 2

| Factor | Semnal |
|---|---|
| Model tb_p_cal 8.64% | ✅ |
| TennisStats 16% Over 12.5 | ⚠️ cel mai ridicat risc azi |
| Garland 0/7 S2 TB grass | ✅✅✅ |
| **Grant 0 meciuri grass** | 🔴 ELIMINARE |
| UNSTABLE (fatigue + hold=0) | cap max 7/10 |
| Garland 31% TB/meci all surfaces | 🔴 risc structural |

**VERDICT: PASS** ❌

Chiar dacă ignorăm Grant's 0 grass sample, scorul ar fi: **max 5/10** din cauza:
1. UNSTABLE = max 7/10
2. Garland 31% TB/meci = -1pp
3. TennisStats 16% (cel mai ridicat risc de azi) = -1pp
4. 50/50 model (nicio direcție clară) = incertitudine
5. Grant 0 grass = -2pp (nevalidat)

Nu există edge clar. Este cel mai slab pick U12.5 din lista de azi.

---

## 9. PREDICȚIE CÂȘTIGĂTOARE

**Grant: ~57-60%** — formă mai bună (76.7% vs 41.2%), mai odihnită, mai puternică mental. Dezavantaj: 0 meciuri pe iarbă.

**Garland: ~40-43%** — a bătut surprinzător Naef #88, cunoaște iarbă, home advantage pe Wimbledon courts. Dezavantaj: obosită, formă slabă.

---

## 10. VERDICT FINAL

| Market | Status | Scor | Decizie |
|---|---|---|---|
| **U12.5 Set 2** | **PASS** | N/A | **❌ Nu recomandăm** |

**Triple filter rezumat:**
- Pasul 1: ✅ (gap 0.55pp — trece)
- Pasul 2: ❌ Grant 0 meciuri iarbă → PASS automat
- Bonus red flags: Garland 31% TB, avg 10.12 games/set, UNSTABLE fatigue

**Din toate picks-urile Wimbledon qualifying analizate azi:**
- Kawa/Stefanini: **7/10** ✅ (7+8 meciuri, pattern consistent)
- Semenistaja/Hibino: PASS (1+4 meciuri)
- **Garland/Grant: PASS** (0 meciuri Grant, + cel mai mare risc TennisStats 16%)

---

## SURSE

- [TennisAbstract JS — Joanna Garland](https://www.tennisabstract.com/jsmatches/JoannaGarland.js)
- [TennisAbstract JS — Tyra Caterina Grant](https://www.tennisabstract.com/jsmatches/TyraCaterinaGrant.js)
- [TennisStats H2H — Garland vs Grant](https://www.tennisstats.com)
- [Wimbledon 2026 Official Qualifying Draw](https://www.wimbledon.com/en_GB/scores/draws/2026_RS_draw.pdf)
- [Olympics.com — Wimbledon 2026 Qualifying](https://www.olympics.com/en/news/wimbledon-2026-qualifying-draw-order-of-play-schedule-results)
- [Wikipedia — Wimbledon 2026 Women's Qualifying](https://en.wikipedia.org/wiki/2026_Wimbledon_Championships_%E2%80%93_Women's_singles_qualifying)
- Model Markov+WElo: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-06-24)
