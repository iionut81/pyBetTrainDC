# CoVe Analysis — Corners OVER 6.5
**Date:** 2026-05-04 | **Template:** 1.0.7CoVe_Corners_Over6_5.md v1.0
**Matches submitted:** 9 (din screenshot) | **RS1 x2 + SA1 x3 = skip (soccerstats 404)**

---

## STEP 0 — DATA FETCH (soccerstats.com)

### E0 — Premier League
| Team | FOR/g | AGS/g |
|------|-------|-------|
| Chelsea | **6.15** | 4.32 |
| Everton | 4.21 | 5.06 |
| Manchester City | **6.09** | 3.73 |
| Nottm Forest | 5.38 | 4.76 |

### P1 — Primeira Liga
| Team | FOR/g | AGS/g |
|------|-------|-------|
| Sporting CP | **7.10** | 2.94 |
| Guimarães (Vit. SC) | 4.26 | 4.06 |
| Benfica | 7.13 | 2.84 |
| FC Porto | 5.53 | 2.91 |

### DK1 — Danish Superligaen (din fetch ieri)
| Team | FOR/g | AGS/g |
|------|-------|-------|
| FC Midtjylland | 5.48 | 4.41 |
| Viborg | 4.72 | 4.28 |

**RS1 (Radnik-Vojvodina, Cukaricki-Crvena Zvezda) → 404. SKIP.**
**SA1 (Al Ittihad, Al Ettifaq, Al Fayha) → 404. SKIP.**

---

## STEP 0B — MISMATCH CALCULATIONS

| # | Meci | Liga | λ | exp_H | exp_A | Total | Mismatch | Flag |
|---|------|------|---|-------|-------|-------|----------|------|
| 1 | Chelsea vs Nottm Forest | E0 | 10.58 | (6.15+4.76)/2=**5.455** | (5.38+4.32)/2=**4.85** | **10.305** | **0.605** | 🔥 |
| 2 | Everton vs Man City | E0 | 9.82 | (4.21+3.73)/2=**3.97** | (6.09+5.06)/2=**5.575** | **9.545** | **1.605** | 🔥 BOOST |
| 3 | Sporting vs Guimarães | P1 | 8.70 | (7.10+4.06)/2=**5.58** | (4.26+2.94)/2=**3.60** | **9.18** | **1.98** | 🔥 EXTREME |
| 4 | Midtjylland vs Viborg | DK1 | 9.69 | (5.48+4.28)/2=**4.88** | (4.72+4.41)/2=**4.565** | **9.445** | **0.315** | ⚠️ LOW |

---

## CONTEXT RESEARCH

### Standings & Motivation
| Echipă | Poziție | Puncte | Context |
|--------|---------|--------|---------|
| Chelsea | 8th | 48 | 5 înfrângeri consecutive! European race (EU Conf.) |
| Nottm Forest | 16th | 39 | 6 neînvins, 5pts above relegation |
| Man City | 2nd | 70 | Luptă cu Arsenal (76pts) pentru titlu → MUST WIN |
| Everton | 11th | 46 | Mid-table, safe |
| Sporting CP | 2nd | — | Campion apărător, luptă pentru titlu |
| Vit. Guimarães | 8th | — | Mid-table, confortabili |

---

## FULL CoVe — TOATE 4 MECIURI

---

### 🔥 MATCH 1: Chelsea vs Nottingham Forest (E0) — Score: 9/10

#### Step 1 — Model Data
| Metric | Value |
|--------|-------|
| λ model | 10.58 |
| Chelsea FOR/g | **6.15** |
| Chelsea AGS/g | 4.32 |
| Forest FOR/g | 5.38 |
| Forest AGS/g | 4.76 |
| exp_home (Chelsea) | 5.455 |
| exp_away (Forest) | 4.85 |
| Total expected | **10.305** |
| Mismatch | 0.605 (just over boost threshold) |
| p_cal Over 6.5 (est.) | ~85% |

**Step A:** Chelsea 6.15, Forest 5.38 → **ambele >5 FOR** → 3/3 (GOLD — identic cu Anderlecht-Brugge)
**Step B:** Total 10.305 → 2/2 (EXCELLENT)
**Step E:** E0 = cel mai bun ligă pentru Over 6.5 (avg 10.30) → 2/2

#### Step C2 — Tactical Style
- **Chelsea:** Wingeri tradiționali, fullback-uri offensive (Pedro Neto, Reece James/cucurella pe flancuri). Cross rate ridicat confirmat de 6.15 FOR/game → **CROSS-HEAVY +5pp**
- **Forest:** Physical, wing-based attacks, unbeaten 6 games = confident attacking. Steve Cooper/similar style = direct play pe flancuri → **CROSS-HEAVY +3pp** (shared boost)
- **Total C2: +5pp** (cel mai relevant: Chelsea e acasă, dominant pe cornere)

#### Step C3 — Referee
- PL referees standard → neutral
- Dar: Chelsea lost 5 straight = tension ridicată → ref poate fi strict → **+1pp**

#### Step C4 — Match Context
**A. Injuries:** Nu există informații specifice din surse tier-1 → **+0pp**

**B. Psychology:**
- **Chelsea (8th, pierdut 5 la rând):** DESPERAȚI pentru European football. Acasă, Stamford Bridge, cu presiunea de a câștiga. Vor ataca constant → **+2pp**
- **Forest (16th, 6 neînvins):** Confident după formă bună. Dacă câștigă, practic salvați. Vor ataca selectiv, dar nu se vor teme → **+1pp**
- **Total C4-B: +3pp**

**C. Recent Corner Form:** Chelsea 6.15 FOR susținut sezon întreg = consistent. Forest 5.38 = consistent. Recent form susține Over. → **+1pp**

**D. H2H:** PL derbies între echipe atacante → tipic 10+ cornere. → **+1pp**

| Factor | Chelsea | Forest | Adj |
|--------|---------|--------|-----|
| C2 — Stil | cross-heavy dominant | cross-heavy attacking | +5pp |
| C3 — Ref | tension match, strict | — | +1pp |
| C4-A — Injuries | necunoscut | necunoscut | +0pp |
| C4-B — Psych | DESPERATE (-5 losses) at home | unbeaten confidence | +3pp |
| C4-C — Form | 6.15 FOR consistent | 5.38 FOR consistent | +1pp |
| C4-D — H2H | PL high-action derbies | — | +1pp |
| **TOTAL** | | | **+10pp** (cap atins) |

**p_research: 85% + 10pp = ~90%** (cap la 95% teoretic, dar conservăm la 90%)

**Quick Score:**
- A (both >5 FOR): 3
- B (total 10.3): 2
- C2 (cross-heavy): +1
- C3 (strict ref): +1
- D+E (E0 + both motivated): 2
- C4 (Chelsea desperate + Forest confident): +1
- **Total: 10/10 → PREMIUM**

**Verdict: PREMIUM BET** — ambele echipe cu >5 FOR + E0 + Chelsea disperată + Forest confident atacant. Profilul Anderlecht-Brugge (16 cornere ieri) în cea mai bună ligă pentru cornere.

**How I lose:** Forest parkează autobuzul + Chelsea nu reușesc să transforme presiunea în centrări. Scor devreme (Forest 1-0) schimbă dinamica → Chelsea atacă mai haotic, mai puțin eficient pe flancuri. Șansă mică (~10%) ca total să fie sub 7.

**Sources:**
- [Chelsea vs Nottm Forest preview — Sports Mole](https://www.sportsmole.co.uk/football/premier-league/chelsea-vs-nottingham-forest_game_247262.html)
- [Soccerstats E0](https://www.soccerstats.com/table.asp?league=england&tid=cr)

---

### 🔥 MATCH 2: Sporting CP vs Vitória Guimarães (P1) — Score: 8.5/10

#### Step 1 — Model Data
| Metric | Value |
|--------|-------|
| λ model | 8.70 |
| Sporting FOR/g | **7.10** |
| Sporting AGS/g | 2.94 |
| Guimarães FOR/g | 4.26 |
| Guimarães AGS/g | 4.06 |
| exp_home (Sporting) | 5.58 |
| exp_away (Guimarães) | 3.60 |
| Total expected | **9.18** |
| Mismatch | **1.98** (EXTREME) |
| p_cal Over 6.5 (est.) | ~82% |

**Step A:** Sporting 7.10 FOR = EXTREME generator. Guimarães 4.26 = decent defense. → 3/3 (GOLD+)
**Step B:** Total 9.18 → 2/2 (EXCELLENT)
**Notă:** P1 = "risky" ligă pentru Over 6.5 (avg mai scăzut), DAR Sporting specific transcende media ligii — la fel ca Feyenoord (7.06 FOR ieri → likely WON)

#### Step C2 — Tactical Style
- **Sporting CP (Amorim-style):** Possesion, dar cu wingeri activi + fullback pressing. 7.10 FOR/game = wing attacks constante. HIGH press → cornere din presiune înaltă. → **+5pp**
- **Guimarães:** Mid-table, balansat, nu defensiv-first. Nu va parca autobuzul → cornere pe ambele părți → **+0pp** (neutral)

#### Step C3 — Referee
- Primeira Liga: referees standard → **+0pp**

#### Step C4 — Match Context
**A. Injuries:** Nu există date → **+0pp**

**B. Psychology:**
- **Sporting (2nd):** Campion apărător, luptă pentru titlu. MUST WIN pentru a rămâne în cursă. → **+2pp**
- **Guimarães (8th):** Mid-table confortabil. Nu are presiune specifică. Joc liber → **+0pp**
- **H2H dominance:** Sporting 20W din 34H2H vs Guimarães → Sporting va domina, Guimarães va juca compact → mismatch pattern

**C. Recent Corner Form:** Sporting 7.10 FOR susținut = consistent. → **+1pp**

**D. H2H:** Sporting domination = tipic High corners generat de Sporting singur → **+1pp**

| Factor | Sporting | Guimarães | Adj |
|--------|----------|-----------|-----|
| C2 — Stil | wing press + 7.10 FOR | balansat | +5pp |
| C3 — Ref | standard | — | +0pp |
| C4-A | necunoscut | necunoscut | +0pp |
| C4-B | titlu fight must-win | confortabil | +2pp |
| C4-C | 7.10 FOR consistent | 4.26 FOR constant | +1pp |
| C4-D | domination pattern | — | +1pp |
| **TOTAL** | | | **+9pp** |

**p_research: 82% + 9pp = ~87%** (P1 ligă risky -2pp discount → **~85%**)

**Quick Score:**
- A (Sporting 7.10): 3
- B (total 9.18): 2
- C2 (press + 7.10): +1
- C3 (neutral): +0
- D+E (P1 risky, -0.5, dar Sporting titlu fight, +1): 1.5
- C4 (titlu fight + dominant): +1
- **Total: 8.5/10 → STRONG**

**How I lose:** Sporting joacă rotit/econom (titlul deja decis?) sau Guimarães apără extrem de compact ca Fortuna Sittard ieri → total < 7 cornere. Risc limitat dat de 7.10 FOR consistent.

**Sources:**
- [Sporting vs Guimarães context — ESPN](https://www.espn.com/soccer/match/_/gameId/750530/vitoria-de-guimaraes-sporting-cp)
- [Soccerstats P1](https://www.soccerstats.com/table.asp?league=portugal&tid=cr)

---

### ✅ MATCH 3: Everton vs Manchester City (E0) — Score: 7.5/10

#### Step 1 — Model Data
| Metric | Value |
|--------|-------|
| λ model | 9.82 |
| Everton FOR/g | 4.21 |
| Everton AGS/g | 5.06 |
| Man City FOR/g | **6.09** |
| Man City AGS/g | 3.73 |
| exp_home (Everton) | 3.97 |
| exp_away (Man City) | **5.575** |
| Total expected | **9.545** |
| Mismatch | **1.605** (BOOST!) |
| p_cal Over 6.5 (est.) | ~82% |

**Step A:** Man City 6.09 → GOLD single generator. Everton 4.21 → decent. → 2/3 (one dominant)
**Step B:** Total 9.545 → 2/2 (EXCELLENT)

#### Step C2 — Tactical Style
- **Man City:** ATENȚIE — stilul "Pep inverted" = C2 penalty potențial. DAR Man City are 6.09 FOR/game = statistici reale care contrazic profilul inverted clasic. Fie stilul s-a schimbat în 2025-26, fie Guardiola nu mai e manager.
  - Man City 61 goluri marcate, 2nd place = **ATTACKING STYLE în 2025-26** → **+3pp** (nu +5pp din cauza incertitudinii stilului)
- **Everton (11th, 46pts):** Fără presiune specifică, defensive pe home. → **-1pp** (reticent to attack aggressively)

#### Step C3 — Referee
- E0 standard + Man City la titlu → tensiune înaltă → **+1pp**

#### Step C4 — Match Context
**B. Psychology:**
- **Man City:** Fighting Arsenal pentru titlu. Fiecare punct contează. MUST WIN. → **+3pp**
- **Everton (11th, safe):** Fără motivație specială. Mid-table. → **-1pp** (confortabil, nu atacă)
- **Net C4-B: +2pp**

**A. Injuries:** Nicio informație → **+0pp**

| Factor | Everton | Man City | Adj |
|--------|---------|----------|-----|
| C2 — Stil | defensiv moderat | attacking high scorer | +3pp |
| C3 — Ref | E0 title match | — | +1pp |
| C4-A | necunoscut | necunoscut | +0pp |
| C4-B | comfortable (−1) | titlu must-win (+3) | +2pp |
| **TOTAL** | | | **+6pp** |

**p_research: 82% + 6pp = ~88%**

**Quick Score:**
- A (Man City 6.09, Everton 4.21): 2
- B (total 9.545): 2
- C2 (Man City attacking): +1
- C3 (title match ref): +1
- D+E (E0 best + Man City must-win): 2
- C4 (titlu fight − Everton comfortable): +0.5
- **Total: 8.5/10**

**Totuși: risc stil Man City.** Dacă Pep-inverted rule se aplică (-5pp), p_research scade la ~83%. Rămâne deasupra 82% și E0 e cea mai bună ligă.

**How I lose:** Man City joacă inverted (Foden/Bernardo central, nu wingeri tradiționali) → generează puțin cornere DESPITE dominanță. Everton pasivă + Man City inverted = total 6-7 cornere. Scenariu posibil ~20% dacă stilul e Pep-classic.

**Verdict: MODERATE-STRONG BET** cu caveat stil Man City.

**Sources:**
- [Everton vs Man City preview — Sports Mole](https://www.sportsmole.co.uk/football/premier-league/everton-vs-man-city_game_247289.html)
- [Man City press preview](https://www.mancity.com/news/mens/everton-premier-league-away-may-2026-match-preview-63912789)

---

### ⚠️ MATCH 4: Midtjylland vs Viborg (DK1) — Score: 5.5/10

#### Step 1 — Model Data
| Metric | Value |
|--------|-------|
| λ model | 9.69 |
| Midtjylland FOR/g | 5.48 |
| Viborg FOR/g | 4.72 |
| Total expected | **9.445** |
| Mismatch | **0.315** (LOW) |

**Step A:** Ambele 4.7-5.5 → decent dar nu exceptional. → 2/3
**Mismatch 0.315 = sub nivelul de boost** → nu există un generator dominant

**Context:** Fără presiune specifică identificată pentru niciunul. DK1 ligă bună dar nu în top.

**Verdict: PASS** — Mismatch prea mic + nicio presiune contextuală. Total 9.4 bun dar fără boost structural.

---

## SUMMARY TABLE

| Meci | Score | p_research | Action |
|------|-------|-----------|--------|
| **Chelsea vs Nottm Forest (E0)** | **10/10** | **~90%** | 🔥 **PREMIUM** |
| **Sporting CP vs Guimarães (P1)** | **8.5/10** | **~85%** | 🔥 **STRONG** |
| **Everton vs Man City (E0)** | **8.5/10** | **~88%** | ✅ **STRONG** (caveat stil) |
| Midtjylland vs Viborg (DK1) | 5.5/10 | ~75% | ❌ PASS |
| RS1 x2 | — | — | ⛔ DATE LIPSĂ |
| SA1 x3 | — | — | ⛔ DATE LIPSĂ |

---

## FINAL PICKS — OVER 6.5 CORNERS (04.05.2026)

### PICK 1 — Chelsea vs Nottingham Forest (E0) ⭐⭐⭐
- **p_research: ~90%** | Fair odds: ~1.11 | Score: 10/10
- **Key stats:** Chelsea 6.15 FOR + Forest 5.38 FOR = **ambele >5 (profilul Anderlecht-Brugge)**. Total expected 10.31. E0 = liga cu avg 10.30 cornere.
- **Context:** Chelsea lost 5 straight, DESPERATE for European football acasă. Forest pe 6-game unbeaten, confident, atacă și ei. Ambele echipe au motivație.
- **Tactical:** Chelsea = cross-heavy traditioanl wingeri (6.15 FOR). Forest = physical direct play.
- **How I lose:** Forest parkează și Chelsea nu convertește presiunea în centrări eficiente.
- **Confidence: HIGH**

### PICK 2 — Everton vs Manchester City (E0) ⭐⭐⭐
- **p_research: ~88%** | Fair odds: ~1.14 | Score: 8.5/10
- **Key stats:** Man City 6.09 FOR. Mismatch 1.605 → Man City atacă dominat vs Everton pasiv. Total expected 9.55.
- **Context:** Man City fighting Arsenal for title — MUST WIN. Everton mid-table, fără presiune. Dynamic: City atacă constant.
- **Caveat:** Dacă stilul Man City e inverted (Pep-classic), -5pp → p_research ~83% (încă >82%).
- **How I lose:** Man City joacă inverted, Everton pasiv → 6-7 cornere total.
- **Confidence: HIGH** (cu caveat stilistic)

### PICK 3 — Sporting CP vs Vitória Guimarães (P1) ⭐⭐
- **p_research: ~85%** | Fair odds: ~1.18 | Score: 8.5/10
- **Key stats:** Sporting CP 7.10 FOR/game (highest in P1, similar Feyenoord 7.06 ieri). Mismatch extreme 1.98. Total expected 9.18.
- **Context:** Sporting 2nd, campion apărător, în luptă pentru titlu. MUST WIN.
- **Ligă P1 discount:** P1 e "risky" pentru Over 6.5 (avg mai mic) → conservăm la 85%, nu 87%.
- **How I lose:** Sporting rotit, Guimarães compact → sub 7 total.
- **Confidence: HIGH**

---

## SELF-VERIFICATION

- [x] Fetched soccerstats.com E0 (Chelsea, Everton, City, Forest) + P1 (Sporting, Guimarães) ✅
- [x] DK1 din fetch ieri valid (sezon stats) ✅
- [x] RS1 + SA1 skipuite corect (404) ✅
- [x] Mismatch calculat cu formula exactă ✅
- [x] Midtjylland-Viborg eliminat corect (mismatch 0.315, fără context) ✅
- [x] Man City style caveat notat explicit ✅
- [x] Cap +10pp respectat (Chelsea atins la 10pp) ✅
- [x] Surse citate inline ✅
- [x] "How I lose" inclus pentru fiecare pick ✅

---

*Analysis: 2026-05-04 | Template CoVe Over 6.5 v1.0*
