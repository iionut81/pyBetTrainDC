# CoVe — Football Corners UNDER 12.5 — Batch Analysis
## Date: 2026-04-24 (Friday)
## Template: Prompts/1.0.3CoVe_Corners.md v1.1
## Pool: 7 recommended picks from daily model

---

## 📋 STEP 1: MODEL + MISMATCH FILTER

Per-team corner profiles (`corners_team_profiles.csv`), computed λ per team and mismatch ratio:

| Meci | Liga | λ_H | λ_A | Mismatch | H h_for | A a_for | Model P | Filter |
|---|---|---|---|---|---|---|---|---|
| Alverca vs Arouca | P1 | 4.54 | 3.13 | **0.18** | 4.81 | 3.26 | 87.6% | ✅ PASS |
| Napoli vs Cremonese | I1 | 6.58 | 2.78 | 0.41 | **6.10** | **3.11** | 85.7% | ❌ **HARD PASS** |
| Avellino vs Bari | I2 | 6.55 | 2.88 | 0.39 | **6.00** | **2.90** | 84.7% | ❌ HARD PASS |
| Brest vs Lens | F1 | 4.21 | 4.58 | **0.04** | 4.82 | 4.88 | 84.5% | 🔥 EXCELLENT |
| Başakşehir vs Kasımpaşa | TR1 | 4.17 | 3.41 | **0.10** | 4.90 | 3.71 | 84.4% | 🔥 EXCELLENT |
| FCSB vs Petrolul | RO1 | 5.82 | 3.99 | 0.19 | **6.54** | 4.36 | 82.7% | ⚠️ FCSB=corner machine |
| Hermannstadt vs Csíkszereda | RO1 | 5.25 | 3.60 | 0.19 | 5.20 | **2.10** | 82.3% | ✅ PASS + Csik GOLD |

**FADES** (HARD PASS — Step A/mismatch violation):
- Napoli-Cremonese: title chase blowout setup (the Sassuolo-Como analog)
- Avellino-Bari: >6 vs <4 + Avellino small sample (n=7 home)
- FCSB-Petrolul: FCSB h_for 6.54 = corner machine, title race pressure

**Survivors**: Alverca-Arouca, Brest-Lens, Başakşehir-Kasımpaşa, Hermannstadt-Csíkszereda.

---

## 📋 STEP 2: EXTERNAL RESEARCH (survivors only)

### 🥇 MATCH 1: Brest vs Lens (F1) — Mismatch 0.04 EXCELLENT

**Sources:** [Sports Mole](https://www.sportsmole.co.uk/football/ligue-1-title-race/preview/brest-vs-lens-prediction-team-news-lineups_596161.html), [Football Whispers](https://footballwhispers.com/blog/brest-vs-lens-prediction-24-04-2026/)

**Brest (12th):**
- **Winless 6 weeks**, 1 pt from last 4 matches, drew Nantes 1-1 Sunday
- **Failed to cover 3.5 corners in 3 consecutive matches** ← LOW corner generation recent
- In a slump, passive play

**Lens (title race):**
- Battling PSG for Ligue 1 title
- 4W from last 5, best season
- **Conceded under 3.5 corners in past 8 away games** ← LOW corners conceded away
- H2H: Lens 4-0 last 4 encounters

**Game state analysis:**
- Likely outcome: Lens wins (probably 1-3 / 0-2)
- **Risk:** If Lens leads 2-0 early, Brest passive → less pressing → fewer corners (both directions)
- **Mitigation:** Recent form of BOTH teams = low corners (Brest can't win, Lens cruises)

**Score:**
```
Corner baseline : 3/3 (both 3-5, both recent trends <4)
Expected total  : 2/2 (exp 8.79, recent actual <7)
Style fit       : 2/2 (Lens cruises, Brest demoralized)
Game state      : 1/2 (-1: Lens title chase might push late)
Intuition       : 1/1 (pattern aligned perfectly)
TOTAL           : 9/10 HIGH CONFIDENCE
```

**Verdict:** Model 84.5% → Research-adj **~87-89%** | Fair: 1.14 | Accept @ >= 1.20 | **Stake: 7 RON**

---

### 🥈 MATCH 2: Başakşehir vs Kasımpaşa (TR1) — Mismatch 0.10 EXCELLENT

**Sources:** [Dailysports](https://dailysports.net/predictions/stanbul-basaksehir-vs-kasmpasa-prediction-h2h-and-probable-lineups-april-24-2026/), [Sports Mole](https://www.sportsmole.co.uk/football/istanbul-basaksehir/preview/istanbul-vs-kasimpasa-prediction-team-news-lineups_596211.html)

**Başakşehir (5th, 48 pts):**
- Unbeaten last 4 (1W 3D), 3 clean sheets
- **NOT covered 5.5 team corners in past 8 home games** ← VERY LOW corner generation
- Solid defensive block

**Kasımpaşa (13th, 31 pts):**
- Unbeaten 3 (2W 1D), 1-0 vs Alanyaspor
- **Conceded UNDER 5.5 corners in past 3 away** ← LOW concession
- 6 pts from relegation, but trending up

**H2H:** Başakşehir **9W 2D last 11** vs Kasımpaşa (dominance), 3-1 this season

**Game state analysis:**
- H2H dominance + O2.5 goals last 10 meetings → some chasing risk
- BUT: recent Başakşehir home corner total < 5.5 ← hard data
- Kasımpaşa away corner concession < 5.5 ← hard data
- Combined: total probably 8-10 corners ← clear Under 12.5

**Score:**
```
Corner baseline : 3/3 (both 3-5, Kasımpaşa a_for 3.71 ideal)
Expected total  : 2/2 (exp 7.58 — one of lowest)
Style fit       : 2/2 (both defensive-minded recent)
Game state      : 1/2 (O2.5 goals history -1)
Intuition       : 1/1 (two recent trends align)
TOTAL           : 9/10 HIGH CONFIDENCE
```

**Verdict:** Model 84.4% → Research-adj **~86-88%** | Fair: 1.18 | Accept @ >= 1.22 | **Stake: 6 RON**
**League caveat:** TR1 hit rate 78.4% (mid-tier) — slight discount for variance.

---

### 🥉 MATCH 3: Alverca vs Arouca (P1) — Mismatch 0.18

**Sources:** [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/alverca-vs-arouca-prediction-lineups-odds-2026-04-24/), [Sports Mole](https://www.sportsmole.co.uk/football/primeira-liga/alverca-vs-arouca_game_246660.html)

**Alverca (10th, 35 pts):**
- Home: 2W from last 3 home, 5 goals scored
- ⚠️ **Corner avg 6.80 last 5 home matches** ← HIGHER than profile (4.81) — TREND UP
- Home-favorite style

**Arouca (11th, 35 pts):**
- Won last vs Estrela 1-0 home
- **Corner avg 3.10 last 10 matches** ← LOW
- Defensive travel style

**H2H / context:**
- Both on 35 pts, mid-table, no desperation
- Match probability: Alverca win 44.3%, Draw 24.8%, Arouca win 30.9%
- Alverca corner match bet favorite ← meaning they'll dominate corners
- Total recent: 6.80 + 3.10 ≈ 9.9 ← near the fair line

**Game state concern:**
- Alverca 6.80 home trend is much higher than profile → **profile STALE**
- Arouca defensive shell → Alverca wing attacks = corners
- Mid-table low-motivation context helps (no late chase)

**Score:**
```
Corner baseline : 2/3 (Alverca recent 6.80 UP, Arouca 3.10 OK)
Expected total  : 1/2 (recent actual ≈ 9.9, close to 10.5)
Style fit       : 1/2 (Alverca wingy vs Arouca low block = corner risk)
Game state      : 2/2 (both mid-table, no pressure)
Intuition       : 1/1 (stale profile trigger warning)
TOTAL           : 7/10 MODERATE
```

**Verdict:** Model 87.6% → Research-adj **~82-84%** (stale profile discount -5pp) | Fair: 1.20 | Accept @ >= 1.25 | **Stake: 4 RON**
**League caveat:** P1 hit rate 77% (weak) — additional discount.

---

### MATCH 4: Hermannstadt vs Csíkszereda (RO1) — Mismatch 0.19

**Sources:** [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/hermannstadt-vs-csikszereda-prediction-lineups-odds-2026-04-24/), [Sibiu Independent](https://sibiuindependent.ro/2026/04/21/meci-hermannstadt-miercurea-ciuc/), [Leaguelane](https://leaguelane.com/predictions/hermannstadt-vs-csikszereda-24-april-2026/)

**Hermannstadt:**
- **3 consecutive home wins**: 3-0 Botosani, 1-0 Farul, Petrolul
- h_for 5.20 (borderline), strong form
- 1.03 goals/match (low scoring)
- Chitu scored in 5 straight SuperLiga matches — form

**Csíkszereda:**
- **Lost 12 of last 18 away matches** ← TRAVEL STRUGGLES EXTREME
- a_for 2.10 ← GOLD level low corners
- Defensive away

**H2H/Game state:**
- Hermannstadt 1.76 favorite (57% match win)
- Likely: Hermannstadt scores early (Chitu streak) → Csíkszereda retreats
- **Game state risk:** If Csíkszereda parks bus → Hermannstadt wing attacks → corner spike?
- Counter-argument: Hermannstadt scoring only 1.03 goals/match = not a corner machine

**Concerns:**
- Mismatch per profile is only 0.19 but real-world quality gap is larger (Csik lost 12/18 away)
- If this becomes a blowout (3-0 scenario like Hermannstadt recent home wins), corners could spike
- Csik a_for = 2.10 is a cushion — they don't generate much even if pushed

**Score:**
```
Corner baseline : 2/3 (Herm 5.20 borderline, Csik 2.10 GOLD)
Expected total  : 2/2 (exp 8.85 comfortable)
Style fit       : 1/2 (Herm wing-capable, Csik low block = potential spike)
Game state      : 1/2 (-1: quality gap > profile says)
Intuition       : 0/1 (Hermannstadt 3 home wins show dominance pattern)
TOTAL           : 6/10 BORDERLINE
```

**Verdict:** Model 82.3% → Research-adj **~78-82%** (quality gap discount) | Fair: 1.22 | Accept @ >= 1.28 | **Stake: 3 RON** (speculative)

---

## 📋 STEP 3: SELF-VERIFICATION

| Check | Answer |
|---|---|
| Used actual corner data? | ✅ team profiles + recent corner trends |
| Verified tempo/style? | ✅ recent form narratives included |
| Avoided narrative bias? | ✅ discounted Alverca model probability for stale profile |
| Included risk factors? | ✅ game-state spike risk noted for each |
| Model contradicted reality? | ✅ Alverca profile 4.81 vs recent 6.80 — flagged |
| Capped +10pp? | ✅ max upgrade was Brest-Lens +4pp |
| Considered match state? | ✅ Lens title chase, Hermannstadt quality gap, Kasımpaşa relegation |

🔴 **Final question:** "Can this match reach 12+ corners?"
- Brest-Lens: **Unlikely** — both recent trends <8 combined
- Başakşehir-Kasımpaşa: **Unlikely** — both recent <5.5 team
- Alverca-Arouca: **Possible** — Alverca 6.80 recent, trend up
- Hermannstadt-Csik: **Possible** — if Herm 3-0 dominance pattern repeats

---

## 🎯 STEP 4: CORRECTIONS TABLE

| Pick | Model | Research | Score | Action | Reason |
|---|---|---|---|---|---|
| Brest-Lens | 84.5% | ~88% | 9/10 | ✅ **STRONG BET** | Both recent trends ultra-low corners |
| Başakşehir-Kasımpaşa | 84.4% | ~87% | 9/10 | ✅ **STRONG BET** | Two aligned recent corner trends <5.5 |
| Alverca-Arouca | 87.6% | ~83% | 7/10 | 🟡 MODERATE | Stale profile (-5pp) + weak P1 league |
| Hermannstadt-Csik | 82.3% | ~80% | 6/10 | 🟡 SPECULATIVE | Quality gap > profile indicates |
| Napoli-Cremonese | 85.7% | — | — | ❌ HARD PASS | Mismatch trap (Sassuolo analog) |
| Avellino-Bari | 84.7% | — | — | ❌ HARD PASS | Mismatch + small sample |
| FCSB-Petrolul | 82.7% | — | — | ❌ HARD PASS | FCSB 6.54 corner machine |

---

## 🎯 STEP 5: FINAL PICKS

### ⭐ TOP CONFIDENCE (9/10)

#### 1. Brest vs Lens U12.5 | 7 RON | Min odds 1.20
- **Checklist:** 9/10 HIGH
- **Model:** 84.5% | **Research:** ~88%
- **Fair:** 1.14 | **Key stat:** Brest failed 3.5 corner line 3 consecutive matches, Lens <3.5 concede 8 away
- **How I lose:** Lens 2-0 at 30min, Brest counter-rushes, late corner spike → 13 corners
- **Kickoff:** Friday 24.04, Ligue 1 M31
- **Source:** [Sports Mole](https://www.sportsmole.co.uk/football/ligue-1-title-race/preview/brest-vs-lens-prediction-team-news-lineups_596161.html)

#### 2. Başakşehir vs Kasımpaşa U12.5 | 6 RON | Min odds 1.22
- **Checklist:** 9/10 HIGH
- **Model:** 84.4% | **Research:** ~87%
- **Fair:** 1.18 | **Key stat:** Başakşehir NOT covered 5.5 team corners last 8 home, Kasımpaşa <5.5 away concede last 3
- **How I lose:** Basaksehir blows Kasımpaşa away with 3-0, wing dominance → 12+ corners
- **Kickoff:** Friday 24.04, Süper Lig M31
- **Source:** [Dailysports](https://dailysports.net/predictions/stanbul-basaksehir-vs-kasmpasa-prediction-h2h-and-probable-lineups-april-24-2026/)

### 🟡 MODERATE (7/10)

#### 3. Alverca vs Arouca U12.5 | 4 RON | Min odds 1.25
- **Checklist:** 7/10 MODERATE
- **Model:** 87.6% | **Research:** ~83%
- **Fair:** 1.20 | **Key stat:** Alverca 6.80 corners recent home — profile stale warning
- **How I lose:** Alverca wing attacks vs Arouca low block → 10+ corners generated by Alverca alone

### ⚠️ SPECULATIVE (6/10)

#### 4. Hermannstadt vs Csíkszereda U12.5 | 3 RON | Min odds 1.28
- **Checklist:** 6/10 LOW-MOD
- **Model:** 82.3% | **Research:** ~80%
- **Fair:** 1.22 | **Key stat:** Csik a_for 2.10 GOLD, but Hermannstadt on 3W home streak
- **How I lose:** Hermannstadt repeats 3-0 pattern, Csik bunker → Herm generates 10+ alone

---

## 🎯 MultiMarket Accumulator option

**2-fold (TOP2):** Brest-Lens U12.5 × Başakşehir-Kasımpaşa U12.5
- Joint P ≈ 0.88 × 0.87 = **76.6%**
- Fair odds ≈ 1.14 × 1.18 = **1.34**
- Accept @ >= **1.38** for small edge
- Stake: 4 RON (half normal since accumulator)

**3-fold (TOP3):** + Alverca-Arouca
- Joint P ≈ 0.88 × 0.87 × 0.83 = **63.5%**
- Fair odds ≈ 1.14 × 1.18 × 1.20 = **1.61**
- Accept @ >= **1.70**
- Stake: 3 RON

---

## 📚 Lessons Reinforced

- **Mismatch filter = absolutely critical** — 3 of 7 HARD PASS on this rule (Napoli, Avellino, FCSB all failed)
- **Recent trends > stale profile** — Alverca 6.80 vs profile 4.81 (-5pp research adjustment)
- **"Not covered X corners" stats** from preview sites = gold signals (Başakşehir 5.5 pattern)
- **Title chase + low-tempo opponent = corner spike risk** — the Sassuolo-Como pattern recurring (would have been Napoli)
- **Csík 2.10 a_for extreme low** = even with Hermannstadt dominance, hard cap on total corners
- **Lens away low corner concede** (<3.5 in 8 away) = unusual and strong signal

---

## 📖 Sources (full list)

### Survivors
- [Brest vs Lens preview (Sports Mole)](https://www.sportsmole.co.uk/football/ligue-1-title-race/preview/brest-vs-lens-prediction-team-news-lineups_596161.html)
- [Brest vs Lens betting tips (Football Whispers)](https://footballwhispers.com/blog/brest-vs-lens-prediction-24-04-2026/)
- [Alverca vs Arouca preview (Sports Mole)](https://www.sportsmole.co.uk/football/primeira-liga/alverca-vs-arouca_game_246660.html)
- [Alverca vs Arouca tips (Sportsgambler)](https://www.sportsgambler.com/betting-tips/football/alverca-vs-arouca-prediction-lineups-odds-2026-04-24/)
- [Başakşehir vs Kasımpaşa (Dailysports)](https://dailysports.net/predictions/stanbul-basaksehir-vs-kasmpasa-prediction-h2h-and-probable-lineups-april-24-2026/)
- [Başakşehir vs Kasımpaşa (Sports Mole)](https://www.sportsmole.co.uk/football/istanbul-basaksehir/preview/istanbul-vs-kasimpasa-prediction-team-news-lineups_596211.html)
- [Hermannstadt vs Csíkszereda (Sportsgambler)](https://www.sportsgambler.com/betting-tips/football/hermannstadt-vs-csikszereda-prediction-lineups-odds-2026-04-24/)
- [Hermannstadt vs Csík (Sibiu Independent)](https://sibiuindependent.ro/2026/04/21/meci-hermannstadt-miercurea-ciuc/)