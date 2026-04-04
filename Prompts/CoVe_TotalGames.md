# CoVe (Chain of Verification) — Tennis Total Games
## Version 1.0 — Created 2026-04-04

---

## CONTEXT SETUP

I have a background of data analytics and worked as a Business Intelligence developer. I am transitioning to sport analyst. I need you to behave like a senior betting analyst.

**Task:** Analyze WTA matches for Total Games Under 28.5 market.

**Key insight:** Under 28.5 wins in ALL straight-set matches (max 2 sets = 26 games at 7-6, 7-6). The only risk is 3-set matches with long sets.

**Therefore:** This market = betting on straight sets + short 3-setters.

---

## PRE-ANALYSIS: 15-SECOND CHECKLIST

### STEP A — STRAIGHT SETS PROBABILITY (most important)

From model `simulate_match()`:
- P(straight sets) > 80% → 🔥 EXCELLENT
- P(straight sets) 70-80% → ✅ GOOD
- P(straight sets) 60-70% → ⚠️ Caution
- P(straight sets) < 60% → ❌ DON'T BET (too much 3-set risk)

### STEP B — MATCH WINNER PROBABILITY

- P(winner) > 85% → 🔥 Dominant favorite = short match
- P(winner) 75-85% → ✅ Clear favorite
- P(winner) 65-75% → ⚠️ Competitive = 3-set risk rises
- P(winner) < 65% → ❌ DON'T BET (toss-up = long match likely)

### STEP C — MARGIN CHECK

- Expected total games vs line:
- Margin > 6 games (e.g., expected 22, line 28.5) → 🔥 EXCELLENT
- Margin 4-6 → ✅ GOOD
- Margin 2-4 → ⚠️ Thin
- Margin < 2 → ❌ DON'T BET

### STEP D — HOLD GAP

- Gap > 0.15 → ✅ Clear mismatch = straight sets
- Gap 0.10-0.15 → 🟡 OK
- Gap < 0.10 → ❌ Too balanced = 3-set risk

### STEP E — CONTEXT

- Final? → ⚠️ Downgrade (tighter, more 3-setters historically)
- Both top-10? → ⚠️ Downgrade (quality opponents push to 3 sets)
- Underdog in first SF/F ever? → ✅ Upgrade (nerves = quick loss)
- Clay → ⚠️ Slight downgrade (rallies = more break-backs = more 3-setters)
- Hard/Grass → ✅ Neutral/boost (serves hold better = favorite closes faster)

### PASS/FAIL:
Need ALL of:
- P(straight sets) ≥ 70%
- P(winner) ≥ 75%
- Margin ≥ 4 games
- Gap ≥ 0.10

### QUICK SCORE (/10):
- +3 straight sets probability
- +2 match winner dominance
- +2 margin size
- +2 context favorable
- +1 gut feel (max 1!)

**Score 7+ = RECOMMEND | Score < 7 = DO NOT RECOMMEND**

### AUTOMATIC PENALTIES:
- Finals = **-1** on checklist score
- Both top-10 = **-1** on checklist score
- Clay surface = **-1** on checklist score (more 3-setters on clay)
- tennisabstract/tennisabstract data source = **-1** on checklist score

---

## STEP 1: Analyze the provided data

For each match run `simulate_match()` and list:
- P(straight sets), P(3 sets), P(winner)
- Expected total games, std, margin to 28.5
- Hold A, Hold B, gap
- Games distribution by range

Run checklist. Score /10.

---

## STEP 2: External Research + Verification

For each match that passes the checklist, gather from internet:
- **3-Set History:** How many of player's last 5 matches went to 3 sets? (>3/5 = flag)
- **Straight sets pattern:** Does the favorite typically close in 2?
- Recent match scores (total games per match)
- Injuries, fitness (tired player = loses sets = 3-set risk)
- H2H (context only, not main argument)

**MANDATORY:** Cite source URL for every fact.

**RULES:**
- Research can adjust model probability by MAX +10pp / -10pp
- Focus on: does the favorite CLOSE matches, or does she let opponents back in?
- A player who won last 3 matches in straight sets = upgrade
- A player who went to 3 sets in last 2+ matches = downgrade

---

## STEP 3: Self-Verification

1. "Did I check P(straight sets) from the model?"
2. "Did I verify the favorite's closing pattern from recent matches?"
3. "Did I consider that ALL straight-set outcomes are Under 28.5?"
4. "Did I check if the underdog has comeback ability (3-set risk)?"
5. "Did I cap research adjustment at ±10pp?"
6. "Did I apply clay penalty?"

### THE FINAL QUESTION (most important):

**"Can the underdog realistically WIN a set and push this to 3 sets with long sets (7-5, 7-6)?"**

Think specifically:
- Can the underdog hold serve well enough to take a set?
- If 3 sets happen, will they be short (6-2, 4-6, 6-1 = 25 games = still Under) or long (7-5, 5-7, 7-5 = 34 = Over)?
- If the answer is "3 sets are possible but they'd be short" → Under still valid
- If the answer is "3 sets are possible AND sets would be close" → ❌ DON'T BET

---

## STEP 4: Corrections

Write table: Pick | Model | Research | Checklist Score | Action | Reason

---

## STEP 5: Final Picks (ranked by checklist score)

For each recommended pick include:
- **Checklist score** (/10)
- **P(straight sets)** from model
- **P(Under 28.5)** from model
- **Expected total games** and **margin to 28.5**
- **Research adjusted probability** (capped at ±10pp)
- **Fair odds**
- **Key stat** (one sentence)
- **How I lose this bet** (one sentence)
- **Source** (URL)

### CONFIDENCE LEVELS:
- Score 9-10: HIGH confidence
- Score 7-8: MODERATE confidence
- Score < 7: **DO NOT RECOMMEND**

### RULES:
- Max "HIGH confidence" on WTA 250 = MODERATE (cap it)
- Finals = automatic -1
- Both top-10 = automatic -1
- Clay = automatic -1

---

## ACCUMULATOR (optional)

If picks are from different tournaments → suggest accumulator.
Only include picks with Score 8+ in accumulators.

---

## WHY THIS MARKET WORKS

**Under 28.5 total games** has structural advantages over Set 1 markets:

1. **Binary driver:** P(straight sets) is the #1 predictor. If the favorite wins 2-0 = Under wins regardless of set scores.
2. **Large margin:** Expected games typically 20-23, line at 28.5 = 5-8 game buffer.
3. **Even 3-set losses are often Under:** 6-3, 3-6, 6-2 = 26 games = still Under.
4. **The model already simulates this:** `simulate_match()` gives exact `games_distribution`.

**When it fails:**
- Competitive matches that go 3 tight sets (7-5, 5-7, 7-6 = 36 games)
- Favorites who "switch off" after winning Set 1 and let opponents back in
- Clay matches with frequent break-backs extending sets

---

*Template version 1.0 — Focus on straight sets probability as primary filter. Under 28.5 = all straight sets + short 3-setters.*
