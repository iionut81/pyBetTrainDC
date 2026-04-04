# CoVe (Chain of Verification) — Multi-Market Accumulator
## Version 1.0 — Created 2026-04-04

---

## CONTEXT SETUP

I have a background of data analytics and worked as a Business Intelligence developer. I am transitioning to sport analyst. I need you to behave like a senior betting analyst.

**Task:** Analyze a mixed slate of betting markets (Football + Tennis), filter each through its own checklist, then combine the best picks into a ranked selection and optional accumulator.

**Markets covered:**
1. ⚽ Double Chance (1X / X2 / 12)
2. ⚽ Under Goals (U3.5 / U4.5)
3. ⚽ Under Corners (U11.5 / U12.5)
4. 🎾 WTA Over 7.5 Games Set 1
5. 🎾 WTA Under 12.5 Games Set 1 (No Tiebreak)

**What's missing:** Impact information, injuries, tactical lineups, psychological momentum, match context.

---

## HOW TO USE THIS TEMPLATE

1. Paste the full list of candidate picks (any mix of markets).
2. The analyst runs **each pick through its own market-specific checklist** (Section A–E below).
3. Every pick gets a **/10 score**.
4. Only picks scoring **7+** survive to Step 2 (research).
5. After research + verification, a **final ranking** picks the top selections.

---

# ═══════════════════════════════════════════
# MARKET-SPECIFIC CHECKLISTS
# ═══════════════════════════════════════════

---

## MARKET A: ⚽ DOUBLE CHANCE (1X / X2 / 12)

### WHY THIS MARKET

Double Chance removes one of three outcomes. The edge comes from finding matches where **two outcomes are structurally dominant** and the third is near-impossible. Typical profile: strong home team where even a draw is fine (1X), or a top team away where a draw is acceptable (X2).

### PRE-ANALYSIS: 15-SECOND CHECKLIST

**STEP A — FORM & DOMINANCE (most important)**
- Team covers DC in 8+ of last 10 → 🔥 GOLD
- Team covers DC in 6-7 of last 10 → ✅ GOOD
- Team covers DC in 5 of last 10 → ⚠️ Caution
- Team covers DC in < 5 of last 10 → ❌ DON'T BET

**STEP B — HOME/AWAY SPLIT**
- For 1X: home record W+D ≥ 80% of home matches → ✅
- For X2: away record W+D ≥ 60% of away matches → ✅
- For 12: both teams rarely draw (< 20% draw rate) → ✅
- Doesn't meet thresholds → ❌

**STEP C — OPPONENT QUALITY**
- Opponent in bottom 5 / relegation zone → 🔥 Boost
- Opponent mid-table, no motivation → ✅ Neutral
- Opponent fighting for Europe / title → ⚠️ Downgrade
- Opponent in form (W3+ streak) → ❌ DON'T BET

**STEP D — ODDS VALUE**
- Odds < 1.08 → ❌ No value (juice eaten by margin)
- Odds 1.08–1.15 → ⚠️ Thin — only for accumulators
- Odds 1.15–1.30 → ✅ Good range
- Odds 1.30+ → 🔥 Investigate deeper

**STEP E — CONTEXT**
- Derby? → ⚠️ Downgrade (chaotic, upsets common)
- Last match of season / nothing to play for? → ⚠️ Motivation risk
- Cup match? → ⚠️ Rotation risk
- Must-win for the DC side? → ✅ Upgrade

**PASS RULE:** DC coverage ≥ 6/10 recent AND opponent not in form AND odds ≥ 1.08.

### QUICK SCORE (/10):
- +3 form / DC coverage rate
- +2 home/away split
- +2 opponent quality
- +2 context favorable
- +1 gut feel (max 1!)

**Score 7+ = RECOMMEND | Score < 7 = PASS**

### AUTOMATIC PENALTIES:
- Derby = **-1**
- Cup match with rotation = **-1**
- Odds < 1.10 = **-1** (thin value)

### FINAL QUESTION:
**"Can the opponent realistically WIN outright (for 1X) / Can the home team realistically WIN outright (for X2)?"**
→ If YES and it's likely → ❌ DON'T BET
→ If NO or very unlikely → ✅ DC valid

---

## MARKET B: ⚽ UNDER GOALS (U3.5 / U4.5)

### PRE-ANALYSIS: 15-SECOND CHECKLIST

**STEP A — GOAL BASELINE (most important)**
- Both teams combined avg goals > 3.2 → ❌ DON'T BET
- One > 3.2, other < 2.5 → ⚠️ Caution
- Both between 2.2–3.0 → ✅ GOOD
- One team total < 2.2 → 🔥 GOLD

Logic: need at least one team that kills the rhythm.

**STEP B — ATTACK vs DEFENSE GAP**
- Both strong attacks vs weak defenses → ❌
- Big mismatch (top vs bottom) → ⚠️ (risk 4-0 / 5-0 blowout)
- Balanced or defensive teams → ✅
- One weak attack vs solid defense → 🔥

**STEP C — SHOT & xG PROFILE**
- Combined xG > 3.2 → ❌
- Combined xG 2.5–3.2 → ⚠️
- Combined xG 2.0–2.5 → ✅
- Combined xG < 2.0 → 🔥

**STEP D — GAME STATE RISK**
- Early goal likely? → ⚠️ (game opens up)
- One team heavy favorite? → ⚠️ (can snowball)
- Must-win match? → ❌ (late chaos)
- Low stakes / mid-table → ✅
- Knockout cautious match → 🔥

**STEP E — LEAGUE PROFILE**
- Bundesliga / Eredivisie → ❌ high scoring
- Premier League → ⚠️ mixed
- Serie A → ✅ strong for under
- Ligue 1 → ✅
- La Liga → 🟡 neutral

**PASS RULE:** Combined avg goals < 3.0, xG < 2.8, no extreme mismatch or chaos context.

### QUICK SCORE (/10):
- +3 goal baseline
- +2 xG profile
- +2 matchup balance
- +2 game state
- +1 intuition (max 1)

**Score 7+ = RECOMMEND | Score < 7 = PASS**

### LINE-SPECIFIC ADJUSTMENTS:
- **Under 3.5:** Strict. Need combined avg < 2.5 AND xG < 2.2. One high-xG team = DON'T BET.
- **Under 4.5:** More forgiving. Combined avg < 3.0 is enough. Risk is only blowout (5-0 type) or open game (3-2).

### FINAL QUESTION:
**"Can this match realistically explode to [line+1]+ goals?"**
→ If early goal → open game → chaos → ❌ DON'T BET
→ If controlled tempo + defensive structure + low xG → ✅ UNDER valid

---

## MARKET C: ⚽ UNDER CORNERS (U11.5 / U12.5)

### PRE-ANALYSIS: 15-SECOND CHECKLIST

**STEP A — CORNER BASELINE (most important)**
- Both teams avg corners FOR > 6 → ❌ DON'T BET (too offensive)
- One team > 6, other < 4 → ⚠️ Caution
- Both teams 3–5 → ✅ GOOD
- One team < 3.5 → 🔥 GOLD

Logic: need at least one team "dead offensively" from the wings.

**STEP B — TOTAL CORNER EXPECTATION**
- Expected total > 11.5 → ❌ DON'T BET
- Expected total 10–11.5 → ⚠️ borderline
- Expected total 9–10 → ✅ GOOD
- Expected total < 9 → 🔥 EXCELLENT

**STEP C — STYLE / TEMPO CHECK**
- Both crossing-heavy teams → ❌ (corner machine)
- One crossing team vs low block → ⚠️
- Both low tempo / static possession → ✅
- Direct play, few wing attacks → 🔥

**STEP D — GAME STATE RISK**
- Likely early goal? → ❌ (chasing team = more corners)
- Derby / high tension? → ⚠️ chaos
- One team big favorite? → ⚠️ (late pressure → corner spike)
- Mid-table / low motivation → ✅

**STEP E — LEAGUE PROFILE**
- Bundesliga / Eredivisie → ❌ high tempo
- Premier League → ⚠️ medium-high
- Serie A / Ligue 1 → ✅ lower tempo
- La Liga → 🟡 neutral
- Segunda División → ✅ generally lower pace

**PASS RULE:** At least ONE team avg corners < 4, expected total < 10.5, no high-tempo matchup.

### QUICK SCORE (/10):
- +3 corner baseline
- +2 expected total
- +2 style fit
- +2 game state
- +1 intuition (max 1)

**Score 7+ = RECOMMEND | Score < 7 = PASS**

### FINAL QUESTION:
**"Can this match realistically reach 12+ corners?"**
→ If one team dominates wide play / early goal leads to pressure / many crosses → ❌ DON'T BET
→ If slow buildup / central play / low urgency → ✅ UNDER valid

---

## MARKET D: 🎾 WTA OVER 7.5 GAMES SET 1

### PRE-ANALYSIS: 15-SECOND CHECKLIST

**STEP A — ANTI-BLOWOUT (most important)**
- Hold < 0.45 for either → ❌ DON'T BET
- Gap > 0.18 → ❌ DON'T BET
- If mismatch visible → STOP

**STEP B — MIN HOLD LEVEL**
- Both < 0.50 → ❌ Chaos → avoid
- One 0.50–0.60, other similar → ✅ Good
- Both > 0.60 → 🔥 Very good

**STEP C — MOMENTUM CHECK (Slow Starter)**
- Did player lose Set 1 in last 2+ consecutive matches? → ⚠️ Flag as "Slow Starter", apply **-2pp** penalty
- If slow starter + opponent in dominant form → risk of 6-0/6-1 → ❌ DON'T BET

**STEP D — CAN IT REACH 3-3?**
- Yes → ✅ OK
- No (one player too weak) → ❌ DON'T BET

**STEP E — SURFACE**
- Hard → ✅ Best for Over
- Clay → ⚠️ Downgrade
- Grass → 🔥 Over heaven

**PASS RULE:** No mismatch, both can hold 2-3 service games, no hold < 0.45.

### DEFINITION:
**Over 7.5** = Set 1 has **≥ 8 total games** (6-2, 2-6, 6-3, 6-4, 7-5, 7-6…). Under 7.5 = only ≤ 7 games: 6-0, 0-6, 6-1, 1-6.

### QUICK SCORE (/10):
- +3 hold quality
- +2 gap appropriate
- +2 surface fits market
- +2 context favorable
- +1 gut feel (max 1!)

**Score 7+ = RECOMMEND | Score < 7 = PASS**

### AUTOMATIC PENALTIES:
- Finals = **-1**
- Both top-10 = **-1**
- tennisabstract data source = **-1**

### FINAL QUESTION:
**"Is a bagel / breadstick set (6-0 / 6-1 / 0-6 / 1-6) realistically likely?"**
→ If YES → ❌ DON'T BET Over
→ If NO → ✅ Over valid

Note: **6-2** = 8 games → WINS Over 7.5. Do not list it as a losing scenario.

---

## MARKET E: 🎾 WTA UNDER 12.5 GAMES SET 1 (No Tiebreak)

### PRE-ANALYSIS: 15-SECOND CHECKLIST

**STEP A — HOLD CHECK (most important)**
- Both > 0.65 → ❌ DON'T BET (both hold too well → TB likely)
- One 0.50–0.65 → ⚠️ Caution
- One < 0.50 → ✅ GOOD
- One < 0.42 → 🔥 GOLD

**STEP B — SURFACE & LEVEL**
- Clay WTA 500/1000/GS → ✅ Boost (more breaks)
- Clay WTA 250 or lower → ❌ HARD PASS (too volatile)
- Hard → 🟡 Neutral
- Grass → ❌ TB risk → avoid

**STEP C — GAP CHECK**
- Gap < 0.12 → ❌ Avoid (too balanced → TB risk)
- Gap 0.12–0.15 → 🟡 OK
- Gap 0.15+ → ✅ Good
- Gap 0.20+ → 🔥 Excellent

**STEP D — CONTEXT**
- Final? → ⚠️ Downgrade (tighter, more holds)
- Both top players? → ⚠️ Downgrade
- One unstable (DFs, weak serve)? → ✅ Upgrade

**PASS RULE:** Need at least one hold < 0.50 AND gap > 0.12 AND not WTA 250 clay AND not elite hard court match.

### DEFINITION:
**Under 12.5** = No 6-6 (no tiebreak at 6-6). `p_set_under_12_5 = 1 - P(6-6)`. Check if bookmaker defines the line differently.

### QUICK SCORE (/10):
- +3 hold quality
- +2 gap appropriate
- +2 surface fits market
- +2 context favorable
- +1 gut feel (max 1!)

**Score 7+ = RECOMMEND | Score < 7 = PASS**

### AUTOMATIC PENALTIES:
- Finals = **-1**
- Both top-10 = **-1**
- tennisabstract data source = **-1**

### FINAL QUESTION:
**"Can BOTH players hold serve 5-6 times to reach 6-6?"**
→ If YES → ❌ DON'T BET Under
→ If NO → ✅ Under valid

---

# ═══════════════════════════════════════════
# UNIFIED ANALYSIS PIPELINE
# ═══════════════════════════════════════════

---

## STEP 1: Run All Checklists

For every candidate pick:
1. Identify the market (A/B/C/D/E).
2. Run the matching checklist above.
3. Assign score /10.
4. **Eliminate everything < 7** immediately. Do not research eliminated picks.

Output table:

| # | Match | Market | Score /10 | Pass? |
|---|-------|--------|-----------|-------|
| 1 | ... | DC 1X | ... | ✅/❌ |
| 2 | ... | U3.5 | ... | ✅/❌ |

---

## STEP 2: External Research + Verification

**Only for picks that scored 7+.**

### Football research:
- Last 3-5 match results (scores, corners, cards)
- League standings & motivation
- Key absences (injuries, suspensions) — **cite source**
- Tactical setup (formation, style)
- H2H (context only, not main argument)

### Tennis research:
- Last 2-3 SET SCORES (Set 1 game count specifically)
- Tiebreak frequency in recent matches
- Slow Starter check: lost Set 1 in last 2+ matches? → **-2pp penalty**
- Injuries, fitness concerns
- Playing style on current surface
- H2H (context only)

**MANDATORY:** Cite source URL for every fact.

**RULES:**
- Research can upgrade model probability by MAX **+10pp** (not more)
- Never say "impossible" — use probabilities
- H2H with < 10 meetings = context only, not statistical evidence
- Small sample / weak league = flag as less reliable

---

## STEP 3: Self-Verification (Honest Answers)

Answer each question honestly:

1. "Did I analyze objectively the specific numbers for each market?"
2. "Did I crosscheck with internet? Are sources reliable?"
3. "Did I make assumptions or just analyzed?"
4. "Did I apply market-specific penalties (slow starter, clay, league profile)?"
5. "Did I cap my research upgrade at +10pp?"
6. "Did I consider match context (final, pressure, level, motivation)?"
7. **"Flag when model data contradicts external research"**
8. **"Did I mix up market definitions?"** (Over 7.5 ≠ Under 12.5 — different filters!)

### MARKET-SPECIFIC FINAL QUESTIONS:

- **Double Chance:** "Can the opponent realistically WIN outright against my DC side?"
- **Under Goals:** "Can this match realistically explode to [line+1]+ goals?"
- **Under Corners:** "Can this match realistically reach 12+ corners?"
- **Over 7.5 Set 1:** "Is a bagel/breadstick (6-0/6-1) realistically likely?"
- **Under 12.5 Set 1:** "Can BOTH players hold serve 5-6 times to reach 6-6?"

---

## STEP 4: Corrections Table

| Pick | Market | Model | Research | Score /10 | Action | Reason |
|------|--------|-------|----------|-----------|--------|--------|
| ... | DC 1X | ... | ... | ... | BET/PASS | ... |

Remove or flag anything:
- Research upgrade > +10pp → cap it
- H2H used as main argument → downgrade to context
- "impossible/certain" language → replace with probabilities
- Match context ignored → apply penalties
- Slow starter penalty not applied → apply it
- Market definitions confused → fix

---

## STEP 5: Final Picks (ranked by checklist score)

For each recommended pick include:
- **Market** (DC / U Goals / U Corners / O7.5 / U12.5)
- **Checklist score** (/10)
- **Model probability** (if available)
- **Research-adjusted probability** (capped at model +10pp)
- **Fair odds** (1 / research_adj_probability)
- **Market odds** (as provided)
- **Edge** (market odds − fair odds, or implied prob vs model prob)
- **Key stat** (one sentence)
- **How I lose this bet** (one sentence)
- **Source** (URL)

### CONFIDENCE LEVELS:
- Score 9-10: **HIGH** confidence
- Score 7-8: **MODERATE** confidence
- Score < 7: **DO NOT RECOMMEND**

### CROSS-MARKET RULES:
- DC at odds < 1.10 = max MODERATE confidence (thin value singles, accumulator-only)
- Football lower leagues = max MODERATE confidence
- WTA 250 on clay = max MODERATE confidence for Under 12.5
- Finals = automatic **-1**
- Both top-10 (tennis) = automatic **-1**
- tennisabstract data source = automatic **-1**

---

## STEP 6: Top 2 Picks

From all surviving picks across all markets, select the **top 2** ranked by:

1. Checklist score (highest first)
2. Edge vs market odds (biggest edge first)
3. Research confidence (most supported first)

Present as:

### 🥇 PICK 1
**[Match] — [Market] @ [Odds]**
Score: X/10 | Confidence: HIGH/MODERATE
Model: X% | Research: X% | Fair odds: X.XX
Key stat: ...
Risk: ...

### 🥈 PICK 2
**[Match] — [Market] @ [Odds]**
Score: X/10 | Confidence: HIGH/MODERATE
Model: X% | Research: X% | Fair odds: X.XX
Key stat: ...
Risk: ...

---

## ACCUMULATOR (optional)

If the top picks are from **different events** (different matches / different sports) → suggest accumulator.

**Rules:**
- Only include picks with Score **8+** in accumulators
- Max 3 legs (more = too much variance)
- DC legs at odds < 1.12 are acceptable as "anchor" legs only
- Calculate combined odds
- Combined fair odds must be < combined market odds (positive expected value)
- Never combine two picks from the same match

### Format:

| Leg | Match | Market | Odds | Score |
|-----|-------|--------|------|-------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| **Combined** | | | **X.XX** | |

---

## INVERSE MARKET WARNING

Some markets are structurally **inverse** on the same match. Flag explicitly:

- **Over 7.5 + Under 12.5** on the same tennis match: balanced holds favor Over 7.5 but increase TB risk (bad for Under 12.5). Large gap favors Under 12.5 but risks blowout (bad for Over 7.5). Analyze both but acknowledge the tension.
- **Under Goals + Double Chance** on the same football match: a dominant team (good for DC) may also create a blowout (bad for Under). Check whether the DC side wins low-scoring or high-scoring.

---

*Template version 1.0 — Multi-market CoVe unifying Double Chance, Under Goals, Under Corners, WTA Over 7.5, WTA Under 12.5 into a single ranked analysis pipeline with top-2 selection and optional accumulator.*
