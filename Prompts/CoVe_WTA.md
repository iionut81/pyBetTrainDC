# CoVe (Chain of Verification) — Tennis Set 1 Markets
## Version 2.3 — Updated 2026-04-04

---

## CONTEXT SETUP

I have a background of data analytics and worked as a Business Intelligence developer. I am transitioning to sport analyst. I need you to behave like a senior betting analyst.

**Task:** Analyze the provided data and gather information from the internet for further analysis and recommendation.

**What happened:** New daily fresh data received.

**What's missing:** Impact information, injuries, psychological momentum, match context.

---

## PRE-ANALYSIS: 15-SECOND CHECKLIST

Before any analysis, run this checklist. If it doesn't pass → DON'T BET.

### FOR UNDER 12.5 (No Tiebreak):

**STEP A — HOLD CHECK (most important)**
- Both > 0.65 → ❌ DON'T BET (both hold too well → TB likely)
- One 0.50–0.65 → ⚠️ Caution
- One < 0.50 → ✅ GOOD
- One < 0.42 → 🔥 GOLD

**STEP B — SURFACE & LEVEL**
- Clay WTA 500/1000/GS → ✅ Boost (more breaks)
- Clay WTA 250 or lower → ❌ HARD PASS (too volatile for Under 12.5)
- Hard → 🟡 Neutral
- Grass → ❌ TB risk → avoid for Under

**STEP C — GAP CHECK**
- < 0.12 → ❌ Avoid (too balanced → TB risk)
- 0.12–0.15 → 🟡 OK
- 0.15+ → ✅ Good
- 0.20+ → 🔥 Excellent

**STEP D — CONTEXT**
- Final? → ⚠️ Downgrade (tighter, more holds)
- Both top players? → ⚠️ Downgrade
- One unstable (DFs, weak serve)? → ✅ Upgrade

**PASS/FAIL:** Need at least one hold < 0.50 AND gap > 0.12 AND not WTA 250 clay AND not elite hard court match.

### FOR OVER 7.5:

**STEP A — ANTI-BLOWOUT**
- Hold < 0.45 for either → ❌ DON'T BET
- Gap > 0.18 → ❌ DON'T BET
- If mismatch visible → STOP

**STEP B — MIN HOLD LEVEL**
- Both < 0.50 → ❌ Chaos → avoid
- One 0.50–0.60, other similar → ✅ Good
- Both > 0.60 → 🔥 Very good

**STEP C — MOMENTUM CHECK (Slow Starter)**
- Did player lose Set 1 in last 2+ consecutive matches? → ⚠️ Flag as "Slow Starter", apply **-2pp** penalty
- If slow starter + opponent in dominant form → risk of 6-0/6-1 increases → ❌ DON'T BET

**STEP D — CAN IT REACH 3-3?**
- Yes → ✅ OK
- No (one player too weak) → ❌ DON'T BET

**STEP E — SURFACE**
- Hard → ✅ Best for Over
- Clay → ⚠️ Downgrade
- Grass → 🔥 Over heaven

**PASS/FAIL:** No mismatch, both can hold 2-3 service games, no hold < 0.45.

### QUICK SCORE (/10):
- +3 hold quality
- +2 gap appropriate
- +2 surface fits market
- +2 context favorable
- +1 gut feel (max 1!)

**Score 7+ = RECOMMEND | Score < 7 = DO NOT RECOMMEND (PASS)**

### AUTOMATIC PENALTIES:
- Finals = **-1** on checklist score
- Both top-10 players = **-1** on checklist score
- tennisabstract/tennisabstract data source = **-1** on checklist score

---

## STEP 1: Analyze the provided data

List all matches with: holds, gap, expected games, p_cal, p_cal_adj, fair odds, blowout score.
Run the 15-second checklist on each match. Score /10.

---

## STEP 2: External Research + Verification

For each match that passes the checklist, gather from internet:
- **Set 1 History:** Did any player lose Set 1 in last 2+ matches? → Flag as **Slow Starter**, apply -2pp
- Last 2-3 SET SCORES (specifically Set 1 game count)
- Tiebreak frequency in recent matches
- Injuries, fitness concerns
- Playing style on current surface
- H2H if available (use as CONTEXT only, not as main argument)

**MANDATORY:** Cite source URL for every fact.

**RULES:**
- Research can upgrade model probability by MAX +10pp (not more)
- Never say "impossible" — use "TB probability ~X%"
- H2H with < 10 meetings = context only, not statistical evidence
- ITF/tennisabstract data = flag as less reliable

---

## STEP 3: Self-Verification (Honest Answers)

Answer each question honestly:

1. "Did I analyze objectively the specific numbers?"
2. "Did I crosscheck with internet? Are sources reliable?"
3. "Did I make assumptions or just analyzed?"
4. "Did I apply the -2pp penalty for slow starters?"
5. "Is this a WTA 250 on Clay? (If yes, did I skip Under 12.5?)"
6. **"Flag when model data contradicts external research"**
7. **"Did I cap my research upgrade at +10pp?"**
8. **"Did I consider match context (final, pressure, level)?"**

### THE FINAL QUESTION (most important):

**For UNDER 12.5:** "Can BOTH players hold serve 5-6 times to reach 6-6?"
→ If YES → don't bet Under
→ If NO → Under is valid

**For OVER 7.5:** Align with model (`wta_markov.py`): **Over 7.5** = set 1 has **≥8 total games** (e.g. **6-2**, 2-6, 6-3, 6-4, 7-5, 7-6…). **Under 7.5** on this line = only **≤7 games**: **6-0**, **0-6**, **6-1**, **1-6**.

**Question:** "Is a **bagel / breadstick** set (6-0 / 6-1 / 0-6 / 1-6) realistically likely — i.e. one side too weak to extend the set to 8+ games?"
→ If **YES** → don't bet Over
→ If **NO** → Over is valid

**Do not** list **6-2** here: 6+2 = **8 games** → that outcome **wins** Over 7.5, it is not a losing scenario for this market.

---

## STEP 4: Corrections

Write table: Pick | Model | Research | Checklist Score | Action | Reason

Remove or flag anything:
- Where research upgrade exceeds +10pp → cap it
- Where H2H was used as main argument → downgrade to context
- Where "impossible/certain" language was used → replace with probabilities
- Where match context (final, pressure) was ignored
- Where slow starter penalty was not applied → apply it

---

## STEP 5: Final Picks (ranked by checklist score)

For each recommended pick include:
- **Checklist score** (/10)
- **Model probability** (p_cal_adj)
- **Research probability** (capped at model +10pp)
- **Fair odds**
- **Key stat** (one sentence)
- **How I lose this bet** (one sentence)
- **Source** (URL)

### CONFIDENCE LEVELS:
- Score 9-10: HIGH confidence
- Score 7-8: MODERATE confidence
- Score < 7: **DO NOT RECOMMEND**

### RULES:
- Max "HIGH confidence" on clay WTA 125/250 = MODERATE (cap it)
- Finals = automatic -1 on checklist score
- Both top-10 players = automatic -1 on checklist score
- tennisabstract/tennisabstract data source = automatic -1 on checklist score

---

## ACCUMULATOR (optional)

If picks are from different tournaments/markets → suggest accumulator with combined odds.
Only include picks with Score 8+ in accumulators.

---

*Template version 2.3 — v2.1 base + WTA 250 Clay hard-pass for Under, Slow Starter -2pp penalty, minimum score 7 threshold, bagel/breadstick final question with 0-6/1-6 variants*