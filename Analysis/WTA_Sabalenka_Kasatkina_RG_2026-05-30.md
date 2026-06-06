# CoVe — Aryna Sabalenka vs Daria Kasatkina
**Competiție:** Roland Garros 2026 — R3
**Data:** 30 Mai 2026 | 13:30
**Piață analizată:** Over 7.5 games Set 1
**Verdict:** ❌ PASS — HARD PASS structural

---

## AVERTISMENT PRELIMINAR

Modelul flaghează această piață cu **HARD PASS structural** la Step A (gap > 0.20). Analiza este completă și transparentă — includ motivele exacte și argumentele research contradictorii.

---

## DATE MODEL

| Metrică | Sabalenka (A) | Kasatkina (B) |
|---|---|---|
| p_hold | **0.754** | **0.511** |
| **gap (hold_asym)** | **0.2427** | ⚠️ **> 0.20 = HARD PASS O7.5** |
| p_cal_adj O7.5 | **74.54%** | — |
| blowout_score | **10** (maxim!) | — |
| competitive_set | False | — |
| good_to_go | No | — |
| O7.5 model flag | no | — |
| U12.5 model flag | YES | — |
| tb_p_raw / tb_p_cal | 0.1056 / 10.2% | — |
| expected_games | **21.1** | — |
| days_rest | 2 | 2 |
| fatigue | False | False |

---

## PROFILURI JUCĂTOARE

### Aryna Sabalenka | #1 mondial

**Rezultate Roland Garros 2026 (Set1 scores):**

| Meci | Scor | Set1 | Games S1 | O7.5? |
|---|---|---|---|---|
| R1 vs Bouzas Maneiro | 6-4, 6-2 | **6-4** | **10** | ✅ |
| R2 vs Jacquemot | 7-5, 6-2 | **7-5** | **12** | ✅ |
| **Pattern RG 2026** | | | | **2/2 ✅** |

**Context:**
- 45 winners / **31 unforced errors** în R2 — inconsistentă
- Eliminată în R2 la Roma → clay season nu perfect
- A fost 5-5 vs Jacquemot în Set1 înainte de a trage

### Daria Kasatkina | Rank ~53

**Rezultate Roland Garros 2026 (Set1 scores):**

| Meci | Scor | Set1 | Games S1 | O7.5? |
|---|---|---|---|---|
| R1 vs Sonmez | 6-4, 6-4 | **6-4** | **10** | ✅ |
| R2 vs Bandecchi | 7-5, 7-6(13/11) | **7-5** | **12** | ✅ |
| **Pattern RG 2026** | | | | **2/2 ✅** |

**Context R2 — extraordinar:**
- Bandecchi conducea **5-1 în Set1** + set point la 5-4 → Kasatkina a salvat
- Set2: a salvat **7 set points** în tiebreak de **17 minute** (11-13 TB!)
- Mentalitate de luptătoare, nu cedează niciodată

**Formă clay 2026:** **12/14 victorii** — "clickul" s-a întors.

---

## H2H — CRITIC

| An | Turneu | Suprafață | Scor | Set1 | O7.5 S1? |
|---|---|---|---|---|---|
| **2020** | **Roland Garros** | **Clay** | **Sabalenka def.** | **6-0** | **❌ LOSE** |
| 2023 | Cincinnati | Hard | Kasatkina def. | — | — |
| Ultimele 4 | — | — | Sabalenka 4-0 | — | — |

**Record total: Sabalenka 7-2 | Clay: 3-0 Sabalenka**

**⚠️ 2020 Roland Garros: Sabalenka a baghelatuat Kasatkina 6-0 în Set1 = 6 gameuri → O7.5 PIERDE.**

---

## ANALIZA O7.5 — FILTRE

### STEP A — Anti-blowout (DECISIVE)

| Criteriu | Valoare | Status |
|---|---|---|
| Hold < 0.45 either | 0.511 > 0.45 | ✅ |
| **Gap > 0.20** | **0.2427 > 0.20** | **❌ HARD PASS** |

**Gap = 0.24 depășește pragul → HARD PASS automat.**

### STEP B — Hold stability
Kasatkina 0.511 < 0.60 → Risk ⚠️

### STEP D — Surface
Clay → downgrade

---

## ARGUMENTE PRO O7.5 (research)

1. **4/4 seturi la RG 2026 = O7.5** (ambele jucătoare: 6-4, 7-5 tipic)
2. **Kasatkina nu cedează** → 7 set points salvate în R2, mentalitate excepțională
3. **Sabalenka oscilantă** → 31 UE în R2, eliminată la Roma R2
4. **expected_games = 21.1** → ~10.5 gameuri/set = 6-4 tipic = O7.5 WINS

---

## ARGUMENTE CONTRA O7.5

1. **Step A HARD PASS — gap 0.24 > 0.20** (filtru automat)
2. **H2H RG 2020: 6-0 în Set1** → precedent direct de blowout la Paris pe zgură
3. **blowout_score = 10** → cel mai mare risc blowout din toate meciurile analizate recent
4. **p_cal_adj = 74.54%** → cu 7.46pp sub pragul de 82%
5. **Kasatkina hold = 0.511** → se rupe des față de o servitoare dominantă (Sabalenka 0.754)

---

## CALCUL PROBABILITATE

| Scenariu | Probabilitate | Games S1 | O7.5? |
|---|---|---|---|
| Sabalenka câștigă 6-3 sau 6-4 | ~40% | 9-10 | ✅ |
| Sabalenka câștigă 6-2 | ~20% | 8 | ✅ |
| **Sabalenka câștigă 6-1 sau 6-0** | **~18%** | **6-7** | **❌** |
| Set competitiv 7-5 sau 7-6 | ~15% | 12-13 | ✅ |
| Kasatkina câștigă Set1 | ~7% | ~10 | ✅ |

| Factor | Ajustare |
|---|---|
| Base model | 74.54% |
| Step A HARD PASS structural | trigger |
| Gap > 0.20 | -10pp |
| H2H 6-0 la RG 2020 | -4pp |
| 4/4 seturi O7.5 la RG 2026 | +3pp |
| Kasatkina 12/14 clay wins | +2pp |
| **p_research final** | **~65-68%** |

---

## SCORING O7.5

| Criteriu | Puncte | Notă |
|---|---|---|
| Hold structure (Kasatkina 0.51 = Risk) | 1/3 | Sub 0.60 |
| Matchup fit (gap > 0.20 = HARD PASS) | 0/2 | Filtru eșuat |
| Gap quality (0.24 = prea mare) | 0/2 | Structural PASS |
| Context (RG pattern O7.5, toughness) | 1.5/2 | Compensare parțială |
| Feel | 0.5/1 | Risc real |
| **Total** | **3/10** | — |

---

## VERDICT FINAL

| Piață | Research p | Score | Odds min | Acțiune |
|---|---|---|---|---|
| **Sabalenka-Kasatkina O7.5** | **~66%** | **3/10** | — | ❌ **PASS** |

**3 motive structurale de PASS:**
1. Step A HARD PASS: gap = 0.24 > 0.20 → filtrul automat eșuat
2. blowout_score = 10 + p_cal_adj = 74.54% → maximum risc, minimum model confidence
3. H2H RG 2020: 6-0 în Set1 → precedent direct de blowout pe aceeași suprafață

**La odds tipice 1.25 (implică 80%), avem negative edge de ~-13pp.**

---

## CONCLUZIE PRACTICĂ

Kasatkina joacă bine și nu cedează ușor. Totuși, diferența structurală de serviciu (75% vs 51%) face ca Sabalenka să poată câștiga Set1 în 6-1 sau 6-2 oricând — H2H clay confirmă cu 6-0 precedent la Roland Garros.

**Regula din 29 mai confirmată:** sub 82% model + HARD PASS structural = PASS indiferent de research.

---

## SURSE

- [Sabalenka R1 RG 2026 — Roland Garros Official](https://www.rolandgarros.com/en-us/article/2026-edition-r1-sabalenka-bouzas-maneiro)
- [Sabalenka R2 stats — WTA Official](https://www.wtatennis.com/news/4510839/by-the-numbers-sabalenka-powers-into-third-round-of-roland-garros)
- [Kasatkina R2 comeback — Tennis Australia](https://www.tennis.com.au/fan-zone/news/2026/05/29/roland-garros-kasatkina-triumphs-to-set-up-sabalenka-showdown)
- [Kasatkina form clay 2026 — Tennis Australia](https://www.tennis.com.au/fan-zone/news/2026/05/30/daria-kasatkina-back-in-the-mix)
- [Kasatkina vs Bandecchi score — Yahoo Sports](https://sports.yahoo.com/tennis/2026/roland-garros/womens-singles/daria-kasatkina-susan-bandecchi-15751817/)
- [Sabalenka-Kasatkina H2H — BeatzeBook](https://beatzebook.ai/en/compare/kasatkina-vs-sabalenka/)
- [Preview Sabalenka-Kasatkina — Sports Mole](https://www.sportsmole.co.uk/tennis/french-open/preview/aryna-sabalenka-vs-daria-kasatkinaprediction-form-head-to-head_598338.html)
- [Sabalenka RG 2026 seed report — SI](https://www.si.com/tennis/2026-roland-garros-womens-seed-report-aryna-sabalenka-favorite-open-field)
