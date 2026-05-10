# CoVe Analysis — WTA Under 28.5 Total Games
**Date:** 2026-05-06 | **Template:** CoVe_WTA_Under.28.5.md v1.0
**Tournament:** Internazionali BNL d'Italia ROME WTA 1000 + Istanbul WTA 125 Clay

---

## SCREENING — 25 meciuri procesate

### Pre-filtre eliminatorii

| Criteriu | Eliminați |
|----------|-----------|
| Oră neconfirmată (23:59) | Townsend/Brancaccio, Pliskova/Bouzas, Boulter/Lys, Basiletti/Tomljanovic, Siniakova/Boisson, Putintseva/Valentova, Sierra/Korpatsch, Golubic/Urgesi, Siegemund/Bejlek, Jeanjean/Haddad Maia, Galfi/Potapova, Pigato/Grant, Stearns/Tjen, Selekhmeteva/Masarova, Starodubtseva/Waltert (r7) |
| UNSTABLE (extreme p_match) | Selekhmeteva/Masarova (p=0.027), Starodubtseva/Waltert r7 (p=0.008) |
| P(winner) < 75% | Eala/Frech (59.2%), Tagger/Sakkari (62%), Starodubtseva/Waltert Istanbul (55.6%) |

**Candidați cu oră confirmată + P(win) ≥ 75%:**

| Meci | Ora | P(win) | Expected | Margin | Gap | Sursă |
|------|-----|--------|----------|--------|-----|-------|
| Ostapenko vs Stefanini | 11:00+02:00 | 81.7% | 22.49 | 6.01 | 0.167 | sackmann |
| Udvardy vs Korneeva | 11:00+02:00 | 77.6% | 23.13 | 5.37 | 0.134 | sackmann |
| Linette vs T.Maria | 11:00+02:00 | 76.5% | 23.34 | 5.16 | 0.122 | sackmann |
| **Cocciaretto vs Kraus** | **19:00+02:00** | **92.9%** | **20.56** | **7.94** | **0.264** | sackmann |
| Gasanova vs Kudermetova | 13:00+03:00 | 76.9% | 23.13 | 5.37 | 0.137 | tennisabstract |
| Vekic vs Cengiz | 14:30+03:00 | 88.4% | 21.91 | 6.59 | 0.201 | sackmann |

---

## STEP 1 — CHECKLIST DETALIAT

### Cocciaretto vs Kraus (Rome, Clay, 19:00)

| Step | Check | Status | Score |
|------|-------|--------|-------|
| A — P(straight sets) | ~87% estimate (p_markov=0.929, hold gap=0.264) | >80% 🔥 | +3 |
| B — P(winner) | 92.9% > 85% | Dominant 🔥 | +2 |
| C — Margin | 28.5 − 20.56 = **7.94** > 6 | Excelent 🔥 | +2 |
| D — Hold gap | 0.264 > 0.15 | Clear mismatch ✅ | — |
| E — Context | Nu finală, nu ambele top-10 | Favorabil | +2 |
| Penalizare clay | Clay = −1 | | −1 |
| **TOTAL** | | | **8/10** |

| Metric | Valoare |
|--------|---------|
| p_hold_a (Cocciaretto) | 0.7102 |
| p_hold_b (Kraus) | 0.4464 |
| p_markov | 0.9289 |
| p_elo | 0.6887 |
| expected_games (total) | 20.56 |
| blowout_score | 10/10 |
| data_source | sackmann/sackmann |

---

### Vekic vs Cengiz (Istanbul WTA 125, Clay, 14:30)

| Step | Check | Status | Score |
|------|-------|--------|-------|
| A — P(straight sets) | ~82% (p_markov=0.884, gap=0.201) | >80% 🔥 | +3 |
| B — P(winner) | 88.4% > 85% | Dominant 🔥 | +2 |
| C — Margin | 28.5 − 21.91 = **6.59** > 6 | Excelent 🔥 | +2 |
| D — Hold gap | 0.201 > 0.15 | Clear ✅ | — |
| E — Context | WTA 125, Cengiz joacă acasă Istanbul | Neutru | +1 |
| Penalizare clay | −1 | | −1 |
| **TOTAL raw** | | | **7/10** |

---

### Ostapenko vs Stefanini (Rome, Clay, 11:00) — PASS

- P(straight sets) ~74% (borderline), Ostapenko 2-3 clay 2026
- Scor raw: 7/10 → cu research downgrade: **6/10 → PASS**

### Udvardy/Korneeva, Linette/T.Maria, Gasanova/Kudermetova — PASS

- P(straight sets) <70% → Step A fail
- Gasanova/Kudermetova: tennisabstract −1 → 4/10

---

## STEP 2 — EXTERNAL RESEARCH

### Cocciaretto vs Kraus

**Cocciaretto — forma:**
- Câștigat Hobart WTA 250 în ianuarie 2026 — primul titlu WTA
- Post-Hobart: câștigă aproape toate meciurile în **straight sets**
- Italiancă pe clay la Roma = avantaj de teren de acasă structural
- Ajustare: **+3pp** (closing pattern în straight sets)

**Kraus — profil underdog:**
- Career high WTA #99 (23 februarie 2026)
- 2 finale WTA 125 în 2026 — **pierde ambele finale în straight sets**
- Debut WTA 1000 main draw = Miami 2026, pierdut R1 vs Alycia Parks
- Această partidă este un salt de nivel față de WTA 125
- Ajustare: **−2pp** (mai competitivă decât modelul la nivel WTA 125, mai puțin la WTA 1000)

**Net research adj: +1pp → p(Under 28.5) ≈ 89%**

Sources:
- [Sinja Kraus — Wikipedia](https://en.wikipedia.org/wiki/Sinja_Kraus)
- [Sinja Kraus WTA Official](https://www.wtatennis.com/players/327061/sinja-kraus)
- [Italian Open 2026 Results | WTA Tournament Centre](https://rallyher.com/italian-open-2026-wta-results-draw-scores-schedule/)
- [WTA Rome Day 2 Predictions | Last Word on Sports](https://lastwordonsports.com/tennis/2026/05/05/wta-rome-day-2-predictions-eala-frech/)

---

### Vekic vs Cengiz

**Vekic R1 Istanbul (05.05.2026):** vs Aliona Falei (#231 WTA) → **6-2, 6-7(3), 6-4**
- Total: 8 + 13 + 10 = **31 games → OVER 28.5** ⚠️
- Template rule: "player who went to 3 sets in last 2+ matches = downgrade"
- Semnificativ: a pierdut un set vs o jucătoare #231 → nu este în modul straight-sets
- Ajustare: **−5pp → p(Under 28.5) ≈ 77%**
- Scor revizuit: **6/10 → PASS**

Sources:
- [Vekic R1 Istanbul 3 seturi | Germanijak.hr](https://www.germanijak.hr/ostalo/vijesti/donna-vekic-uspjesno-otvorila-istanbul-break-u-trecem-setu-presudio-bjeloruskoj-suparnici/161233)
- [Istanbul Cup 2026 — Wikipedia](https://en.wikipedia.org/wiki/2026_%C4%B0stanbul_Cup)
- [HTS.hr — Vekic Istanbul R2](https://hts.hr/donna-vekic-u-2-kolu-wta-125-turnira-u-istanbulu/)

---

### Ostapenko vs Stefanini

**Ostapenko 2026 clay record: 2-3**
- Madrid R3 (26 apr): pierdut vs Potapova **4-6, 6-4, 6-4** = 3 seturi, 29 games
- Madrid R2: bătut Waltert în meci dificil (3 seturi)
- Stuttgart: pierdut vs Andreeva
- Tendință clară: merge în 3 seturi pe clay sistematic
- Ajustare: **−5pp → Scor: 6/10 → PASS**

Sources:
- [Ostapenko vs Stefanini preview | Tennis Tonic](https://tennistonic.com/tennis-news/993619/h2h-prediction-of-jelena-ostapenko-vs-lucrezia-stefanini-in-rome-with-odds-preview-pick-6th-may-2026/)
- [WTA Rome Day 2 Predictions | Last Word on Sports](https://lastwordonsports.com/tennis/2026/05/05/wta-rome-day-2-predictions-stefanini-ostapenko/)

---

## STEP 3 — SELF-VERIFICATION (Cocciaretto)

- [x] P(straight sets) verificat din model (~87% via p_markov + gap)
- [x] Closing pattern Cocciaretto: straight sets post-Hobart ✅
- [x] Under 28.5 câștigă ALL 2-seturi (max 26 < 28.5) ✅
- [x] Kraus comeback ability: scăzut la WTA 1000, 2 finale pierdute în straight sets ✅
- [x] Cap ±10pp: +1pp aplicat ✅
- [x] Clay penalty aplicat: −1 în scor ✅

**FINAL QUESTION:** "Can Kraus realistically WIN a set?"

Kraus ține serviciul doar 44.6% → pierde serviciul >55% din game-uri. Ca să câștige un set trebuie să țină serviciul constant ȘI să facă break. Probabilitate scenariul de set câștigat: ~13%. Și dacă se întâmplă, cu hold rates atât de asimetrice, al 3-lea set va fi tot scurt (6-2 sau 6-3 tipic). → Under 28.5 rămâne valid chiar și în scenariu 3-set scurt.

---

## STEP 4 — CORRECTIONS TABLE

| Pick | Model P | Research adj | Scor | Action | Motiv |
|------|---------|--------------|------|--------|-------|
| Cocciaretto vs Kraus U28.5 | ~88% | +1pp → **89%** | **8/10** | ✅ BET | Straight-sets closer post-Hobart, Kraus step-up WTA 1000, margin 7.94, gap 0.264 |
| Vekic vs Cengiz U28.5 | ~82% | −5pp → **77%** | **6/10** | ❌ PASS | Vekic 3 seturi vs #231 ieri (31 games) = trend negativ clar |
| Ostapenko vs Stefanini U28.5 | ~76% | −5pp → **71%** | **6/10** | ❌ PASS | Ostapenko 2-3 clay 2026, 2+ meciuri 3 seturi consecutive |
| Udvardy/Korneeva, Linette/Maria | <75% | — | <6/10 | ❌ PASS | P(straight sets) <70%, criteriu Step A fail |
| Gasanova/Kudermetova | 76.9% | tennisabstract −1 | 4/10 | ❌ PASS | Sursă de date penalizată |

---

## STEP 5 — FINAL PICK

### ✅ Cocciaretto vs Kraus — Under 28.5 Total Games | Score: 8/10 — MODERATE

```
P(straight sets):     ~87%
P(Under 28.5):        ~89% (research adj +1pp)
Expected total games: 20.56 | Margin to line: 7.94
Fair odds:            1/0.89 = 1.124
Hold A (Cocciaretto): 71.0% | Hold B (Kraus): 44.6% | Gap: 0.264
Tournament:           Rome WTA 1000, Clay, Q3 / R1
Ora:                  19:00+02:00 (confirmată)
Data source:          sackmann/sackmann ✅
```

**Key stat:** Cocciaretto ține serviciul 71% + Kraus pierde serviciul >55% din game-uri → Kraus nu poate construi seturi lungi. Expected 20.56 total games = buffer de aproape 8 față de linia 28.5. Post-Hobart, Cocciaretto a închis aproape orice meci în straight sets.

**How I lose:** Cocciaretto are o zi slabă pe serviciu, Kraus intră în ritm pe clay (jucătoare europeană pe suprafață preferată), scoate un set tight (7-5) → 3 seturi de câte 9-10 game-uri = 29-31 games → OVER 28.5. Probabilitate scenariul: ~11%.

### EV Check

| Odds oferite | EV U28.5 (89%) | EV U29.5 (92%) |
|-------------|----------------|----------------|
| 1.15 | +2.4% ✅ | +5.8% ✅ |
| 1.20 | +6.8% ✅ | +10.4% ✅ |
| 1.25 | +11.3% ✅ | +15.0% ✅ |

> **Notă linie alternativă:** Dacă bookmaker-ul oferă Under **29.5** (nu 28.5), marja crește la 8.94 games → p ≈ 92% → EV și mai bun la aceleași cote.

---

## SUMAR PICKS

| Pick | Market | Scor | p_research | Fair odds | Acțiune |
|------|--------|------|-----------|-----------|---------|
| Cocciaretto vs Kraus | Under 28.5 games | **8/10** | **89%** | **1.124** | ✅ BET la ≥ 1.15 |
| Vekic vs Cengiz | Under 28.5 games | 6/10 | 77% | 1.299 | ❌ PASS |
| Ostapenko vs Stefanini | Under 28.5 games | 6/10 | 71% | 1.408 | ❌ PASS |

---

*Analysis: 2026-05-06 | Template CoVe_WTA_Under.28.5.md v1.0 | Rome WTA 1000 + Istanbul WTA 125*

### All Sources

- [Sinja Kraus — Wikipedia](https://en.wikipedia.org/wiki/Sinja_Kraus)
- [Sinja Kraus WTA Official Profile](https://www.wtatennis.com/players/327061/sinja-kraus)
- [Elisabetta Cocciaretto WTA Official Profile](https://www.wtatennis.com/players/327909/elisabetta-cocciaretto)
- [Italian Open 2026 Results & Draw | WTA Tournament Centre](https://rallyher.com/italian-open-2026-wta-results-draw-scores-schedule/)
- [WTA Rome Day 2 Predictions incl. Cocciaretto context | Last Word on Sports](https://lastwordonsports.com/tennis/2026/05/05/wta-rome-day-2-predictions-eala-frech/)
- [Vekic R1 Istanbul 3 seturi vs Falei | Germanijak.hr](https://www.germanijak.hr/ostalo/vijesti/donna-vekic-uspjesno-otvorila-istanbul-break-u-trecem-setu-presudio-bjeloruskoj-suparnici/161233)
- [Vekic Istanbul R2 confirmat | HTS.hr](https://hts.hr/donna-vekic-u-2-kolu-wta-125-turnira-u-istanbulu/)
- [Istanbul Cup 2026 Overview — Wikipedia](https://en.wikipedia.org/wiki/2026_%C4%B0stanbul_Cup)
- [Ostapenko vs Stefanini H2H & preview | Tennis Tonic](https://tennistonic.com/tennis-news/993619/h2h-prediction-of-jelena-ostapenko-vs-lucrezia-stefanini-in-rome-with-odds-preview-pick-6th-may-2026/)
- [WTA Rome Day 2 Predictions incl. Ostapenko | Last Word on Sports](https://lastwordonsports.com/tennis/2026/05/05/wta-rome-day-2-predictions-stefanini-ostapenko/)
- [McNally vs Kasatkina context — Rome WTA | Last Word on Sports](https://lastwordonsports.com/tennis/2026/05/06/wta-rome-best-bets-mcnally-kasatkina/)
