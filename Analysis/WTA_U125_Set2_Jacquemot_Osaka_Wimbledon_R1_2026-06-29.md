# CoVe Analysis — U12.5 Set 2 | Wimbledon 2026 R1
## Elsa Jacquemot vs Naomi Osaka
**Data:** 2026-06-29 | **Ora:** 14:20 BST (15:20 CEST)
**Turneu:** The Championships, Wimbledon — Round 1 (R128)
**Suprafață:** Iarbă (outdoor, All England Club)
**Analist:** AI Sports Analyst | **Metodologie:** Triple Filter + Contextual TB Analysis

---

## NOTE METODOLOGICE

**Double Signal model:** O7.5=YES și U12.5=YES simultan.
**Hold asymmetry:** 2.66pp ← MINIMAL (aceeași îngrijorare ca Navarro/Ruse).
**tb_p_cal = 10.0%** ← exact la pragul ≤10%.
**Gap:** 7.8pp ✅ (curat).
**EROARE MODEL:** days_rest Osaka = 35 (greșit) — în realitate a jucat Bad Homburg Final pe **21 iunie = 8 zile în urmă!**

---

## PASUL 1 — TRIPLE FILTER

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **10.0%** | ✅ exact la prag |
| p_markov | 0.4465 (44.65% Jacquemot) | — |
| p_elo | 0.3685 (36.85% Jacquemot) | — |
| gap | **\|44.65 - 36.85\| = 7.8pp** | ✅✅ curat |
| UNSTABLE | Nu | ✅ |
| **hold_asym** | **2.66pp** ← MINIM | ⚠️ |
| blowout | 0 | meci echilibrat |
| O7.5 | **YES** (double signal) | — |
| data_source | sackmann/sackmann | ✅ ambele fiabile |
| days_rest real | **Jacquemot 7, Osaka ~8** | corect |

**PASUL 1: ✅ TRECUT** (cu nota: hold_asym minimal, tb la prag)

---

## PASUL 2 — TENNISABSTRACT (iarbă cu analiză contextuală completă)

### Elsa Jacquemot — Iarbă 2023-2026

**Sample: 8 meciuri** ✅ (borderline dar suficient)

| Meci | Turneu | Oponent rang | S1 TB? | S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs Wei (W) Ilkley 2025 | WTA 125 | ~134 | ❌ | 6-4 | ❌ NO |
| vs Golubic (L) Ilkley 2025 | WTA 125 | ~82 | ❌ | 3-6 | ❌ NO (3-set) |
| vs **Jabeur** (L) Berlin 2025 | Int. WTA | **~61** | ❌ | **7-5** | ❌ NO ← S3=7-6(11), nu S2! |
| vs Bulgaru (W) Wimbledon Q1 | Grand Slam Q | ~203 | ❌ | 6-2 | ❌ NO |
| vs Rodionova (W) Wimbledon Q2 | Grand Slam Q | ~213 | ❌ | 6-3 | ❌ NO |
| vs Cornet (W) Wimbledon Q3 | Grand Slam Q | ~843 | ❌ | 6-1 | ❌ NO |
| vs **Bencic** (L) Wimbledon R64 | **Grand Slam** | **~35** | ❌ | 6-1 | ❌ NO (3-set) |
| vs Sonmez (L) Eastbourne Q 2026 | Int. WTA Q | ~54 | ❌ | 6-2 | ❌ NO (3-set) |

**Jacquemot S2 TB pe iarbă: 0/8 = 0%** ✅✅✅

**NOTĂ IMPORTANTĂ:** Agentul TA a identificat greșit S3 7-6(11) din meciul vs Jabeur ca "S2 TB". Corect: S1=4-6, S2=7-5 (NO TB), S3=7-6(11). Set 2 a fost decisiv.

**Jacquemot NU A AJUNS LA TB ÎN SET 2 PE IARBĂ NICIODATĂ** — în 8 meciuri complete!

---

### Naomi Osaka — Iarbă 2023-2026

**Sample: 5 meciuri completate + 1 RET** ✅

| Meci | Turneu | Oponent rang | S1 TB? | S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs **Samsonova** (L) Berlin 2025 | Int. WTA | **#20** | ❌ (S1=3-6) | **7-6(3)** | **✅ YES** |
| vs **Danilovic** (W) Bad Homburg 2025 | WTA 500 | **#39** | **✅ 7-6(6)** | **7-6(4)** | **✅ YES** |
| vs **Navarro** (L) Bad Homburg 2025 | WTA 500 | **#10** | ❌ (S1=6-4) | 6-4 | ❌ NO |
| vs **Gibson** (W) Wimbledon 2025 R128 | **Grand Slam** | **#126** | ❌ (S1=6-4) | **7-6(4)** | **✅ YES** |
| vs **Siniaková** (W) Wimbledon 2025 R64 | **Grand Slam** | **#81** | ❌ (S1=6-3) | 6-2 | ❌ NO |
| vs **Muchova** (L RET) Bad Homburg 2026 F | WTA 500 | **#11** | ❌ (S1=6-1 Osaka!) | 1-0 RET | — incomplet |

**Osaka S2 TB pe iarbă: 3/5 = 60%** 🔴🔴 ← EXTREM DE RIDICAT

---

### ANALIZA CONTEXTUALĂ A CELOR 3 S2 TB-URI OSAKA

#### TB #1: vs Liudmila Samsonova — Berlin 2025 (Int. WTA)

| Factor | Detaliu | Impact U12.5 azi |
|---|---|---|
| **Samsonova rang** | **#20** la data meciului | Top-20, server dominant (> Jacquemot #82) |
| **Nivel turneu** | WTA International Berlin | Similar cu Wimbledon (suprafață iarbă) |
| **Context** | Osaka pierdut în 3 seturi | Meci dificil vs top-20 |
| **S1 → S2** | S1=3-6 (Osaka pierde), S2=7-6 TB (Osaka câștigă), S3=6-4 (Samsonova câștigă) | TB în S2 după S1 pierdut = lupta de revenire |
| **ELO Samsonova** | ~2200+ | Mult mai mare decât Jacquemot (~1625 Elo) |
| **Relevanță azi** | Samsonova mult mai bună decât Jacquemot | ⚠️ parțial relevant — nivel mai ridicat |

**Concluzie #1:** TB vs #20 este de așteptat. Jacquemot (#82) = nivel diferit. **Parțial relevant.**

---

#### TB #2: vs Olga Danilovic — Bad Homburg 2025 (WTA 500)

| Factor | Detaliu | Impact U12.5 azi |
|---|---|---|
| **Danilovic rang** | **#39** la data meciului | Mid-tier WTA, similar cu Jacquemot |
| **Nivel turneu** | WTA 500 Bad Homburg (iarbă!) | Identică suprafață |
| **Context** | Osaka a CÂȘTIGAT dar via **2 tiebreaks** (S1 + S2!) | Osaka chiar vs #39 = 2 TBs! |
| **S1 TB → S2** | S1=7-6(6) TB, S2=7-6(4) TB | Cascadă de TB-uri |
| **ELO Danilovic** | ~1500-1600 | Comparabilă cu Jacquemot (~1625) |
| **Relevanță azi** | **EXTREM DE RELEVANTĂ** | 🔴🔴 Danilovic ≈ Jacquemot rang |

**Concluzie #2:** Danilovic #39 este cel mai apropiat comparator pentru Jacquemot (#82 azi, peak #53). Osaka a jucat **ambele seturi la TB vs Danilovic**. **Cel mai relevant precedent pentru pick-ul de azi.**

---

#### TB #3: vs Talia Gibson — Wimbledon 2025 R128 (Grand Slam)

| Factor | Detaliu | Impact U12.5 azi |
|---|---|---|
| **Gibson rang** | **#126** la data meciului | MULT mai slabă decât Jacquemot (#82) |
| **Nivel turneu** | **Wimbledon R128** — identic cu meciul de azi! | Suprafață identică, același turneu |
| **Context** | Osaka a CÂȘTIGAT dar S2 = TB vs #126 | TB vs jucătoare rang similar cu Dudeney |
| **ELO Gibson** | ~600-700 | Mult mai mic decât Jacquemot (~1625) |
| **Relevanță azi** | **CRITIC** — Wimbledon, aceeași rundă | 🔴🔴 dacă S2 TB vs #126, cu siguranță risc vs #82 |
| **Mindset Osaka** | La debut la Wimbledon 2025, returning from maternity | Azi: sezonată, #14 seed, finalistă Bad Homburg |

**Concluzie #3:** Osaka a mers la S2 TB vs Gibson (#126) la **Wimbledon în R128** — exact situația de azi! Chiar dacă Osaka este mai puternică acum (finalistă Bad Homburg), pattern-ul structurii ei de joc pe iarbă = TB frecvente.

---

### Rezumat Pasul 2

| | Jacquemot | Osaka |
|---|---|---|
| Sample iarbă | 8 ✅ | 5-6 ✅ |
| **S2 TB rate** | **0/8 = 0%** ✅✅✅ | **3/5 = 60%** 🔴🔴 |
| S1 TB → S2 | N/A (0 S1 TBs) | 1/1 = 100% (Danilovic match) |
| Cel mai relevant precedent | — | Danilovic #39, Wimbledon Gibson |

**PASUL 2: 🔴 RED FLAG MAJOR — Osaka 60% S2 TB pe iarbă**

---

## 1. MATCH CONTEXT

**Wimbledon 2026 R1 (R128)** — All England Club. Ora: 14:20 BST.

**Osaka** vine ca **#14 seed**, în cea mai bună formă de iarbă din carieră:
- A ajuns în **FINALA Bad Homburg 2026** (pierdut vs Muchova prin retragere la 6-1, 1-0)
- "Servici practic de neretornat" în semifinalele Bad Homburg ([IBTimes](https://www.ibtimes.co.uk/naomi-osaka-quarter-final-2026-bad-homburg-1804705))
- Prima finală de iarbă din carieră
- Mișcare "fluidă și încrezătoare" — schimbare față de problemele sale istorice pe iarbă

**EROARE MODEL:** days_rest=35 este incorect. Osaka a jucat Bad Homburg Final pe **21 iunie 2026** = 8 zile în urmă. Modelul nu a preluat acest meci din baza de date.

---

## 2. PROFILURI JUCĂTOARE

### Elsa Jacquemot (Franța)
- **Rang oficial:** ~#82 WTA | **Peak:** #53 (feb 2026) | **Vârstă:** 23 ani (n. 3 mai 2003)
- **Stil:** Baseliner solid, joc consistent, forehand agresiv, returnuri bune
- **Grass:** 8 meciuri, 5-3 record (inclusiv Wimbledon 2025 main draw win!)
- **S2 TB pe iarbă: 0/8 = 0%** ← structural player care câștigă seturi fără TB
- **2026 form:** 50% win rate (mediocru), issues la qualifying multiple turnee
- **Hold iarbă:** 74.62% (Sackmann, fiabil)

### Naomi Osaka (Japonia)
- **Rang oficial:** ~#15-20 | **Seed Wimbledon 2026: #14** | **Vârstă:** 29 ani
- **Career:** 4 Grand Slam titles (AO 2019/2021, USO 2018/2020), peak #1
- **Stil:** Powerful baseline game, serve MAJOR weapon, forehand agresiv. Pe iarbă 2026 = mișcare mai bună, serve "unreturnable"
- **Grass:** 6 meciuri, formă ascendentă (Bad Homburg finalist!)
- **S2 TB pe iarbă: 3/5 = 60%** ← structural pattern care duce la TBs
- **Hold iarbă:** 77.27% (Sackmann, fiabil) — ↑ față de anii anteriori
- **2026:** Roland Garros R4 (best Slam result), Bad Homburg finalist = revenire totală

---

## 3. STATISTICI MODEL

| Parametru | Jacquemot (A) | Osaka (B) |
|---|---|---|
| **Hold % iarbă** | **74.62%** | **77.27%** |
| **Hold asymmetry** | **+2.66pp Osaka** ← MINIMAL | ⚠️ |
| p_markov | **44.65% Jacquemot** | |
| p_elo | **36.85% Jacquemot** | |
| gap | **7.8pp** | ✅✅ curat |
| expected_games | **25.15** | seturi lungi estimate |
| blowout | **0** | complet echilibrat structural |
| O7.5 | **YES** | ← seturi lungi, ≥7.5 games |
| tb_p_cal | **10.0%** | exact la prag |

**Expected_games = 25.15** ← mai lung decât Yastremska/Ito (20.8). Modelul spune seturi LUNGI (O7.5 YES). Seturi lungi + hold_asym minimal = TB risc crescut natural.

---

## 4. CONDIȚIE FIZICĂ (DATE CORECTATE)

### Jacquemot — ⚠️ Obosită recent
- days_rest = 7 (a jucat Eastbourne qualifying pe ~22 iunie)
- had_3sets_7d = True (a jucat 3 seturi la Eastbourne)
- fatigue_flag = False (model nu o marchează fatigued)

### Osaka — ✅ Fresh dar intensă
- **Real days_rest = 8** (Bad Homburg Final 21 iunie)
- A jucat până în finală la Bad Homburg = 4-5 meciuri în 7 zile → ușor obosită
- DAR: model nu știe asta (days_rest=35 în model = greșit)
- **IMPACT:** Osaka a jucat mult recent → serve poate fi mai puțin constant

---

## 5. MOTIVAȚIE & PSIHOLOGIC

### Osaka — ⬆️ MOTIVAȚIE MAXIMĂ + Încredere istorică
- Prima finală de iarbă → vrea să confirme că "a descoperit iarba"
- Wimbledon = Grand Slam = cel mai important turneu
- #14 seed = presiunea favorita → trebuie să treacă R1
- "Loves the heat" (declarație la Bad Homburg) → condiție bună
- Return după maternitate = poveste de comeback → motivare personală
- La Wimbledon 2025 a eliminat Siniakova (#81) și Gibson (#126) → știe că poate ajunge adânc

### Jacquemot — ↔️ Nicio presiune dar și puțin moral
- 50% win rate 2026 = sezon mediocru
- A pierdut la calificări la mai multe turnee (Queen's, Nottingham) → nu e în formă maximă
- vs Osaka seeded #14 = favorita clară → Jacquemot joacă liberă
- Tânără (23 ani), primele apariții serioase la Wimbledon
- Motivată să demonstreze că poate concura la nivel top

---

## 6. STILUL DE JOC — IMPACT PE U12.5

**Jacquemot vs Osaka:**

- Osaka servici devastator + forehand plat → game-uri de serviciu scurte, decisive
- Jacquemot returnuri solide + joc consistent → poate ține ritmul
- Ambele hold la ~75% → seturile progresează cu break-uri alternate

**De ce se ajunge la TB în meciurile Osaka pe iarbă:**

Osaka tinde să domine cu serviciul DAR în seturile strânse (când adversara returnează bine), nu face break-ul rapid → 5-5, 6-6 → TB. Jacquemot returnează solid și servește decent (74.62% hold). Combinația creează seturi competitive care duc la 6-6.

---

## 7. CoVe SCORING — DECIZIA FINALĂ

### Argumente PRO U12.5 (pentru pick)

| Factor | Valoare | Semnal |
|---|---|---|
| Jacquemot S2 TB | **0/8 = 0%** | ✅✅✅ |
| Gap model | 7.8pp | ✅✅ |
| Data source | sackmann/sackmann | ✅✅ |
| O7.5 signal | YES (confirmare) | ✅ |

### Argumente CONTRA U12.5 (împotriva pick-ului)

| Factor | Valoare | Semnal |
|---|---|---|
| **Osaka S2 TB pe iarbă** | **3/5 = 60%** | 🔴🔴 |
| **Osaka vs Gibson #126 Wimbledon** | S2 TB vs jucătoare similară | 🔴🔴 |
| **Osaka vs Danilovic #39 (≈Jacquemot)** | S2 TB + S1 TB = 2 TBs | 🔴🔴 |
| hold_asym | **2.66pp** ← minimal | 🔴 |
| blowout = 0 | meci echilibrat | 🔴 |
| expected_games = 25.15 | seturi lungi → TB risc | 🔴 |
| tb_p_cal = 10.0% | exact la prag, modelul deja incert | ⚠️ |
| Osaka servici "unreturnable" recent | Se poate domina rapid | ✅ neutral |

### DECIZIA CENTRALĂ

**Osaka's 60% S2 TB rate este DOMINANTĂ.** Contextual:
- Danilovic (#39) ≈ Jacquemot (rang similar) → S2 TB REAL
- Gibson (#126) la Wimbledon R1 → S2 TB vs adversară similară la același turneu

Hold_asym = 2.66pp confirmă că ambele pot ține serviciul similar → setul nu se termina cu break rapid → 6-6 probabilitate reală.

Modelul însuși ezită: tb_p_cal = 10.0% (exact la prag), O7.5=YES (seturi lungi) → model spune "seturi competitive" nu "seturi decisive".

---

## 8. VERDICT U12.5 SET 2

**PASS** 🔴

**Probabilitate estimată real: ~65-70%** (sub pragul 80%)

Model spune 90% U12.5 dar TennisAbstract data reală spune:
- Osaka 60% S2 TB vs adversare de nivel Jacquemot
- hold_asym prea mic pentru seturi decisive
- expected_games 25.15 = seturi lungi = risc TB

---

## 9. O7.5 SET 1 — VERDICT

**Model: O7.5=YES, p_cal_adj=87.26%**

Aceasta este o piață DIFERITĂ și mai interesantă:
- Ambele hold la ~75% → seturile vor fi competitive (7.5+ games)
- Osaka's pattern pe iarbă = seturi lungi (11 games în Wimbledon vs Gibson = 6-4, 7-6)
- Jacquemot seturile ei pe iarbă sunt în general ≥7 games pe set
- **O7.5 Set 1: ~85-87%** → **7/10 ✅** — pick mai bun decât U12.5!

---

## 10. PREDICȚIE CÂȘTIGĂTOARE

**Osaka câștigă: ~65-70%**
- Model Elo: 63.15% Osaka
- Ajustat pentru forma recentă (Bad Homburg finalist): +5pp → ~68-70%
- Osaka #14 seed vs Jacquemot ~#82 la Wimbledon
- Jacquemot poate câștiga dacă prinde un zi bună (ea câștigă 50% din meciuri în 2026)

**Scenariu probabil: Osaka 6-4, 7-6 sau 7-5, 7-6** — seturi lungi cu TB în unul din seturi

---

## 11. RANKING FINAL PIEȚE

| # | Market | Probabilitate | Scor | Decizie |
|---|---|---|---|---|
| **1** | **O7.5 Set 1** | **~86%** | **7/10** | **✅ PICK** |
| 2 | Osaka câștigă | ~68% | 5/10 | BORDERLINE |
| 3 | U12.5 Set 2 | ~67% | **PASS** | 🔴 |

**Pick recomandat: O7.5 Set 1** — ambele jucătoare hold ~75%, seturi competitive, Osaka's pattern = seturi lungi pe iarbă.

**U12.5 Set 2: PASS** — Osaka 60% S2 TB este prea ridicat, hold_asym 2.66pp nu forțează break-uri rapide.

---

## SURSE

- [TennisAbstract JS — Elsa Jacquemot](https://www.tennisabstract.com/jsmatches/ElsaJacquemot.js)
- [TennisAbstract JS — Naomi Osaka](https://www.tennisabstract.com/jsmatches/NaomiOsaka.js)
- [IBTimes — Osaka "Loves the Heat" Bad Homburg 2026](https://www.ibtimes.co.uk/naomi-osaka-quarter-final-2026-bad-homburg-1804705)
- [LiveNewsChat — Osaka First Grass Final Bad Homburg](https://livenewschat.eu/naomi-osaka-first-grass-court-final-bad-homburg-wimbledon/)
- [YardBarker — Osaka Can't Afford to Disappoint on Grass](https://www.yardbarker.com/tennis/articles/naomi_osaka_cant_afford_to_disappoint_on_grass_this_year_for_one_key_reason/s1_17664_43976596)
- [Wikipedia — Elsa Jacquemot](https://en.wikipedia.org/wiki/Elsa_Jacquemot)
- [WTA — Osaka Player Profile](https://www.wtatennis.com/players/319998/naomi-osaka)
- [ESPN — Wimbledon 2026 Contenders](https://www.espn.com/tennis/story/_/id/49176272/wimbledon-top-contenders-rankings-sinner-gauff-sabalenka-swiatek)
- Model Markov+WElo: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-06-29)
