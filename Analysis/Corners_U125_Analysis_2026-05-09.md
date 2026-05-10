# CoVe — Football Corners Under 12.5
## Template: 1.0.1 (v1.5) | Date: 2026-05-09 (Sâmbătă)
## Model: 51 meciuri → 32 recomandate

---

## STEP 0 — FILTRE AUTOMATE (mismatch + dead rubber)

### Surse: soccerstats.com I1, SP2, SP1

| Meci | Liga | Mismatch | Observatie | Verdict |
|------|------|----------|------------|---------|
| Lazio vs Inter | I1 | **2.06** | + Inter FOR 6.46 > 6 | **DOUBLE HARD PASS** |
| Lecce vs Juventus | I1 | **1.26** | Lecce relegation must-attack | **HARD PASS** |
| Cagliari vs Udinese | I1 | **0.74** | Udinese passive, asimetrie | **HARD PASS** |
| Burgos vs Almería | SP2 | **0.76** | Almería playoff-motivated | **HARD PASS** |
| Ceuta vs Castellón | SP2 | **1.74** | + Castellón FOR 6.13 > 6 | **DOUBLE HARD PASS** |
| Valladolid vs Zaragoza | SP2 | **1.28** | Zaragoza relegation | **HARD PASS** |
| Albacete vs Cultural Leonesa | SP2 | 0.46 ✅ | ⛔ Cultural RELEGATĂ = dead rubber | **DEAD RUBBER HARD PASS** |
| **Málaga vs Gijón** | SP2 | **0.09** | OK | **PROCEED** ✅ |
| **Sevilla FC vs Espanyol** | SP1 | **0.175** | OK | **PROCEED** ✅ |
| Crvena Zvezda vs Novi Pazar | RS1 | ~3.25 est. | + Red Star FOR 8.50 > 6 | **DOUBLE HARD PASS** |
| Partizan vs OFK Beograd | RS1 | ~1.18 est. | date estimate | **HARD PASS** |
| Zeleznicar vs Radnik | RS1 | ~0.35 est. | soccerstats 404 → neconfirmat | **PENDING** |
| TR1 cluster (8 meciuri) | TR1 | — | toate sub 82% p_cal | **PASS DIRECT** |
| D1 cluster (3 meciuri) | D1 | — | toate sub 80% p_cal | **PASS DIRECT** |

**Serie A data (soccerstats.com):** [soccerstats.com/table.asp?league=italy&tid=cr](https://www.soccerstats.com/table.asp?league=italy&tid=cr)
**SP2 data:** [soccerstats.com/table.asp?league=spain2&tid=cr](https://www.soccerstats.com/table.asp?league=spain2&tid=cr)
**SP1 data:** [soccerstats.com/table.asp?league=spain&tid=cr](https://www.soccerstats.com/table.asp?league=spain&tid=cr)

---

## NOTE SERIE A — MD37 STAKES

| Echipa | Poz | Pts | Miză | Impact pe cornere |
|--------|-----|-----|------|-------------------|
| Inter | 1 | 82 | CAMPIONI deja. Coppa finala 13 mai vs Lazio → ROTAȚIE garantată | Rotație = cornere impredictibile |
| Lazio | 8 | 51 | Urmăresc Europa (7th la 4 pts), Coppa final | Motivată dar obosite după Coppa semi |
| Juventus | 4 | 65 | CL asigurat, pozitie vs Roma/Milan | Semi-motivată |
| Lecce | 17 | 32 | SUPRAVIEȚUIRE — 4 pts deasupra zonei | MUST-ATTACK = cornere spike |
| Udinese | 11 | 47 | Mid-table, nimic de jucat | PASIVĂ = unpredictabil |
| Cagliari | 15 | 37 | Aproape safe (9 pts buffer), confirmare posibilă | Motivație moderată |

**Concluzie I1:** Toate mecurile I1 cad la filtrul mismatch ÎNAINTE de dead rubber check. Dar chiar dacă ar trece, contextele sunt problematice:
- Lazio vs Inter: Inter cu rotație masivă = impredictibil
- Lecce vs Juventus: Lecce must-attack = corner spike garantat
- Cagliari vs Udinese: Udinese pasivă = unknown

---

---

## MECI 1: MÁLAGA vs SPORTING GIJÓN
## SP2 (LaLiga 2) | Matchday 39/42 | La Rosaleda, Málaga | KO ~21:00 CET

---

### Step 0 — Date verificate (soccerstats.com SP2)

| Echipa | FOR/meci | AGAINST/meci |
|--------|----------|-------------|
| Málaga CF | 4.13 | 4.76 |
| Sporting Gijón | 4.74 | 5.18 |

**MISMATCH:**
- exp_home (Málaga) = (4.13 + 5.18) / 2 = **4.655**
- exp_away (Gijón) = (4.74 + 4.76) / 2 = **4.75**
- **mismatch = |4.655 − 4.75| = 0.095** ← EXCELENT
- **Expected total soccerstats: 9.41 | λ model: 8.84** ← model < 9 = EXCELLENT

### Model
- p_cal: **84.71%** | λ=8.84 | fair odds: 1.180

### Stakes
- Málaga (6th, 63pts): **ULTIMUL LOC DE PLAYOFF** — altă echipă la 1-2 puncte în spate. Trebuie să câștige.
- Gijón (12th, 52pts): **SAFE, nimic de jucat.** Entertainment mode posibil.

### Dead rubber: NU — Málaga chasing playoff (exception aplicat)

### QUICK SCORE (/10)

**STEP A — Corner Baseline:**
- Málaga FOR 4.13 → ✅ GOOD (3-5 range)
- Gijón FOR 4.74 → ✅ GOOD (3-5 range)
- Niciuna < 3.5 → nu e GOLD
- ⚠️ **PASS RULE: "At least ONE team avg corners < 4"** — AMBELE sunt > 4. Acesta este un semnal structural slab.
- Score: **+2** (GOOD profile, dar nu GOLD)

**STEP B — Expected Total:**
- λ model = 8.84 → < 9 → 🔥 EXCELLENT
- Score: **+2**

**STEP C2 — Stil tactic:**
- Málaga (SP2 club, Pellicer style): joc de atac direct, nu inverted possession
- Gijón away: pasivi, nimic de jucat → nu vor genera corners agresiv
- Mixed → **+0**

**STEP C3 — Vreme (Málaga, 21:00 CET):**
- 22°C, 0.10mm precipitații avg primele 10 zile mai, vânt 4mph E → CLEAR
- **+1**

**STEP D — Game State:**
- Málaga MUST WIN (playoff) = vor ataca agresiv → risc cornere spike
- Gijón pasivă = va absorbi, contracara
- Nu e derby, nu e supraviețuire directă
- Score: **+0** (nu e profilul "controlled match" dorit)

**STEP E — League Profile:**
- SP2 = 85.0% hit rate U12.5 → BEST 5 leagues
- Score: **+1**

**TOTAL PRE-C4: 2+2+0+1+0+1 = 6/10**

### C4 — Match Context

**C4-B Psihologie:**
- Málaga MUST WIN = mode atacant agresiv → **-2pp** Under (template: "-2pp — Echipă MUST-WIN si știe că trebuie să marcheze")
- Gijón nimic de jucat → joc deschis fără urgență → risc dead rubber la Gijón individual

**C4 flag contra Under:** Málaga MUST WIN = cornere spike risc
→ C4 contribuție Quick Score: **-1** (multiple flags contra Under)

**TOTAL QUICK SCORE: 6 - 1 = 5/10** → PASS

### Research probability

| Ajustare | pp |
|----------|----|
| C2 — stil | +0 |
| C3 — vreme | +0 |
| C4-B Málaga must-win | -2pp |
| **Total** | **-2pp** |

- p_cal: 84.71%
- p_research: 84.71 - 2 = **82.71%** → marginale la prag
- Dar: **PASS RULE neîndeplinit** (niciuna < 4 FOR) → risc structural suplimentar

**VERDICT: PASS** ❌

**Motivul principal:** Málaga în mode MUST-WIN pentru playoff → vor ataca cu wings și crossing, generând cornere. Gijón cu nimic de jucat → nu va controla disciplinat. Ambele echipe > 4 FOR/meci. Scor 5/10 sub pragul de 8+.

---

## MECI 2: SEVILLA FC vs ESPANYOL
## SP1 (LaLiga) | Matchday 37 | Ramón Sánchez Pizjuán, Sevilla | KO ~18:30 CET

---

### Step 0 — Date verificate (soccerstats.com SP1)

| Echipa | FOR/meci | AGAINST/meci |
|--------|----------|-------------|
| Sevilla FC | 4.91 | 4.50 |
| Espanyol | 4.68 | 4.62 |

**MISMATCH:**
- exp_home (Sevilla) = (4.91 + 4.62) / 2 = **4.765**
- exp_away (Espanyol) = (4.68 + 4.50) / 2 = **4.59**
- **mismatch = |4.765 − 4.59| = 0.175** ← bun
- **Expected total: 9.355 | λ model: 9.13**

### Model
- p_cal: **82.62%** | λ=9.13 | fair odds: 1.210

### Stakes — CRITIC
- Sevilla (17th, 37pts): **ÎN ZONA AUTOMATICĂ DE RELEGARE**. DESPERARE MAXIMĂ.
- Espanyol (13th, 39pts): Safe, nimic de jucat la MD37.

### SCORING — REDFLAGS

**STEP A:** Sevilla FOR 4.91, Espanyol FOR 4.68 → ambele >4, niciuna <4 → PASS RULE FAIL
Score: **+2** (GOOD dar nu GOLD)

**STEP B:** λ=9.13 → ⚠️ borderline (9-10) → **+1**

**STEP C2 — Stil tactic Sevilla:**
- Sevilla în retrogradare cu backing de "must score" = se aruncă pe flancuri cu crosses
- Sevilla este un club din Andaluzia cu stilul tradițional cross-heavy
- **→ -5pp potential DOWNGRADE Under / HARD PASS signal din C2**

**STEP C3:** Sevilla, seară de mai, clar → **+1**

**STEP D:** Sevilla relegation = MUSTATTACK = **-2** (contra Under)
**STEP E:** SP1 = 83.3% → **+1**

**TOTAL: 2+1+0+1+0+1 = 5/10** și **C2 = HARD PASS dacă confirmăm cross-heavy Sevilla**

**Research probability:**
- C2 Sevilla cross-heavy (relegation must-score = confirmă stilul): -5pp
- C4-B Sevilla MUST WIN relegation: -2pp
- Total: -7pp → p_research = 82.62 - 7 = **75.6%** → FAR BELOW 82%

**VERDICT: HARD PASS** ❌

**Motivul principal:** Sevilla în retrogradare directă = maximum desperation = vor ataca pe flancuri cu crossing = cornere spike garantat. Profilul Sevilei + MD37 + must-win = exact tiparul de meci care generează 13-15 cornere când echipa de acasă atacă disperat.

---

## TABEL FINAL

| Meci | Liga | Mismatch | Score | p_research | Verdict |
|------|------|----------|-------|------------|---------|
| Lazio vs Inter | I1 | 2.06 | — | — | HARD PASS ❌ |
| Lecce vs Juventus | I1 | 1.26 | — | — | HARD PASS ❌ |
| Cagliari vs Udinese | I1 | 0.74 | — | — | HARD PASS ❌ |
| Ceuta vs Castellón | SP2 | 1.74 | — | — | HARD PASS ❌ |
| Valladolid vs Zaragoza | SP2 | 1.28 | — | — | HARD PASS ❌ |
| Burgos vs Almería | SP2 | 0.76 | — | — | HARD PASS ❌ |
| Albacete vs Cultural | SP2 | 0.46 | — | — | Dead rubber ❌ |
| **Málaga vs Gijón** | SP2 | 0.09 | **5/10** | **82.7%** | **PASS** ❌ |
| **Sevilla vs Espanyol** | SP1 | 0.175 | **5/10** | **75.6%** | **HARD PASS** ❌ |
| Red Star vs Novi Pazar | RS1 | ~3.25 | — | — | HARD PASS ❌ |
| Partizan vs OFK | RS1 | ~1.18 | — | — | HARD PASS ❌ |
| Zeleznicar vs Radnik | RS1 | ~0.35 | — | — | Neconfirmat |
| TR1 (8 matches) | TR1 | — | — | <82% | PASS direct ❌ |
| D1 (3 matches) | D1 | — | — | <80% | PASS direct ❌ |

---

## ZI SLABĂ — ZERO PICKS Under 12.5

**Motivele principale:**

1. **Serie A MD37 = mismatch epidemic.** Inter cu 6.46 FOR generează o dominanță asimetrică masivă (exp_away=5.36 în Lazio-Inter). Lecce relegation + Juventus attacking = mismatch. Cagliari-Udinese = mismatch 0.74.

2. **SP2 MD39 = MUST-WIN context pentru mai toate echipele.** Málaga (playoff), Almería (playoff), Zaragoza (relegare), Cultural (deja retrogradată) → fiecare meci are presiune de o parte sau alta.

3. **Sevilla SP1 = relegation desperate = guaranteed corner spike.** Chiar dacă mismatch e mic, contextul de degradare directă = cel mai rău profil pentru Under.

4. **RS1 = date neconfirmate** (soccerstats 404). Nu se poate face CoVe fără date verificate.

5. **Zilele de sâmbătă cu multe meciuri de Serie A aproape de finalul sezonului = problematice sistematic** din cauza mismatch-urilor generate de echipele Top-4 care domina colțurile și echipele de jos care trebuie să atace.

---

## NOTĂ — RS1 ZELEZNICAR vs RADNIK (PENDING)

Dacă reușești să verifici manual datele de colțuri pentru RS1 pe soccerstats sau windrawwin, această pereche (estimated mismatch ~0.35) ar putea fi cel mai bun candidat al zilei. Zaleznicar vs Radnik are λ=9.08 din model (p_cal=85.48%).

Pentru verificare: încearcă [soccerstats.com/latest.asp?league=serbia](https://www.soccerstats.com/latest.asp?league=serbia) → tab Corners.

---

## SURSE

- [soccerstats.com — I1 corners](https://www.soccerstats.com/table.asp?league=italy&tid=cr)
- [soccerstats.com — SP2 corners](https://www.soccerstats.com/table.asp?league=spain2&tid=cr)
- [soccerstats.com — SP1 corners](https://www.soccerstats.com/table.asp?league=spain&tid=cr)
- [Wikipedia 2025-26 Serie A](https://en.wikipedia.org/wiki/2025%E2%80%9326_Serie_A)
- [Wikipedia 2025-26 Segunda División](https://en.wikipedia.org/wiki/2025%E2%80%9826_Segunda_Divisi%C3%B3n)
- [BeSoccer Serie A standings](https://www.besoccer.com/competition/table/serie_a/2026)
- [DailySports Lazio-Inter preview](https://dailysports.net/predictions/lazio-vs-inter-prediction-facts-lineups-and-h2h-may-9-2026/)
- [DailySports Malaga-Gijon](https://dailysports.net/predictions/malaga-vs-sporting-gijon-prediction-h2h-and-probable-lineups-may-9-2026/)
