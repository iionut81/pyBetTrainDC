# CoVe Analysis — WTA U12.5 Set 2 (Manual, fără model)
# Karatancheva vs Rus | W75 The Hague | Clay | QF | 18:00 CEST | 10 iulie 2026

---

## META

| Field | Value |
|---|---|
| Turneu | W75 The Hague, Olanda (clay outdoor) |
| Nivel | ITF Women's W75 ($60,000) |
| Suprafață | Clay |
| Rundă | Quarterfinal |
| Data/Ora | 10 iulie 2026, 18:00 CEST |
| Player A | **Lia Karatancheva** (BUL, WTA 472, Elo 124) |
| Player B | **Arantxa Rus** (NED, WTA 178, Elo 428) |
| H2H | 0-0 (primul meci profesionist) |
| Market analizat | Under 12.5 games Set 2 |
| Robinhood | Rus 63%, Karatancheva 37% |
| Model CSV | **NOT IN MODEL** — W75 ITF, nu apare în CSV WTA oficial |

---

## NOTĂ PRELIMINARĂ: COVe MANUAL

Meciul NU apare în CSV-ul modelului (W75 ITF = circuit ITF, nu WTA oficial/WTA 125). Nu există `tb_p_cal`, `p_markov`, `p_elo` din model.

CoVe manual bazat pe date empirice CoreTennis (≥10 meciuri clay fiecare) + TennisStat + Robinhood, conform precedentului `feedback_pelo_zero_manual_cove`.

---

## PASUL 1 — CSV + ROBINHOOD CHECK

**CSV Model:** N/A — W75 ITF nu intră în pipeline model.

**Robinhood check:**
- Rus (favorită): **63%** → zona 60-74% → continuă cu notă de divergență
- Karatancheva: 37%
- Divergență vs p_markov: N/A (fără p_markov din model)
- Robinhood < 75% → class gap **neconfirmat** de piață

**TennisStat — semnal surogat pentru model:**

| Metric | Karatancheva | Rus | Combined |
|---|---|---|---|
| TB/match rate | 22% | **38%** | **30%** ⚠️ |
| Over 12.5/set | 16% | 17% | **17%** ⚠️ |
| Avg games/set | 9.41 | **9.88** ⚠️ | 9.65 |
| DF/match | 2.20 | **4.43** | 6.63 |
| Breaks/match | 3.55 | 4.93 | 8.48 |
| S2 Win% | 47% | 42% | — |

Semnale de alarmă TennisStat:
- Combined TB rate 30% >> baseline WTA 125 din sesiunea de astăzi (15-22%)
- Rus avg 9.88 games/set — cel mai ridicat din toate meciurile analizate azi
- Over 12.5/set 17% combinat → baseline 83% U12.5 (vs 88-91% model optim)

**Pasul 1 STATUS: PARȚIAL / DEGRADAT** — fără model, continui cu incertitudine structurală ridicată

---

## PASUL 2 — CORE TENNIS (Clay, 2-set matches, 2024-2026)

Surse: [CoreTennis Rus](https://www.coretennis.net/tennis-player/arantxa-rus/470/results.html) | [CoreTennis Karatancheva](https://www.coretennis.net/tennis-player/lia-karatancheva/97647/results.html)

---

### Karatancheva — Clay S2 TB Analysis

**Sample:** ~50 meciuri 2-set pe clay (2024-2026) ✅ (≥10)

**S2 Tiebreaks în 2-set clay — identificate și verificate manual:**

| Data | Turneu | Scor complet | Tip |
|---|---|---|---|
| Apr 2025 | Megasaray W115 R1 vs Veronika Erjavec | **7-6(1) 7-6(4)** | CASCADE (S1 TB → S2 TB) |
| Sep 2025 | Bucharest W75 R2 vs Mia Ristic | **7-6(6) 7-6(8)** | CASCADE (S1 TB → S2 TB) |

**S2 TB rate (2-set clay):** 2/50 ≈ **4%** ✅ ≤15%

**Cascade analysis (S1 TB → S2 TB în 2-set clay):**

S1 TBs identificate în 2-set clay (~10 cazuri): Lucie Urbanova, Mia Ristic, Nastasja Schunk, Veronika Erjavec, Caroline Werner, Tena Lukas, Maileen Nuudi, Daria Kuczer, Liv Hovde.
- Cascade: 2/10 = **20%** ✅ ≤33%

Context cascade:
- Erjavec (Megasaray): WTA ~250-350 (nivel W115)
- Ristic (Bucharest W75): WTA ~350+ (nivel W75)
- Ambele cascade contra jucătoare de nivel mediu-inferior, nu WTA top-150

**Standalone S2 TB (fără S1 TB în 2-set clay):** 0/40 = **0%** ← excelent

*Notă sesiune curentă:* Karatancheva a avut S2 TB în qualifying R2 vs Vismane (6-3, **7-6(5)**, 6-0) — dar meci pe 3 seturi, nu relevant pentru grila 2-set.

---

### Rus — Clay S2 TB Analysis

**Sample:** ~35 meciuri 2-set pe clay (2024-2026) ✅ (≥10)

**S2 Tiebreaks în 2-set clay — verificate manual:**

| Data | Turneu | Scor complet | Tip |
|---|---|---|---|
| Jul 2025 | Gran Canaria W100 SF vs Caroline Werner | **7-6(6) 7-6(1)** | CASCADE (S1 TB → S2 TB) |
| Jul 2024 | Hamburg W500 F vs Noma Noha Akugue | **6-0 7-6(3)** | Standalone S2 TB |

**S2 TB rate (2-set clay):** 2/35 ≈ **5.7%** ✅ ≤15%

**Cascade analysis (S1 TB → S2 TB în 2-set clay):**

S1 TBs identificate în 2-set clay (2 cazuri): Clara Tauson RG 2025 (76(2) 75 → no cascade), Caroline Werner Gran Canaria 2025 (cascade).
- Cascade: 1/2 = **50%** ❌ >33% — **TRIGGER CAP per grid**

Context cascade Gran Canaria 2025:
- Caroline Werner: WTA ~70-90 la momentul meciului, top-100 confirmată
- Turneu W100 SF (nivel ridicat), meci competitiv → **relevant**, nu outlier pe opponent slab

Context standalone Hamburg 2024 Final:
- vs Noma Noha Akugue (WTA ~120): Rus dominase 6-0 în S1, S2 TB în finală mare (presiune excepțională)
- Hamburg W500 Final = context de presiune cel mai ridicat posibil pentru acest circuit

**Pasul 2 STATUS:** S2 TB rates OK ✅ | Cascade Rus 50% → **MAX 6/10 per grid** (>33% cap)

---

## PASUL 3 — CONTEXT

### Profiluri Jucătoare

**Lia Karatancheva (BUL, 21 ani, WTA 472, Elo 124)**
- Stil: agresivă de linia a doua, winner count mare, service relativ slab (2.20 DF/match)
- Form 2026: WWLWWWL → win rate 40.6% (13/32), inconsistentă global dar în formă la The Hague

*The Hague 2026 — meciuri jucate:*
| Rundă | Adversar | Scor | Format |
|---|---|---|---|
| Qualifying R1 | Charlotte Pikkaart | **6-0 6-0** | 2 seturi (dominant!) |
| Qualifying R2 | Daniela Vismane | **6-3 7-6(5) 6-0** | **3 seturi** ← |
| Main R1 | Iva Primorac Pavicic | **5-7 6-1 6-3** | **3 seturi** ← |
| Main R2 | Chloe Paquet | **2-6 6-4 6-1** | **3 seturi** ← |

**FATIGUE CRITIC:** 4 meciuri jucate în total, din care **3 la 3 seturi**. Estimat ~6-7 ore pe teren în 5 zile.

**Arantxa Rus (NED, 35 ani, WTA 178, Elo 428)**
- Stil: baselinera consistentă, slice backhand, răbdare tactică, experiență uriașă ($4M+ career prize)
- Career high: WTA 66. Veterana cu experiență QF/SF la toate nivelurile.
- Form 2026: WLLLLWW → 4 înfrângeri consecutive înainte de The Hague; dar a câștigat ambele meciuri de aici

*The Hague 2026 — meciuri jucate:*
| Rundă | Adversar | Scor | Format |
|---|---|---|---|
| Main R1 | Sarah Van Emst | **6-3 6-4** | 2 seturi |
| Main R2 | Federica Urgesi | **4-6 6-2 6-1** | 3 seturi |

**2 meciuri jucate** (1 la 3 seturi). Semnificativ mai odihnită față de Karatancheva.

### Asimetrie Fizică

| Factor | Karatancheva | Rus |
|---|---|---|
| Meciuri jucate săpt. | **4** (2 cal + 2 main) | 2 (main) |
| 3-seturi jucate | **3** ← CRITIC | 1 |
| Ore estimate pe teren | ~6-7h | ~2.5h |
| Stare fizică intrare QF | Obosită | Relativ fresh |

Karatancheva intră în QF cu **deficit fizic semnificativ**. Poate fi avantaj pentru U12.5 S2 (cedează mai rapid), dar poate crea și irregularitate tactică.

### H2H & Motivație

- **H2H: 0-0** — primul meci profesionist direct
- **Motivație Rus:** acasă (Olanda), QF în propriul turneu, șansă reală la titlu la 35 ani → RIDICATĂ
- **Motivație Karatancheva:** underdog, a făcut un parcurs solid, joacă relaxat → RIDICATĂ
- **Miza:** QF W75 = SF la WTA 125 echivalent, ambele au motive să lupte

### Condiții & Context

- The Hague, iulie: tipic 20-23°C, posibil vânt moderat atlantic — clay lent, setes tend to be structural
- Antrenor Rus: Julian Alonso (coach spaniol, specialist clay)
- Antrenor Karatancheva: necunoscut
- Rus joacă acasă în fața publicului propriu (home crowd advantage)

---

## PROBABILITATE AJUSTATĂ (bazată pe TennisStat baseline)

**Baseline fără model:** TennisStat → 83% U12.5 (17% over 12.5/set)

| Factor | Direcție | Impact pp |
|---|---|---|
| CoreTennis Karatancheva S2 TB 4% (≤15%) | pozitiv | +3 |
| CoreTennis Rus S2 TB 5.7% (≤15%) | pozitiv | +3 |
| Karatancheva standalone S2 TB 0% | pozitiv | +2 |
| Karatancheva cascade 20% (OK ≤33%) | neutru | 0 |
| Rus cascade 50% — risc real la S1 TB | negativ | -4 |
| Rus standalone S2 TB (Hamburg Final) | negativ mic | -1 |
| Karatancheva fatigue (3×3seturi) → S2 mai lung / erratic | negativ | -2 |
| Rus freshness + home → breaks mai clare | pozitiv | +2 |
| No model data (incertitudine structurală maximă) | negativ | -3 |
| Robinhood <75% (class gap neconfirmat de piață) | negativ | -1 |

**p_cal_adj ≈ 83 + 3 + 3 + 2 + 0 - 4 - 1 - 2 + 2 - 3 - 1 = 82%**

Exact pe pragul de 82%, dar scorul rămâne limitat de regula cascade.

---

## SCOR FINAL

Aplicând grila U12.5 S2 v1.1:

| Criteriu | Status |
|---|---|
| Model CSV (Pasul 1) | ⚪ N/A (W75 ITF — CoVe manual) |
| Robinhood ≥60% | ✅ Rus 63% |
| Robinhood ≥75% (class gap confirmat) | ❌ 63% (zona 60-74%) |
| S2 TB Karatancheva ≤15% | ✅ **4%** |
| S2 TB Rus ≤15% | ✅ **5.7%** |
| Cascade Karatancheva ≤33% | ✅ **20%** |
| Cascade Rus ≤33% | ❌ **50%** → trigger cap max 6/10 |
| TennisStat TB baseline | ⚠️ 30% combinat (semnificativ peste WTA 125) |
| No model data | ⚠️ incertitudine structurală ridicată |

**Scor: 6/10 — PASS**

Grid: cascade Rus > 33% → max 6/10 ← regulă strictă aplicată.
Clay minimum necesar: **8/10 + RH ≥60%** → 6/10 sub prag.

---

## ATENȚIONARE

Meciurile ITF W75 nu sunt incluse în backtestul modelului. Baseline TennisStat 83% vs model optim 91.2% = gap de 8pp neacoperit. Cascade Rus 50% în 2-set clay (din 2 instanțe — sample mic, dar cascada Gran Canaria contra Werner WTA top-100 este relevant). Fără model, nu putem valida structural signal-ul U12.5.

---

## PREDICȚIE MECI

**Winner: Arantxa Rus (NED)** — favorită clară

Motivație:
- WTA 178 vs 472 (gap 294 poziții), Elo 428 vs 124
- Acasă (Olanda), 2 meciuri vs 4 meciuri jucate
- Veteran experience QF/SF vs tânăr extrem de obosit (3 trei-seturi)
- Rus ca baselinera consistentă domină jucătoare erratice la scoruri strânse

**Scor estimat:** Rus W **6-3 6-2** sau **6-4 6-3**

S2 estimat: Rus controlează ritm, Karatancheva cedează fizic după oboselă acumulată.

---

## VERDICT FINAL

| Market | Recomandare | Scor | Motiv |
|---|---|---|---|
| U12.5 Set 2 | ❌ **PASS** | 6/10 | Sub minimul clay (8/10+RH); cascade Rus 50%; fără model; TB baseline 30% |

**Nu recomandăm.** Cascade Rus 50% în 2-set clay (1/2 instanțe, dar contra opponent de calitate) + fără model structurat + TennisStat TB rate 30% = incertitudine prea ridicată pentru un pick clay.

---

*Surse:*
- *[CoreTennis — Arantxa Rus](https://www.coretennis.net/tennis-player/arantxa-rus/470/results.html) (clay results 2024-2026)*
- *[CoreTennis — Lia Karatancheva](https://www.coretennis.net/tennis-player/lia-karatancheva/97647/results.html) (clay results 2024-2026)*
- *TennisStat H2H stats — furnizat de user*
- *Robinhood prediction markets (07.07.2026): Rus 63% / Karatancheva 37%*
- *[TennisTemple — Paquet vs Karatancheva R2](https://en.tennistemple.com/match/paquet-karatancheva-the-hague-2026/9471895/) (draw structure)*

*Generat: 10 iulie 2026 | Sesiune Contrexeville + The Hague*
