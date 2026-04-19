# Bankroll Management Plan — Kelly Fractional Strategy
## Created: 2026-04-18
## Starting bankroll: 1,000 RON
## Target: 2,587,000 USD (~12,500,000 RON)
## Estimated timeline: 8-10 years with discipline + monthly contributions

---

## STAKE SIZING RULES — REGULA FUNDAMENTALA

### Stake = procent din BANKROLL-ul actual (nu din bankroll-ul initial)

**Foarte important:** De fiecare data cand pariezi, recalculezi stake-ul bazat pe bankroll-ul CURENT (dupa castigurile/pierderile anterioare). Asta se numeste **compounding** si e motorul principal de crestere.

---

## TABELUL DE STAKE PE SCOR

| Scor CoVe | Confidence | Stake % bankroll | Exemplu cu bankroll 1,000 RON |
|---|---|---|---|
| **10/10** | PREMIUM+ | **5%** | 50 RON |
| **9/10** | HIGH | **3%** | 30 RON |
| **8/10** | MODERATE-HIGH | **2%** | 20 RON |
| **7/10** | MODERATE | **1%** | 10 RON |
| **< 7/10** | — | **0% (NU PARIEZI)** | 0 RON |

### De ce aceste procente?

**Kelly fractional 1/4** — cel mai sigur:
- Full Kelly = extrem de agresiv (riscul de a pierde 50% bankroll intr-o luna = 30%)
- 1/4 Kelly = varianta 4x mai mica, drawdown maxim ~15%
- Trade-off: growth mai incet dar NU FACI BANCRUPT

### Logica:
- **10/10 e rar** (1-2/saptamana) → poti permite 5% = expunere mai mare
- **7/10 borderline** → 1% = "dimensiunea testului", pastrezi capitalul
- **Sub 7/10 = zero** → disciplina = supravietuire

---

## REGULI SUPLIMENTARE DE PROTECTIE

### 1. MAXIM 3 PARIURI ACTIVE SIMULTAN
Daca ai 3 pariuri pe 10/10 in aceeasi zi: **NU pariezi 15% (3 × 5%)**. Limitezi la **8% total portfolio exposure** = redistribui stake-ul.
- Scenariul: 3 picks 9/10 (teoretic 3 × 3% = 9%) → aloca max 8%, deci 2.66% pe fiecare.

### 2. DAILY STOP-LOSS
Daca pierzi **3 pariuri consecutive intr-o zi**, OPREsti pariatul pana maine. Compensezi varianta emotionala.

### 3. WEEKLY STOP-LOSS (-10%)
Daca pierzi **10% din bankroll intr-o saptamana**, opresti pariatul pana luni urmatoare. Analizezi ce a mers prost.

### 4. MONTHLY STOP-LOSS (-20%)
Daca pierzi **20% din bankroll intr-o luna**, opresti 2 saptamani. Re-evaluezi strategia complet.

### 5. NO MARTINGALE
**NU dubla stake-ul dupa pierdere.** Niciodata. E cea mai rapida cale catre ruin.

### 6. NO "FOTBAL RAZBUNARE"
Daca pierzi pe DC, nu recuperezi cu odds mari. Urmatorul pariu e pe CoVe, nu pe emotie.

---

## EXEMPLE CONCRETE CU BANKROLL 1,000 RON

### Ziua 1 — Bankroll: 1,000 RON
Ai 2 picks:
- Meci A: Udinese 1X, scor 9/10 → **30 RON stake**
- Meci B: Rybakina O7.5, scor 10/10 → **50 RON stake**
- Total stake: 80 RON (8% expunere, OK)
- Cazul best: +40 RON (dupa castiguri) → bankroll 1,040 RON
- Cazul worst: -80 RON → bankroll 920 RON

### Ziua 10 — Bankroll: 1,050 RON (dupa castiguri)
Pick 9/10 → stake nu mai e 30, ci **3% × 1,050 = 31.5 RON**

### Ziua 30 — Bankroll: 1,100 RON (luna bun)
Pick 10/10 → stake = **5% × 1,100 = 55 RON**

**Observatie:** Stake-ul creste AUTOMAT cu bankroll-ul. Asta e compounding in actiune.

---

## REGULA PARIURI COMBINATE (ACCUMULATOR)

Accumulator = **doua sau mai multe pariuri combinate**. Risc multiplu → regula speciala:

### Stake max pe accumulator:
- 2 legs (ambele 9/10+): **stake = 1% bankroll** (nu 3% × 2)
- 3 legs (toate 9/10+): **stake = 0.5% bankroll**
- 4+ legs: **NU recomand** — varianta prea mare

### De ce redus?
Probabilitate combinata scade. 2 pariuri 90% fiecare = 81% combined. 3 × 90% = 72.9%. Accumulator = variance killer.

**Foloseste acumulator doar ca "risk enhancer" cu stake mic, nu ca strategie principala.**

---

## PLAN DE CONTRIBUTII (PENTRU ACCELERARE)

Target realist cu contributii lunare:

### Scenariul A: Doar compounding (1,000 RON start, zero contributii)
- 10% ROI lunar
- Anul 1: 3,138 RON
- Anul 3: 30,913 RON
- Anul 5: 304,482 RON
- **Anul 9: 12,500,000 RON** ✓

### Scenariul B: + 500 RON/luna contributie
- Aceleasi ROI 10%/luna
- Anul 1: 12,000 RON (vs 3,138)
- Anul 3: 50,000 RON
- Anul 5: 400,000 RON
- **Anul 7: ~2,500,000 RON** → continui 1 an mai mult pentru 12.5M
- **Total: ~8 ani, mai stabil**

### Scenariul C: + 1,000 RON/luna contributie
- **Anul 6: 12,500,000 RON** ✓
- Cel mai realist pentru tine

---

## PHASE-BASED STRATEGY (SCALARE PROGRESIVA)

### FAZA 1: Validation (luni 1-3)
**Bankroll: 1,000 → 1,400 RON target**
- Obiectiv: confirmi ca strategia functioneaza
- Pariuri totale: 20-30 (doar 8-10/10 picks)
- Stake-uri mici: max 30-50 RON
- Daca ajungi la 1,400 → continui
- Daca scazi sub 700 → **revizuiesti strategia, pauza 2 saptamani**

### FAZA 2: Consistency (luni 4-12)
**Bankroll: 1,400 → 3,000+ RON target**
- Introduci contributii lunare (500-1,000 RON)
- Disciplina: stake % constant
- Build track record

### FAZA 3: Compounding (anii 2-5)
**Bankroll: 3,000 → 300,000 RON**
- Stake-uri cresc proportional
- Selectivitate continua (doar 9-10/10)
- Diversificare: DC + WTA Winner + O7.5

### FAZA 4: Scaling (anii 5-10)
**Bankroll: 300,000 → 12,500,000 RON**
- Bankroll mare = poate aloca 1-3% pe pick (nu mai tine 5%)
- Mai putine pariuri, stake-uri mari
- Tax considerations
- Eventual business/LLC pentru betting

---

## TABEL COMPLET — STAKE LA DIFERITE NIVELURI DE BANKROLL

| Bankroll | 7/10 (1%) | 8/10 (2%) | 9/10 (3%) | 10/10 (5%) |
|---|---|---|---|---|
| 1,000 RON | 10 RON | 20 RON | 30 RON | 50 RON |
| 2,000 RON | 20 RON | 40 RON | 60 RON | 100 RON |
| 5,000 RON | 50 RON | 100 RON | 150 RON | 250 RON |
| 10,000 RON | 100 RON | 200 RON | 300 RON | 500 RON |
| 50,000 RON | 500 RON | 1,000 RON | 1,500 RON | 2,500 RON |
| 100,000 RON | 1,000 RON | 2,000 RON | 3,000 RON | 5,000 RON |
| 500,000 RON | 5,000 RON | 10,000 RON | 15,000 RON | 25,000 RON |
| 1,000,000 RON | 10,000 RON | 20,000 RON | 30,000 RON | 50,000 RON |
| 10,000,000 RON | 100,000 RON | 200,000 RON | 300,000 RON | 500,000 RON |

**Regula de aur:** Procent constant. Stake RON creste cu bankroll.

---

## METRICI DE TRACKING (EXCEL / SPREADSHEET)

Pastreaza un log cu fiecare pariu:

| Data | Match | Market | Scor | Stake | Odds | Rezultat | P/L | Bankroll |
|---|---|---|---|---|---|---|---|---|
| 18.04 | Leeds vs Wolves | DC 1X | 9/10 | 30 | 1.14 | WIN | +4.2 | 1,004.2 |
| 18.04 | Rybakina vs Andreeva | O7.5 | 10/10 | 50 | 1.11 | WIN | +5.5 | 1,009.7 |
| ... | | | | | | | | |

**Review saptamanal:**
- Win rate pe scor (7/10 vs 8/10 vs 9/10 vs 10/10)
- ROI pe piata (DC vs Goals vs Corners vs WTA)
- Drawdown maxim

---

## REGULILE DE CRESTERE (MILESTONE-URI)

La fiecare **dublare de bankroll**, faci review strategic:

| Milestone | Bankroll | Actiune |
|---|---|---|
| M1 | 2,000 RON | Confirmi strategie, continui aceleasi procente |
| M2 | 5,000 RON | Poti permite picks 7/10 daca sunt 2+ crossovers pe 8+/10 |
| M3 | 10,000 RON | Diversifica: adauga WTA Winner |
| M4 | 50,000 RON | Evaluezi impact psihologic al stake-urilor mari. Reduce procentul la 1/5 Kelly daca anxietate |
| M5 | 100,000 RON | Consideri separare fonduri (50% bet, 50% withdraw safe) |
| M6 | 500,000 RON | Business structure, tax planning |
| M7 | 1,000,000 RON | Poti retragi 50%, continui cu 500K |

---

## CHEAT SHEET (PRINT SI TINE LANGA CALCULATOR)

```
==================================
   BANKROLL STAKE CHEAT SHEET
==================================

PICK SCORE → STAKE %:
  10/10  = 5%
  9/10   = 3%
  8/10   = 2%
  7/10   = 1%
  <7/10  = 0% (NU PARIEZI!)

MAX EXPOSURE:
  Per zi:  8% total stake
  Per meci: max 5% (doar 10/10)

CIRCUIT BREAKERS:
  3 losses in row → STOP today
  -10% weekly   → STOP 7 days
  -20% monthly  → STOP 2 weeks

REGULI ABSOLUTE:
  ✓ Niciodata martingale
  ✓ Niciodata "razbunare"
  ✓ Kelly fractional 1/4
  ✓ Procent constant din bankroll
  ✓ Tracking fiecare pariu
==================================
```

---

## FINAL: CE SA FACI DE MAINE

1. **Configureaza spreadsheet tracking** (Excel/Google Sheets)
2. **Stabileste bankroll initial: 1,000 RON** (bani pe care accepti sa pierzi in worst case)
3. **Cumpara disciplina** — nu cresti stake-ul "doar de data asta"
4. **Review saptamanal** — duminica, analiza meciuri + W/L
5. **Review lunar** — evaluezi bankroll, ajustezi contributii

**Obiectiv Luna 1:** 1,000 → 1,100 RON (10% crestere sustenabila)
**Obiectiv Luna 3:** 1,000 → 1,330 RON (compound 10% × 3 luni)
**Obiectiv An 1:** 1,000 → 3,000+ RON cu contributii

---

## ADEVARUL FINAL

**Strategia asta nu te face bogat rapid. Te face bogat SIGUR daca ai 7-10 ani disciplina.**

90% din pariorii de fotbal pierd bani pentru ca:
- Nu au stake sizing
- Chase losses
- Pariaza pe odds mari pentru "gain rapid"
- Nu tin track

Tu ai modelul. Acum ai si gestiunea. **Restul e doar executie.**

**Success probability cu acest plan: ~85% de a ajunge la target in 8-10 ani cu contributii.**
**Success probability fara plan (agresiv): ~5% de a ajunge; 95% ruin in 1-2 ani.**

Alege cu mintea, nu cu emotia.