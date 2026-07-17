# WTA U12.5 Set 2 — Triple Filter CoVe
## Nuria Brancaccio vs Eva Vedder
### WTA 125 Roma (ATV Bancomat Tennis Open) | Clay | R16 | 20:00 CEST July 16, 2026

---

## MODEL DATA (din 1.5_WTA_Under12_5.csv + 1.1_WTA_Winner.csv — run 2026-07-16)

| Câmp | Valoare | Interpretare |
|---|---|---|
| p_hold_a (Brancaccio) | 0.6651 | Ține 66.5% din servicii pe clay — solidă |
| p_hold_b (Vedder) | 0.5007 | Ține 50.1% din servicii — la limita min_hold, dar nu catastrofal |
| hold_asym | 0.1644 | > 0.15 (prag premium) |
| min_hold | 0.5007 | **Ratează premium cu doar 0.0007** (prag 0.50) |
| bci | 0.0821 | |
| tb_p_raw | 0.0445 | |
| **tb_p_cal** | **0.0927** | **9.27% probabilitate TB în Set 2 — sub prag 10% ✅** |
| p_u125 | 0.9073 | 90.73% probabilitate U12.5 S2 |
| blowout_score | 8 | |
| premium_elite | no | |
| **premium_u125** | **no** | min_hold=0.5007 vs prag 0.50 — miss marginal |
| danger_zone | no | |
| UNSTABLE flag | — (gol) | NO UNSTABLE |
| fatigue_flag_a/b | False / False | Ambele fără semnal de oboseală |
| days_rest_a/b | 1 / 2 | Brancaccio cu o zi mai puțin, dar meci R1 scurt |
| p_markov | 0.8189 | Brancaccio câștigă 81.9% prin simulare hold-rate |
| p_elo | 0.7041 | Brancaccio câștigă 70.4% prin Elo istoric clay |
| predicted_winner | Nuria Brancaccio | |
| p_cal (Winner) | 0.7019 | Brancaccio câștigă 70.2% — calibrat |

**Notă critică — divergență explicată:** Surse externe (TennisStats.com) arată Brancaccio în "Bad Form" (35.7% win rate 2026) vs Vedder în formă foarte bună (67.6%) — și rank general Vedder #179 mai bun ca Brancaccio #209. Totuși modelul o favorizează CLAR pe Brancaccio. Research confirmă: divergența e reală și **complet explicabilă** — vezi Pasul 1.3 și Pasul 3.

---

## PASUL 1 — CSV Model + Market Check

### 1.1 — Verificare tb_p_cal
```
□ tb_p_cal = 0.0927 ≤ 0.10 ✅ — semnal primar U12.5 S2 activ
```

### 1.2 — Elo/Markov Double Guard
- p_markov = 0.8189, p_elo = 0.7041
- Gap = |81.89 − 70.41| = **11.5pp < 35pp** → ✅ NU e SKIP — gap mic, ambele metode converg clar spre Brancaccio

```
□ p_elo ≠ 0.0 ✅ | Gap Elo/Markov = 11.5pp < 35pp ✅ (consistență foarte bună)
```

### 1.3 — Robinhood / 4-Proxy Market Check + Explicație Divergență
Căutare directă: **niciun bookmaker (Robinhood, Oddsportal) nu are linie publicată** pentru acest meci.

**4-Proxy Market Check (pentru predicted_winner = Brancaccio):**

| Proxy | Valoare | Concluzie |
|---|---|---|
| p_markov | 81.9% | ≥75% ✅ |
| p_elo | 70.4% | <75% ❌ (dar apropiat) |
| Ranking gap general | #209 (Brancaccio) vs #179 (Vedder) | Vedder mai bine clasată general ⚠️ |
| **Record clay 2026 (proxy suplimentar, verificat prin research)** | **Brancaccio 11-4 (73.3%) pe clay** vs formă generală 35.7% | **✅ Confirmă puternic direcția modelului** |

**Explicație divergență (research confirmat):** "Bad Form" generală a lui Brancaccio e artefact al altor suprafețe + accidentare — a suferit o **accidentare la încheietura mâinii stângi** în calificările Australian Open 2026 (vs Korpatsch), operată chirurgical, revenire abia la finalul lunii martie. Pe **hard 2026: 0-2**. Pe **clay 2026: 11-4 (73.3%)**, inclusiv titlu la Santa Margherita di Pula (aprilie) și campioană Modena Memorial Fontana — la Modena, presa locală confirmă "never lost her serve" în R16 (6-2, 6-2). Acest record clay-specific e un proxy suplimentar solid care confirmă direcția modelului, chiar dacă ranking-ul general și Robinhood nu sunt disponibile.

```
□ PASUL 1: VALID, DIVERGENȚĂ EXPLICATĂ ✅
□ tb_p_cal ≤ 0.10 ✅ | Elo/Markov gap 11.5pp ✅ (consistent) | Record clay 2026 confirmă modelul (73.3% Brancaccio) ✅
```

---

## PASUL 2 — TennisAbstract / CoreTennis (suprafață clay)

**Surse utilizate:**
- [CoreTennis — Nuria Brancaccio](https://www.coretennis.net/tennis-player/nuria-brancaccio/71197/results.html)
- [TennisExplorer — Eva Vedder](https://www.tennisexplorer.com/player/vedder/)
- [matchstat.com — Brancaccio](https://matchstat.com/tennis/player/Nuria%20Brancaccio/)
- [TennisRatio — Vedder](https://www.tennisratio.com/players/EvaVedder.html)

Notă: TennisAbstract și Flashscore nu au randat complet via fetch (conținut JS) — date verificate încrucișat prin CoreTennis, TennisExplorer, matchstat și presă italiană/spaniolă.

---

### 2A. NURIA BRANCACCIO — Clay 2025-2026

**Sample:** N ≈ 19-20 meciuri clay — peste minim 10 ✅

#### Set 2 TB Rate pe Clay

| Data | Turneu | Adversar | Rank adversar | Scor | S2 |
|---|---|---|---|---|---|
| iul. 2025 | Bucharest 6 ITF, SF | Andrea Lázaro García | ~WTA 177-183 (**comparabil cu Vedder azi**) | 6-4, **7-6(6)** | TB |

**S2 TB count: 1/19-20 ≈ 5-6%** → **MULT SUB 15% ✅ (confirmatoriu puternic)** — cel mai bun rezultat din analizele făcute azi.

**Context:** Unicul TB Set2 a venit contra unei adversare de nivel APROAPE IDENTIC cu Vedder (~180 WTA) — deci e un semnal real, nu se poate respinge ca irelevant, dar rămâne un eveniment singular pe eșantion de ~20 meciuri (5-6%, mult sub prag).

Alte TB-uri identificate au fost în **Set 1** (Bertea, Schunk) sau **Set 3** (Gorgodze, Bejlek) — NU Set 2, verificate manual scor-cu-scor pentru a evita confuzia frecventă TB Set1/Set2/Set3.

#### S1 TB → S2 Pattern
2 meciuri cu Set1 TB (Bertea, Schunk) → **0/2 = 0%** → **✅ +1pp confirmare**

---

### 2B. EVA VEDDER — Clay 2025-2026

**Sample:** N ≈ 74 meciuri clay — foarte robust ✅

#### Set 2 TB Rate pe Clay

| Data | Turneu | Nivel | Adversar | Rank adversar | Scor | S2 |
|---|---|---|---|---|---|---|
| 18.03.25 | Vacaria ITF (BRA), R1 | ITF | Pedretti | ~300-500 | 6-4, 7-6(10) | TB |
| 17.04.25 | Madrid19 ITF, R16 | ITF | Charaeva | ITF slab | 6-4, 7-6(3) | TB (pierdut) |
| 13.07.25 | Hamburg WTA quali R1 | Tour | Zhenikhova | ~250-350 | 6-4, 6-7(2), 6-4 | TB |
| 21.05.26 | Portoroz2 ITF, R16 | ITF | Tsygourova | ITF slab | 4-6, 7-6(6), 6-2 | TB |
| 28.05.26 | Zaragoza ITF, R16 | ITF | Day | ITF slab | 6-3, 7-6(5) | TB (pierdut) |
| 13.06.26 | Ceska Lipa ITF, SF | ITF | Bertea | ITF slab | 6-4, 7-6(2) | TB |

**S2 TB count: 7/74 ≈ 9.5%** (2026: 4/25=16%, 2025: 3/49=6.1%) → **sub pragul de risc, dar cu rezervă**

**⚠️ Flag important:** 6 din 7 TB-uri vin din meciuri ITF de nivel scăzut sau meciuri pe echipe — **niciunul contra unei adversare cu profil de returnare comparabil cu Brancaccio**. Rata brută de 9.5% e favorabilă, dar riscul real vs o adversară de calitate mai mare (Brancaccio, clay specialist) e ușor subestimat de acest eșantion — nu poate fi tratat ca semnal 100% direct-transferabil.

#### S1 TB → S2 Pattern
8 meciuri cu Set1 TB → **0/8 = 0%** → **✅ semnal foarte curat**

```
□ PASUL 2: VALID, CU O REZERVĂ MINORĂ ✅⚠️
□ Brancaccio N≈20 ≥ 10 ✅ | S2 TB = 5-6% < 15% ✅ | S1→S2 = 0/2 = 0% ✅
□ Vedder N≈74 ≥ 10 ✅ | S2 TB = 9.5% < 15% ✅ (dar 6/7 vs adversare slabe) | S1→S2 = 0/8 = 0% ✅
```

---

## PASUL 3 — Context Manual

### 3.1 — Profil Brancaccio

**Biografie:** Italia, 26 ani, WTA #209 (Elo general 360, dar mult mai solidă pe clay). Antrenor: necunoscut public (confuzie frecventă online cu antrenorii fratelui ei, Raúl Brancaccio, ATP — nu se aplică).

**Context "Bad Form":** Accidentare încheietură mână stângă (calificări Australian Open 2026 vs Korpatsch) → operație → revenire finalul lunii martie. Pe hard 2026: 0-2. Eliminare rapidă în calificări Roland Garros vs Quevedo. **Pe clay specific: 11-4 (73.3%)** — titlu Santa Margherita di Pula, campioană Modena. R1 Roma (13 iul.) — **victorie facilă 6-2, 6-2 vs Jessica Pieri**, fără game de serviciu pierdut.

### 3.2 — Profil Vedder

**Biografie:** Olanda, 26 ani, WTA #179, career-high #198 atins 29 iunie 2026. Antrenor: necunoscut public.

**Formă excepțională:** **Serie de 11 victorii consecutive** (22 iunie → 14 iulie 2026), cu **2 titluri ITF consecutive pe zgură** (Palma del Río W50 — finală 6-4, 6-4 vs Micic; Haag2 W75 — finală 6-4, 6-3 vs Boskovic). Record 52 săpt.: 41-23 (64.1%), clay 30-16 (65.2%). R1 Roma (14 iul.) — victorie 6-4, 6-4 vs Francesca Pace. **Contra Top 100 carieră: 0-5 (0%)** — nu a bătut niciodată o jucătoare din top 100.

**Serviciu:** Hold clay 50%, dar DF scăzut (nu regalează puncte gratuit) — pierde game-uri de serviciu prin joc de bază insuficient de puternic pentru a domina schimburile, nu prin erori proprii.

### 3.3 — Factori Fizici și Oboseală

| Parametru | Brancaccio | Vedder |
|---|---|---|
| Meci R1 Roma | 6-2, 6-2 vs Pieri — facil, fără seturi pierdute la serviciu | 6-4, 6-4 vs Pace — facil |
| Zile odihnă | 1 | 2 |
| fatigue_flag model | False — **confirmat corect** | False |

**Concluzie:** Ambele au avut meciuri R1 curate, scurte, fără drenaj fizic. Diferența de 1 zi odihnă nu e un factor de risc real — modelul a evaluat corect absența fatigue-ului.

### 3.4 — Condiții Meteo

Roma, 16 iulie, ora 20:00 (seară) — condiții tipice mediteraneene de vară: cald-secetos, cer clar, precipitații practic zero. Seara rămâne caldă (peste 25°C probabil), dar fără riscul de caniculă extremă din meciurile de după-amiază analizate anterior azi. Fără impact structural pe dinamica seturilor.

### 3.5 — Miză și Motivație

Ambele au câștigat R1. Pentru **Brancaccio** (sub career-high #150): meciul e relevant pentru revenire în top 150-170 post-accidentare, pe suprafața ei preferată. Pentru **Vedder** (career-high recent #198, în plin breakthrough): oportunitate de a continua urcarea, presiune pozitivă de a susține valul de formă. Ambele motivate, fără asimetrie clară.

### 3.6 — H2H și Stil de Joc

**Fără H2H anterior** — prima confruntare pro. Brancaccio: date interne model arată profil agresiv la fileu (26.5 puncte fileu/meci) — nu confirmat extern granular, dar consistent cu un stil de clay specialist ofensiv. Vedder: descrisă generic ca baseline olandeză clasică, fără date publice contrare.

```
□ PASUL 3: Fără fatigue real | Meteo neutru | Motivație ridicată ambele | Vedder în serie de 11 victorii — factor de moment psihologic real, contrar direcției modelului
```

---

## SCOR FINAL

### Evaluare conform tabelului de scoring

| Criteriu | Valoare | Status |
|---|---|---|
| Pasul 1 — tb_p_cal | 0.0927 ≤ 0.10 | ✅ |
| Pasul 1 — Elo/Markov gap | 11.5pp, consistent | ✅ |
| Pasul 1 — direcție confirmată | Record clay 2026 Brancaccio 73.3% confirmă modelul | ✅ |
| Sample size Brancaccio | N≈20 ≥ 10 | ✅ |
| Sample size Vedder | N≈74 ≥ 10 | ✅ |
| Brancaccio S2 TB rate clay | 5-6% | **< 15% ✅ (excelent)** |
| Vedder S2 TB rate clay | 9.5% (dar 6/7 vs adversare slabe) | **< 15% ✅ cu rezervă** |
| Brancaccio S1→S2 | 0/2 = 0% | ✅ |
| Vedder S1→S2 | 0/8 = 0% | ✅ |
| Fatigue | Absent, confirmat corect ambele | ✅ |
| premium_u125 | no (min_hold 0.5007 — miss marginal 0.0007) | La limită, nu premium |
| Vedder momentum | Serie 11 victorii, career-high | ⚠️ factor de precauție psihologic |

**Conform tabel:** Toți pașii tehnici OK, S2 TB ≤15% pentru ambele, S1→S2 = 0% pentru ambele — pilonul Pasul 2 e printre cele mai solide analizate azi. Divergența de formă generală e complet explicată prin record clay-specific.

### **SCOR FINAL: 8/10 — RECOMANDĂM**

---

## VERDICT FINAL

### 🎯 SCOR: 8/10 — RECOMANDĂM

**Decizie:** U12.5 Set 2 — Brancaccio vs Vedder, Roma 125, Clay

**Argumente pentru:**
1. tb_p_cal=0.0927 sub prag, Elo/Markov foarte consistente (gap doar 11.5pp)
2. Brancaccio S2 TB doar 5-6% — cel mai bun rezultat din toate analizele de azi
3. Vedder S2 TB 9.5%, S1→S2 0/8 — foarte curat, chiar cu rezerva de calitate a adversarelor
4. Divergența "Bad Form" e explicată solid: Brancaccio 73.3% pe clay 2026, problema e pe alte suprafețe/accidentare, nu pe zgură
5. Ambele fără fatigue real, meteo neutru

**Factori de risc (motivul pentru 8/10, nu 9/10):**
- premium_u125 ratat cu doar 0.0007 — la limită, nu o confirmare completă
- Vedder în serie de 11 victorii consecutive + 2 titluri — moment psihologic real care poate face meciul mai competitiv decât indică modelul (70% Brancaccio)
- Vedder S2 TB rate posibil subestimat de eșantion (6/7 vs adversare ITF slabe, nu de calibru Brancaccio)
- Fără confirmare de piață (Robinhood/bookmaker indisponibile) — compensat parțial de record clay concret

---

## PREDICȚIE CÂȘTIGĂTOARE

**Brancaccio câștigă (probabilitate ~65-70%, ajustat pentru momentum-ul lui Vedder):**
- Scenariu principal: **Brancaccio 6-4 / 6-3** (experiență de clay specialist + serviciu solid domină, dar Vedder rezistă cu joc constant de bază)
- Alternativ: **Brancaccio 7-5 / 6-4** (Vedder ține pasul mai mult timp datorită formei excelente)

**Vedder câștigă (probabilitate ~30-35%, susținută de valul de formă):**
- Scenariu: **Vedder 6-4 / 3-6 / 6-3** — momentum-ul de 11 victorii consecutive o duce la un upset, posibil în 3 seturi

**Predicție scor U12.5 S2:** Indiferent de câștigătoare, Set 2 se încadrează probabil 8-10 game-uri (6-3, 6-4, 7-5). Rata combinată S2 TB (5-6% / 9.5%) susține solid Under 12.5 — acesta e, alături de Bassols-Rus, cel mai "clean" pick din cele analizate azi.

---

## SURSE

| Sursă | Utilizare |
|---|---|
| [CoreTennis — Nuria Brancaccio](https://www.coretennis.net/tennis-player/nuria-brancaccio/71197/results.html) | Clay match scores, S2 TB analysis |
| [matchstat.com — Brancaccio](https://matchstat.com/tennis/player/Nuria%20Brancaccio/) | Record clay 2026 (11-4), record hard (0-2) |
| [TennisExplorer — Eva Vedder](https://www.tennisexplorer.com/player/vedder/) | Clay match scores, S2 TB analysis |
| [TennisRatio — Vedder](https://www.tennisratio.com/players/EvaVedder.html) | Statistici serviciu (hold 62.7% general, DF) |
| [modenasportiva.it — Brancaccio Modena](https://modenasportiva.it/sport-vari/tennis-memorial-fontana-la-brancaccio-avanza-a-sorpresa-oggi-i-quarti-di-finale-78545/) | Confirmare formă clay, "never lost serve" |
| [spaziotennis.com — RG quali Brancaccio](https://www.spaziotennis.com/roland-garros/qualificazioni-roland-garros-2026-subito-fuori-brancaccio-sconfitta-da-quevedo-in-tre-set/131415) | Context accidentare/formă slabă |
| [tennisteen.it — titlu Santa Margherita di Pula](https://www.tennisteen.it/articoli/wta-tour/24344-week-17-brancaccio-vince-a-santa-margherita-di-pula.html) | Confirmare titlu clay aprilie 2026 |
| [livetennis.it — R1 Brancaccio vs Pieri](https://www.livetennis.it/post/467995/) | Confirmare scor R1, fatigue check |
| [WTA Official — Eva Vedder](https://www.wtatennis.com/players/325011/eva-vedder) | Career-high ranking, profil |
| [WTA Official — Rome 125 Draws](https://www.wtatennis.com/tournaments/1130/rome-125/2026/draws) | Confirmare R1 Vedder vs Pace |
| [Wikipedia — Eva Vedder](https://en.wikipedia.org/wiki/Eva_Vedder) | Background carieră |
| Model CSV 1.5_WTA_Under12_5.csv + 1.1_WTA_Winner.csv run 2026-07-16 | Model flags, tb_p_cal, hold rates, p_markov/p_elo |
