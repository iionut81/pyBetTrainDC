# WTA U12.5 Set 2 — Triple Filter CoVe
## Aurora Zantedeschi vs Martina Trevisan
### WTA 125 Roma (ATV Bancomat Tennis Open) | Clay | R16 | 17:00 CEST July 16, 2026

---

## MODEL DATA (din 1.5_WTA_Under12_5.csv + 1.1_WTA_Winner.csv — run 2026-07-16)

| Câmp | Valoare | Interpretare |
|---|---|---|
| p_hold_a (Zantedeschi) | 0.3985 | Ține doar 39.85% din servicii pe clay — servitoare foarte slabă |
| p_hold_b (Trevisan) | 0.5291 | Ține 52.9% din servicii — mai solidă la serviciu |
| hold_asym | 0.1306 | **Sub pragul premium (>0.15)** |
| min_hold | 0.3985 | Zantedeschi = jucătoarea mai slabă la serviciu — la 0.0015 de danger_zone (prag 0.40) |
| bci | 0.0786 | |
| tb_p_raw | 0.0252 | |
| **tb_p_cal** | **0.0865** | **8.65% probabilitate TB în Set 2 — sub prag 10% ✅** |
| p_u125 | 0.9135 | 91.35% probabilitate U12.5 S2 |
| blowout_score | 10 | Maxim — dar direcția "blowout" e neclară (vezi 1.3) |
| premium_elite | no | |
| **premium_u125** | **no** | hold_asym=0.1306 < 0.15 — NU e pick premium |
| danger_zone | no | min_hold=0.3985 — marginal sub prag (0.40) |
| UNSTABLE flag | — (gol) | NO UNSTABLE |
| fatigue_flag_a | True | **CONFIRMAT real (vezi 3.3)** — Zantedeschi 3 seturi cu 2 zile în urmă |
| fatigue_flag_b | False | |
| p_markov | 0.2387 | Zantedeschi câștigă doar 23.9% prin simulare hold-rate |
| p_elo | 0.3376 | Zantedeschi câștigă 33.8% prin Elo istoric |
| predicted_winner | Martina Trevisan | |
| p_cal (Winner) | 0.6606 | Trevisan câștigă 66.1% — calibrat |

**Notă critică — direcție neclară:** Rank/Elo general (surse externe: Zantedeschi #292/Elo 237 vs Trevisan #405/Elo 154) ar sugera Zantedeschi favorită. Modelul nostru (Elo surface-specific + hold rates clay) o dă pe **Trevisan** favorită (66%). Aceasta NU e o eroare — Trevisan e ex-semifinalistă Roland Garros 2022 (calitate de clay istorică foarte ridicată), în revenire după accidentare, nu în declin natural de vârstă. Vezi Pasul 3 pentru context complet.

---

## PASUL 1 — CSV Model + Market Check

### 1.1 — Verificare tb_p_cal
```
□ tb_p_cal = 0.0865 ≤ 0.10 ✅ — semnal primar U12.5 S2 activ
```

### 1.2 — Elo/Markov Double Guard
- p_markov = 0.2387 (Zantedeschi câștigă 23.9%)
- p_elo = 0.3376 (Zantedeschi câștigă 33.8%)
- Gap = |23.87 − 33.76| = **9.9pp < 35pp** → ✅ NU e SKIP — de fapt gap FOARTE mic, cele două metode sunt consistente ca direcție (ambele favorizează Trevisan)

```
□ p_elo ≠ 0.0 ✅ (0.3376 — date reale)
□ Gap Elo/Markov = 9.9pp < 35pp ✅ (consistență bună între metode)
```

### 1.3 — Robinhood / 4-Proxy Market Check
Căutare directă: **niciun bookmaker (Robinhood, Oddsportal) nu are linie publicată pentru acest meci specific** — confirmat prin cercetare directă.

**4-Proxy Market Check (fallback, pentru predicted_winner = Trevisan):**

| Proxy | Valoare (pentru Trevisan) | Concluzie |
|---|---|---|
| p_markov | 76.1% | ≥75% ✅ (marginal) |
| p_elo | 66.2% | <75% ❌ |
| Ranking gap | #292 (Zantedeschi) vs #405 (Trevisan) | **Trevisan e favorizată de model DAR e mai slab clasată** — direcție neconvențională ⚠️ |
| H2H | Fără istoric (prima confruntare pro) | N/A |

**Rezultat: proxy-urile NU converg clar.** Acesta e un caz special: modelul favorizează jucătoarea cu ranking general mai slab (Trevisan #405) datorită calității istorice de clay (ex-top-40, SF Roland Garros 2022) și hold rate mai bun în simulare, NU datorită unui "class gap" convențional. Nu e un semnal de eroare, dar nu poate fi tratat ca "class gap confirmat pe toate proxy-urile" gen Jones-Urgesi.

```
□ PASUL 1: VALID CU REZERVĂ ⚠️ (semnal TB solid, dar market-check direcție neconvențională/neconfirmată)
□ tb_p_cal ≤ 0.10 ✅ | Elo/Markov gap 9.9pp ✅ (consistent) | 4-Proxy: MIXT, fără confirmare piață
```

---

## PASUL 2 — TennisAbstract / CoreTennis (suprafață clay)

**Surse utilizate:**
- [CoreTennis — Aurora Zantedeschi](https://www.coretennis.net/tennis-player/aurora-zantedeschi/83021/results.html)
- [CoreTennis / TennisExplorer — Martina Trevisan] (cross-verificat, CoreTennis avea erori de an corectate prin tennis.com)
- [TennisRatio — Zantedeschi](https://www.tennisratio.com/players/AuroraZantedeschi.html)

---

### 2A. AURORA ZANTEDESCHI — Clay 2025-2026

**Sample:** N ≈ 32 meciuri clay — peste minim 10 ✅

#### Set 2 TB Rate pe Clay

| Data | Turneu | Nivel | Scor | S2 | Context adversar |
|---|---|---|---|---|---|
| Contrexeville W125, QF | WTA 125 | 6-4, **7-6(5)** | TB | Anna Blinkova, ~#102, ex-top 50 — **calibru comparabil cu Trevisan** |
| Aix-les-Bains W35, SF | ITF W35 | 6-2, **7-6(3)** | TB | Madison Sieg, ~#426, ITF domestic — sub nivelul Trevisan |
| Caserta W75, 1/8 | ITF W75 | 3-6, **7-6(3)**, 6-2 | TB | Alessandra Mazzola, ~#375-400 |
| Chiasso W75 quals, 1/8 | ITF W75 | 6-4, **7-6(6)** | TB | Alessandra Mazzola |

**S2 TB count: 4/32 = 12.5%** → **SUB 15% ✅ (confirmatoriu)**

**Context:** Doar 1/4 TB (Blinkova) a venit contra unei adversare de nivel comparabil cu Trevisan (ex-top). Restul, nivel ITF slab. Nu susține un pattern specific de risc contra "ex-jucătoare de top în revenire".

#### S1 TB → S2 Pattern
Meciuri cu Set1 TB: Lew Yan Foon (7-6→6-3), Gabriela Ce (7-6→6-1), Maria Sara Popa (7-6→5-7).
**Rate: 0/3 = 0%** → **✅ +1pp confirmare**

#### Statistici serviciu (context pentru hold=39.85%)
TennisRatio (52 săpt.): ace rate 1.4/meci, break points saved 56%, 1st serve won ~61%. Profil coerent cu o servitoare structural slabă — hold-ul scăzut din model NU e o eroare de date, e consistent cu profilul real. Notă: în analiza CoVe precedentă (Ferro-Zantedeschi, 14 iulie, deja salvată în repo), modelul o dăduse pe Zantedeschi cu p_hold=0.3793 — aproape identic, confirmă consistență între adversari diferiți.

---

### 2B. MARTINA TREVISAN — Clay 2026 (perioada de comeback)

**Sample:** N ≈ 19 meciuri clay (2026, post-revenire) — peste minim 10 ✅

#### Set 2 TB Rate pe Clay

| Data | Turneu | Nivel | Adversar (rank la moment) | Scor | Rezultat |
|---|---|---|---|---|---|
| 16.06.2026 | Brescia | WTA 125, R1 | Carole Monnet, ~#190-200 | 7-5, **7-6(5)** | **Pierdut** |
| 10.06.2026 | Modena | WTA 125, R16 | Dominika Salkova, ~#124-129 (cap serie #4) | 6-1, **7-6(5)** | **Pierdut** |
| 03.04.2026 | Sta. Margherita di Pula | ITF W35, QF | Rositsa Dencheva, 19 ani, ~#428 | 6-3, **7-6(4)** | **Pierdut** |

**S2 TB count: 3/19 ≈ 16%** → **PESTE pragul de confirmare (15%), dar SUB pragul de risc (33%)** — zonă de precauție moderată

**Context critic:** Toate 3 TB-uri vin din perioada actuală de comeback (2026), la nivel WTA125/ITF direct comparabil cu Zantedeschi (#124-#428) — NU din era ei de top-40. Ipoteza "TB-urile sunt irelevante, din trecutul de elită" e **falsă**. **Toate 3 au fost pierderi pentru Trevisan** — semnal că, atunci când Set 2 se prelungește la TB, Trevisan tinde să cedeze (posibil legat de rezistență fizică post-operație).

#### S1 TB → S2 Pattern
Niciun meci din eșantion cu Set1 TB → **N/A**, nu 0% (nu s-a produs premisa, nu poate fi evaluat).

```
□ PASUL 2: VALID CU ATENȚIONARE ⚠️
□ Zantedeschi N=32 ≥ 10 ✅ | S2 TB = 12.5% < 15% ✅ | S1→S2 = 0/3 = 0% ✅
□ Trevisan N=19 ≥ 10 ✅ | S2 TB = 16% — PESTE prag confirmare, SUB prag risc ⚠️ | S1→S2 = N/A
□ Toate 3 TB-uri Trevisan = pierderi, la nivel relevant pentru meciul de azi — semnal real, nu zgomot istoric
```

---

## PASUL 3 — Context Manual

### 3.1 — Profil Zantedeschi

**Biografie:** Italia, 25 ani, WTA #292, Elo 237. Stil declarat: "prevalentemente aggressiva" — serviciu, forehand, backhand, drop shot. Antrenor: necunoscut/neconfirmat (nu e semnal negativ, doar lipsă de informație publică).

**Formă 2026:** 66.7% win rate (24/36) — solidă, în ascensiune. Recent: SF Contrexeville W125, finală Aix-les-Bains W35 (începutul lui iulie). R1 Roma (14 iul.) — victorie în 3 seturi vs Fiona Ferro (favorită 79% în model): **6-4, 2-6, 6-3**.

### 3.2 — Profil Trevisan

**Biografie:** Italia, 32 ani, 1.60m/54kg, WTA #405, Elo 154. Ex-#18 mondial, **semifinalistă Roland Garros 2022**, carieră $4.16M premii. Antrenor: **Matteo Catarsi**.

**Context declin — comeback, nu retragere:** Operație pentru **sindrom Haglund** (călcâi, picior stâng) în martie 2025, ~9 luni pauză, revenire iulie 2025 la turnee mici (ITF→W75→WTA125), reconstruind rankingul de la zero. Declarații recente: *"fisicamente sto meglio e sogno l'Australia"* (14 iul. 2026), *"sogno uno Slam e la top 100"* — obiectiv declarat: calificare Australian Open, NU ultimul sezon de carieră.

**Formă 2026:** 41.7% win rate (10/24) — dar explicabil prin recuperare fizică, nu declin de nivel tehnic. R1 Roma (14 iul.) — victorie dominantă vs Tyra Grant: **6-0, 6-3** (derby italian).

**Problema serviciului:** 6.62 DF/meci (foarte ridicat) — plauzibil biomecanic: operația a fost exact la piciorul de impuls al serviciului, motor pattern încă afectat. Contradicție parțială cu hold_b=52.91% din model — posibil model captează holduri din meciuri câștigate facil (adversare ITF slabe), nu robustețe sub presiune reală.

### 3.3 — Factori Fizici și Oboseală (CONFIRMAT, nu artefact)

| Parametru | Zantedeschi | Trevisan |
|---|---|---|
| Meci R1 Roma | 6-4, 2-6, 6-3 vs Ferro — **3 seturi reale** | 6-0, 6-3 vs Grant — straight sets |
| Zile odihnă | **2** | 2 |
| fatigue_flag model | **True — CONFIRMAT** | False |

**Diferență față de analiza Bassols-Rus:** aici flag-ul de oboseală al modelului e **real, verificat cu sursă**, nu fals-pozitiv. Zantedeschi joacă la 2 zile după un meci solicitant de 3 seturi câștigat ca underdog. Impact posibil: serviciu (deja slab, 39.85%) suplimentar afectat de oboseală → break-uri și mai facile pentru Trevisan → paradoxal, poate întări semnalul Under (game-uri scurte, decisive), nu riscul de TB.

### 3.4 — Condiții Meteo (factor de risc real)

**Sursă:** [romadailynews.it](https://www.romadailynews.it/senza-categoria/meteo-roma-e-lazio-16-luglio-2026-caldo-estremo-fino-a-40-gradi-0955988/), [meteo.it — alertă caniculă](https://www.meteo.it/notizie/meteo-allerta-caldo-16-luglio-2026-italia-citta-a-rischio-2d1e292d)

| Parametru | Valoare |
|---|---|
| Temperatură | **26-38°C** |
| Alertă | **Caniculă activă 08:00-19:59** — meciul (17:00) e ÎN interiorul alertei |
| Precipitații | Neglijabile |

**Impact:** Căldură extremă favorizează schimburi mai scurte și oboseală accelerată — poate afecta disproporționat Trevisan (fitness încă în reconstrucție post-operație) sau Zantedeschi (deja obosită din meciul de 3 seturi). Efect net ambiguu pe cine câștigă, dar tinde să FAVORIZEZE Under (jucătoare obosite = servicii mai scurte = break-uri rapide, nu schimburi lungi la egalitate).

### 3.5 — Miză și Motivație

Ambele au câștigat deja R1 (13-14 iul.) — motivație ridicată pentru amândouă. Derby 100% italian, presiune publică egală (nu asimetrică). Trevisan e susținută public de Tathiana Garbin (căpitan echipă Italia) ca simbol al reconstrucției tenisului italian feminin — presiune simbolică suplimentară, dar și motivație pozitivă puternică.

### 3.6 — H2H și Context Psihologic

**Fără H2H anterior** — prima confruntare la nivel profesionist. Fără bagaj psihologic din meciuri trecute. Trevisan e veterana cu experiență de Grand Slam (avantaj mental sub presiune), Zantedeschi e underdog-ul de ranking dar cu formă net superioară 2026.

```
□ PASUL 3: Fatigue Zantedeschi CONFIRMAT real | Meteo cald extrem (posibil accelerează break-urile) | Motivație ridicată ambele | Fără H2H
```

---

## SCOR FINAL

### Evaluare conform tabelului de scoring

| Criteriu | Valoare | Status |
|---|---|---|
| Pasul 1 — tb_p_cal | 0.0865 ≤ 0.10 | ✅ |
| Pasul 1 — Elo/Markov gap | 9.9pp, consistent | ✅ |
| Pasul 1 — 4-Proxy market | Mixt, fără confirmare piață, direcție neconvențională | ⚠️ |
| Sample size Zantedeschi | N=32 ≥ 10 | ✅ |
| Sample size Trevisan | N=19 ≥ 10 | ✅ |
| Zantedeschi S2 TB rate clay | 12.5% | **< 15% ✅** |
| Trevisan S2 TB rate clay | 16% | **15-25% ⚠️ (peste confirmare, sub risc)** |
| Zantedeschi S1→S2 | 0/3 = 0% | ✅ |
| Trevisan S1→S2 | N/A (fără sample) | neutru |
| Fatigue flag Zantedeschi | CONFIRMAT real | ⚠️ risc autentic |
| premium_u125 | no (hold_asym 0.1306 < 0.15) | Pick standard, nu premium |
| danger_zone | no (marginal, 0.3985 vs prag 0.40) | ✅ dar la limită |

**Conform tabel de scoring:** "Pași OK, S2 TB 15-25%, S1→S2 20-33% → 8/10" — Trevisan cade în banda 15-25%, dar combinat cu fatigue confirmat real, market check nemixt, și lipsa de premium flag, ajustez la:

### **SCOR FINAL: 7/10 — RECOMANDARE MODERATĂ**

---

## ⚠️ ATENȚIONARE (conform regulă backtest warning mode)

Acest pick e **sub minimul de suprafață pentru clay (8/10 + confirmare piață)**. Motive concrete:
1. Trevisan S2 TB rate = 16%, peste pragul de confirmare de 15% — toate 3 TB-uri din eșantion au fost pierderi ale ei, la nivel relevant (nu istoric irelevant)
2. Fără confirmare de piață (Robinhood/bookmaker indisponibile), iar 4-proxy e mixt/neconvențional (modelul favorizează jucătoarea cu ranking mai slab)
3. Fatigue confirmat real pentru Zantedeschi (nu doar teoretic) — introduce incertitudine fizică autentică
4. Nu e pick premium (hold_asym 0.1306 < 0.15)

**HR de referință (backtest general clay, tb_p_cal≤0.10, fără premium):** ~86-90%. Acest pick e la marginea inferioară a acelui interval, nu în zona premium (93%+).

---

## VERDICT FINAL

### 🎯 SCOR: 7/10 — RECOMANDARE MODERATĂ (nu HIGH confidence)

**Decizie:** U12.5 Set 2 — Zantedeschi vs Trevisan, Roma 125, Clay

**Argumente pentru:**
1. tb_p_cal=0.0865 sub prag, Elo/Markov consistente (gap doar 9.9pp)
2. Zantedeschi S2 TB doar 12.5%, S1→S2 0%
3. Ambele jucătoare au servicii vulnerabile (39.85% / potențial mai fragil decât arată 52.91% dat fiind DF ridicat) → break-uri probabile, game-uri scurte
4. Căldură extremă tinde să scurteze schimburile, nu să le prelungească

**Factori de risc (motivul pentru 7/10, nu 8-9):**
- Trevisan S2 TB=16%, peste prag, toate din eșantion relevant, toate pierdute
- Fatigue Zantedeschi confirmat real (nu teoretic) — 3 seturi cu 2 zile în urmă, în căldură extremă
- Fără confirmare de piață, 4-proxy mixt
- Nu e pick premium

---

## PREDICȚIE CÂȘTIGĂTOARE

**Trevisan câștigă (probabilitate ~55%, per model + calitate istorică clay):**
- Scenariu principal: **Trevisan 6-4 / 6-3** (retur solid + experiență clay îi permite să exploateze serviciul slab al lui Zantedeschi, în ciuda propriilor DF-uri)
- Alternativ 3 seturi: **Zantedeschi 6-4 / Trevisan 6-3 / Trevisan 6-4** (Trevisan cedează un set din cauza propriei fragilități pe serviciu, dar se impune fizic pe final)

**Zantedeschi câștigă (probabilitate ~45%, per formă 2026 + fitness relativ mai proaspăt):**
- Scenariu: **Zantedeschi 7-5 / 6-4** (profită de căldură + DF-urile lui Trevisan, formă superioară 2026 compensează oboseala)

**Predicție scor U12.5 S2:** Indiferent de câștigătoare, Set 2 se încadrează probabil 8-10 game-uri (6-3, 6-4, 7-5). Risc principal de Over: dacă Trevisan repetă pattern-ul din eșantionul recent (S2 TB 16%, toate pierdute în circumstanțe similare) — un scenariu de minoritate, dar nu negligibil.

---

## SURSE

| Sursă | Utilizare |
|---|---|
| [CoreTennis — Aurora Zantedeschi](https://www.coretennis.net/tennis-player/aurora-zantedeschi/83021/results.html) | Clay match scores, S2 TB analysis |
| [TennisRatio — Zantedeschi](https://www.tennisratio.com/players/AuroraZantedeschi.html) | Statistici serviciu |
| [tennis.com — Monnet vs Trevisan Brescia](https://www.tennis.com/tournaments/internazionali-femminili-di-brescia/matches/c-monnet-vs-m-trevisan-2026-06-16) | Confirmare scor/dată S2 TB |
| [Sky Sport — accidentare Trevisan](https://sport.sky.it/tennis/2025/03/03/martina-trevisan-infortunio-news) | Context sindrom Haglund |
| [Virgilio Sport — calvar accidentare](https://sport.virgilio.it/martina-trevisan-calvario-infortunio-sindrome-haglund-operazione-899935) | Context medical |
| [TennisWorldItalia — interviu Trevisan](https://www.tennisworlditalia.com/tennis/news/Interviste_Tennis/103476/) | Recuperare, mindset |
| [tiebreaktennis.it — R1 Trevisan vs Grant](https://www.tiebreaktennis.it/wta-125-roma-2026-martina-trevisan-domina-il-derby-con-tyra-grant-garbin-promuove-il-tennis-azzurro/) | Confirmare R1, declarații |
| [ventidisport.it — Trevisan interviu](https://ventidisport.it/2026/tennis/wta-125-roma-brilla-trevisan-nel-derby-contro-grant-fisicamente-sto-meglio-e-sogno-laustralia-garbin-torneo-importante-per-le-nostre-ragazze/) | Motivație, obiectiv Australian Open |
| [oasport.it — Trevisan vs Gibson mai 2026](https://www.oasport.it/2026/05/live-trevisan-gibson-6-0-6-3-wta-roma-2026-in-diretta-lazzurra-torna-nel-tennis-che-conta-dopo-una-lunga-assenza/) | Confirmare fragilitate serviciu (44% 2nd serve) |
| [WTA Official — Martina Trevisan](https://www.wtatennis.com/players/316266/martina-trevisan) | Antrenor Matteo Catarsi |
| [spaziotennis.com — interviu Zantedeschi](https://www.spaziotennis.com/interv/esclusiva-zantedeschi-professionista-dai-19-anni-tennis-viaggionon-contano-solo-risultati/135793) | Stil de joc |
| [romadailynews.it — meteo Roma 16 iulie](https://www.romadailynews.it/senza-categoria/meteo-roma-e-lazio-16-luglio-2026-caldo-estremo-fino-a-40-gradi-0955988/) | Meteo, alertă caniculă |
| [meteo.it — alertă caniculă](https://www.meteo.it/notizie/meteo-allerta-caldo-16-luglio-2026-italia-citta-a-rischio-2d1e292d) | Confirmare fereastră alertă |
| Analysis/WTA_U125_Set2_Ferro_Zantedeschi_Rome_2026-07-14.md (repo intern) | Confirmare meci R1 Zantedeschi + p_hold istoric |
| Model CSV 1.5_WTA_Under12_5.csv + 1.1_WTA_Winner.csv run 2026-07-16 | Model flags, tb_p_cal, hold rates, p_markov/p_elo |
