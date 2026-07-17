import pandas as pd
import numpy as np
import re

hist = pd.read_csv('data/historical/wta_matches_combined.csv', low_memory=False)
preds = pd.read_csv('simulations/WTA/backtests/wta_predictions.csv')

def parse_set2_tb(score):
    if not isinstance(score, str):
        return None
    score = re.sub(r'\s*(RET|W/O|DEF|Def\.?).*', '', score, flags=re.IGNORECASE).strip()
    sets = score.split()
    if len(sets) < 2:
        return None
    s2 = sets[1]
    if re.match(r'^7-6', s2) or re.match(r'^6-7', s2):
        return 1
    if re.match(r'^\d-\d', s2):
        return 0
    return None

hist['s2_tb'] = hist['score'].apply(parse_set2_tb)
valid = hist[hist['s2_tb'].notna()].copy()
valid['match_date'] = pd.to_datetime(valid['match_date'])
preds['match_date'] = pd.to_datetime(preds['match_date'])

merged = pd.merge(
    valid[['match_date','surface','tourney_name','winner_name','loser_name','s2_tb','winner_rank','loser_rank']],
    preds[['match_date','surface','winner_name','loser_name','p_tiebreak','p_hold_w','p_hold_l','p_elo','p_markov']],
    on=['match_date','surface','winner_name','loser_name'],
    how='inner'
)

merged['hold_asym'] = abs(merged['p_hold_w'] - merged['p_hold_l'])
merged['min_hold']  = merged[['p_hold_w','p_hold_l']].min(axis=1)
merged['fav_elo']   = merged['p_elo'].apply(lambda x: max(x, 1-x))

# Calibration thresholds: raw p_tiebreak that maps to tb_p_cal ~= 0.10
# Grass:  raw < 0.098 -> cal ~0.086 (OK)
# Clay:   raw < 0.158 -> cal ~0.093 (OK, based on isotonic breakpoints)
# Hard:   raw < 0.100 -> cal ~0.102 (approximate)
THRESHOLDS = {'Grass': 0.098, 'Clay': 0.155, 'Hard': 0.100}

# Robinhood proxy: fav_elo >= 0.60 = match not too even
# p_tb < 0.020 = proxy 9/10
# p_tb 0.020-0.060 = proxy 9/10 (but surface-adjusted)
# p_tb 0.060-thresh = proxy 8/10
# p_tb thresh-0.15  = FAIL (tb_p_cal ~0.127)

SURFACES = ['Grass', 'Clay', 'Hard']

print('='*80)
print('BACKTEST U12.5 SET 2 — TOATE SUPRAFETELE')
print('Nota: p_tiebreak = RAW (necalibrat). Thresholds ajustate per suprafata.')
print('Robinhood proxy: fav_elo >= 0.60 (favorita clara)')
print('='*80)

summary_rows = []

for surf in SURFACES:
    df = merged[merged['surface'] == surf].copy()
    thresh = THRESHOLDS[surf]
    baseline_hr = 1 - df['s2_tb'].mean()
    n_total = len(df)

    print(f'\n{"="*80}')
    print(f'  SUPRAFATA: {surf.upper()}')
    print(f'  Total meciuri: {n_total} | Baseline S2 TB rate: {df["s2_tb"].mean()*100:.1f}% | Baseline HR: {baseline_hr*100:.1f}%')
    print(f'  Threshold raw p_tb (tb_cal~0.10): {thresh}')
    print(f'{"="*80}')

    below = df[df['p_tiebreak'] < thresh]
    above = df[df['p_tiebreak'] >= thresh]

    print(f'\n  {"Segment":<50} {"N":>5}  {"HR%":>7}  {"TB losses":>10}')
    print('  ' + '-'*75)

    # Main buckets
    if surf == 'Grass':
        buckets_def = [
            ('BASELINE (toate)',                    df,  '—'),
            ('Sub prag (tb_cal~0.086, CoVe OK)',   below, 'OK'),
            ('  p_tb < 0.020  → 9/10 proxy',       df[df['p_tiebreak'] < 0.020], '9/10'),
            ('  p_tb 0.020-0.050 → 9/10',          df[(df['p_tiebreak']>=0.020)&(df['p_tiebreak']<0.050)], '9/10'),
            ('  p_tb 0.050-0.070 → 8/10',          df[(df['p_tiebreak']>=0.050)&(df['p_tiebreak']<0.070)], '8/10'),
            ('  p_tb 0.070-0.098 → 7/10',          df[(df['p_tiebreak']>=0.070)&(df['p_tiebreak']<0.098)], '7/10'),
            ('Peste prag (FAIL, tb_cal~0.127)',     above, 'PASS'),
        ]
    elif surf == 'Clay':
        buckets_def = [
            ('BASELINE (toate)',                    df,  '—'),
            ('Sub prag (tb_cal~0.093, CoVe OK)',   below, 'OK'),
            ('  p_tb < 0.020  → 9/10 proxy',       df[df['p_tiebreak'] < 0.020], '9/10'),
            ('  p_tb 0.020-0.060 → 9/10',          df[(df['p_tiebreak']>=0.020)&(df['p_tiebreak']<0.060)], '9/10'),
            ('  p_tb 0.060-0.100 → 8/10',          df[(df['p_tiebreak']>=0.060)&(df['p_tiebreak']<0.100)], '8/10'),
            ('  p_tb 0.100-0.155 → 7/10',          df[(df['p_tiebreak']>=0.100)&(df['p_tiebreak']<0.155)], '7/10'),
            ('Peste prag (FAIL)',                   above, 'PASS'),
        ]
    else:  # Hard
        buckets_def = [
            ('BASELINE (toate)',                    df,  '—'),
            ('Sub prag (tb_cal~0.102, CoVe OK)',   below, 'OK'),
            ('  p_tb < 0.020  → 9/10 proxy',       df[df['p_tiebreak'] < 0.020], '9/10'),
            ('  p_tb 0.020-0.050 → 9/10',          df[(df['p_tiebreak']>=0.020)&(df['p_tiebreak']<0.050)], '9/10'),
            ('  p_tb 0.050-0.080 → 8/10',          df[(df['p_tiebreak']>=0.050)&(df['p_tiebreak']<0.080)], '8/10'),
            ('  p_tb 0.080-0.100 → 7/10',          df[(df['p_tiebreak']>=0.080)&(df['p_tiebreak']<0.100)], '7/10'),
            ('Peste prag (FAIL)',                   above, 'PASS'),
        ]

    for label, sub, score in buckets_def:
        if len(sub) == 0:
            print(f'  {label:<50} {"0":>5}  {"N/A":>7}')
            continue
        hr = 1 - sub['s2_tb'].mean()
        tb_l = int(sub['s2_tb'].sum())
        print(f'  {label:<50} {len(sub):>5}  {hr*100:>6.1f}%  {tb_l:>10}')
        summary_rows.append({'surface': surf, 'bucket': score, 'n': len(sub), 'hr': hr*100})

    # Robinhood proxy filter
    print(f'\n  --- Cu filtru Robinhood proxy (fav_elo >= 0.60) ---')
    for label, sub, score in buckets_def:
        if score in ('—', 'OK', 'PASS') or len(sub) == 0:
            continue
        sub_rh = sub[sub['fav_elo'] >= 0.60]
        if len(sub_rh) == 0:
            continue
        hr_rh = 1 - sub_rh['s2_tb'].mean()
        print(f'  {label+" + RH>=0.60":<50} {len(sub_rh):>5}  {hr_rh*100:>6.1f}%')

    # Best combined filter
    print(f'\n  --- Cel mai bun filtru combinat ---')
    combos = [
        ('p_tb < thresh + fav_elo >= 0.65',
            below[below['fav_elo'] >= 0.65]),
        ('p_tb < thresh + min_hold < 0.70',
            below[below['min_hold'] < 0.70]),
        ('p_tb < thresh/2 + fav_elo >= 0.60',
            df[(df['p_tiebreak'] < thresh/2) & (df['fav_elo'] >= 0.60)]),
        ('p_tb < thresh/2 (elite, fara altceva)',
            df[df['p_tiebreak'] < thresh/2]),
    ]
    for label, sub in combos:
        if len(sub) == 0:
            continue
        hr = 1 - sub['s2_tb'].mean()
        print(f'  {label:<50} {len(sub):>5}  {hr*100:>6.1f}%')

# Summary table
print('\n\n' + '='*80)
print('REZUMAT — HR% PER SUPRAFATA SI SCOR COVE (proxy)')
print('='*80)
print(f'  {"Suprafata":<10} {"Baseline":>9} {"7/10":>8} {"8/10":>8} {"9/10":>8}')
print('  ' + '-'*50)
for surf in SURFACES:
    df = merged[merged['surface'] == surf]
    thresh = THRESHOLDS[surf]
    baseline = (1 - df['s2_tb'].mean()) * 100

    # 9/10: p_tb < thresh/2
    s9 = df[df['p_tiebreak'] < thresh/2]
    # 8/10: p_tb thresh/2 to thresh*0.7
    if surf == 'Grass':
        s8 = df[(df['p_tiebreak']>=0.050)&(df['p_tiebreak']<0.070)]
        s7 = df[(df['p_tiebreak']>=0.070)&(df['p_tiebreak']<thresh)]
    elif surf == 'Clay':
        s8 = df[(df['p_tiebreak']>=0.060)&(df['p_tiebreak']<0.100)]
        s7 = df[(df['p_tiebreak']>=0.100)&(df['p_tiebreak']<thresh)]
    else:
        s8 = df[(df['p_tiebreak']>=0.050)&(df['p_tiebreak']<0.080)]
        s7 = df[(df['p_tiebreak']>=0.080)&(df['p_tiebreak']<thresh)]

    hr7 = f'{(1-s7["s2_tb"].mean())*100:.1f}%(N={len(s7)})' if len(s7)>0 else 'N/A'
    hr8 = f'{(1-s8["s2_tb"].mean())*100:.1f}%(N={len(s8)})' if len(s8)>0 else 'N/A'
    hr9 = f'{(1-s9["s2_tb"].mean())*100:.1f}%(N={len(s9)})' if len(s9)>0 else 'N/A'
    print(f'  {surf:<10} {baseline:>8.1f}% {hr7:>16} {hr8:>16} {hr9:>16}')

print('''
NOTA IMPORTANTA:
- Robinhood nu e in model, e filtru manual. Proxy: fav_elo >= 0.60/0.65
- TennisAbstract S2 TB rate (< 15%) e filtrul real al scorului 8/10 vs 7/10
- Calibrarea isotonica face ca valorile calibrate sa fie constante pe bucketuri:
    Grass:  cal = 0.086 (raw < 0.098) sau 0.127 (raw 0.098-0.148)
    Clay:   cal = 0.086-0.093 (raw < 0.155) sau 0.105+ (raw > 0.155)
    Hard:   calibrare mai fina, mai multe puncte de inflexiune
- Sample mic pe Clay si Grass → CI larg, concluzii orientative
''')
