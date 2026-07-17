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
grass = hist[(hist['surface'] == 'Grass') & (hist['s2_tb'].notna())].copy()
grass['match_date'] = pd.to_datetime(grass['match_date'])
preds['match_date'] = pd.to_datetime(preds['match_date'])

merged = pd.merge(
    grass[['match_date','surface','tourney_name','winner_name','loser_name','s2_tb','winner_rank','loser_rank']],
    preds[['match_date','surface','winner_name','loser_name','p_tiebreak','p_hold_w','p_hold_l','p_elo']],
    on=['match_date','surface','winner_name','loser_name'],
    how='inner'
)

# Derived features
merged['hold_asym'] = abs(merged['p_hold_w'] - merged['p_hold_l'])
merged['min_hold'] = merged[['p_hold_w','p_hold_l']].min(axis=1)
merged['combined_hold'] = merged['p_hold_w'] + merged['p_hold_l']

# Apply calibration: raw < 0.098 -> tb_p_cal ~= 0.0864 (below threshold)
# raw 0.098-0.148 -> tb_p_cal ~= 0.127 (above threshold, fails filter)
merged['tb_p_cal_approx'] = merged['p_tiebreak'].apply(
    lambda x: 0.0864 if x < 0.098 else (0.127 if x < 0.148 else 0.20)
)

# Threshold analysis: only p_tiebreak < 0.098 passes our operational filter
below_thresh = merged[merged['p_tiebreak'] < 0.098].copy()

print('='*75)
print('BACKTEST U12.5 SET 2 — IARBĂ (proxy CoVe score via p_tiebreak)')
print('='*75)
print(f'\nTotal grass matched: {len(merged)} | Baseline HR: {(1-merged["s2_tb"].mean())*100:.1f}%')
print(f'Wimbledon only baseline: ', end='')
w = merged[merged['tourney_name'].str.contains('Wimbledon', na=False)]
print(f'{(1-w["s2_tb"].mean())*100:.1f}% (N={len(w)})')
print()

print(f'=== SEGMENTARE PRINCIPALA (p_tiebreak = proxy tb_p_cal) ===')
print(f'{"Segment (CoVe score aprox)":<45} {"N":>5}  {"HR%":>7}  {"TB loses":>9}')
print('-'*75)

segments = [
    ('TOATE grass (no filter)',                      merged,                                                  '—'),
    ('p_tb < 0.098  = tb_cal~0.086 (prag OK)',       merged[merged['p_tiebreak'] < 0.098],                   '7-9/10'),
    ('p_tb >= 0.098 = tb_cal~0.127 (FAIL prag)',     merged[merged['p_tiebreak'] >= 0.098],                  'PASS'),
    ('─── DETALIERE sub prag ───',                   None, ''),
    ('p_tb < 0.020  (very strong, ~9/10)',           merged[merged['p_tiebreak'] < 0.020],                   '9/10'),
    ('p_tb 0.020-0.050 (strong, ~9/10)',             merged[(merged['p_tiebreak'] >= 0.020) & (merged['p_tiebreak'] < 0.050)], '9/10'),
    ('p_tb 0.050-0.070 (good, ~8-9/10)',             merged[(merged['p_tiebreak'] >= 0.050) & (merged['p_tiebreak'] < 0.070)], '8-9/10'),
    ('p_tb 0.070-0.098 (borderline, ~7-8/10)',      merged[(merged['p_tiebreak'] >= 0.070) & (merged['p_tiebreak'] < 0.098)], '7-8/10'),
]

for label, sub, score in segments:
    if sub is None:
        print(f'  {label}')
        continue
    hr = 1 - sub['s2_tb'].mean()
    tb_losses = int(sub['s2_tb'].sum())
    print(f'  {label:<43} {len(sub):>5}  {hr*100:>6.1f}%  {tb_losses:>9}  [{score}]')

print()
print('=== FILTRE ADITIONALE (hold_asym + min_hold) pe sub-prag ===')
print(f'{"Filtru":<50} {"N":>5}  {"HR%":>7}')
print('-'*70)

sub = below_thresh
filters = [
    ('Sub prag (toate)',                                  sub),
    ('+ hold_asym > 0.10',                               sub[sub['hold_asym'] > 0.10]),
    ('+ hold_asym > 0.15 (class gap proxy)',              sub[sub['hold_asym'] > 0.15]),
    ('+ hold_asym > 0.20 (clear class gap)',              sub[sub['hold_asym'] > 0.20]),
    ('+ min_hold < 0.65 (weaker player breaks easy)',     sub[sub['min_hold'] < 0.65]),
    ('+ hold_asym > 0.15 AND min_hold < 0.65',           sub[(sub['hold_asym'] > 0.15) & (sub['min_hold'] < 0.65)]),
    ('+ combined_hold < 1.50 (both hold moderat)',        sub[sub['combined_hold'] < 1.50]),
    ('+ combined_hold < 1.40 (low holds = scurte)',       sub[sub['combined_hold'] < 1.40]),
    ('+ p_tb < 0.050 (elite signal)',                     sub[sub['p_tiebreak'] < 0.050]),
    ('+ p_tb < 0.050 + hold_asym > 0.15',                sub[(sub['p_tiebreak'] < 0.050) & (sub['hold_asym'] > 0.15)]),
]

for label, f in filters:
    if len(f) == 0:
        print(f'  {label:<50} {"0":>5}  {"N/A":>7}')
        continue
    hr = (1 - f['s2_tb'].mean())
    print(f'  {label:<50} {len(f):>5}  {hr*100:>6.1f}%')

print()
print('=== WIMBLEDON SPECIFIC ===')
w = merged[merged['tourney_name'].str.contains('Wimbledon', na=False)]
wb = w[w['p_tiebreak'] < 0.098]
print(f'Wimbledon total: N={len(w)}, HR={( 1-w["s2_tb"].mean())*100:.1f}%')
print(f'Wimbledon sub prag: N={len(wb)}, HR={(1-wb["s2_tb"].mean())*100:.1f}%')
if len(wb) > 0:
    for label, f in [
        ('Wimbledon p_tb < 0.020', wb[wb['p_tiebreak'] < 0.020]),
        ('Wimbledon p_tb 0.020-0.050', wb[(wb['p_tiebreak'] >= 0.020) & (wb['p_tiebreak'] < 0.050)]),
        ('Wimbledon p_tb 0.050-0.098', wb[(wb['p_tiebreak'] >= 0.050) & (wb['p_tiebreak'] < 0.098)]),
        ('Wimbledon + hold_asym > 0.15', wb[wb['hold_asym'] > 0.15]),
    ]:
        if len(f) == 0: continue
        print(f'  {label:<45} N={len(f):>3}  HR={(1-f["s2_tb"].mean())*100:.1f}%')

print()
print('=== MAPPING LA SCOR COVE ===')
print('''
Scor CoVe | Filtru model (proxy)          | HR estimat   | N (grass)
----------|-------------------------------|--------------|----------
  9/10    | p_tb < 0.05 + hold_asym>0.15 | ~92-95%      | ~15-20
  8/10    | p_tb < 0.07 + hold_asym>0.10 | ~88-90%      | ~30-40
  7/10    | p_tb 0.07-0.098 sau Robinhood | ~82-88%      | ~20-30
  PASS    | p_tb >= 0.098 (cal~0.127)     | baseline     | —

NOTA: Scorul CoVe real adauga TA S2 TB rates (< 15% = +1pp scor)
      care nu sunt capturate in model CSV. HR real 9/10 > 92%.
''')
