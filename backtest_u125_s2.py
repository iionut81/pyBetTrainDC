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
print(f'Grass matches with valid S2: {len(grass)}')
print(f'S2 TB rate baseline (grass): {grass["s2_tb"].mean()*100:.1f}%')
print()

grass['match_date'] = pd.to_datetime(grass['match_date'])
preds['match_date'] = pd.to_datetime(preds['match_date'])

merged = pd.merge(
    grass[['match_date','surface','winner_name','loser_name','s2_tb','winner_rank','loser_rank']],
    preds[['match_date','surface','winner_name','loser_name','p_tiebreak','p_hold_w','p_hold_l']],
    on=['match_date','surface','winner_name','loser_name'],
    how='inner'
)
print(f'Merged grass matches: {len(merged)}')
print(f'Date range: {merged["match_date"].min().date()} - {merged["match_date"].max().date()}')
print()

# Buckets by p_tiebreak (proxy for CoVe score)
print(f'{"Bucket":<45} {"N":>5}  {"HR(noTB)":>9}  {"p_tb avg":>9}')
print('-'*75)

buckets = [
    ('Toate grass (no filter)',         merged),
    ('p_tb <= 0.127  (rec=True model)', merged[merged['p_tiebreak'] <= 0.127]),
    ('p_tb <= 0.100  (operational)',    merged[merged['p_tiebreak'] <= 0.100]),
    ('p_tb <= 0.090',                   merged[merged['p_tiebreak'] <= 0.090]),
    ('p_tb <= 0.080  (strong signal)',  merged[merged['p_tiebreak'] <= 0.080]),
    ('p_tb <= 0.070  (elite)',          merged[merged['p_tiebreak'] <= 0.070]),
    ('p_tb 0.090-0.100 (~7-8/10)',      merged[(merged['p_tiebreak'] > 0.090) & (merged['p_tiebreak'] <= 0.100)]),
    ('p_tb 0.070-0.090 (~8-9/10)',      merged[(merged['p_tiebreak'] > 0.070) & (merged['p_tiebreak'] <= 0.090)]),
    ('p_tb <= 0.050  (max signal)',     merged[merged['p_tiebreak'] <= 0.050]),
]

for label, sub in buckets:
    if len(sub) == 0:
        print(f'{label:<45} {"0":>5}  {"N/A":>9}')
        continue
    hr = 1 - sub['s2_tb'].mean()
    avg_p = sub['p_tiebreak'].mean()
    print(f'{label:<45} {len(sub):>5}  {hr*100:>8.1f}%  {avg_p:>9.3f}')

print()
print('=== Wimbledon-only ===')
wimb = merged[merged['tourney_name'].str.contains('Wimbledon', case=False, na=False)] if 'tourney_name' in merged.columns else merged[merged['surface']=='Grass']

# Try joining with hist to get tourney_name
merged2 = pd.merge(
    merged,
    grass[['match_date','winner_name','loser_name','tourney_name']],
    on=['match_date','winner_name','loser_name'],
    how='left'
)
wimb = merged2[merged2['tourney_name'].str.contains('Wimbledon', case=False, na=False)]
print(f'Wimbledon matches: {len(wimb)}')
if len(wimb) > 0:
    print(f'Wimbledon S2 TB baseline: {wimb["s2_tb"].mean()*100:.1f}%')
    for label, sub in [
        ('Wimbledon toate', wimb),
        ('Wimbledon p_tb <= 0.127', wimb[wimb['p_tiebreak'] <= 0.127]),
        ('Wimbledon p_tb <= 0.100', wimb[wimb['p_tiebreak'] <= 0.100]),
        ('Wimbledon p_tb <= 0.090', wimb[wimb['p_tiebreak'] <= 0.090]),
        ('Wimbledon p_tb <= 0.080', wimb[wimb['p_tiebreak'] <= 0.080]),
    ]:
        if len(sub) == 0:
            continue
        hr = 1 - sub['s2_tb'].mean()
        print(f'  {label:<40} N={len(sub):>4}  HR={hr*100:.1f}%')

print()
print('=== Distribution p_tiebreak on grass (filtered <= 0.10) ===')
filtered = merged[merged['p_tiebreak'] <= 0.100]
print(filtered['p_tiebreak'].describe())
print()
tb_counts = filtered['s2_tb'].value_counts()
print(f'No TB (win): {tb_counts.get(0,0)}  |  TB (loss): {tb_counts.get(1,0)}')
