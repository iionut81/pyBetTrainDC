import fs from 'node:fs';
import path from 'node:path';
import {serializeSelections} from './selection-store.mjs';

export function makeFixture(root){
 const put=(name,rows)=>{const file=path.join(root,name);fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,serializeSelections(rows,[...new Set(rows.flatMap(Object.keys))]));};
 put('Analysis/Selection_History/2026-11-04_goal70_candidates.csv',[
  {id:'g1',home:'Test A',away:'Test B',competition:'eng.2',kickoff_ro:'22:00',p_raw:'.91',p_stress:'.85',first_goal_success:'9',first_goal_n:'10',venue_success:'4',venue_n:'5',source:'https://www.espn.com/soccer/match/_/gameId/g1',analyst_selected:'true',recorded_at_ro:'2026-11-04T08:00:00+02:00'},
  {id:'g2',home:'Test C',away:'Test D',competition:'eng.2',kickoff_ro:'23:00',p_raw:'.89',p_stress:'.8',source:'https://www.espn.com/soccer/match/_/gameId/g2',recorded_at_ro:'2026-11-04T08:00:00+02:00'}]);
 put('Analysis/Selection_History/2026-11-04_corners1h_candidates.csv',[{id:'1001',goal70_espn_id:'g1',home:'Test A',away:'Test B',competition:'eng.2',date:'2026-11-04T20:00:00Z',p_raw:'.95',p_stress:'.9',unique_n:'10',unique_under:'10',venue_unique_n:'5',venue_under:'5',eligible:'true',analyst_selected:'true',source:'https://www.fotmob.com/api/data/matchDetails?matchId=1001'}]);
 put('Analysis/Corners1H_2026-11-04/prospective_selection.csv',[{match_id:'1001',home:'Test A',away:'Test B',final_p_raw:'.96',final_p_stress:'.92',historical_success:'9',historical_n:'10',venue_success:'5',venue_n:'5',recorded_at_ro:'2026-11-04T09:00:00+02:00'}]);
 put('Analysis/Selection_History/2026-11-03_goal70_candidates.csv',[{id:'old',home:'Old A',away:'Old B',competition:'eng.2',kickoff_ro:'22:00',p_raw:'.9',source:'https://example.test'}]);
 put('Analysis/Selection_History/user_selections.csv',[{selection_id:'old-choice',analysis_date:'2026-11-03',selected_at_ro:'2026-11-03T12:00:00+02:00',market:'GOAL_BY_70',match_id:'old',home:'Old A',away:'Old B',p_raw:'.9',result:'success',actual_first_goal_minute:'20'}]);
 for(const date of ['2026-11-03','2026-11-04'])fs.writeFileSync(path.join(root,`Analysis/Selection_History/${date}.md`),'# Test archive');
}
