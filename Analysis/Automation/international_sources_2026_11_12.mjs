import {read,save,writeCsv,pool,fetch,api,ro} from './research_2026_11_12.mjs';
const b='Analysis/Goal70_2026-11-12/International';
const urls={
 uefa_nations:'https://www.uefa.com/uefanationsleague/news/02a2-1fea18abbcbc-456e846509e7-1000–2026-27-uefa-nations-league-all-the-league-phase-fixtures-a/',
 uefa_postponed:'https://www.uefa.com/news-media/news/02ab-2209e8267c40-cadb2ec0869e-1000--uefa-matches-in-portugal-postponed-due-to-storm-charles/',
 caf_u23:'https://www.cafonline.com/caf-u-23-africa-cup-of-nations/news/u-23-afcon-qualifiers-nigeria-dr-congo-and-ghana-libya-headline-high-stakes-second-round-ties/',
 england_u20:'https://www.englandfootball.com/england/youth/Mens-U20',
 italy_u20:'https://www.padovacalcio.it/under-20-italia-portogallo-giovedi12102026/',
 france_u20:'https://angers-sco.fr/treve-internationale-le-programme-des-matchs-pour-nos-internationaux-2/',
 qatar_u23:'https://www.olympic.qa/node/20682',
 chelsea:'https://www.chelseafc.com/en/news/article/chelsea-international-call-ups-and-schedules-for-november-2026',
 extra_friendly:'https://ligatoday.com/fixtures/2026-11-12/',
 unconfirmed:'https://scoremondo.com/en/tomorrow',
 ksi_preview:'https://www.ksi.is/um-ksi/frettir/landslid/u21-karla-tekur-a-moti-grikklandi/',
 ksi_squad:'https://www.ksi.is/um-ksi/frettir/landslid/hopur-u21-karla-fyrir-umspilsleiki-gegn-grikklandi/',
 epo:'https://www.epo.gr/el/arthra/ethniki-elpidon-stin-islandia-gia-proto-bima-prokrisis',
 uefa_u21:'https://www.uefa.com/under21/match/2050682--iceland-vs-greece/matchinfo/',
 live_score:'https://www.livescore.mobi/football/2026-11-12/?tz=3'
};
save(b+'/sources.json',{at_ro:ro(),urls,notes:['UEFA/KSI/EPO confirm U21 at13UTC=15RO; secondary display16:00 uses UTC+3, not Romania.', 'UEFA rescheduling notice supersedes stale U17/U19 calendars; Portugal hosting creates neutral designated-home matches.', 'CAF U23 official times are GMT; secondary clocks are not assumed Romania.', 'No bookmaker prices or tipster probabilities used. ScoreMondo reports without independent identity remain unconfirmed, not modeled.', 'Nations League current edition has4prior fixtures/team, AFCON normally2; unchanged eligibility minimum5/3.', 'Global three adjacent UTC dates were converted to Europe/Bucharest; late CONCACAF22/23UTC12Nov belongs13NovRO.']});
const extras=[
 ['ger_swe_u19','Germany U19','Sweden U19','UEFA U19 Qualification','2026/2027','13:00',urls.uefa_postponed,'official_rescheduled_neutral'],
 ['por_ltu_u19','Portugal U19','Lithuania U19','UEFA U19 Qualification','2026/2027','15:00',urls.uefa_postponed,'official_rescheduled'],
 ['bul_srb_u17','Bulgaria U17','Serbia U17','UEFA U17 Qualification','2026/2027','15:00',urls.uefa_postponed,'official_rescheduled_neutral'],
 ['por_and_u17','Portugal U17','Andorra U17','UEFA U17 Qualification','2026/2027','15:00',urls.uefa_postponed,'official_rescheduled'],
 ['eth_mar_u23','Ethiopia U23','Morocco U23','CAF U23 Qualification','2026/2027','13:00',urls.caf_u23,'official'],
 ['ang_cgo_u23','Angola U23','Congo U23','CAF U23 Qualification','2026/2027','15:00',urls.caf_u23,'official'],
 ['tun_gab_u23','Tunisia U23','Gabon U23','CAF U23 Qualification','2026/2027','17:00',urls.caf_u23,'official'],
 ['ita_por_u20','Italy U20','Portugal U20','U20 Elite League','2026/2027','16:00',urls.italy_u20,'official_club_confirms_time'],
 ['eng_cze_u20','England U20','Czechia U20','U20 Elite League','2026/2027','19:30',urls.england_u20,'official'],
 ['fra_den_u20','France U20','Denmark U20','Friendlies U20','2026','16:00',urls.france_u20,'official_club_confirms_time'],
 ['tha_qat_u23','Thailand U23','Qatar U23','Gia Lai VTV Cup U23','2026','08:00',urls.qatar_u23,'official_date_secondary_time_unconfirmed_neutral'],
 ['sin_vie_u23','Singapore U23','Vietnam U23','Gia Lai VTV Cup U23','2026','12:00',urls.extra_friendly,'secondary_only_time_unconfirmed'],
 ['usa_jpn_u20','USA U20','Japan U20','Friendlies U20','2026','01:00',urls.chelsea,'club_reported_time_date_requires_confirmation_already_past'],
 ['egy_tan_u20','Egypt U20','Tanzania U20','Friendlies U20','2026','',urls.unconfirmed,'unconfirmed_single_schedule'],
 ['rus_kgz_u23','Russia U21','Kyrgyzstan U23','Friendlies youth mixed categories','2026','',urls.unconfirmed,'unconfirmed_mixed_squad_identity']
];
const fixtures=read(b+'/candidate_fixtures.json');
for(const [key,home,away,competition,edition,time,source,note]of extras){const id='intl_20261112_'+key;if(fixtures.some(f=>f.id===id))continue;fixtures.push({id,competition,league_id:'manual_'+competition.replace(/[^A-Za-z0-9]+/g,'_'),date:time?'2026-11-12T'+time+':00.000Z':'',fixture_date_ro:'2026-11-12',home,away,home_id:'manual_'+home.replaceAll(' ','_'),away_id:'manual_'+away.replaceAll(' ','_'),category:'youth',edition,source,history_basis:'national_current_competition_edition',status_verified:false,status:{started:time?Date.parse('2026-11-12T'+time+':00Z')<Date.now():false,finished:false},inventory_note:note+'; exact current-edition history not available in saved numeric feed',venue_note:note.includes('neutral')?'Neutral venue; designated home is not own home':'Venue advantage unconfirmed'});}
save(b+'/candidate_fixtures.json',fixtures);
writeCsv(b+'/supplementary_inventory.csv',fixtures.filter(f=>f.id.startsWith('intl_')));
await pool(['5842','5843'],async id=>{const r=await fetch(api('teams?id='+id));save(b+'/Context/team_'+id+'.json',r);},2);
const url='https://api.open-meteo.com/v1/forecast?latitude=64.14155&longitude=-21.87765&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m&start_date=2026-11-12&end_date=2026-11-12&timezone=UTC';
save(b+'/Context/weather_reykjavik.json',await fetch(url));
console.log(JSON.stringify({inventory:fixtures.length,supplementary:extras.length,sourceCount:Object.keys(urls).length}));
