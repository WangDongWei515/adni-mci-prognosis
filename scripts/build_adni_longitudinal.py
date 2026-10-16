"""Build an auditable clinical-date longitudinal dataset and feasibility pairs.
Source: downloaded ADNIMERGE2 package. Never modifies raw data.
Dependencies: pandas, numpy, pyreadr, pyarrow. This is not a locked cohort.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import pyreadr

SCALES={'MMSE':('MMSE','MMSCORE',30),'MoCA':('MOCA','MOCA',30),
        'ADAS13':('ADAS','TOTAL13',85),'CDRSB':('CDR','CDRSB',18)}
WINDOWS={6:2,12:3,24:3}
MIN_DATE=pd.Timestamp('2004-01-01') # broad ADNI-era plausibility bound, not imputation

def load(src,name):
    return pyreadr.read_r(str(src/f'{name}.rda'))[name].copy()

def common(d,name,date_col,snapshot):
    o=pd.DataFrame(index=d.index)
    o['subject_id']=pd.to_numeric(d.RID,errors='coerce').astype('Int64')
    o['source_table']=name;o['source_row']=np.arange(len(d))
    o['source_record_id']=d.ID.astype('string');o['protocol']=d.COLPROT.astype('string')
    o['visit_code']=d.VISCODE.astype('string');o['visit_code2']=d.VISCODE2.astype('string')
    o['date_raw']=d[date_col].astype('string');o['date']=pd.to_datetime(d[date_col],errors='coerce')
    o['date_field']=date_col
    o['date_provenance']='assessment_or_registry_fallback_not_distinguishable' if date_col=='VISDATE' else 'diagnostic_form_completion_date'
    o['qc_status']=d.HAS_QC_ERROR.astype('string')
    o['qc_error']=o.qc_status.isin(['1','Has QC error'])
    o['date_missing']=o.date.isna();o['date_outside_snapshot']=o.date.notna()&(~o.date.between(MIN_DATE,snapshot))
    o['date_eligible']=~o.date_missing&~o.date_outside_snapshot&o.subject_id.notna()
    o['screening']=o.visit_code2.str.startswith('sc',na=False)|o.visit_code.str.startswith('sc',na=False)
    return o

def asof_join(left,right,right_date):
    result=pd.merge_asof(left.sort_values('date'),right.sort_values(right_date),
                         left_on='date',right_on=right_date,by='subject_id',direction='backward',allow_exact_matches=True)
    assert len(result)==len(left)
    assert (result[right_date].isna()|result[right_date].le(result.date)).all()
    return result

def pair_candidates(clean):
    rows=[]
    for (sid,scale),d in clean.groupby(['subject_id','scale'],sort=True):
        d=d.sort_values('date').reset_index(drop=True)
        first=d.date.iloc[0]
        for i,start in d.iterrows():
            if start.diagnosis!='MCI' or pd.isna(start.diagnosis_lag_days) or start.diagnosis_lag_days>90:
                continue
            for months,tolerance in WINDOWS.items():
                nominal=start.date+pd.DateOffset(months=months)
                low=start.date+pd.DateOffset(months=months-tolerance);high=start.date+pd.DateOffset(months=months+tolerance)
                future=d.loc[d.date.between(low,high)&d.date.gt(start.date)&~d.screening].copy()
                if future.empty:continue
                future['distance']=(future.date-nominal).abs()
                end=future.sort_values(['distance','date']).iloc[0]
                rows.append({'subject_id':int(sid),'scale':scale,'index_date':start.date,'target_date':end.date,
                    'horizon_months':months,'window_tolerance_months':tolerance,'actual_gap_days':(end.date-start.date).days,
                    'index_score':start.score,'target_score':end.score,'score_change':end.score-start.score,
                    'history_count_same_scale':i+1,'history_span_days':(start.date-first).days,
                    'diagnosis_date':start.diagnosis_date,'diagnosis_lag_days':start.diagnosis_lag_days,
                    'diagnosis_same_day':start.diagnosis_lag_days==0,'index_screening':bool(start.screening),
                    'sex':start.sex,'education_years':start.education_years,'approx_age_years':start.approx_age_years})
    return pd.DataFrame(rows)

def main(root):
    src=root/'data/raw/adni/ADNIMERGE2/data';out=root/'data/processed/adni_v0';out.mkdir(parents=True,exist_ok=True)
    reports=root/'reports';reports.mkdir(exist_ok=True)
    snapshot=pd.Timestamp(load(src,'DATA_DOWNLOADED_DATE').iloc[0,0])
    sources={n:load(src,n) for n in ['MMSE','MOCA','ADAS','CDR','DXSUM','PTDEMOG']}
    chunks=[]
    for scale,(name,field,maximum) in SCALES.items():
        d=sources[name];o=common(d,name,'VISDATE',snapshot)
        o['scale']=scale;o['score_field']=field;o['score']=pd.to_numeric(d[field],errors='coerce')
        o['score_missing']=o.score.isna();o['score_out_of_range']=o.score.notna()&~o.score.between(0,maximum)
        o['not_done']=d.DONE.isin(['No','0','NOT DONE']) if 'DONE' in d else False
        o['initial_eligible']=o.date_eligible&~o.qc_error&~o.score_missing&~o.score_out_of_range&~o.not_done
        chunks.append(o)
    observations=pd.concat(chunks,ignore_index=True)
    keys=['subject_id','date','scale'];eligible=observations.loc[observations.initial_eligible].copy()
    key_stats=eligible.groupby(keys,dropna=False).score.agg(['size','nunique']).rename(columns={'size':'same_day_rows','nunique':'same_day_score_values'})
    observations=observations.merge(key_stats,on=keys,how='left',validate='many_to_one')
    observations['same_day_conflict']=observations.same_day_score_values.gt(1)
    clean=observations.loc[observations.initial_eligible&~observations.same_day_conflict].sort_values(keys+['screening','source_row']).drop_duplicates(keys).copy()
    assert not clean.duplicated(keys).any()
    # Latest diagnosis at/before an assessment, including unknown/conflicting barriers.
    dxraw=sources['DXSUM'];dx=common(dxraw,'DXSUM','EXAMDATE',snapshot)
    dx['diagnosis']=dxraw.DIAGNOSIS.replace({'Dementia':'DEM'}).where(dxraw.DIAGNOSIS.isin(['CN','MCI','Dementia']))
    dxrows=[]
    for (sid,date),d in dx.loc[dx.date_eligible].groupby(['subject_id','date']):
        labels=d.diagnosis.dropna().unique();conflict=len(labels)>1;bad=d.qc_error.any()
        dxrows.append({'subject_id':sid,'diagnosis_date':date,'diagnosis':labels[0] if len(labels)==1 and not bad else None,
                       'diagnosis_conflict':conflict,'diagnosis_qc_error':bool(bad),'diagnosis_source_rows':len(d)})
    diagnoses=pd.DataFrame(dxrows);diagnoses.subject_id=diagnoses.subject_id.astype('Int64')
    clean=asof_join(clean,diagnoses,'diagnosis_date')
    clean['diagnosis_lag_days']=(clean.date-clean.diagnosis_date).dt.days
    # Demographics: only records dated on/before the prediction; retain missingness.
    demo_raw=sources['PTDEMOG'];demo=common(demo_raw,'PTDEMOG','VISDATE',snapshot)
    demo['date_provenance']='registry_date_for_matching_visit'
    demo['sex']=demo_raw.PTGENDER.where(demo_raw.PTGENDER.isin(['Female','Male']))
    demo['birth_year']=pd.to_numeric(demo_raw.PTDOBYY,errors='coerce').where(lambda v:v.between(1900,1980))
    demo['education_years']=pd.to_numeric(demo_raw.PTEDUCAT,errors='coerce').where(lambda v:v.between(0,30))
    demo_rows=[]
    for (sid,date),d in demo.loc[demo.date_eligible].groupby(['subject_id','date']):
        row={'subject_id':sid,'demographics_date':date}
        for col in ['sex','birth_year','education_years']:
            values=d[col].dropna().unique();row[col]=values[0] if len(values)==1 and not d.qc_error.any() else None
        demo_rows.append(row)
    demographics=pd.DataFrame(demo_rows);demographics.subject_id=demographics.subject_id.astype('Int64')
    clean=asof_join(clean,demographics,'demographics_date')
    clean['approx_age_years']=clean.date.dt.year-clean.birth_year
    clean=clean.sort_values(keys).reset_index(drop=True)
    clean['history_count_same_scale']=clean.groupby(['subject_id','scale']).cumcount()+1
    clean['previous_gap_days']=clean.groupby(['subject_id','scale']).date.diff().dt.days
    clean['mci_index_90d']=clean.diagnosis.eq('MCI')&clean.diagnosis_lag_days.between(0,90)
    # A history is all retained records <= a selected index date, including pre-MCI observations.
    visits=clean.pivot(index=['subject_id','date'],columns='scale',values='score').reset_index()
    shared=clean.drop_duplicates(['subject_id','date'])[['subject_id','date','diagnosis','diagnosis_date','diagnosis_lag_days','sex','birth_year','education_years','demographics_date','approx_age_years']]
    visits=visits.merge(shared,on=['subject_id','date'],validate='one_to_one').sort_values(['subject_id','date'])
    pairs=pair_candidates(clean)
    summaries=[]
    for (scale,horizon),g in pairs.groupby(['scale','horizon_months']):
        repeat=g.history_count_same_scale.ge(2)&g.history_span_days.ge(30)
        same=g.diagnosis_same_day
        summaries.append({'scale':scale,'horizon_months':int(horizon),'pairs':len(g),'subjects':g.subject_id.nunique(),
            'pairs_with_history_2plus_span30d':int(repeat.sum()),'subjects_with_history_2plus_span30d':g.loc[repeat,'subject_id'].nunique(),
            'pairs_same_day_diagnosis':int(same.sum()),'subjects_same_day_diagnosis':g.loc[same,'subject_id'].nunique(),
            'repeat_pairs_same_day_diagnosis':int((same&repeat).sum()),'repeat_subjects_same_day_diagnosis':g.loc[same&repeat,'subject_id'].nunique(),
            'pairs_nonscreen_index':int((~g.index_screening).sum()),'subjects_nonscreen_index':g.loc[~g.index_screening,'subject_id'].nunique(),
            'pairs_with_one_assessment':int(g.history_count_same_scale.eq(1).sum()),'subjects_with_one_assessment':g.loc[g.history_count_same_scale.eq(1),'subject_id'].nunique(),
            'pairs_with_demographics_complete':int(g[['sex','education_years','approx_age_years']].notna().all(axis=1).sum())})
    availability=pd.DataFrame(summaries)
    assert pairs.target_date.gt(pairs.index_date).all()
    assert pairs.diagnosis_date.le(pairs.index_date).all()
    assert pairs.diagnosis_lag_days.between(0,90).all()
    assert not pairs.duplicated(['subject_id','scale','horizon_months','index_date']).any()
    assert len(observations)==sum(len(sources[n]) for n in ['MMSE','MOCA','ADAS','CDR'])
    metrics={'snapshot':str(snapshot.date()),'raw_observation_rows':len(observations),'retained_observation_rows':len(clean),
        'retained_subjects':int(clean.subject_id.nunique()),'visit_rows':len(visits),'candidate_pair_rows':len(pairs),
        'issue_counts_nonexclusive':{c:int(observations[c].sum()) for c in ['date_missing','date_outside_snapshot','qc_error','score_missing','score_out_of_range','not_done','same_day_conflict']},
        'same_day_conflicting_keys':int(key_stats['same_day_score_values'].gt(1).sum()),
        'duplicate_consistent_rows_collapsed':int((observations.initial_eligible&~observations.same_day_conflict).sum()-len(clean)),
        'diagnosis_conflicting_dates':int(diagnoses.diagnosis_conflict.sum()),
        'diagnosis_qc_error_rows':int(dx.qc_error.sum()),
        'missing_diagnosis_on_clean_rows':int(clean.diagnosis.isna().sum()),
        'moca_by_phase':sources['MOCA'].groupby('COLPROT',observed=True).MOCA.agg(['size','count']).reset_index().astype({'COLPROT':str}).to_dict('records'),
        'availability':availability.to_dict('records'),
        'source_sha256':{n:hashlib.sha256((src/f'{n}.rda').read_bytes()).hexdigest() for n in sources},
        'notes':['Clinical assessment/form dates used; real-time data availability not established.',
                 'Date fallback provenance cannot be separated row-by-row from these package tables.',
                 'Feasibility only; no split, imputation, model, or final cohort selection.']}
    for name,df in [('observations_audit',observations),('diagnoses_audit',dx),('demographics_audit',demo),('cognition_long',clean),('visits_wide',visits),('candidate_pairs',pairs)]:
        path=out/f'{name}.parquet';df.to_parquet(path,index=False)
        readback=pd.read_parquet(path);assert readback.shape==df.shape
    (reports/'adni_longitudinal_feasibility.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=int)+'\n')
    print(json.dumps(metrics,ensure_ascii=False,indent=2,default=int))
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);main(parser.parse_args().root)
