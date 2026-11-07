#!/usr/bin/env python3
"""Evidence-preserving postprocessing of municipal OCR candidate rows.

This is an auxiliary export; it does not change or overwrite the original
OCR build, matching code, or source CSV. It does not fix wrong matched names.

Usage:
  python code/postprocess_communes.py --repo .

Only infer five primary financial fields from raw leading columns if the
column positions are independently supported by existing mapped rows in the
same year. Duplicate year/Cassini groups with incompatible numbers go to
review, not to the provisional output.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = [
    'population',
    'superficie_hectares',
    'revenus_annuels_francs',
    'valeur_centime_francs',
    'centimes_total',
]
RAW = [f'raw_col_{i:02d}' for i in range(2,7)]
REQUIRED = ['row_id', 'year', 'cassini_id', 'commune_raw',
            'commune_cleaned', 'cm_status', 'extraction_flags',
            'review_flags', *MAIN, *RAW]


def parse_number(value: str, decimal: bool) -> float:
    """Parse plausible digit-only numeric cells, never OCR words/ditto symbols.

    Allows standard 1,234 (thousands), 42,65 (decimal), 2,849.27
    (grouped thousands plus decimal) and OCR-inserted spaces e.g. '42, 65'.
    Does not infer malformed integer groups or unusual punctuation.
    """
    s = re.sub(r'\s+', '', str(value).strip())
    if not s or s in {'-', '"', "'", '—', '“', '”'}:
        return np.nan
    if not re.fullmatch(r'[0-9.,]+', s):
        return np.nan
    if re.fullmatch(r'\d+',s):
        return float(s)
    chunks = re.split(r'[,.]',s)
    if decimal:
        if len(chunks)==2 and 1 <= len(chunks[1]) <= 2:
            return float(chunks[0]+'.'+chunks[1])
        if len(chunks)>=3 and 1<=len(chunks[0])<=3 and all(len(v)==3 for v in chunks[1:-1]) and 1<=len(chunks[-1])<=2:
            return float(''.join(chunks[:-1])+'.'+chunks[-1])
    if len(chunks)>=2 and 1<=len(chunks[0])<=3 and all(len(v)==3 for v in chunks[1:]):
        return float(''.join(chunks))
    return np.nan


def setup_values(df: pd.DataFrame, min_anchors: int=25, min_agreement: float=0.995):
    """Return copy, validation report, mapped-cell discrepancies.

    Use only established leading positions; a new department/year with
    unsupported positions does not receive speculative recovered values.
    """
    df = df.copy()
    is_unmapped = df.extraction_flags.str.contains(r'(?:^|;)unmapped_columns(?:;|$)',regex=True)
    parsed = {}
    for field,raw in zip(MAIN,RAW):
        parsed[field] = df[raw].map(lambda x, field=field: parse_number(x, field == 'valeur_centime_francs'))
        df[field] = pd.to_numeric(df[field],errors='coerce')
    val_report, discrepancies = [], []
    for year,ids in df.groupby('year',sort=True).groups.items():
        g = df.loc[ids]; unmapped = is_unmapped.loc[ids]
        checks = []
        for field,raw in zip(MAIN,RAW):
            anchor = (~unmapped)&g[field].notna()&parsed[field].loc[ids].notna()
            agree = np.isclose(g.loc[anchor,field].values,
                               parsed[field].loc[ids].loc[anchor].values,atol=1e-6,rtol=0)
            n = int(anchor.sum()); rate=float(agree.mean()) if n else 0
            checks.append((field,n,rate))
            for k in g.loc[anchor].index[~agree]:
                discrepancies.append({'year':year,'row_id':df.at[k,'row_id'], 'field':field,
                  'parsed_value':float(df.at[k,field]), 'raw_value':df.at[k,raw],
                  'possible_value':float(parsed[field].at[k])})
        approved = all(n>=min_anchors and rate>=min_agreement for _,n,rate in checks)
        val_report.append({'year':year,'inference_enabled':approved,'anchor_counts':json.dumps({f:n for f,n,_ in checks}),
              'agreement_rates':json.dumps({f:round(r,6) for f,_,r in checks})})
        if approved:
            for field in MAIN:
                idx = g.index[unmapped & g[field].isna()]
                df.loc[idx,field] = parsed[field].loc[idx]
        df.loc[ids,'layout_inference_allowed'] = bool(approved)
    df['layout_recovered_field_count'] = 0
    # Count recovered fields for provenance, not a truth/verification stamp.
    for field,raw in zip(MAIN,RAW):
        recovered = is_unmapped & df.layout_inference_allowed & df[field].notna()
        df['layout_recovered_field_count'] += recovered.astype(int)
    df['core_complete'] = df[MAIN].notna().all(axis=1)
    return df, pd.DataFrame(val_report), pd.DataFrame(discrepancies,
        columns=['year','row_id','field','parsed_value','raw_value','possible_value'])


def build_panel(df: pd.DataFrame):
    matched = df[(df.cassini_id!='') & (df.cm_status=='MATCH')].copy()
    unmatched = df[~((df.cassini_id!='')&(df.cm_status=='MATCH'))].copy()
    provisional, conflicts, conflict_rows = [],[],[]
    for (year,cid),g in matched.groupby(['year','cassini_id'],sort=True):
        values={f: sorted(set(g[f].dropna().tolist())) for f in MAIN}
        bad=[f for f,v in values.items() if len(v)>1]
        row_ids=list(g.row_id)
        names=list(dict.fromkeys(v for v in g.commune_raw if v))
        repaired=int((g.layout_recovered_field_count>0).sum())
        source_images=list(dict.fromkeys(g.source_image)) if 'source_image' in g else []
        meta={'year':int(year), 'cassini_id':cid,'n_candidate_rows':len(g),
              'source_row_ids':json.dumps(row_ids,ensure_ascii=False),
              'source_images':json.dumps(source_images,ensure_ascii=False),
              'ocr_names':json.dumps(names,ensure_ascii=False),
              'layout_recovered_candidate_rows':repaired}
        if bad:
            conflicts.append({**meta, 'conflicting_fields':';'.join(bad),
                'distinct_values':json.dumps({f:values[f] for f in bad},ensure_ascii=False)})
            conflict_rows.append(g)
            continue
        nms=g.commune_cleaned[g.commune_cleaned!='']
        name=nms.mode().iloc[0] if not nms.empty else ''
        rep=next((v for v in names if v),'')
        row={**meta,'commune_cleaned':name,'commune_raw_example':rep,
             'all_core_present':all(len(values[f])==1 for f in MAIN),
             'contains_recovered_layout':repaired>0,
             'has_canton_subtotal_warning':any('canton_ocr_subtotal_discrepancy_review' in s for s in g.review_flags),
             'has_tax_warning':any('tax:' in s for s in g.review_flags),
             'has_other_ocr_review_flag':any('ocr_uncertain' in s or 'numeric_ambiguous' in s for s in g.extraction_flags),
             'validation_status':'provisional_unverified'}
        for f in MAIN: row[f]=values[f][0] if values[f] else np.nan
        provisional.append(row)
    panel=pd.DataFrame(provisional)
    # Make deterministic empty outputs too.
    conflict_details=pd.concat(conflict_rows,ignore_index=True) if conflict_rows else matched.iloc[0:0].copy()
    conflicts=pd.DataFrame(conflicts)
    if not panel.empty:
        panel=panel.sort_values(['year','cassini_id']).reset_index(drop=True)
    complete=panel[panel.all_core_present].copy() if not panel.empty else panel
    return panel,complete,conflicts,conflict_details,unmatched


def run(repo:Path, input_path:Path|None, output:Path|None):
    repo=repo.expanduser().resolve()
    path=input_path.expanduser().resolve() if input_path else repo/'derived/situation_financiere/output/situation_financiere_commune_year_candidates.csv'
    out=output.expanduser().resolve() if output else repo/'derived/situation_financiere/econometric_export_v2'
    if not path.is_file(): raise FileNotFoundError(f'Input not found: {path}')
    data=pd.read_csv(path,dtype=str,keep_default_na=False,low_memory=False)
    missing=[s for s in REQUIRED if s not in data.columns]
    if missing:raise ValueError(f'Missing required columns: {missing}')
    df,validation,discrepancies=setup_values(data)
    panel,complete,conflicts,conflict_rows,unmatched=build_panel(df)
    out.mkdir(parents=True,exist_ok=True)
    for filename,frame in [('provisional_panel.csv',panel),('complete_core_panel.csv',complete),
       ('conflicting_ids.csv',conflicts),('conflicting_candidate_rows.csv',conflict_rows),
       ('unmatched_candidate_rows.csv',unmatched),('layout_validation.csv',validation),
       ('numeric_parse_discrepancies.csv',discrepancies),
       ('unmapped_layout_review.csv',df[df.extraction_flags.str.contains('unmapped_columns')].copy())]:
        frame.to_csv(out/filename,index=False)
    for basename,frame in [('provisional_panel',panel),('complete_core_panel',complete)]:
        try: frame.to_parquet(out/f'{basename}.parquet',index=False)
        except ImportError:pass
    yearly=[]
    for year,d in df.groupby('year',sort=True):
        mp=panel[panel.year==int(year)] if not panel.empty else panel
        cp=complete[complete.year==int(year)] if not complete.empty else complete
        cf=conflicts[conflicts.year==int(year)] if not conflicts.empty else conflicts
        u=d[d.extraction_flags.str.contains('unmapped_columns')]
        yearly.append({'year':int(year),'candidate_rows':len(d),
         'matched_candidate_rows':int(((d.cassini_id!='')&(d.cm_status=='MATCH')).sum()),
         'unmatched_or_ambiguous_rows':int(((d.cassini_id=='')|(d.cm_status!='MATCH')).sum()),
         'originally_unmapped_rows':len(u),
         'unmapped_rows_with_five_core_values_after_inference':int(u.core_complete.sum()),
         'provisional_unconflicted_id_years':len(mp),
         'provisional_with_all_core_variables':len(cp),
         'conflicting_id_years_excluded':len(cf),
         'duplicated_id_years_safely_collapsed':int((mp.n_candidate_rows>1).sum()) if len(mp) else 0})
    summary=pd.DataFrame(yearly)
    summary.to_csv(out/'year_quality_v2.csv',index=False)
    totals=summary.select_dtypes(include='number').drop(columns=['year']).sum().astype(int).to_dict()
    report={'input_file':str(path),'out_dir':str(out), 'warning':'ALL VALUES ARE UNVERIFIED OCR. Recovered layouts need visual review.',
      'year_totals':totals,'layout_year_validation':validation.to_dict(orient='records'),
      'anomalous_mapped_numeric_values':len(discrepancies),
      'method':'Verify leading core columns against already mapped rows year by year. Infer fields only on unmapped rows. Merge identical/compatible year-Cassini core values. Hold any conflicts. No manual names corrections, no tax resolution, no sample selection.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(summary.to_string(index=False))
    print('\nTotals:',json.dumps(totals,indent=2))
    print('Output directory:',out)
    print('Unverified provisional panel, NOT a validated historical-data release.')
    return summary,report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path('.'))
    ap.add_argument('--input',type=Path,default=None)
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    run(args.repo,args.input,args.out)

if __name__=='__main__':main()
