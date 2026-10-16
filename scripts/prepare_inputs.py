"""Prepare local analysis inputs from an independently authorised ADNI download.

Composes the original data preparation and the v1 notebook's feature/split steps.
No data are downloaded or redistributed by this script.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.path.insert(0,str(ROOT))
    from build_adni_longitudinal import main as prepare_longitudinal
    from src.data_engineering.diagnosis_labels import build_diagnosis_transitions
    from src.data_engineering.mri_volumes import build_mri_volumes
    from src.dataset import build_dataset,assign_splits,verify_no_leakage

    required=['DATA_DOWNLOADED_DATE','MMSE','MOCA','ADAS','CDR','DXSUM','PTDEMOG',
              'UCSFFSX7','APOERES','UCBERKELEY_AMY_6MM']
    raw=ROOT/'data/raw/adni/ADNIMERGE2/data'
    missing=[name+'.rda' for name in required if not (raw/(name+'.rda')).is_file()]
    if missing:
        parser.error('Missing authorised local ADNI inputs: '+', '.join(missing))
    prepare_longitudinal(ROOT)
    output=ROOT/'data/processed/adni_v0'
    labels,_=build_diagnosis_transitions(ROOT,horizon_months=24,tolerance_months=3)
    labels.to_parquet(output/'diagnosis_transitions.parquet',index=False)
    build_mri_volumes(ROOT).to_parquet(output/'mri_volumes.parquet',index=False)
    samples,sequences,_=build_dataset(ROOT)
    samples,_=assign_splits(samples,ratios=(0.70,0.15,0.15),seed=42)
    check=verify_no_leakage(samples)
    assert check['通过'], 'Participant split overlap detected'
    v1=ROOT/'data/processed/dataset_v1'
    v1.mkdir(parents=True,exist_ok=True)
    samples.to_parquet(v1/'samples.parquet',index=False)
    sequences.to_parquet(v1/'sequences.parquet',index=False)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_dataset_v2.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
