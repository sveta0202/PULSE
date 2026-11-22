"""AIRI baseline: leakage-safe EMG feature regression and official CSV submission."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
import joblib

FINGERS = ['Index', 'Middle', 'Little', 'Ring', 'Thumb']
ACTIONS = {x.lower(): i for i, x in enumerate(FINGERS)}

def features(emg):
    x = np.asarray(emg, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f'Expected (T,6), got {x.shape}')
    # Per-recording robust scaling: no target/test label information.
    center = np.median(x, axis=0)
    scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    x = np.clip((x-center)/np.maximum(scale, 1e-5), -12, 12)
    parts = [x]
    for width in (25, 100, 250):
        mean = uniform_filter1d(x, size=width, axis=0, mode='nearest')
        rms = np.sqrt(np.maximum(uniform_filter1d(x*x, size=width, axis=0, mode='nearest'), 0))
        mav = uniform_filter1d(np.abs(x), size=width, axis=0, mode='nearest')
        parts.extend([mean, rms, mav])
    # Delayed EMG context (~40, 100, 200 ms); causal shifts.
    for lag in (20, 50, 100):
        shifted = np.empty_like(x)
        shifted[:lag] = x[0]
        shifted[lag:] = x[:-lag]
        parts.append(uniform_filter1d(np.abs(shifted), size=100, axis=0, mode='nearest'))
    return np.concatenate(parts, axis=1).astype(np.float32)

def load_record(data, rid):
    x = np.load(data/'train'/'signals'/f'{rid}.npy', mmap_mode='r')
    y = np.load(data/'train'/'targets'/f'{rid}.npy', mmap_mode='r')
    if x.ndim != 2 or x.shape[1] != 6 or y.ndim != 2 or y.shape[1] != 20 or len(x) != len(y):
        raise ValueError(f'Bad shape for {rid}: {x.shape}, {y.shape}')
    return x,y

def fit(data, rows, stride=20, alpha=100.0):
    xx, yy = [], []
    for row in rows.itertuples(index=False):
        x,y = load_record(data, row.recording_id)
        xx.append(features(x)[1000::stride]); yy.append(np.asarray(y[1000::stride], dtype=np.float32))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='lsqr'))
    model.fit(np.concatenate(xx), np.concatenate(yy))
    return model

def record_error(y, pred, action):
    err = np.abs(np.asarray(y)[1000:] - np.asarray(pred)[1000:]).reshape(-1,5,4).mean(axis=(0,2))
    active = ACTIONS.get(str(action).lower())
    if active is None:
        return float(err.mean())
    return float(0.5*err[active]+0.125*(err.sum()-err[active]))

def validate(data, meta, out, stride, alpha):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, va = next(splitter.split(meta, groups=meta.session_id))
    train, valid = meta.iloc[tr], meta.iloc[va]
    print(f'Train sessions: {train.session_id.nunique()}, validation sessions: {valid.session_id.nunique()}')
    model = fit(data, train, stride, alpha)
    rows=[]
    for row in valid.itertuples(index=False):
        x,y=load_record(data,row.recording_id)
        pred=np.clip(model.predict(features(x)), -np.pi, np.pi)
        rows.append({'recording_id':row.recording_id,'session_id':row.session_id,'action':row.action,'weighted_mae':record_error(y,pred,row.action)})
    scores=pd.DataFrame(rows)
    # Each session has eight actions: average within sessions, then across sessions.
    session_scores=scores.groupby('session_id').weighted_mae.mean()
    print('Session-balanced weighted MAE (rad):', float(session_scores.mean()))
    out.mkdir(parents=True,exist_ok=True)
    scores.to_csv(out/'validation_per_record.csv',index=False)
    session_scores.rename('weighted_mae').to_csv(out/'validation_per_session.csv')
    return model

def train_all(data, meta, out, stride, alpha):
    model=fit(data,meta,stride,alpha)
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/'model.joblib')
    print('Saved:',out/'model.joblib')

def submit(data, template_path, test_folder, model_path, output):
    template=pd.read_csv(template_path)
    ids=['row_id','recording_id','sample_index']
    cols=[f'angle_{i:02d}' for i in range(20)]
    if template.columns.tolist() != ids+cols:
        raise ValueError('Unexpected submission columns/order; use official sample submission')
    if template[ids].isna().any().any() or template.row_id.duplicated().any():
        raise ValueError('Missing IDs or duplicate row_id in template')
    model=joblib.load(model_path)
    output_data=template.copy()
    for rid, positions in template.groupby('recording_id',sort=False).groups.items():
        path=Path(test_folder)/f'{rid}.npy'
        if not path.exists(): raise FileNotFoundError(path)
        x=np.load(path,mmap_mode='r')
        if x.ndim != 2 or x.shape[1] != 6: raise ValueError(f'Invalid EMG shape: {path}: {x.shape}')
        pred=np.clip(model.predict(features(x)), -np.pi, np.pi)
        idx=template.loc[positions,'sample_index'].to_numpy()
        if (idx<1000).any() or (idx>=len(x)).any(): raise ValueError(f'Bad indices for {rid}')
        output_data.loc[positions,cols]=pred[idx]
    output_data[cols]=output_data[cols].astype('float64')
    if not np.isfinite(output_data[cols].to_numpy()).all(): raise ValueError('NaN or Inf predictions')
    if not output_data[ids].equals(template[ids]): raise ValueError('Identifiers changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    output_data.to_csv(output,index=False)
    print(f'Saved {len(output_data)} rows to {output}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['validate','train','submit'])
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,default=Path('outputs'))
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--alpha',type=float,default=100)
    p.add_argument('--template',type=Path)
    p.add_argument('--test-folder',type=Path)
    p.add_argument('--model',type=Path,default=Path('outputs/model.joblib'))
    p.add_argument('--output',type=Path,default=Path('outputs/submission.csv'))
    a=p.parse_args()
    if a.command=='submit':
        if a.template is None or a.test_folder is None: p.error('--template and --test-folder required')
        submit(a.data,a.template,a.test_folder,a.model,a.output)
    else:
        meta=pd.read_csv(a.data/'train'/'metadata.csv')
        if a.command=='validate':validate(a.data,meta,a.out,a.stride,a.alpha)
        else:train_all(a.data,meta,a.out,a.stride,a.alpha)
if __name__=='__main__':main()
