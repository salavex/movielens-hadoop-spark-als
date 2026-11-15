"""Display actual saved experiment outputs for terminal screenshots."""
import json
import sys
from datetime import datetime

with open(sys.argv[1], encoding='utf-8') as f:
    r = json.load(f)
mode = sys.argv[2]
print('MOVIELENS / SPARK MLLIB - ' + mode.upper())
print('Captured:', datetime.now().isoformat(timespec='seconds'))
print('Application:', r['application_id'], 'Spark:', r['spark_version'], 'Master:', r['master'])
print()
if mode == 'metrics':
    print('Fixed hash split (80/10/10), seed=42; no user/movie pair overlap')
    print('Train: {train}  Validation: {validation}  Test: {test}'.format(**r['split_counts']))
    print()
    print('Validation set (select parameters by RMSE):')
    print('rank   regParam    RMSE       MAE        coverage    fit_seconds')
    for e in r['validation_experiments']:
        print(f"{e['rank']:4d}   {e['regParam']:8.2f}    {e['rmse']:.6f}   {e['mae']:.6f}   {e['coverage']:.2%}      {e['fit_seconds']:.2f}")
    print()
    print('Independent test set (same eligible rows for ALS and baseline):')
    for label, e in [('ALS', r['test_metrics']), ('Mean baseline',r['baseline_metrics'])]:
        print(f"{label:14s} RMSE={e['rmse']:.6f}  MAE={e['mae']:.6f}  rows={e['predicted_rows']}/{e['total_rows']}")
    print('Cold-start rows excluded:', r['test_metrics']['total_rows']-r['test_metrics']['predicted_rows'])
    print('Checks:', ', '.join(f'{k}={v}' for k,v in r['checks'].items()))
    print('Pipeline elapsed seconds:', round(r['elapsed_seconds'],2))
elif mode == 'data':
    for k,v in r['data'].items():
        print(f'{k:24s} {v}')
    print('Rating distribution:', ', '.join(f"{x['rating']}:{x['count']}" for x in r['rating_distribution']))
elif mode == 'recommendations':
    print('Candidate movies: at least 20 TRAIN ratings; full known history excluded')
    for uid in ([int(sys.argv[3])] if len(sys.argv) > 3 else [1,414]):
        print(f'\nUSER {uid} - TOP 10 (raw ALS predicted scores)')
        for x in r['recommendations']:
            if x['userId'] == uid:
                print(f"{x['position']:2d}. {x['title'][:75]:75s} {x['prediction']:.3f}")
