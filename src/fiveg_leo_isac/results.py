"""Aggregate stored receiver outputs into detection and timing statistics."""
import gzip,json,hashlib
from importlib.resources import files
from pathlib import Path
import numpy as np
from scipy.stats import beta
from .scoring import score_emissions


def resource(name):
    return json.loads(gzip.decompress(files('fiveg_leo_isac').joinpath('_config/'+name+'.json.gz').read_bytes()))


def correct(event):
    error=event['errors_samples_Hz']
    value=bool(event['alert'] and error is not None and abs(error[0])<=30.72e6/18.36e6 and abs(error[1])<=1.)
    if 'recorded_correct' in event and value!=event['recorded_correct']:raise RuntimeError('Stored correctness differs from its endpoint errors.')
    return value


def timing_score(attempt):
    truth={}
    if attempt['present']:
        for event in attempt['events']:
            if event['alert']:
                truth[event['observation_s']]=[event['prediction'][i+1]-event['errors_samples_Hz'][i] for i in range(2)]
    value=score_emissions(attempt['events'],present=attempt['present'],truth_at=lambda t:truth[t],terminal_s=attempt['terminal_s'])
    for a,b in zip(value['events'],attempt['events']):
        if a['correct']!=b['recorded_correct']:raise RuntimeError('Timing endpoint differs from the recorded source outcome.')
    return value


def outcomes(*,bootstrap=True):
    data=resource('recorded_results');methods=data['methods'];quality={};false={};timing={}
    for name in methods:
        attempts=[a for row in data['quality'] for a in row['methods'][name]]
        final=sum(correct(a['events'][-1]) for a in attempts)
        ever=sum(any(correct(e) for e in a['events']) for a in attempts)
        wrong=sum(not correct(positive[0]) for a in attempts if (positive:=[e for e in a['events'] if e['alert']]))
        expected=data['expected_quality'][name]['uav-quality']
        assert (final,ever,wrong)==(expected['final_report_correct'],expected['ever_correct'],expected['wrong_first_alerts'])
        quality[name]=dict(records=len(data['quality']),attempts=len(attempts),final_correct=final,final_correct_pct=100*final/len(attempts),ever_correct=ever,wrong_first_alerts=wrong)
        n=len(data['false_alerts']);k=sum(name in a['any_alert'] for a in data['false_alerts']);last=sum(name in a['final_alert'] for a in data['false_alerts'])
        upper=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.
        expected=data['expected_false_alerts'][name];assert k==expected['any_alert_count'] and last==expected['final_alert_count'] and abs(upper-expected['any_alert_one_sided_95_upper'])<1e-14
        false[name]=dict(independent_recordings=n,any_alert_count=k,final_alert_count=last,any_alert_probability=k/n,any_alert_one_sided_95_upper=upper)
    order=data['timing_record_order'];lookup={(r['episode_id'],r['method']):r for r in data['timing']}
    rng=np.random.default_rng(20261101);reps=20000
    ti=rng.integers(0,len(order['target']),size=(reps,len(order['target']))) if bootstrap else None
    ei=rng.integers(0,len(order['empty']),size=(reps,len(order['empty']))) if bootstrap else None
    for name in methods:
        target=[[timing_score(a) for a in lookup[eid,name]['attempts']] for eid in order['target']]
        empty=[[timing_score(a) for a in lookup[eid,name]['attempts']] for eid in order['empty']]
        target_op=np.array([np.mean([s['operational_s'] for s in row]) for row in target]);empty_term=np.array([np.mean([s['terminal_s'] for s in row]) for row in empty])
        successes=np.array([sum(s['first_correct_available_s'] is not None for s in row) for row in target]);success_times=np.array([sum(s['first_correct_available_s'] or 0. for s in row) for row in target])
        mean=float(success_times.sum()/successes.sum());balanced=float(.5*target_op.mean()+.5*empty_term.mean());expected=data['expected_timing'][name]
        assert abs(mean-expected['first_correct_given_success']['mean_s'])<1e-12 and abs(balanced-expected['balanced_operational_mean_s'])<1e-12
        result=dict(target_records=len(target),empty_records=len(empty),attempts_per_record=4,successful_target_attempts=int(successes.sum()),
                    first_correct_given_success_mean_s=mean,target_operational_mean_s=float(target_op.mean()),empty_terminal_mean_s=float(empty_term.mean()),balanced_operational_mean_s=balanced)
        if bootstrap:
            conditional=success_times[ti].sum(1)/np.maximum(successes[ti].sum(1),1)
            mixed=.5*target_op[ti].mean(1)+.5*empty_term[ei].mean(1)
            result['first_correct_given_success_ci95_s']=np.nanquantile(conditional,[.025,.975]).tolist()
            result['balanced_operational_ci95_s']=np.nanquantile(mixed,[.025,.975]).tolist()
            np.testing.assert_allclose(result['first_correct_given_success_ci95_s'],expected['first_correct_given_success_ci95_s'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(result['balanced_operational_ci95_s'],expected['balanced_operational_ci95_s'],rtol=0,atol=1e-12)
        timing[name]=result
    return dict(passed=True,quality=quality,false_alerts=false,recorded_timing=timing,
                inference_performed=False,new_timing_measurement=False,bootstrap=dict(replicates=reps if bootstrap else 0,seed=20261101,unit='Whole recording, paired across methods; target/empty strata separate'),
                limitation='Recomputes saved outcomes and recorded-session statistics. Does not claim repeatable wall-clock timing or session-to-session uncertainty.')


def export_inputs(output,cohort,*,waveform_root):
    data=resource('cohort_inputs')
    if cohort not in data:raise ValueError('Unknown cohort; choose '+', '.join(data))
    output=Path(output)
    if output.exists():raise FileExistsError('Existing input directory is preserved.')
    value=data[cohort];manifest=value['manifest']
    for wave in manifest['waveforms']:wave['cache_path']=str((Path(waveform_root)/Path(wave['cache_path']).name).resolve())
    output.mkdir(parents=True);(output/(manifest['split']+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'scene.json').write_text(json.dumps(manifest['scenes'][0],indent=2)+'\n')
    (output/'waveform.json').write_text(json.dumps(manifest['waveforms'][0],indent=2)+'\n')
    (output/'contract.json').write_text(json.dumps(dict(reference_choice=dict(choice='aligned_zenith',status='approved',scope='Frozen received-reference choice; unchanged physical inputs.')),indent=2)+'\n')
    metadata=dict(cohort=cohort,split=manifest['split'],episode_id=manifest['episodes'][0]['episode_id'],scene_id=manifest['scenes'][0]['scene_id'],duration_s=manifest['duration_s'],records=len(manifest['episodes']),manifest_sha256=hashlib.sha256((output/(manifest['split']+'_manifest.json')).read_bytes()).hexdigest(),
                purpose='Scenes and seeds for explicit regeneration of a completed evaluation cohort.',generation_performed=False)
    (output/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata



def cohorts():
    """Describe available scene/seed exports without generating samples."""
    data=resource('cohort_inputs')
    return {name:dict(records=len(value['manifest']['episodes']),
                      duration_s=value['manifest']['duration_s'])
            for name,value in data.items()}
