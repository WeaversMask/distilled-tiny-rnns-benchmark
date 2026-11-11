"""Fixed fixture checking reward-cue replay, diagnosis and family recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from distilled_rnns.evaluation import binary_kl, binary_nll, make_trial_masks
from distilled_rnns.simulation import replay_policy
from foundation.classical_baselines import DEFAULT_GRID, candidate_probabilities
from foundation.reward_cue import CUE_GRID, cue_probabilities, fit_family_mixture, simulate_fixture


LENGTHS = (100, 2000)
GENERATOR_SEED = 99309
SPLIT_SEED = 99310


def hand_probe():
    h = {'choice': np.zeros((2,2),dtype=int), 'state': np.array([[1,0],[1,0]]),
         'second_choice': np.zeros((2,2),dtype=int), 'reward': np.array([[1,0],[0,0]])}
    cue = cue_probabilities(h,[.75])[:,0,1]
    hybrid = replay_policy(h,.75,.4,4)[:,1]
    expected = 1/(1+np.exp(-np.array([.14,.10])))
    np.testing.assert_array_equal(cue,[.75,.25])
    np.testing.assert_allclose(hybrid,expected,rtol=0,atol=1e-14)
    return {'previous_trial': 'first action 0, state 1, second action 0; reward 1 vs 0',
            'cue_rewarded_unrewarded':cue.tolist(),'hybrid_rewarded_unrewarded':hybrid.tolist(),
            'cue_contrast':float(cue[0]-cue[1]),'hybrid_contrast':float(hybrid[0]-hybrid[1])}


def score_group(truth, choice, p, mask):
    # Same person weighting for overall and diagnostic scores. Omit people
    # without any diagnostic support only for that diagnostic summary.
    loss = binary_nll(choice,p)
    kl = binary_kl(truth,p)
    supported = mask.any(1)
    def person_mean(values):
        means = [row[use].mean() for row, use in zip(values,mask) if use.any()]
        return float(np.mean(means)) if means else None
    return {'support_people': int(supported.sum()), 'support_trials': int(mask.sum()),
            'nll': person_mean(loss),'kl':person_mean(kl)}


def run_length(d, length):
    h={key:d[key][:,:length] for key in ('choice','state','second_choice','reward')}
    truth=d['true_prob'][:,:length]
    reference=d['reference_prob'][:,:length]
    n=len(truth)
    mask=make_trial_masks(n,length,SPLIT_SEED)
    fit=fit_family_mixture(h,mask['train'])
    if not fit.converged: raise RuntimeError('Family fit did not reach its stopping tolerance')
    hp, cp = candidate_probabilities(h), cue_probabilities(h)
    reproduced=np.zeros_like(truth)
    nearest_wrong_family=[]
    for i, rare in enumerate(d['rare']):
        if rare:
            reproduced[i]=cp[i,CUE_GRID.index(d['cue_reliability'][i])]
            wrong=hp[i]
        else:
            reproduced[i]=hp[i,d['hybrid_grid_index'][i]]
            wrong=cp[i]
        # The executable information check uses the true policy after fitting.
        rows=wrong[:,mask['train'][i]]
        correct=np.broadcast_to(truth[i,mask['train'][i]],rows.shape)
        nearest_wrong_family.append(float(binary_kl(correct,rows).mean(1).min()))
    maximum_replay_error=float(abs(reproduced-truth).max())
    reference_error=float(abs(replay_policy(h,.75,d['alpha'],d['beta'])-reference).max())
    if maximum_replay_error>1e-12 or reference_error>1e-12 or min(nearest_wrong_family)<=0:
        raise AssertionError('Truth, reference replay or candidate-family separation check failed')
    np.testing.assert_array_equal(d['x'][:,0],0)
    np.testing.assert_array_equal(d['x'][:,1:length],2*np.stack(list(h.values()),axis=-1)[:,:-1]-1)
    diagnostic=(abs(truth-reference)>.2)&mask['test']
    hard=fit.family_posterior.argmax(1).astype(bool)
    group_data={}
    for family, selector in [('hybrid',~d['rare']),('cue',d['rare'])]:
        scores={}
        for name,p in [('family_mixture',fit.probability),('hybrid_only',fit.hybrid_only_probability),
                       ('cue_only',fit.cue_only_probability)]:
            scores[name]={which:score_group(truth[selector],h['choice'][selector],p[selector],which_mask[selector])
                          for which,which_mask in [('overall',mask['test']),('diagnostic',diagnostic)]}
        group_data[family]={'people':int(selector.sum()),'correct_family_map':int((hard[selector]==d['rare'][selector]).sum()),
                            'mean_posterior_cue':float(fit.family_posterior[selector,1].mean()), 'test_scores':scores}
    return {'length':length,'training_choices_per_person':int(mask['train'][0].sum()),
            'test_choices_per_person':int(mask['test'][0].sum()),'groups':group_data,
            'prior_family_weights_hybrid_cue':fit.family_prior.tolist(),'posterior_cue_by_person':fit.family_posterior[:,1].tolist(),
            'iterations':fit.iterations,'converged':fit.converged,'maximum_truth_replay_error':maximum_replay_error,
            'maximum_reference_replay_error':reference_error,'causal_history_matches':True,
            'minimum_mean_training_kl_to_any_other_family_candidate':min(nearest_wrong_family),
            'realized_common_transitions_fraction':float((h['choice']==h['state']).mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON result file; defaults to stdout.')
    args=parser.parse_args()
    d=simulate_fixture(max(LENGTHS),GENERATOR_SEED)
    paths=('foundation/reward_cue.py','foundation/classical_baselines.py','scripts/check_rare_mechanism.py',
           'src/distilled_rnns/simulation.py','src/distilled_rnns/evaluation.py')
    result={'scope':'Fixed synthetic 20-hybrid/4-cue foundation fixture; same people and histories at both lengths.',
            'generator_seed':GENERATOR_SEED,'split_seed':SPLIT_SEED,'numpy_version':np.__version__,
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
            'cue_grid':list(CUE_GRID),'hybrid_grid':list(DEFAULT_GRID),'family_pseudocount':.1,
            'within_family_prior':'uniform within each family, averaged evidence; family prior estimated by EM',
            'true_cue_indices':np.flatnonzero(d['rare']).tolist(),'hand_derived_probe':hand_probe(),
            'runs':[run_length(d,length) for length in LENGTHS]}
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output is None:
        print(text,end='')
    else:
        args.output.write_text(text)


if __name__=='__main__':
    main()
