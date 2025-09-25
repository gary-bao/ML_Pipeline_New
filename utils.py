# ml_pipeline/utils.py
from typing import List, Optional
import numpy as np
import pandas as pd
import logging

def prefiler_and_sample(pred_data: pd.DataFrame,
                        score_col: str,
                        n_nominations: int = 400,
                        top_k_per_seed: int = 50,
                        seed_sampling: str = 'uniform',
                        quantile_perc: float = 0.8,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Keep top_k per seed, filter by score quantile, then sample seeds (uniform or score-weighted).
    Returns DataFrame with selected rows (n_nominations or fewer).
    """
    qValue = pred_data[score_col].quantile(quantile_perc)

    # 1) keep top K candidates per seed
    topk = (pred_data
            .sort_values(['seed', score_col], ascending=[True, False])
            .groupby('seed')
            .head(top_k_per_seed)
            .reset_index(drop=True))
    topk = topk[topk[score_col] >= qValue]

    seeds = topk['seed'].unique().tolist()
    rng = np.random.RandomState(random_state)

    # 2) sample seeds (either uniformly or weighted by seed max score)
    if seed_sampling == 'uniform':
        chosen = []
        pools = {s: topk[topk['seed'] == s].copy().reset_index(drop=True) for s in seeds}
        # shuffle rows within each pool
        for s in pools:
            pools[s] = pools[s].sample(frac=1, random_state=random_state).reset_index(drop=True)
        i = 0
        cnt = 0
        while len(chosen) < n_nominations and any(len(df) > i for df in pools.values()):
            for s in seeds:
                df = pools[s]
                if i < len(df) and len(chosen) < n_nominations:
                    chosen.append(df.iloc[i])
            i += 1
            if cnt > 10000:
                break
            cnt += 1
        result = pd.DataFrame(chosen).reset_index(drop=True).head(n_nominations)
        return result

    elif seed_sampling == 'score':
        # seed weights proportional to seed's best score
        seed_scores = topk.groupby('seed')[score_col].max().reset_index()
        seed_scores['weight'] = seed_scores[score_col] / seed_scores[score_col].sum()
        probs = seed_scores['weight'].to_numpy()
        probs = probs / probs.sum()
        chosen_keys = set()
        chosen_rows = []
        max_attempts = 10000
        attempts = 0

        while len(chosen_rows) < n_nominations and attempts < max_attempts:
            if len(seed_scores) == 0:
                break
            sampled_seeds = rng.choice(seed_scores['seed'], size=min(2, len(seed_scores)), replace=False, p=probs)
            for seed in sampled_seeds:
                pool = topk[topk['seed'] == seed]
                for _, row in pool.iterrows():
                    key = (row['seed'], row['sequence'])
                    if key not in chosen_keys:
                        chosen_keys.add(key)
                        chosen_rows.append(row)
                        break
                if len(chosen_rows) >= n_nominations:
                    break
            attempts += 1
        # if not enough unique entries
        if len(chosen_rows) < n_nominations:
            remaining = topk[~topk.set_index(['seed', 'sequence']).index.isin(chosen_keys)]
            for _, row in remaining.iterrows():
                key = (row['seed'], row['sequence'])
                if key not in chosen_keys:
                    chosen_keys.add(key)
                    chosen_rows.append(row)
                if len(chosen_rows) >= n_nominations:
                    break

        result = pd.DataFrame(chosen_rows).reset_index(drop=True).head(n_nominations)
        return result
    else:
        raise ValueError("seed_sampling must be 'uniform' or 'score'")


def get_seeds(df_all: pd.DataFrame, positive_controls: List[str], negative_controls: List[str], top_n: int = 200) -> pd.DataFrame:
    """
    Extract seed data by computing pos_neg_ratio and sorting. Returns top_n unique sequences.
    """
    cols = [col for col in df_all.columns if col.endswith("CPM") or col.startswith("FC")]
    required_cols = ['sequence', 'seq_origin', 'SDB', 'GroupID']
    available = [c for c in required_cols if c in df_all.columns]
    df_all = df_all[available + cols] if available else df_all[cols]

    # filter out zero CPM rows if any selected control columns are present
    all_controls = positive_controls + negative_controls
    if all_controls:
        df_all = df_all[(df_all[all_controls] != 0).all(axis=1)]

    if negative_controls:
        df_all['pos_neg_ratio'] = df_all[positive_controls].min(axis=1) / df_all[negative_controls].max(axis=1)
    else:
        df_all['pos_neg_ratio'] = df_all[positive_controls].min(axis=1)
    df_sorted = df_all.sort_values(by='pos_neg_ratio', ascending=False)
    df_unique = df_sorted.drop_duplicates(subset="sequence", keep='first').head(top_n)
    return df_unique
