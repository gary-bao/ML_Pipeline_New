# ml_pipeline/cli.py
import argparse
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from .features import featurize
from .models.xgb_model import XGBModel
from .utils import prefiler_and_sample, get_seeds
from .mutations import mutate, mutate_double, is_valid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--sdb", default="")
    p.add_argument("--fp-size", type=int, default=2048)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--min-fc", type=float, default=0.1)
    p.add_argument("--should-bind", default="DLL31ug")
    p.add_argument("--should-not-bind", default="")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--double-scan", action="store_true")
    p.add_argument("--title", default="DLL3-XGBoost-1D")
    p.add_argument("--n-nominations", type=int, default=400)
    return p.parse_args()

def run_pipeline(args):
    input_path = Path(args.input)
    base_dir = input_path.parent
    output_dir = base_dir / "result" / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    if input_path.suffix.lower() == ".csv":
        data = pd.read_csv(str(input_path))
    else:
        data = pd.read_excel(str(input_path))
    logging.info(f"Loaded {data.shape[0]} rows")

    # basic filtering
    data = data[data["sequence"].apply(is_valid)]
    data = data[data["Input_CPM"] >= 2]
    if data.shape[0] < 10:
        logging.warning("Not enough rows after filtering; exiting")
        return
    
    if 'SDB' in data.columns:
        data = data.sort_values(['GroupID', 'Position', 'seq_origin', 'SDB'])
        sdbs = data['SDB'].unique().tolist()
    else:
        data = data.sort_values(['GroupID', 'Position', 'seq_origin'])

    # featurize
    sequences = data['sequence'].tolist()
    fp_size = args.fp_size
    radius = args.radius
    n_jobs = args.n_jobs

    X = np.stack(Parallel(n_jobs=n_jobs)(delayed(featurize)(seq, fp_size, radius) for seq in tqdm(sequences, desc="Featurize")))
    logging.info(f"Featurized to X shape {X.shape}")

    # targets extraction
    targets = [col.split('.')[1] for col in data.columns if "FC" in col]
    if args.sdb:
        data = data[data["SDB"] == args.sdb]

    # prepare data_dict
    data_dict = {}
    for target in targets:
        fc_column = f'FC.{target}.vs.Input'
        y = data[fc_column].to_numpy()
        mask = y > args.min_fc
        y_processed = np.log10(y[mask])
        data_dict[target] = {'X': X[mask], 'y': y_processed}
        logging.info(f"Prepared target {target} with {len(y_processed)} examples")

    # build config mapping positive/negative
    config = {}
    for t in (args.should_bind or "").split(','):
        t = t.strip()
        if t:
            config[t] = True
    for t in (args.should_not_bind or "").split(','):
        t = t.strip()
        if t:
            config[t] = False

    positive_controls = []
    negative_controls = []
    for target, is_pos in config.items():
        fc_column = f'FC.{target}.vs.Input'
        if is_pos:
            positive_controls.append(fc_column)
        else:
            negative_controls.append(fc_column)

    # get seeds
    seed_data = get_seeds(data.copy(), positive_controls, negative_controls, top_n=200)
    seeds = seed_data['sequence'].tolist()
    seq_to_group = dict(zip(seed_data["sequence"].tolist(), seed_data["GroupID"].tolist()))
    logging.info(f"Seeds: {len(seeds)}")

    # train models for config targets
    parameters = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
    }

    model_dict = {}
    for target in config:
        X_train = data_dict[target]['X']
        y_train = data_dict[target]['y']
        # if args.tune:
        #     best_params = 
        model = XGBModel(params=parameters, n_estimators=args.n_estimators)
        model.fit(X_train, y_train)   # point model
        model_dict[target] = model
        logging.info(f"Trained model for {target}")

    # predict children
    pred_rows = []
    for seed in seeds:
        children = mutate_double(seed) if args.double_scan else mutate(seed)
        X_novel = np.stack(Parallel(n_jobs=-1)(delayed(featurize)(child, fp_size, radius) for child in children))
        row = {'seed': seed, 'GroupID': seq_to_group.get(seed, None), 'sequence': children}
        for target, model in model_dict.items():
            ypred_log = model.predict(X_novel)
            row[target] = 10 ** ypred_log  # back-transform
        # expand to rows
        import pandas as pd
        df = pd.DataFrame(row)
        pred_rows.append(df)
    pred_data = pd.concat(pred_rows, ignore_index=True)
    logging.info(f"Predictions made for {len(pred_data)} candidate sequences")

    # sample nominations
    pos_targets = [t.strip() for t in (args.should_bind or "").split(',') if t.strip()]
    if pos_targets:
        child_nominations = prefiler_and_sample(pred_data, score_col=pos_targets[0], n_nominations=args.n_nominations)
        child_nominations.to_excel(output_dir / f"{args.title}-Nominations.xlsx", index=False)
        logging.info("Saved nominations")
