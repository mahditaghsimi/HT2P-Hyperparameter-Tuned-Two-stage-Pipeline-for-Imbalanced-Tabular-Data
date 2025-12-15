import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier
import time
from tqdm.auto import tqdm
import warnings
import gc
import json
import sys
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Settings
PARAMS_CSV_PATH = "input_grid_modes_split/grid_params_split_06.csv"
N_RANDOM_SAMPLES = 7000
CHECKPOINT_FREQUENCY = 10  # Save checkpoint every 10 models

try:
    SPLIT_NUMBER = int(Path(PARAMS_CSV_PATH).stem.split('_')[-1])
except:
    SPLIT_NUMBER = 1

print("=" * 80)
print(f"Parameter File: {PARAMS_CSV_PATH}")
print(f"Split Number: {SPLIT_NUMBER}")
print(f"Random Samples: {N_RANDOM_SAMPLES}")
print(f"Checkpoint Frequency: Every {CHECKPOINT_FREQUENCY} models")
print("=" * 80 + "\n")

# Create Checkpoint Directory
checkpoint_dir = Path('output_grid')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = checkpoint_dir / f'checkpoint_split_{SPLIT_NUMBER:02d}.pkl'
results_file = checkpoint_dir / f'results_split_{SPLIT_NUMBER:02d}_incremental.csv'
completed_file = checkpoint_dir / f'completed_ids_split_{SPLIT_NUMBER:02d}.txt'

# File Check
params_file = Path(PARAMS_CSV_PATH)
if not params_file.exists():
    print(f"ERROR: File {PARAMS_CSV_PATH} not found!")
    sys.exit(1)

# GPU Detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=" * 80)
print(f"Device: {device.upper()}")
if device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.2f} GB")
else:
    print("ERROR: GPU not found - stopping execution")
    sys.exit(1)
print("=" * 80 + "\n")

# Download and Preprocessing
print("Downloading and preprocessing Adult dataset...\n")

train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

data_path = Path('data')
data_path.mkdir(parents=True, exist_ok=True)

train_file = data_path / 'adult_train.csv'
test_file = data_path / 'adult_test.csv'

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

if not train_file.exists():
    print("Downloading Train...")
    try:
        import urllib.request
        urllib.request.urlretrieve(train_url, train_file)
    except:
        print("ERROR: Download error")
        sys.exit(1)

if not test_file.exists():
    print("Downloading Test...")
    try:
        import urllib.request
        urllib.request.urlretrieve(test_url, test_file)
    except:
        print("ERROR: Download error")
        sys.exit(1)

df_train = pd.read_csv(train_file, header=None, names=columns, na_values=' ?', skipinitialspace=True)
df_test = pd.read_csv(test_file, header=None, names=columns, na_values=' ?', 
                      skipinitialspace=True, skiprows=1)

df_train = df_train.dropna().reset_index(drop=True)
df_test = df_test.dropna().reset_index(drop=True)

df_train['income'] = df_train['income'].str.strip()
df_test['income'] = df_test['income'].str.replace('.', '', regex=False).str.strip()

X_full = pd.concat([df_train.drop('income', axis=1), df_test.drop('income', axis=1)], 
                   ignore_index=True)
y_full = pd.concat([
    df_train['income'].map({'<=50K': 0, '>50K': 1}),
    df_test['income'].map({'<=50K': 0, '>50K': 1})
], ignore_index=True).astype(int)

categorical_columns = X_full.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_full[col] = le.fit_transform(X_full[col].astype(str))
    label_encoders[col] = le

train_size = len(df_train)
X_train_full = X_full.iloc[:train_size]
y_train_full = y_full.iloc[:train_size]
X_test = X_full.iloc[train_size:]
y_test = y_full.iloc[train_size:]

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

cat_idxs = [i for i, col in enumerate(X_train.columns) if col in categorical_columns]
cat_dims = [X_train[col].nunique() for col in categorical_columns]

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train.values)
weights_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

print(f"Data ready:")
print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Class Weights: {{0: {weights_dict[0]:.4f}, 1: {weights_dict[1]:.4f}}}\n")

# Helper Functions
def find_best_threshold(y_true, y_proba):
    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.3, 0.8, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    return best_threshold, best_f1

def train_single_model_gpu(combination_id, params_dict):
    try:
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=params_dict['cat_emb_dim'],
            n_d=params_dict['n_d'],
            n_a=params_dict['n_a'],
            n_steps=params_dict['n_steps'],
            gamma=params_dict['gamma'],
            lambda_sparse=params_dict['lambda_sparse'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params_dict['lr']),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.95},
            mask_type="entmax",
            device_name='cuda',
            verbose=0,
            seed=42
        )
        
        clf.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            max_epochs=30,
            patience=7,
            batch_size=params_dict['batch_size'],
            virtual_batch_size=params_dict['virtual_batch_size'],
            weights=weights_dict
        )
        
        y_val_proba = clf.predict_proba(X_val.values)[:, 1]
        threshold, val_f1 = find_best_threshold(y_val, y_val_proba)
        
        y_val_pred = (y_val_proba >= threshold).astype(int)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        del clf
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'combination_id': combination_id,
            **params_dict,
            'threshold': threshold,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'status': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        torch.cuda.empty_cache()
        return {
            'combination_id': combination_id,
            **params_dict,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def save_checkpoint(checkpoint_data):
    """Save checkpoint to file"""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {checkpoint_file}")

def load_checkpoint():
    """Load checkpoint from file"""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_completed_id(combination_id):
    """Append completed ID to file"""
    with open(completed_file, 'a') as f:
        f.write(f"{combination_id}\n")

def load_completed_ids():
    """Load list of completed IDs"""
    if completed_file.exists():
        with open(completed_file, 'r') as f:
            return set(int(line.strip()) for line in f if line.strip())
    return set()

def save_incremental_results(results_list):
    """Save results incrementally"""
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(results_file, index=False)

# Load or Create Checkpoint
checkpoint_data = load_checkpoint()
completed_ids = load_completed_ids()

if checkpoint_data is not None:
    print("=" * 80)
    print("RESUMING FROM CHECKPOINT")
    print("=" * 80)
    results_list = checkpoint_data['results_list']
    start_time = checkpoint_data['start_time']
    baseline_results = checkpoint_data['baseline_results']
    print(f"Loaded {len(results_list)} previous results")
    print(f"Completed IDs: {len(completed_ids)}")
    print("=" * 80 + "\n")
else:
    print("Starting fresh - no checkpoint found\n")
    
    # Train Baseline
    print("Training Baseline Model...\n")
    clf_baseline = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        device_name='cuda',
        verbose=0,
        seed=42
    )
    
    clf_baseline.fit(
        X_train.values, y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        weights=weights_dict
    )
    
    y_val_proba_base = clf_baseline.predict_proba(X_val.values)[:, 1]
    baseline_threshold, _ = find_best_threshold(y_val, y_val_proba_base)
    y_val_pred_base = (y_val_proba_base >= baseline_threshold).astype(int)
    y_test_pred_base = clf_baseline.predict(X_test.values)
    
    baseline_results = {
        'val_acc': accuracy_score(y_val, y_val_pred_base),
        'val_f1': f1_score(y_val, y_val_pred_base),
        'test_acc': accuracy_score(y_test, y_test_pred_base),
        'test_f1': f1_score(y_test, y_test_pred_base)
    }
    
    print(f"Baseline Results:")
    print(f"   Val  - Acc: {baseline_results['val_acc']:.4f}, F1: {baseline_results['val_f1']:.4f}")
    print(f"   Test - Acc: {baseline_results['test_acc']:.4f}, F1: {baseline_results['test_f1']:.4f}\n")
    
    del clf_baseline
    gc.collect()
    torch.cuda.empty_cache()
    
    results_list = []
    start_time = time.time()

# Read and Sample Parameters
df_params = pd.read_csv(params_file)
total_combinations = len(df_params)

print(f"Total combinations in file: {total_combinations:,}")

if total_combinations <= N_RANDOM_SAMPLES:
    print(f"WARNING: Using all {total_combinations} combinations\n")
    df_params_sample = df_params.copy()
    actual_n_samples = total_combinations
else:
    print(f"Randomly sampling {N_RANDOM_SAMPLES} combinations...\n")
    df_params_sample = df_params.sample(n=N_RANDOM_SAMPLES, random_state=42).reset_index(drop=True)
    actual_n_samples = N_RANDOM_SAMPLES

# Sequential GPU Grid Search with Checkpointing
print(f"Starting Sequential GPU Grid Search...\n")
print(f"Progress: {len(completed_ids)}/{len(df_params_sample)} completed\n")

for idx, row in tqdm(df_params_sample.iterrows(), total=len(df_params_sample), 
                     desc="GPU Training", ncols=100, initial=len(completed_ids)):
    
    combination_id = int(row['combination_id']) if 'combination_id' in row else idx
    
    # Skip if already completed
    if combination_id in completed_ids:
        continue
    
    params_dict = {
        'n_d': int(row['n_d']),
        'n_a': int(row['n_a']),
        'n_steps': int(row['n_steps']),
        'gamma': float(row['gamma']),
        'lambda_sparse': float(row['lambda_sparse']),
        'lr': float(row['lr']),
        'cat_emb_dim': int(row['cat_emb_dim']),
        'batch_size': int(row['batch_size']),
        'virtual_batch_size': int(row['virtual_batch_size'])
    }
    
    result = train_single_model_gpu(combination_id, params_dict)
    results_list.append(result)
    
    # Mark as completed
    save_completed_id(combination_id)
    completed_ids.add(combination_id)
    
    # Save checkpoint every N models
    if len(completed_ids) % CHECKPOINT_FREQUENCY == 0:
        checkpoint_data = {
            'results_list': results_list,
            'start_time': start_time,
            'baseline_results': baseline_results,
            'last_checkpoint': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_checkpoint(checkpoint_data)
        save_incremental_results(results_list)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / len(completed_ids)
        remaining = avg_time * (len(df_params_sample) - len(completed_ids))
        print(f"\n   CHECKPOINT [{len(completed_ids)}/{len(df_params_sample)}] "
              f"Avg: {avg_time:.1f}s/model | Remaining: {remaining/60:.1f} min")

# Final Save
elapsed_time = time.time() - start_time

print(f"\n{'=' * 80}")
print(f"All models completed!")
print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
print(f"{'=' * 80}\n")

# Save final results
save_incremental_results(results_list)

successful_results = [r for r in results_list if r.get('status') == 'success']
failed_results = [r for r in results_list if r.get('status') == 'failed']

print(f"Successful: {len(successful_results):,} | Failed: {len(failed_results):,}\n")

if len(successful_results) == 0:
    print("ERROR: No successful results!")
    sys.exit(1)

results_df = pd.DataFrame(successful_results)
results_df = results_df.sort_values('val_f1', ascending=False).reset_index(drop=True)

# Save Final Results
results_dir = Path('results')
results_dir.mkdir(parents=True, exist_ok=True)

output_file = results_dir / f'results_split_{SPLIT_NUMBER:02d}_final.csv'
results_df.to_csv(output_file, index=False)
print(f"Final results saved: {output_file}\n")

if failed_results:
    failed_df = pd.DataFrame(failed_results)
    error_file = results_dir / f'errors_split_{SPLIT_NUMBER:02d}.csv'
    failed_df.to_csv(error_file, index=False)
    print(f"WARNING: Errors saved: {error_file}\n")

# Display TOP 10
print("=" * 90)
print("TOP 10 Best Combinations")
print("=" * 90)
display_cols = ['combination_id', 'n_d', 'n_a', 'n_steps', 'gamma', 'lambda_sparse', 
                'lr', 'cat_emb_dim', 'batch_size', 'threshold', 'val_acc', 'val_f1']
print(results_df[display_cols].head(10).to_string(index=True))
print()

# Retrain Best Model
best_result = results_df.iloc[0]
best_params = {
    'n_d': int(best_result['n_d']),
    'n_a': int(best_result['n_a']),
    'n_steps': int(best_result['n_steps']),
    'gamma': float(best_result['gamma']),
    'lambda_sparse': float(best_result['lambda_sparse']),
    'lr': float(best_result['lr']),
    'cat_emb_dim': int(best_result['cat_emb_dim']),
    'batch_size': int(best_result['batch_size']),
    'virtual_batch_size': int(best_result['virtual_batch_size'])
}
best_threshold = float(best_result['threshold'])

print("=" * 90)
print("Best Parameters:")
print("=" * 90)
for k, v in best_params.items():
    print(f"   {k:20s}: {v}")
print(f"   {'threshold':20s}: {best_threshold:.4f}")
print(f"   {'combination_id':20s}: {int(best_result['combination_id'])}")
print(f"   {'val_f1':20s}: {best_result['val_f1']:.4f}\n")

print("Retraining best model with max_epochs=100...\n")

best_model = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    **best_params,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=best_params['lr']),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 10, "gamma": 0.95},
    mask_type="entmax",
    device_name='cuda',
    verbose=1,
    seed=42
)

best_model.fit(
    X_train.values, y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    max_epochs=100,
    patience=15,
    batch_size=best_params['batch_size'],
    virtual_batch_size=best_params['virtual_batch_size'],
    weights=weights_dict
)

y_test_proba = best_model.predict_proba(X_test.values)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n" + "=" * 90)
print("Final Test Results:")
print("=" * 90)
print(f"   Accuracy : {test_acc:.4f}")
print(f"   F1-Score : {test_f1:.4f}\n")
print(classification_report(y_test, y_test_pred, target_names=['<=50K', '>50K']))

# Save Model and Config
model_file = results_dir / f'best_model_split_{SPLIT_NUMBER:02d}'
best_model.save_model(model_file.as_posix())

config = {
    'split_number': SPLIT_NUMBER,
    'total_combinations': total_combinations,
    'samples_tested': actual_n_samples,
    'best_combination_id': int(best_result['combination_id']),
    'best_params': best_params,
    'best_threshold': best_threshold,
    'val_f1': float(best_result['val_f1']),
    'val_acc': float(best_result['val_acc']),
    'test_f1': test_f1,
    'test_acc': test_acc,
    'baseline': baseline_results,
    'total_time_minutes': elapsed_time / 60,
    'successful_models': len(successful_results),
    'failed_models': len(failed_results)
}

config_file = results_dir / f'config_split_{SPLIT_NUMBER:02d}.json'
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# Clean up checkpoint files
if checkpoint_file.exists():
    checkpoint_file.unlink()
if completed_file.exists():
    completed_file.unlink()
print("\nCheckpoint files cleaned up\n")

print("=" * 90)
print("Saved Files:")
print("=" * 90)
print(f"   SUCCESS: {output_file}")
print(f"   SUCCESS: {model_file}.zip")
print(f"   SUCCESS: {config_file}")
print()
print("=" * 90)
print(f"Split {SPLIT_NUMBER:02d} completed successfully!")
print("=" * 90)

