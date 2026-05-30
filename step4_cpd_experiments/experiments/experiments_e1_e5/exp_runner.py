"""
Experiment Runner for E1-E5
============================
Comprehensive CPD and feature engineering experiments with metrics tracking
and visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json
from datetime import datetime

import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import precision_recall_curve, auc

# Add parent directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


@dataclass
class ExperimentMetrics:
    """Container for CPD metrics"""
    exp_id: str
    feature_set: str
    cpd_method: str
    precision: float
    recall: float
    f1: float
    avg_time_to_detection: float
    num_detections: int
    sequence_count: int
    
    def to_dict(self):
        return asdict(self)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal using RobustScaler."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    signal_var = np.var(signal)
    if signal_var < 1e-15:
        return signal.flatten()
    
    scaler = RobustScaler()
    normalized = scaler.fit_transform(signal)
    return normalized.flatten()


def smooth_signal(signal: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth signal using uniform filter."""
    return uniform_filter1d(signal, size=window, mode='nearest')


def apply_ewma(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply Exponential Weighted Moving Average smoothing."""
    return pd.Series(signal).ewm(alpha=alpha, adjust=False).mean().values


def compute_cpd_metrics(
    detected_cps: List[int], 
    true_cp: Optional[int], 
    tolerance: int = 1
) -> Optional[Dict[str, float]]:
    """Compute precision, recall, F1 for CPD detection."""
    if true_cp is None or len(detected_cps) == 0:
        return None
    
    within = [d for d in detected_cps if abs(d - true_cp) <= tolerance]
    tp = 1 if within else 0
    fp = len(detected_cps) - tp
    fn = 1 - tp
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    time_to_detection = min(abs(d - true_cp) for d in detected_cps) if detected_cps else float('inf')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time_to_detection': time_to_detection
    }


def run_pelt_cpd(
    signal: np.ndarray,
    min_size: int = 5,
    penalty: float = 1.0,
    model: str = "l2"
) -> List[int]:
    """Run PELT algorithm for CPD."""
    signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
    algo = rpt.Pelt(model=model, min_size=min_size)
    algo.fit(signal)
    bkps = algo.predict(pen=penalty)
    return [cp for cp in bkps if cp < len(signal)]


def run_pelt_cpd_robust(
    signal: np.ndarray,
    min_size: int = 5,
    penalty: float = 1.0,
    cost_model: str = "l2"
) -> List[int]:
    """Run PELT with different cost models."""
    signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
    try:
        algo = rpt.Pelt(model=cost_model, min_size=min_size)
        algo.fit(signal)
        bkps = algo.predict(pen=penalty)
        return [cp for cp in bkps if cp < len(signal)]
    except Exception as e:
        print(f"Warning: PELT with {cost_model} failed: {e}")
        return []


def prepare_signal(
    signal: np.ndarray,
    normalize: bool = True,
    ewma: bool = True,
    smooth: bool = True
) -> np.ndarray:
    """Prepare signal with standard preprocessing."""
    signal = signal.astype(float)
    signal = np.nan_to_num(signal, nan=0)  # Fill NaNs with 0
    
    if normalize:
        signal = normalize_signal(signal)
    if ewma:
        signal = apply_ewma(signal, alpha=0.3)
    if smooth:
        signal = smooth_signal(signal, window=3)
    
    return signal


def create_feature_delta(df: pd.DataFrame, feature: str, lag: int = 1, fillna: float = 0) -> np.ndarray:
    """Create differenced feature."""
    values = df[feature].values.astype(float)
    values = np.nan_to_num(values, nan=fillna)
    delta = np.diff(values, n=lag, prepend=fillna)
    return delta


def create_rolling_variance(df: pd.DataFrame, feature: str, window: int = 3) -> np.ndarray:
    """Create rolling variance feature."""
    values = df[feature].values.astype(float)
    values = np.nan_to_num(values, nan=0)
    variance = pd.Series(values).rolling(window=window, center=True).var().fillna(0).values
    return variance


def plot_feature_vs_label(
    df: pd.DataFrame,
    feature: str,
    label: str,
    output_dir: Path,
    title: str
):
    """Plot feature values against hallucination labels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    data_to_plot = [df[df[label] == 0][feature].dropna(), 
                     df[df[label] == 1][feature].dropna()]
    ax1.boxplot(data_to_plot, labels=['No Hallucination', 'Hallucination'])
    ax1.set_ylabel(feature)
    ax1.set_title(f'{title} - Box Plot')
    ax1.grid(True, alpha=0.3)
    
    # Distribution plot
    ax2.hist(df[df[label] == 0][feature].dropna(), bins=30, alpha=0.6, label='No Hallucination')
    ax2.hist(df[df[label] == 1][feature].dropna(), bins=30, alpha=0.6, label='Hallucination')
    ax2.set_xlabel(feature)
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title} - Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"feature_vs_label_{feature}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved feature plot: {output_path}")


def plot_pr_curve(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    output_dir: Path,
    title: str
):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o', markersize=2, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    output_path = output_dir / f"pr_curve_{title}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve: {output_path}")
    
    return pr_auc


class ExperimentRunner:
    """Runner for CPD experiments with unified interface."""
    
    def __init__(self, data_file: str, output_dir: str, sample_frac: float = 0.5):
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"Loading data from {self.data_file}...")
        self.df = pd.read_csv(self.data_file, low_memory=False)
        print(f"Loaded {len(self.df)} rows")
        
        # Sample for faster experimentation (can disable by setting sample_frac=1.0)
        if sample_frac < 1.0:
            self.df = self.df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            print(f"Sampled to {len(self.df)} rows ({sample_frac*100:.0f}%)")
        
        self.results = []
        
    def add_features_to_df(self):
        """Add new features for E2, E2.5 experiments."""
        print("\nAdding new features...")
        
        # Sort by sequence for proper lag calculation
        sort_cols = ['domain', 'question_id']
        if 'config_idx' in self.df.columns:
            sort_cols.append('config_idx')
        if 'chunk_index' in self.df.columns:
            sort_cols.append('chunk_index')
        
        self.df = self.df.sort_values(sort_cols).reset_index(drop=True)
        
        # E2: Feature differencing
        if 'att_avg_entropy' in self.df.columns:
            self.df['delta_avg_attn_entropy'] = self.df.groupby(
                ['domain', 'question_id'])['att_avg_entropy'].transform(
                lambda x: np.concatenate([[0], np.diff(x.values)]))
        
        if 'distractor_attention_max' in self.df.columns:
            self.df['delta_distractor_max'] = self.df.groupby(
                ['domain', 'question_id'])['distractor_attention_max'].transform(
                lambda x: np.concatenate([[0], np.diff(x.values)]))
        
        # E2.5: Attention level change
        if 'attention_entropy' in self.df.columns:
            self.df['entropy_variance'] = self.df.groupby(
                ['domain', 'question_id'])['attention_entropy'].transform(
                lambda x: pd.Series(x.values).rolling(window=3, center=True).var().fillna(0).values)
        
        # Gold attention drop (if gold_found column exists)
        if 'cross_attention_mass_to_gold' in self.df.columns:
            self.df['gold_attn_drop'] = self.df.groupby(
                ['domain', 'question_id'])['cross_attention_mass_to_gold'].transform(
                lambda x: np.concatenate([[0], -np.diff(x.values)]))  # Negative diff = drop
        
        # Attn reallocation rate
        if 'attention_entropy' in self.df.columns:
            self.df['attn_reallocation_rate'] = self.df.groupby(
                ['domain', 'question_id'])['attention_entropy'].transform(
                lambda x: np.abs(np.concatenate([[0], np.diff(x.values)])))
        
        print("Features added successfully")
    
    def run_e1_penalty_sweep(self, penalties: List[float] = None):
        """E1: Penalty sweep with PELT."""
        if penalties is None:
            base_penalty = 1.0
            penalties = [0.1 * base_penalty, 0.25 * base_penalty, 0.5 * base_penalty]
        
        print(f"\n{'='*70}")
        print(f"E1: Penalty Sweep - Penalties: {penalties}")
        print(f"{'='*70}")
        
        # Use existing features
        feature_cols = ['evidence_pos_category', 'interference_score_lexical_wrt_distractors']
        
        for penalty in penalties:
            print(f"\nRunning PELT with penalty={penalty}...")
            metrics = self._run_cpd_experiment(
                exp_id=f"E1_pen{penalty}",
                feature_cols=feature_cols,
                cpd_fn=lambda sig: run_pelt_cpd(sig, penalty=penalty),
                cpd_method=f"PELT(λ={penalty})"
            )
            
            if metrics:
                self.results.append(metrics)
                print(f"  Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1:.3f}")
    
    def run_e2_feature_differencing(self):
        """E2: Feature differencing."""
        print(f"\n{'='*70}")
        print(f"E2: Feature Differencing (Δ metrics)")
        print(f"{'='*70}")
        
        # Ensure features exist
        if 'delta_avg_attn_entropy' not in self.df.columns:
            self.add_features_to_df()
        
        # Visualize new features
        feature_cols = ['delta_avg_attn_entropy', 'gold_attn_drop', 'delta_distractor_max']
        for feat in feature_cols:
            if feat in self.df.columns:
                plot_feature_vs_label(
                    self.df, feat, 'hallu_label',
                    self.output_dir,
                    title=f"E2: {feat}"
                )
        
        # Run CPD with new features
        print(f"\nRunning CPD with differenced features...")
        metrics = self._run_cpd_experiment(
            exp_id="E2",
            feature_cols=feature_cols,
            cpd_fn=lambda sig: run_pelt_cpd(sig, penalty=1.0),
            cpd_method="PELT"
        )
        
        if metrics:
            self.results.append(metrics)
            print(f"  Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1:.3f}")
    
    def run_e2_5_attn_level_change(self):
        """E2.5: Attention level change."""
        print(f"\n{'='*70}")
        print(f"E2.5: Attention Level Change")
        print(f"{'='*70}")
        
        # Ensure features exist
        if 'entropy_variance' not in self.df.columns:
            self.add_features_to_df()
        
        # Visualize new features
        feature_cols = ['entropy_variance', 'attn_reallocation_rate']
        for feat in feature_cols:
            if feat in self.df.columns:
                plot_feature_vs_label(
                    self.df, feat, 'hallu_label',
                    self.output_dir,
                    title=f"E2.5: {feat}"
                )
        
        # Run CPD with new features
        print(f"\nRunning CPD with attention level features...")
        metrics = self._run_cpd_experiment(
            exp_id="E2.5",
            feature_cols=feature_cols,
            cpd_fn=lambda sig: run_pelt_cpd(sig, penalty=1.0),
            cpd_method="PELT"
        )
        
        if metrics:
            self.results.append(metrics)
            print(f"  Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1:.3f}")
    
    def run_e3_robust_cost_l1_huber(self):
        """E3: Robust cost L1/Huber."""
        print(f"\n{'='*70}")
        print(f"E3: Robust Cost (L1/Huber)")
        print(f"{'='*70}")
        
        feature_cols = ['evidence_pos_category', 'interference_score_lexical_wrt_distractors']
        cost_models = ['l1', 'huber']
        
        for model in cost_models:
            print(f"\nRunning PELT with {model} cost...")
            metrics = self._run_cpd_experiment(
                exp_id=f"E3_{model}",
                feature_cols=feature_cols,
                cpd_fn=lambda sig, m=model: run_pelt_cpd_robust(sig, cost_model=m),
                cpd_method=f"PELT({model})"
            )
            
            if metrics:
                self.results.append(metrics)
                print(f"  Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1:.3f}")
    
    def run_e3_5_robust_cost_gaussian(self):
        """E3.5: Robust cost Gaussian."""
        print(f"\n{'='*70}")
        print(f"E3.5: Robust Cost (Gaussian)")
        print(f"{'='*70}")
        
        feature_cols = ['evidence_pos_category', 'interference_score_lexical_wrt_distractors']
        
        print(f"\nRunning PELT with gaussian cost...")
        metrics = self._run_cpd_experiment(
            exp_id="E3.5_gaussian",
            feature_cols=feature_cols,
            cpd_fn=lambda sig: run_pelt_cpd_robust(sig, cost_model='l2'),  # Gaussian is l2
            cpd_method="PELT(l2/Gaussian)"
        )
        
        if metrics:
            self.results.append(metrics)
            print(f"  Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1:.3f}")
    
    def _run_cpd_experiment(
        self,
        exp_id: str,
        feature_cols: List[str],
        cpd_fn: Callable,
        cpd_method: str,
        max_sequences: int = 500
    ) -> Optional[ExperimentMetrics]:
        """Generic CPD experiment runner."""
        # Filter valid rows
        valid_cols = [c for c in feature_cols if c in self.df.columns]
        if not valid_cols:
            print(f"Warning: No valid feature columns found for {exp_id}")
            return None
        
        df_valid = self.df[self.df[valid_cols].notna().all(axis=1)].copy()
        
        # Group by sequence
        group_cols = ['domain', 'question_id']
        if 'config_idx' in df_valid.columns:
            group_cols.append('config_idx')
        
        all_metrics = []
        all_scores = []
        all_labels = []
        seq_count = 0
        skipped = 0
        
        groups = df_valid.groupby(group_cols)
        total_groups = len(groups)
        print(f"  Processing {total_groups} sequences (limiting to {max_sequences})...")
        
        for i, (seq_id, seq_df) in enumerate(groups):
            if seq_count >= max_sequences:
                skipped = total_groups - seq_count
                break
            
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{total_groups} ({seq_count} valid sequences)")
            
            seq_df = seq_df.sort_values('chunk_index' if 'chunk_index' in seq_df.columns else seq_df.index).reset_index(drop=True)
            
            if len(seq_df) < 5:
                continue
            
            seq_count += 1
            
            # Build combined signal from features
            signal_parts = []
            for col in valid_cols:
                s = seq_df[col].values.astype(float)
                s = np.nan_to_num(s, nan=0)
                s = prepare_signal(s)
                signal_parts.append(s)
            
            signal = np.column_stack(signal_parts) if len(signal_parts) > 1 else signal_parts[0].reshape(-1, 1)
            
            # Run CPD
            try:
                detected_cps = cpd_fn(signal)
            except Exception as e:
                continue
            
            # Get ground truth change point from hallu_label
            labels = seq_df['hallu_label'].values
            if len(labels) > 0 and np.any(labels):
                true_cp = np.argmax(labels)
                metrics = compute_cpd_metrics(detected_cps, true_cp)
                if metrics:
                    all_metrics.append(metrics)
                    all_scores.extend(detected_cps)
                    all_labels.extend([1] * len(detected_cps))
        
        if not all_metrics:
            print(f"Warning: No valid sequences for {exp_id}")
            return None
        
        print(f"  Processed {seq_count} sequences, computed {len(all_metrics)} metrics")
        
        # Aggregate metrics
        metrics_df = pd.DataFrame(all_metrics)
        avg_precision = metrics_df['precision'].mean()
        avg_recall = metrics_df['recall'].mean()
        avg_f1 = metrics_df['f1'].mean()
        avg_time_to_detection = metrics_df['time_to_detection'].mean()
        num_detections = sum(1 for m in all_metrics if m['precision'] > 0)
        
        return ExperimentMetrics(
            exp_id=exp_id,
            feature_set=str(valid_cols),
            cpd_method=cpd_method,
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
            avg_time_to_detection=avg_time_to_detection,
            num_detections=num_detections,
            sequence_count=seq_count
        )
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save results to JSON and CSV."""
        results_data = [m.to_dict() for m in self.results]
        
        # JSON
        json_path = self.output_dir / filename
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nSaved results to {json_path}")
        
        # CSV
        csv_path = self.output_dir / filename.replace('.json', '.csv')
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
        # Print summary table
        print(f"\n{'='*100}")
        print("EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*100}")
        print(results_df.to_string(index=False))
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="Run E1-E5 experiments")
    parser.add_argument(
        "--data_file",
        type=str,
        default="dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv",
        help="Path to prepared dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="E1,E2,E2.5,E3,E3.5",
        help="Comma-separated list of experiments to run"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.data_file, args.output_dir)
    
    exp_list = [e.strip() for e in args.experiments.split(',')]
    
    for exp in exp_list:
        if exp == "E1":
            runner.run_e1_penalty_sweep()
        elif exp == "E2":
            runner.run_e2_feature_differencing()
        elif exp == "E2.5":
            runner.run_e2_5_attn_level_change()
        elif exp == "E3":
            runner.run_e3_robust_cost_l1_huber()
        elif exp == "E3.5":
            runner.run_e3_5_robust_cost_gaussian()
    
    runner.save_results()


if __name__ == "__main__":
    main()
