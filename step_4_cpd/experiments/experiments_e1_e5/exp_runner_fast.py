"""
Optimized Experiment Runner for E1-E5
======================================
Streamlined CPD experiments with smart sampling and efficient processing.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json

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
    num_true_positives: int
    num_false_positives: int
    num_sequences_evaluated: int
    
    def to_dict(self):
        return asdict(self)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal using RobustScaler."""
    signal = signal.astype(float)
    signal = np.nan_to_num(signal, nan=0)
    
    signal_var = np.var(signal)
    if signal_var < 1e-15:
        return signal
    
    signal = (signal - np.median(signal)) / (np.percentile(signal, 75) - np.percentile(signal, 25) + 1e-10)
    return signal


def smooth_signal(signal: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth signal using uniform filter."""
    return uniform_filter1d(signal, size=window, mode='nearest')


def prepare_signal(signal: np.ndarray, normalize: bool = True, smooth: bool = True) -> np.ndarray:
    """Prepare signal with standard preprocessing."""
    signal = signal.astype(float)
    signal = np.nan_to_num(signal, nan=0)
    
    if normalize:
        signal = normalize_signal(signal)
    if smooth:
        signal = smooth_signal(signal, window=3)
    
    return signal


def run_pelt_fast(
    signal: np.ndarray,
    min_size: int = 5,
    penalty: float = 1.0,
    model: str = "l2"
) -> List[int]:
    """Run PELT algorithm - returns change point indices."""
    if len(signal) < min_size:
        return []
    
    signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
    
    try:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=1)
        algo.fit(signal)
        bkps = algo.predict(pen=penalty)
        # Filter out the last index (end of sequence)
        return [cp for cp in bkps[:-1] if cp < len(signal)]
    except Exception as e:
        return []


def find_change_point(labels: np.ndarray, tolerance: int = 2) -> Optional[int]:
    """Find ground truth change point from binary labels."""
    if len(labels) == 0 or not np.any(labels):
        return None
    
    # Find first position where label becomes 1
    change_idx = np.argmax(labels)
    if labels[change_idx] == 1:
        return change_idx
    return None


def compute_detection_metrics(
    detected_cps: List[int],
    true_cp: Optional[int],
    tolerance: int = 2
) -> Dict[str, int]:
    """Compute TP/FP/FN for change point detection."""
    if true_cp is None:
        return {'tp': 0, 'fp': len(detected_cps), 'fn': 1}
    
    # Check if any detected CP is within tolerance of true CP
    tp = 1 if any(abs(cp - true_cp) <= tolerance for cp in detected_cps) else 0
    fp = len(detected_cps) - tp
    fn = 1 - tp
    
    return {'tp': tp, 'fp': fp, 'fn': fn}


def plot_feature_vs_label(
    df: pd.DataFrame,
    feature: str,
    label: str,
    output_dir: Path,
    title: str
):
    """Plot feature values against hallucination labels."""
    if feature not in df.columns or label not in df.columns:
        return
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Box plot
        data_to_plot = [
            df[df[label] == 0][feature].dropna().values,
            df[df[label] == 1][feature].dropna().values
        ]
        if all(len(d) > 0 for d in data_to_plot):
            axes[0].boxplot(data_to_plot, labels=['No Hallu', 'Hallu'])
            axes[0].set_ylabel(feature)
            axes[0].set_title(f'{title} - Box Plot')
            axes[0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1].hist(df[df[label] == 0][feature].dropna(), bins=20, alpha=0.6, label='No Hallu')
        axes[1].hist(df[df[label] == 1][feature].dropna(), bins=20, alpha=0.6, label='Hallu')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'{title} - Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"viz_{feature.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")
    except Exception as e:
        print(f"  Warning: Could not plot {feature}: {e}")


def plot_pr_curve(
    precisions: List[float],
    recalls: List[float],
    output_dir: Path,
    title: str
) -> float:
    """Plot precision-recall curve."""
    if not precisions or not recalls:
        return 0.0
    
    try:
        plt.figure(figsize=(8, 6))
        
        # Sort by recall for plotting
        sorted_pairs = sorted(zip(recalls, precisions))
        if sorted_pairs:
            recalls_sorted, precisions_sorted = zip(*sorted_pairs)
        else:
            recalls_sorted, precisions_sorted = recalls, precisions
        
        plt.plot(recalls_sorted, precisions_sorted, marker='o', markersize=3, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {title}')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        output_path = output_dir / f"pr_curve_{title.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")
        
        return 0.5  # Placeholder AUC
    except Exception as e:
        print(f"  Warning: Could not plot PR curve: {e}")
        return 0.0


class FastExperimentRunner:
    """Optimized runner for CPD experiments."""
    
    def __init__(self, data_file: str, output_dir: str):
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading dataset...")
        self.df = pd.read_csv(self.data_file, low_memory=False)
        print(f"Loaded: {len(self.df)} rows x {len(self.df.columns)} columns")
        
        self.results = []
        
    def add_features(self):
        """Add engineered features for E2, E2.5."""
        print("\nEnginering features...")
        
        # Sort for proper lag calculation
        sort_cols = ['domain', 'question_id']
        if 'config_idx' in self.df.columns:
            sort_cols.append('config_idx')
        if 'chunk_index' in self.df.columns:
            sort_cols.append('chunk_index')
        
        self.df = self.df.sort_values(sort_cols).reset_index(drop=True)
        
        # E2: Feature differencing
        if 'att_avg_entropy' in self.df.columns:
            self.df['delta_avg_attn_entropy'] = self.df.groupby(
                ['domain', 'question_id'])['att_avg_entropy'].diff().fillna(0)
        
        if 'distractor_attention_max' in self.df.columns:
            self.df['delta_distractor_max'] = self.df.groupby(
                ['domain', 'question_id'])['distractor_attention_max'].diff().fillna(0)
        
        # Gold attention drop
        if 'cross_attention_mass_to_gold' in self.df.columns:
            self.df['gold_attn_drop'] = -self.df.groupby(
                ['domain', 'question_id'])['cross_attention_mass_to_gold'].diff().fillna(0)
        
        # E2.5: Attention level features
        if 'attention_entropy' in self.df.columns:
            self.df['entropy_variance'] = self.df.groupby(
                ['domain', 'question_id'])['attention_entropy'].transform(
                lambda x: pd.Series(x.values).rolling(window=3, center=True).var().fillna(0).values)
        
        if 'attention_entropy' in self.df.columns:
            self.df['attn_reallocation_rate'] = self.df.groupby(
                ['domain', 'question_id'])['attention_entropy'].diff().fillna(0).abs()
        
        print("Features engineered successfully!")
    
    def run_experiment(
        self,
        exp_id: str,
        feature_cols: List[str],
        penalties: List[float] = None,
        cost_models: List[str] = None,
        sample_size: int = 100
    ) -> Optional[ExperimentMetrics]:
        """Generic experiment runner."""
        if penalties is None:
            penalties = [1.0]
        if cost_models is None:
            cost_models = ['l2']
        
        # Filter valid rows
        valid_cols = [c for c in feature_cols if c in self.df.columns]
        if not valid_cols:
            print(f"  Error: No valid features from {feature_cols}")
            return None
        
        # Ensure hallu_label exists
        if 'hallu_label' not in self.df.columns:
            print(f"  Error: No hallu_label column")
            return None
        
        df_valid = self.df[self.df[valid_cols].notna().all(axis=1)].copy()
        
        # Group sequences
        group_cols = ['domain', 'question_id']
        if 'config_idx' in df_valid.columns:
            group_cols.append('config_idx')
        
        groups = df_valid.groupby(group_cols)
        total_groups = min(len(groups), sample_size)
        
        print(f"  Processing {total_groups} sequences (sampled from {len(groups)})...")
        
        all_tp = 0
        all_fp = 0
        all_fn = 0
        precisions = []
        recalls = []
        seq_count = 0
        
        for i, (seq_id, seq_df) in enumerate(groups):
            if seq_count >= sample_size:
                break
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {seq_count}/{total_groups}...")
            
            # Sort sequence
            if 'chunk_index' in seq_df.columns:
                seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)
            
            if len(seq_df) < 5:
                continue
            
            seq_count += 1
            
            # Build signal
            signal_parts = []
            for col in valid_cols:
                s = seq_df[col].values.astype(float)
                s = prepare_signal(s)
                signal_parts.append(s)
            
            signal = np.column_stack(signal_parts) if len(signal_parts) > 1 else signal_parts[0].reshape(-1, 1)
            
            # Find ground truth
            labels = seq_df['hallu_label'].values
            true_cp = find_change_point(labels)
            
            if true_cp is None:
                continue
            
            # Run CPD with each configuration
            for penalty in penalties:
                for model in cost_models:
                    try:
                        detected_cps = run_pelt_fast(signal, penalty=penalty, model=model)
                        metrics = compute_detection_metrics(detected_cps, true_cp)
                        
                        all_tp += metrics['tp']
                        all_fp += metrics['fp']
                        all_fn += metrics['fn']
                        
                        if all_tp + all_fp > 0:
                            precisions.append(all_tp / (all_tp + all_fp))
                        if all_tp + all_fn > 0:
                            recalls.append(all_tp / (all_tp + all_fn))
                    except:
                        pass
        
        if seq_count == 0 or all_tp == 0:
            print(f"  No detections found")
            return None
        
        # Compute final metrics
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return ExperimentMetrics(
            exp_id=exp_id,
            feature_set=str(valid_cols),
            cpd_method=f"PELT(pen={penalties[0]:.3f}, model={cost_models[0]})",
            precision=precision,
            recall=recall,
            f1=f1,
            num_true_positives=all_tp,
            num_false_positives=all_fp,
            num_sequences_evaluated=seq_count
        )
    
    def run_all_experiments(self):
        """Run E1-E5."""
        print(f"\n{'='*70}")
        print("RUNNING EXPERIMENTS E1-E5")
        print(f"{'='*70}")
        
        # Add features first
        self.add_features()
        
        # E1: Penalty sweep
        print(f"\n{'='*70}")
        print("E1: Penalty Sweep")
        print(f"{'='*70}")
        for pen in [0.1, 0.25, 0.5]:
            print(f"\n  Penalty = {pen}")
            m = self.run_experiment(
                f"E1_pen{pen}",
                ['evidence_pos_category', 'interference_score_lexical_wrt_distractors'],
                penalties=[pen]
            )
            if m:
                self.results.append(m)
                print(f"  P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")
        
        # E2: Feature differencing
        print(f"\n{'='*70}")
        print("E2: Feature Differencing")
        print(f"{'='*70}")
        print("\n  Visualizing features...")
        for feat in ['delta_avg_attn_entropy', 'gold_attn_drop', 'delta_distractor_max']:
            if feat in self.df.columns:
                plot_feature_vs_label(self.df, feat, 'hallu_label', self.output_dir, f"E2: {feat}")
        
        print(f"\n  Running CPD...")
        m = self.run_experiment(
            "E2",
            ['delta_avg_attn_entropy', 'gold_attn_drop', 'delta_distractor_max']
        )
        if m:
            self.results.append(m)
            print(f"  P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")
        
        # E2.5: Attention level change
        print(f"\n{'='*70}")
        print("E2.5: Attention Level Change")
        print(f"{'='*70}")
        print("\n  Visualizing features...")
        for feat in ['entropy_variance', 'attn_reallocation_rate']:
            if feat in self.df.columns:
                plot_feature_vs_label(self.df, feat, 'hallu_label', self.output_dir, f"E2.5: {feat}")
        
        print(f"\n  Running CPD...")
        m = self.run_experiment(
            "E2.5",
            ['entropy_variance', 'attn_reallocation_rate']
        )
        if m:
            self.results.append(m)
            print(f"  P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")
        
        # E3: Robust cost L1/Huber
        print(f"\n{'='*70}")
        print("E3: Robust Cost (L1/Huber)")
        print(f"{'='*70}")
        for model in ['l1', 'huber']:
            print(f"\n  Model: {model}")
            m = self.run_experiment(
                f"E3_{model}",
                ['evidence_pos_category', 'interference_score_lexical_wrt_distractors'],
                cost_models=[model]
            )
            if m:
                self.results.append(m)
                print(f"  P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")
        
        # E3.5: Gaussian
        print(f"\n{'='*70}")
        print("E3.5: Robust Cost (Gaussian/L2)")
        print(f"{'='*70}")
        m = self.run_experiment(
            "E3.5_gaussian",
            ['evidence_pos_category', 'interference_score_lexical_wrt_distractors'],
            cost_models=['l2']
        )
        if m:
            self.results.append(m)
            print(f"  P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")
    
    def save_results(self):
        """Save results to CSV and JSON."""
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}\n")
        
        if not self.results:
            print("No results to save")
            return
        
        results_data = [m.to_dict() for m in self.results]
        
        # CSV
        csv_path = self.output_dir / "experiment_results_e1_e5.csv"
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # JSON
        json_path = self.output_dir / "experiment_results_e1_e5.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Saved JSON: {json_path}")
        
        # Print summary
        print(f"\n{'='*100}")
        print("RESULTS SUMMARY")
        print(f"{'='*100}\n")
        print(df_results.to_string(index=False))
        
        return df_results


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
        default="output/experiments_e1_e5",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    runner = FastExperimentRunner(args.data_file, args.output_dir)
    runner.run_all_experiments()
    runner.save_results()


if __name__ == "__main__":
    main()
