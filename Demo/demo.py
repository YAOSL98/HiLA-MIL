import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from sklearn.preprocessing import label_binarize

result_dir = "./checkpoints/Virchow"
class_names = ['Negative', 'ITC', 'Micro', 'Macro']
n_classes = len(class_names)
output_dir = "./Results/Virchow"
os.makedirs(output_dir, exist_ok=True)

def specificity_score_multiclass(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for i in range(n_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return specs

def safe_convert_to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.array(data)

def calculate_single_pt_metrics(pt_file_path):
    data = torch.load(pt_file_path, map_location='cpu')
    y_true = safe_convert_to_numpy(data['bag_labels']).astype(int)
    y_pred = safe_convert_to_numpy(data['bag_hat']).astype(int)
    y_score = safe_convert_to_numpy(data['bag_prob'])

    metrics = {
        "auc": roc_auc_score(label_binarize(y_true, classes=range(n_classes)), y_score, multi_class='ovr'),
        "acc": accuracy_score(y_true, y_pred),
        "pre": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "rec": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "spec": np.mean(specificity_score_multiclass(y_true, y_pred, n_classes)),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "y_true": y_true, "y_pred": y_pred, "y_score": y_score
    }

    for i in range(n_classes):
        y_true_cls = (y_true == i).astype(int)
        metrics[f"cls{i}_auc"] = roc_auc_score(y_true_cls, y_score[:, i]) if len(np.unique(y_true_cls)) > 1 else 0.5
        metrics[f"cls{i}_acc"] = accuracy_score(y_true_cls, (y_pred == i))
        metrics[f"cls{i}_pre"] = precision_score(y_true_cls, (y_pred == i), zero_division=0)
        metrics[f"cls{i}_rec"] = recall_score(y_true_cls, (y_pred == i), zero_division=0)
        metrics[f"cls{i}_spec"] = specificity_score_multiclass(y_true, y_pred, n_classes)[i]
        metrics[f"cls{i}_f1"] = f1_score(y_true_cls, (y_pred == i), zero_division=0)
    return metrics

print("Loading fold results...")
all_folds = []
for fold in range(10):
    path = os.path.join(result_dir, f"fold_{fold}_results.pt")
    if os.path.exists(path):
        print(f"Loaded: {path}")
        all_folds.append(calculate_single_pt_metrics(path))
    else:
        print(f"Missing: {path}")

if not all_folds:
    print("No fold results found! Check your file path.")
    exit()

results = {}
for k in all_folds[0].keys():
    if k not in ["y_true", "y_pred", "y_score"]:
        vals = [d[k] for d in all_folds]
        results[k] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1), "raw": vals}

results["all_y_true"] = np.concatenate([d["y_true"] for d in all_folds])
results["all_y_pred"] = np.concatenate([d["y_pred"] for d in all_folds])
results["all_y_score"] = np.concatenate([d["y_score"] for d in all_folds])

rows = []
for prefix, name in [("", "Overall"), ("cls0_", "Negative"), ("cls1_", "ITC"), ("cls2_", "Micro"), ("cls3_", "Macro")]:
    rows.append({"Method": name})
    row = {"Method": "Virchow+HiLA-MIL (Ours)"}
    for k, n in [("auc", "AUC"), ("acc", "Acc"), ("pre", "Pre"), ("rec", "Rec"), ("spec", "Spe"), ("f1", "F1")]:
        key = prefix + k
        if key in results:
            row[n] = f"{results[key]['mean']:.4f}±{results[key]['std']:.4f}"
    rows.append(row)
    rows.append({})

pd.DataFrame(rows).to_csv(os.path.join(output_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
print("Saved metrics.csv")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
ax_list = [
    (axes[0], "Overall", -1),
    (axes[1], "Negative", 0),
    (axes[2], "ITC", 1),
    (axes[3], "Micro", 2),
    (axes[4], "Macro", 3),
]
color = "#2E86AB"

for ax, title, cls_idx in ax_list:
    ax.set_title(title, fontsize=12)
    ax.plot([0,1],[0,1], "k--", lw=1)
    y_true = results["all_y_true"]
    y_score = results["all_y_score"]

    if cls_idx == -1:
        yb = label_binarize(y_true, classes=range(4))
        fpr, tpr, _ = roc_curve(yb.ravel(), y_score.ravel())
    else:
        yb = (y_true == cls_idx).astype(int)
        fpr, tpr, _ = roc_curve(yb, y_score[:, cls_idx])

    ax.plot(fpr, tpr, color=color, lw=3)
    ax.set_xlabel("1 - Specificity", fontsize=10)
    ax.set_ylabel("Sensitivity / Recall", fontsize=10)
    ax.grid(alpha=0.3)

ax_f1 = axes[5]
xs = class_names + ["Mean"]
f1s = [results[f"cls{i}_f1"]["mean"] for i in range(4)]
f1s.append(np.mean(f1s))
ax_f1.bar(xs, f1s, color=color)
ax_f1.set_title("F1-Score", fontsize=12)
ax_f1.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_f1.png"), dpi=300, bbox_inches='tight')
print("Saved roc_f1.png")

plt.figure(figsize=(6,5))
cm = confusion_matrix(results["all_y_true"], results["all_y_pred"])
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
print("Saved confusion_matrix.png")

print("All done! Results saved to:", output_dir)