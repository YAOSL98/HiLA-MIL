import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from modules import eval_model

import numpy as np
def load_model(mode_path, device):
    model = eval_model.SAttention()
    # print(model)
    param = torch.load(mode_path)
    # new_param = {k[6:]: v for k, v in param.items()}
    new_state_dict = {}
    for key, value in param['model'].items():
        new_key = key.replace('online_encoder.', '')
        new_key = new_key.replace('attention.attention.', 'attention.')
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model

def load_test_list(csv_path):
    df = pd.read_csv(csv_path)
    test_names = df['patient'].tolist()
    test_labels = df['label'].tolist()
    return [{'name': name.strip(), 'label': int(label)} for name, label in zip(test_names, test_labels)]

def load_feature_fn(name):
    return torch.load(f"/public/home/jiaqi2/project/datasets_all/{name}.pt")

def compute_per_model_metrics(y_true, y_prob, n_classes):
    y_pred = np.argmax(y_prob, axis=1)
    overall_accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(n_classes)))
    try:
        # aucs = roc_auc_score(y_true, y_prob, multi_class='ovr', average=None)
        aucs = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovo')
    except ValueError:
        aus = [np.nan] * n_classes

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specificity = []
    for i in range(n_classes):
        TP = conf_mat[i, i]
        FN = conf_mat[i, :].sum() - TP
        FP = conf_mat[:, i].sum() - TP
        TN = conf_mat.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity.append(spec)

    return {
        'accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': aucs,
        'specificity': specificity
    }


# 路径设置
csv_path = './seed/seed1/fold1_test.csv'
model_dir = './checkpoints/transmil/'  # 模型文件夹
output_csv = 'model_metrics.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载测试列表
test_list = load_test_list(csv_path)

# 初始化输出 CSV，写入表头（如果文件已存在可以先删除）
if os.path.exists(output_csv):
    os.remove(output_csv)

# 遍历模型
for model_file in os.listdir(model_dir):
    if not model_file.endswith('.pt'):
        continue

    model_path = os.path.join(model_dir, model_file)
    model = load_model(model_path, device)

    all_probs = []
    all_labels = []

    for sample in test_list:
        feat = load_feature_fn(sample['name']).to(device)
        label = sample['label']
        with torch.no_grad():
            prob = model(feat)  # 输出 logits
            prob = torch.softmax(prob, dim=1).squeeze(0).cpu().numpy()
        all_probs.append(prob)
        all_labels.append(label)

    y_true = all_labels
    y_prob = all_probs
    metrics = compute_per_model_metrics(y_true, y_prob, n_classes=4)

    row = {
        'model': model_file,
        # 'AUC': sum(metrics['auc']) / len(metrics['auc']),
        'AUC' : metrics['auc'],
        'ACC': metrics['accuracy'],
        'Precision': sum(metrics['precision']) / len(metrics['precision']),
        'Recall': sum(metrics['recall']) / len(metrics['recall']),
        'F1': sum(metrics['f1']) / len(metrics['f1']),
        'Specificity': sum(metrics['specificity']) / len(metrics['specificity']),
    }

    # 实时保存到 CSV（追加）
    pd.DataFrame([row]).to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))

    # 打印当前模型结果
    print(f"[{model_file}] ->", row)

print(f"\n✅ All results saved to {output_csv}")
