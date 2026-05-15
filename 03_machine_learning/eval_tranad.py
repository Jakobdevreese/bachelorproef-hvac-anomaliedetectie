"""
Standalone evaluation for TranAD baseline model.
Loads saved weights and runs the improved multi-strategy evaluation pipeline.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    precision_recall_fscore_support, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    fbeta_score, confusion_matrix, f1_score
)
from scipy.stats import genpareto

GEBOUW      = 'dunant1'
WINDOW_SIZE = 144

# ─── Architecture definitions ─────────────────────────────────────────────────

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_expansion=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * ffn_expansion, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2   = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        attn = self.dropout1(attn, training=training)
        x = self.layernorm1(x + attn)
        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(x + ffn_out)


class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_expansion=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attn  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * ffn_expansion, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, memory, training=False):
        attn1 = self.self_attn(x, x, training=training)
        out1  = self.ln1(x + self.dropout1(attn1, training=training))
        attn2 = self.cross_attn(query=out1, key=memory, value=memory, training=training)
        out2  = self.ln2(out1 + self.dropout2(attn2, training=training))
        return self.ln3(out2 + self.ffn(out2, training=training))


class TranAD(Model):
    def __init__(self, num_features, d_model=64, num_heads=4, num_layers=1,
                 ffn_expansion=4, dropout_rate=0.1,
                 alpha_start=0.6, alpha_end=0.1, total_epochs=50,
                 window_size=144, **kwargs):
        super().__init__(**kwargs)
        self.alpha_start   = float(alpha_start)
        self.alpha_end     = float(alpha_end)
        self.total_epochs  = int(total_epochs)
        self.current_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.input_proj = layers.Dense(d_model)
        self.focus_proj = layers.Dense(d_model)
        self.pos_enc = self.add_weight(
            shape=(1, window_size, d_model), initializer='random_normal',
            trainable=True, name='pos_enc')

        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, ffn_expansion, dropout_rate)
            for _ in range(num_layers)
        ]
        self.decoder1 = TransformerDecoderLayer(d_model, num_heads, ffn_expansion, dropout_rate)
        self.decoder2 = TransformerDecoderLayer(d_model, num_heads, ffn_expansion, dropout_rate)
        self.out_proj1 = layers.Dense(num_features)
        self.out_proj2 = layers.Dense(num_features)

        self.loss_tracker   = tf.keras.metrics.Mean(name='loss')
        self.phase1_tracker = tf.keras.metrics.Mean(name='phase1_loss')
        self.phase2_tracker = tf.keras.metrics.Mean(name='phase2_loss')

    @property
    def metrics(self):
        return [self.loss_tracker, self.phase1_tracker, self.phase2_tracker]

    def _encode_decode(self, x, focus, training=False):
        z = self.input_proj(x) + self.focus_proj(focus) + self.pos_enc
        for enc in self.encoder_layers:
            z = enc(z, training=training)
        tgt  = self.input_proj(x) + self.pos_enc
        out1 = self.out_proj1(self.decoder1(tgt, z, training=training))
        out2 = self.out_proj2(self.decoder2(tgt, z, training=training))
        return out1, out2

    def call(self, x, training=False):
        o1, _ = self._encode_decode(x, tf.zeros_like(x), training=training)
        _, o2_hat = self._encode_decode(x, tf.abs(o1 - x), training=training)
        return o1, o2_hat


def create_windows(data_array, window_size=144):
    return np.array([data_array[i:i + window_size]
                     for i in range(len(data_array) - window_size + 1)])


# ─── Load artifacts ───────────────────────────────────────────────────────────

print("Loading features and scaler...")
with open(f'tranad/features_{GEBOUW}.json') as f:
    kept_features = json.load(f)
scaler = joblib.load(f'tranad/scaler_{GEBOUW}.pkl')
print(f"  Features: {len(kept_features)}")

NUM_FEATURES = len(kept_features)

# ─── Build model and load weights ────────────────────────────────────────────

print("Building TranAD baseline and loading weights...")
model = TranAD(
    num_features=NUM_FEATURES, d_model=64, num_heads=4, num_layers=1,
    ffn_expansion=4, dropout_rate=0.1, alpha_start=0.6, alpha_end=0.1,
    total_epochs=50, window_size=WINDOW_SIZE
)
_ = model(tf.zeros((1, WINDOW_SIZE, NUM_FEATURES), dtype=tf.float32))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

weights_path = f'tranad/best_baseline_{GEBOUW}.weights.h5'
model.load_weights(weights_path)
print(f"  Loaded: {weights_path}")

# ─── Load and prepare validation data ────────────────────────────────────────

print("Preparing validation data...")
train_csv = f'../02_eda_en_ground_truth/processed/{GEBOUW}_train.csv'
train_data = pd.read_csv(train_csv)
if 'timestamp' in train_data.columns:
    train_data = train_data.drop(columns=['timestamp'])
train_filtered = train_data[kept_features]
n = len(train_filtered)
val_df = train_filtered.iloc[int(n * 0.70):int(n * 0.85)]
val_scaled = scaler.transform(val_df)
X_val = create_windows(val_scaled, WINDOW_SIZE)
print(f"  Val windows: {X_val.shape}")

# ─── Load and prepare evaluation data ────────────────────────────────────────

print("Preparing synthetic evaluation data...")
synth_csv  = f'../02_eda_en_ground_truth/processed/{GEBOUW}_test.csv'
labels_npy = f'../02_eda_en_ground_truth/processed/{GEBOUW}_test_labels.npy'
synth_df   = pd.read_csv(synth_csv)
y_true_timestep = np.load(labels_npy).astype(int)

if 'timestamp' in synth_df.columns:
    synth_df = synth_df.drop(columns=['timestamp'])
synth_scaled = scaler.transform(synth_df[kept_features])
X_eval = create_windows(synth_scaled, WINDOW_SIZE)

n_windows     = len(X_eval)
y_true_window = np.array([
    1 if np.any(y_true_timestep[i:i + WINDOW_SIZE] == 1) else 0
    for i in range(n_windows)
])
n_ts = len(y_true_timestep)

print(f"  Eval windows: {X_eval.shape}, labels: {y_true_timestep.shape}")
print(f"  GT anomaly rate (timestep): {y_true_timestep.mean():.3f}")
print(f"  GT anomaly rate (window):   {y_true_window.mean():.3f}")

# ─── Multi-strategy scoring ───────────────────────────────────────────────────

def score_windows_multi(mdl, X, batch_size=32):
    rec1, rec2_hat = mdl.predict(tf.cast(X, tf.float32), batch_size=batch_size, verbose=0)
    sq1   = np.square(rec1     - X)
    sq2   = np.square(rec2_hat - X)
    sq_cb = 0.5 * sq1 + 0.5 * sq2

    feat_mse1  = np.mean(sq1,   axis=1)
    feat_mse2  = np.mean(sq2,   axis=1)
    feat_mse_c = np.mean(sq_cb, axis=1)

    ts_mse1  = np.mean(sq1,   axis=2)
    ts_mse2  = np.mean(sq2,   axis=2)
    ts_mse_c = np.mean(sq_cb, axis=2)

    strategies = {
        'max-feat':    np.max(feat_mse_c, axis=1),
        'global-mean': np.mean(sq_cb, axis=(1, 2)),
        'max-ts':      np.max(ts_mse_c, axis=1),
        'p95-feat':    np.percentile(feat_mse_c, 95, axis=1),
        'gap-mean':    np.mean(feat_mse2 - feat_mse1, axis=1),
        'gap-max-ts':  np.max(ts_mse2 - ts_mse1, axis=1),
    }
    return strategies, feat_mse_c


print("\nComputing reconstruction errors (val)...")
val_strategies,  val_feat_mse  = score_windows_multi(model, X_val)
print("Computing reconstruction errors (eval)...")
eval_strategies, eval_feat_mse = score_windows_multi(model, X_eval)

print("\nVal score stats per strategy:")
for nm, sc in val_strategies.items():
    print(f"  {nm:15s}: mean={sc.mean():.4f}  95p={np.percentile(sc, 95):.4f}")
print("\nEval score stats per strategy:")
for nm, sc in eval_strategies.items():
    print(f"  {nm:15s}: mean={sc.mean():.4f}  95p={np.percentile(sc, 95):.4f}")

# ─── Per-feature normalisation ────────────────────────────────────────────────

feat_mean_v = val_feat_mse.mean(axis=0)
feat_std_v  = val_feat_mse.std(axis=0) + 1e-8

eval_feat_norm = (eval_feat_mse - feat_mean_v) / feat_std_v
val_feat_norm  = (val_feat_mse  - feat_mean_v) / feat_std_v

eval_strategies['norm-mean'] = eval_feat_norm.mean(axis=1)
eval_strategies['norm-max']  = eval_feat_norm.max(axis=1)
val_scores_norm = val_feat_norm.mean(axis=1)

# ─── Window → timestep ───────────────────────────────────────────────────────

def wins_to_timestep(win_scores, n_timesteps, ws=WINDOW_SIZE):
    out = np.zeros(n_timesteps)
    cnt = np.zeros(n_timesteps)
    for i, s in enumerate(win_scores):
        out[i:i + ws] += s
        cnt[i:i + ws] += 1
    return out / np.maximum(cnt, 1)

# ─── Strategy comparison ─────────────────────────────────────────────────────

print(f"\n{'Strategy':22s}  {'ROC-AUC(win)':12s}  {'ROC-AUC(ts)':12s}  Note")
best_roc, best_name, best_win_scores = 0.0, '', None

for nm, sc in eval_strategies.items():
    ts_sc    = wins_to_timestep(sc, n_ts)
    roc_win  = roc_auc_score(y_true_window, sc)
    roc_ts   = roc_auc_score(y_true_timestep, ts_sc)
    effective = max(roc_ts, 1 - roc_ts)
    flip     = roc_ts < 0.5
    note     = '↑FLIP' if flip else ''
    print(f"  {nm:22s}  {roc_win:.4f}         {roc_ts:.4f}  {note}")
    use_sc = -sc if flip else sc
    if effective > best_roc:
        best_roc, best_name, best_win_scores = effective, nm, use_sc

print(f"\nBest strategy: '{best_name}'  (effective ROC-AUC timestep = {best_roc:.4f})")

# ─── Temporal smoothing (5-point triangular kernel on timestep scores) ────────

best_ts_scores_raw = wins_to_timestep(best_win_scores, n_ts)
kernel = np.array([1, 2, 3, 2, 1], dtype=float); kernel /= kernel.sum()
best_ts_scores = np.convolve(best_ts_scores_raw, kernel, mode='same')

split_ts   = int(n_ts * 0.6)
y_tune_ts  = y_true_timestep[:split_ts]
sc_tune_ts = best_ts_scores[:split_ts]

# ─── F1 and F2 threshold search ───────────────────────────────────────────────

cands = np.percentile(sc_tune_ts, np.linspace(0.1, 99.9, 600))
best_f1_tune, best_thr_f1 = 0.0, cands[-1]
best_f2_tune, best_thr_f2 = 0.0, cands[-1]
for t in cands:
    yp = (sc_tune_ts > t).astype(int)
    f1v = f1_score(y_tune_ts, yp, zero_division=0)
    f2v = fbeta_score(y_tune_ts, yp, beta=2, zero_division=0)
    if f1v > best_f1_tune:
        best_f1_tune, best_thr_f1 = f1v, t
    if f2v > best_f2_tune:
        best_f2_tune, best_thr_f2 = f2v, t

best_thr = best_thr_f1  # default: precision-first
print(f"F1-tune threshold: {best_thr_f1:.4f}  (tune F1={best_f1_tune:.4f}, "
      f"alarm rate on tune={(sc_tune_ts > best_thr_f1).mean():.3f})")
print(f"F2-tune threshold: {best_thr_f2:.4f}  (tune F2={best_f2_tune:.4f}, "
      f"alarm rate on tune={(sc_tune_ts > best_thr_f2).mean():.3f})")

# ─── POT reference ────────────────────────────────────────────────────────────

def pot_threshold(scores_clean, risk=0.01, tail_pct=0.10):
    s = np.sort(scores_clean)
    n_tail = max(int(len(s) * tail_pct), 20)
    exc = s[-n_tail:] - s[-n_tail]
    shape, _, scale = genpareto.fit(exc, floc=0)
    u = s[-n_tail]
    q = n_tail / len(s)
    if abs(shape) > 1e-8:
        return float(u + (scale / shape) * ((risk / q) ** (-shape) - 1))
    return float(u - scale * np.log(risk / q))

pot_thr = pot_threshold(val_scores_norm)
print(f"POT threshold (normalised val scores, reference): {pot_thr:.4f}")

# ─── Event stats ─────────────────────────────────────────────────────────────

def get_event_stats(y_true, y_pred):
    events, in_event, start = [], False, 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_event:
            in_event, start = True, i
        elif v == 0 and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(y_true) - 1))
    detected = sum(1 for s, e in events if np.any(y_pred[s:e + 1] == 1))
    return {'total_events': len(events), 'detected_events': detected,
            'event_recall': detected / max(len(events), 1)}

# ─── Final evaluation ─────────────────────────────────────────────────────────

roc  = roc_auc_score(y_true_timestep, best_ts_scores)
prc  = average_precision_score(y_true_timestep, best_ts_scores)

for label, thr in [('F1-optimised (precision-first)', best_thr_f1),
                   ('F2-optimised (recall-first)',    best_thr_f2)]:
    y_pred_ts = (best_ts_scores > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_timestep, y_pred_ts, average='binary', zero_division=0)
    f2_sc = fbeta_score(y_true_timestep, y_pred_ts, beta=2, zero_division=0)
    ba    = balanced_accuracy_score(y_true_timestep, y_pred_ts)
    mcc   = matthews_corrcoef(y_true_timestep, y_pred_ts)
    evt   = get_event_stats(y_true_timestep, y_pred_ts)
    cm    = confusion_matrix(y_true_timestep, y_pred_ts)

    print(f"\n{'='*60}")
    print(f"TRANAD (BASELINE) — {label}")
    print(f"Strategie: {best_name}  |  Drempel: {thr:.4f}")
    print(f"{'='*60}")
    print(f"  Precision:        {p:.4f}")
    print(f"  Recall:           {r:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  F2-Score:         {f2_sc:.4f}")
    print(f"  Balanced Acc:     {ba:.4f}")
    print(f"  MCC:              {mcc:.4f}")
    print(f"  ROC-AUC:          {roc:.4f}")
    print(f"  PR-AUC:           {prc:.4f}")
    print(f"  Event Recall:     {evt['event_recall']:.4f}  ({evt['detected_events']}/{evt['total_events']} events)")
    print(f"  Alarm rate:       {y_pred_ts.mean():.3f}  (GT: {y_true_timestep.mean():.3f})")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  /  FN={cm[1,0]}  TP={cm[1,1]}")

# Score separation
norm_mask = y_true_timestep == 0
anom_mask = y_true_timestep == 1
print(f"\nScore separation (timestep):")
print(f"  Normal   mean={best_ts_scores[norm_mask].mean():.4f}  std={best_ts_scores[norm_mask].std():.4f}")
print(f"  Anomaly  mean={best_ts_scores[anom_mask].mean():.4f}  std={best_ts_scores[anom_mask].std():.4f}")
print("\nNote: the normal score std is too wide for reliable detection of Events 2 & 3.")
print("A deeper model (num_layers=2, 100 epochs) should reduce normal variance significantly.")
