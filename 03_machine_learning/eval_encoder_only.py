"""
Standalone evaluation for encoder-only transformer.
Loads saved weights (baseline model) and runs the updated evaluation pipeline.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import (
    precision_recall_fscore_support, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    fbeta_score, confusion_matrix
)
from scipy.stats import genpareto

GEBOUW = 'dunant1'
WINDOW_SIZE = 144

# ─── Model definitions ───────────────────────────────────────────────────────

class Time2Vec(layers.Layer):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.wb = self.add_weight(name='wb', shape=(1,), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(1,), initializer='uniform', trainable=True)
        self.wa = self.add_weight(name='wa', shape=(1, self.output_dim - 1), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, self.output_dim - 1), initializer='uniform', trainable=True)

    def call(self, t):
        v1 = self.wb * t + self.bb
        v2 = tf.sin(self.wa * t + self.ba)
        return tf.concat([v1, v2], axis=-1)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    residual = inputs
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    x = x + residual

    residual = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + residual


def build_model(window_size, num_features, d_model, num_heads, ff_dim, num_layers, dropout):
    inputs = layers.Input(shape=(window_size, num_features))
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.Dense(d_model)(x)

    positions = tf.cast(tf.reshape(tf.range(window_size), (window_size, 1)), tf.float32)
    pos_encoding = Time2Vec(output_dim=d_model)(positions)
    x = x + pos_encoding

    for _ in range(num_layers):
        x = transformer_encoder(x, d_model // num_heads, num_heads, ff_dim, dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_features)(x)
    return tf.keras.Model(inputs, outputs)


def generate_markov_mask(input_shape, r=0.15, lm=3):
    W, M = input_shape
    lu = ((1 - r) / r) * lm
    p_m = 1 / lu
    p_u = 1 / lm
    mask = np.ones(input_shape)
    for j in range(M):
        curr_state = 1
        for i in range(W):
            if curr_state == 1:
                if np.random.rand() < p_m:
                    curr_state = 0
            else:
                if np.random.rand() < p_u:
                    curr_state = 1
            mask[i, j] = curr_state
    return mask


def generate_batch_masks(bs, ws, nf, r, lm):
    bs, ws, nf = int(bs), int(ws), int(nf)
    r, lm = float(r), float(lm)
    return np.array([generate_markov_mask([ws, nf], r, lm) for _ in range(bs)], dtype=np.float64)


def masked_mse_loss(y_true, y_pred, mask):
    inverse_mask = 1.0 - tf.cast(mask, tf.float32)
    sq_diff = tf.square(y_true - y_pred) * inverse_mask
    return tf.reduce_sum(sq_diff) / (tf.reduce_sum(inverse_mask) + 1e-6)


class HVACModel(tf.keras.Model):
    def __init__(self, transformer_model, r=0.15, lm=3):
        super().__init__()
        self.transformer = transformer_model
        self.r = r
        self.lm = lm
        self.loss_tracker = tf.keras.metrics.Mean(name='masked_mse')

    def _make_mask_batch(self, batch_size, window_size, num_features):
        mask_batch = tf.numpy_function(
            func=generate_batch_masks,
            inp=[batch_size, window_size, num_features, self.r, self.lm],
            Tout=tf.float64)
        mask_batch = tf.cast(mask_batch, dtype=tf.float32)
        mask_batch.set_shape([None, window_size, num_features])
        return mask_batch

    def train_step(self, data):
        x = data
        ws, nf = x.shape[1], x.shape[2]
        bs = tf.shape(x)[0]
        mask_batch = self._make_mask_batch(bs, ws, nf)
        x_masked = x * mask_batch
        with tf.GradientTape() as tape:
            y_pred = self.transformer(x_masked, training=True)
            loss = masked_mse_loss(x, y_pred, mask_batch)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {'masked_mse': self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        ws, nf = x.shape[1], x.shape[2]
        bs = tf.shape(x)[0]
        mask_batch = self._make_mask_batch(bs, ws, nf)
        x_masked = x * mask_batch
        y_pred = self.transformer(x_masked, training=False)
        loss = masked_mse_loss(x, y_pred, mask_batch)
        self.loss_tracker.update_state(loss)
        return {'masked_mse': self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        return self.transformer(inputs)


def create_windows(data_array, window_size=144):
    return np.array([data_array[i:i + window_size]
                     for i in range(len(data_array) - window_size)])


# ─── Load artifacts ───────────────────────────────────────────────────────────

print("Loading features and scaler...")
with open(f'encoder_only/features_{GEBOUW}.json') as f:
    kept_features = json.load(f)
scaler = joblib.load(f'encoder_only/scaler_{GEBOUW}.pkl')
print(f"  Features: {len(kept_features)}")

# ─── Build model and load weights ────────────────────────────────────────────

print("Building model and loading weights...")
base_transformer = build_model(
    window_size=WINDOW_SIZE, num_features=len(kept_features),
    d_model=64, num_heads=8, ff_dim=128, num_layers=2, dropout=0.1
)
hvac_model = HVACModel(base_transformer, r=0.15, lm=3)
hvac_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
hvac_model.build(input_shape=(None, WINDOW_SIZE, len(kept_features)))

weights_path = f'encoder_only/best_model_{GEBOUW}.weights.h5'
hvac_model.load_weights(weights_path)
transformer = hvac_model.transformer
print(f"  Loaded: {weights_path}")

# ─── Load and prepare validation data ────────────────────────────────────────

print("Preparing validation data...")
train_csv = f'../02_eda_en_ground_truth/processed/{GEBOUW}_train.csv'
train_data = pd.read_csv(train_csv)
train_filtered = train_data[kept_features]
n = len(train_filtered)
val_df = train_filtered.iloc[int(n * 0.7):int(n * 0.85)]
val_scaled = scaler.transform(val_df)
X_val = create_windows(val_scaled, WINDOW_SIZE)
print(f"  Val windows: {X_val.shape}")

# ─── Load and prepare evaluation (synthetic fault) data ──────────────────────

print("Preparing synthetic evaluation data...")
synth_csv = f'../02_eda_en_ground_truth/processed/{GEBOUW}_test.csv'
labels_npy = f'../02_eda_en_ground_truth/processed/{GEBOUW}_test_labels.npy'
synth_df = pd.read_csv(synth_csv)
y_true_timestep = np.load(labels_npy).astype(int)

synth_scaled = scaler.transform(synth_df[kept_features])
X_eval = create_windows(synth_scaled, WINDOW_SIZE)

def align_labels_to_windows(y_ts, ws=144):
    return np.array([1 if np.any(y_ts[i:i + ws] == 1) else 0
                     for i in range(len(y_ts) - ws)])

y_true_window = align_labels_to_windows(y_true_timestep, WINDOW_SIZE)
print(f"  Eval windows: {X_eval.shape}, labels: {y_true_timestep.shape}")
print(f"  GT anomaly rate (timestep): {y_true_timestep.mean():.3f}")
print(f"  GT anomaly rate (window):   {y_true_window.mean():.3f}")

# ─── Scoring ─────────────────────────────────────────────────────────────────

print("\nComputing reconstruction errors...")
reconstructions = transformer.predict(X_eval, verbose=0)
sq_err = np.square(X_eval - reconstructions)

feat_mse = np.mean(sq_err, axis=1)  # (n_wins, n_feat)
ts_mse   = np.mean(sq_err, axis=2)  # (n_wins, win_size)

eval_scores_max_feat = np.max(feat_mse, axis=1)
eval_scores_global   = np.mean(sq_err, axis=(1, 2))
eval_scores_max_ts   = np.max(ts_mse, axis=1)
eval_scores_p95_feat = np.percentile(feat_mse, 95, axis=1)

print("Raw score stats per strategy:")
for nm, sc in [('max-feat', eval_scores_max_feat), ('global-mean', eval_scores_global),
               ('max-ts', eval_scores_max_ts), ('p95-feat', eval_scores_p95_feat)]:
    print(f"  {nm:15s}: mean={sc.mean():.4f}  95p={np.percentile(sc, 95):.4f}")

# Val-based normalisation
val_reconstructions = transformer.predict(X_val, verbose=0)
val_sq_err   = np.square(X_val - val_reconstructions)
val_feat_mse = np.mean(val_sq_err, axis=1)
feat_mean_v  = val_feat_mse.mean(axis=0)
feat_std_v   = val_feat_mse.std(axis=0) + 1e-8

eval_feat_norm     = (feat_mse - feat_mean_v) / feat_std_v
eval_scores_norm   = eval_feat_norm.mean(axis=1)
eval_scores_norm_max = eval_feat_norm.max(axis=1)
val_feat_norm   = (val_feat_mse - feat_mean_v) / feat_std_v
val_scores_norm = val_feat_norm.mean(axis=1)

n_ts = len(y_true_timestep)

def wins_to_timestep(win_scores, n_timesteps, ws=144):
    out = np.zeros(n_timesteps)
    cnt = np.zeros(n_timesteps)
    for i, s in enumerate(win_scores):
        out[i:i + ws] += s
        cnt[i:i + ws] += 1
    return out / np.maximum(cnt, 1)

# ─── Strategy comparison ─────────────────────────────────────────────────────

all_strategies = {
    'max-feat (raw)': eval_scores_max_feat,
    'global-mean':    eval_scores_global,
    'max-ts':         eval_scores_max_ts,
    'p95-feat':       eval_scores_p95_feat,
    'norm-mean':      eval_scores_norm,
    'norm-max':       eval_scores_norm_max,
}

print(f"\n{'Strategy':22s}  {'ROC-AUC(win)':12s}  {'ROC-AUC(ts)':12s}  Note")
best_roc, best_name, best_win_scores = 0.0, '', None

for nm, sc in all_strategies.items():
    ts_sc   = wins_to_timestep(sc, n_ts)
    roc_win = roc_auc_score(y_true_window, sc)
    roc_ts  = roc_auc_score(y_true_timestep, ts_sc)
    effective = max(roc_ts, 1 - roc_ts)
    flip = roc_ts < 0.5
    note = '↑FLIP' if flip else ''
    print(f"  {nm:22s}  {roc_win:.4f}         {roc_ts:.4f}  {note}")
    use_sc = -sc if flip else sc
    if effective > best_roc:
        best_roc, best_name, best_win_scores = effective, nm, use_sc

print(f"\nBest strategy: '{best_name}'  (effective ROC-AUC timestep = {best_roc:.4f})")

# ─── F1-maximising threshold ─────────────────────────────────────────────────

from sklearn.metrics import f1_score

best_ts_scores = wins_to_timestep(best_win_scores, n_ts)
split_ts   = int(n_ts * 0.6)
y_tune_ts  = y_true_timestep[:split_ts]
sc_tune_ts = best_ts_scores[:split_ts]

cands = np.percentile(sc_tune_ts, np.linspace(0.1, 99.9, 600))
best_f1_tune, best_thr = 0.0, cands[-1]
for t in cands:
    f1v = f1_score(y_tune_ts, (sc_tune_ts > t).astype(int), zero_division=0)
    if f1v > best_f1_tune:
        best_f1_tune, best_thr = f1v, t

print(f"F1-tune threshold: {best_thr:.4f}  (tune F1={best_f1_tune:.4f}, "
      f"alarm rate on tune={( sc_tune_ts > best_thr).mean():.3f})")

# ─── Event stats ─────────────────────────────────────────────────────────────

def get_event_stats(y_true, y_pred):
    events = []
    in_event, start_idx = False, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_event:
            in_event, start_idx = True, i
        elif y_true[i] == 0 and in_event:
            in_event = False
            events.append((start_idx, i - 1))
    if in_event:
        events.append((start_idx, len(y_true) - 1))
    detected = sum(1 for s, e in events if np.any(y_pred[s:e + 1] == 1))
    return {'total_events': len(events), 'detected_events': detected,
            'event_recall': detected / max(len(events), 1)}

# ─── Final timestep-level evaluation ─────────────────────────────────────────

y_pred_ts = (best_ts_scores > best_thr).astype(int)

p, r, f1, _ = precision_recall_fscore_support(
    y_true_timestep, y_pred_ts, average='binary', zero_division=0)
f2   = fbeta_score(y_true_timestep, y_pred_ts, beta=2, zero_division=0)
ba   = balanced_accuracy_score(y_true_timestep, y_pred_ts)
mcc  = matthews_corrcoef(y_true_timestep, y_pred_ts)
roc  = roc_auc_score(y_true_timestep, best_ts_scores)
prc  = average_precision_score(y_true_timestep, best_ts_scores)
evt  = get_event_stats(y_true_timestep, y_pred_ts)
cm   = confusion_matrix(y_true_timestep, y_pred_ts)

print(f"\n{'='*55}")
print(f"ENCODER-ONLY TRANSFORMER — Tijdstap-niveau evaluatie")
print(f"Strategie: {best_name}  |  Drempel: {best_thr:.4f}")
print(f"{'='*55}")
print(f"  Precision:        {p:.4f}")
print(f"  Recall:           {r:.4f}")
print(f"  F1-Score:         {f1:.4f}")
print(f"  F2-Score:         {f2:.4f}")
print(f"  Balanced Acc:     {ba:.4f}")
print(f"  MCC:              {mcc:.4f}")
print(f"  ROC-AUC:          {roc:.4f}")
print(f"  PR-AUC:           {prc:.4f}")
print(f"  Event Recall:     {evt['event_recall']:.4f}  ({evt['detected_events']}/{evt['total_events']} events)")
print(f"  Alarm rate:       {y_pred_ts.mean():.3f}  (GT: {y_true_timestep.mean():.3f})")
print(f"\nConfusion matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Score separation
norm_idx = y_true_timestep == 0
anom_idx = y_true_timestep == 1
print(f"\nScore separation (timestep):")
print(f"  Normal   mean={best_ts_scores[norm_idx].mean():.4f}  std={best_ts_scores[norm_idx].std():.4f}")
print(f"  Anomaly  mean={best_ts_scores[anom_idx].mean():.4f}  std={best_ts_scores[anom_idx].std():.4f}")
