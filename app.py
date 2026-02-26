"""
5 Independent Binary TFT Classifiers — one per ETF.
Each model answers: "Will this ETF beat the risk-free rate over the next 5 days?"
At inference time, ETFs are ranked by their YES probability.
This avoids the regime-lock problem of 5-class softmax.
"""

import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    Multiply, Add, Activation, Conv1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Fixed random seed ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def grn_block(x, units, dropout_rate=0.15):
    """Gated Residual Network — works on 2D and 3D tensors."""
    if x.shape[-1] != units:
        residual = Dense(units, use_bias=False)(x)
    else:
        residual = x
    h    = Dense(units)(x)
    h    = Activation('elu')(h)
    h    = Dense(units)(h)
    h    = Dropout(dropout_rate)(h)
    gate = Dense(units, activation='sigmoid')(x)
    h    = Multiply()([h, gate])
    out  = Add()([residual, h])
    out  = LayerNormalization(epsilon=1e-6)(out)
    return out


def build_binary_tft(seq_len, num_features, units=64, num_heads=4,
                     num_attn_layers=2, dropout_rate=0.15):
    """
    Binary TFT classifier: outputs P(ETF beats cash over next 5 days).
    Output: single sigmoid neuron → probability in [0, 1].
    """
    inputs = Input(shape=(seq_len, num_features), name='input')

    # Feature gating
    proj  = Dense(units)(inputs)
    gate  = Dense(units, activation='sigmoid')(inputs)
    x     = Multiply()([proj, gate])
    x     = LayerNormalization(epsilon=1e-6)(x)

    # Local temporal patterns
    x = Conv1D(units, kernel_size=3, padding='causal', activation='relu')(x)

    # GRN
    x = grn_block(x, units, dropout_rate)

    # Positional encoding
    positions = np.arange(seq_len).reshape(-1, 1).astype(np.float32)
    dims      = np.arange(0, units, 2).astype(np.float32)
    angles    = positions / np.power(10000.0, dims / units)
    sin_enc   = np.sin(angles)
    cos_enc   = np.cos(angles)
    pos_enc   = np.concatenate([sin_enc, cos_enc if units % 2 == 0
                                 else cos_enc[:, :-1]], axis=-1)
    x = x + pos_enc[np.newaxis].astype(np.float32)

    # Stacked attention
    key_dim = max(1, units // num_heads)
    for i in range(num_attn_layers):
        attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=dropout_rate, name=f'attn_{i}'
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout_rate)(attn))
        x = grn_block(x, units, dropout_rate)

    x = GlobalAveragePooling1D()(x)
    x = grn_block(x, units // 2, dropout_rate, )
    x = Dropout(dropout_rate)(x)

    # Binary output
    output = Dense(1, activation='sigmoid', name='beat_cash_prob')(x)

    return Model(inputs=inputs, outputs=output, name='BinaryTFT')


def grn_block(x, units, dropout_rate=0.15):
    """Gated Residual Network."""
    if x.shape[-1] != units:
        residual = Dense(units, use_bias=False)(x)
    else:
        residual = x
    h    = Dense(units)(x)
    h    = Activation('elu')(h)
    h    = Dense(units)(h)
    h    = Dropout(dropout_rate)(h)
    gate = Dense(units, activation='sigmoid')(x)
    h    = Multiply()([h, gate])
    out  = Add()([residual, h])
    out  = LayerNormalization(epsilon=1e-6)(out)
    return out


def train_binary_tft(X_train, y_train, X_val, y_val, etf_name="ETF", epochs=150):
    """
    Train one binary TFT for a single ETF.
    y_train/y_val: 1D array of 0/1 (1 = ETF beat cash over next 5 days).
    """
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    model = build_binary_tft(
        seq_len=X_train.shape[1],
        num_features=X_train.shape[2]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-5, verbose=0)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64,
        callbacks=callbacks, verbose=0
    )
    return model, history


def train_all_binary_tfts(X_train, y_train_matrix, X_val, y_val_matrix,
                           etf_names, epochs=150):
    """
    Train one binary TFT per ETF.
    y_train_matrix: shape (N, n_etfs) — each column is 0/1 for one ETF.
    Returns list of trained models.
    """
    models = []
    histories = []
    for j, name in enumerate(etf_names):
        m, h = train_binary_tft(
            X_train, y_train_matrix[:, j],
            X_val,   y_val_matrix[:, j],
            etf_name=name, epochs=epochs
        )
        models.append(m)
        histories.append(h)
    return models, histories


def predict_binary_tfts(models, X_test):
    """
    Run inference for all binary models.
    Returns (N, n_etfs) array of P(beat cash) for each ETF.
    """
    probs = np.hstack([m.predict(X_test, verbose=0) for m in models])
    return probs  # shape (N, n_etfs)
