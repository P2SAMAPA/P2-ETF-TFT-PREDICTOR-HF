"""
TFT-inspired ETF classifier in Keras.
Simplified to be fully compatible with Keras Functional API.

Key components:
  - Gated Residual Network (GRN)
  - Feature projection + gating (replaces VSN for Functional API compatibility)
  - Sinusoidal positional encoding
  - Stacked multi-head self-attention (correct key_dim)
  - Sparse categorical cross-entropy loss
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

# ── Fixed random seed — ensures reproducible results across runs ─────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
from tensorflow.keras.regularizers import l2


def grn_block(x, units, dropout_rate=0.15):
    """
    Gated Residual Network block.
    Works on both 2D (B, units) and 3D (B, T, units) tensors.
    """
    # Project residual if needed
    if x.shape[-1] != units:
        residual = Dense(units, use_bias=False)(x)
    else:
        residual = x

    # Hidden transformation
    h = Dense(units)(x)
    h = Activation('elu')(h)
    h = Dense(units)(h)
    h = Dropout(dropout_rate)(h)

    # Gate
    gate = Dense(units, activation='sigmoid')(x)
    h = Multiply()([h, gate])

    # Add & Norm
    out = Add()([residual, h])
    out = LayerNormalization(epsilon=1e-6)(out)
    return out


def build_tft_model(seq_len, num_features, num_classes,
                    units=64, num_heads=4, num_attn_layers=2,
                    dropout_rate=0.15):
    """
    TFT-inspired classifier compatible with Keras Functional API.

    Architecture:
      Input → Feature Projection → GRN → Positional Encoding
      → N × (Multi-Head Attention + GRN) → GlobalAvgPool
      → GRN → Dense(softmax)
    """
    inputs = Input(shape=(seq_len, num_features), name='input')

    # ── 1. Feature projection + gating (lightweight VSN replacement) ──────
    # Projects all features to 'units' dim with a learned gating mechanism
    proj  = Dense(units)(inputs)                          # (B, T, units)
    gate  = Dense(units, activation='sigmoid')(inputs)    # (B, T, units)
    x     = Multiply()([proj, gate])                      # (B, T, units)
    x     = LayerNormalization(epsilon=1e-6)(x)

    # ── 2. Local temporal convolution (captures short-range patterns) ────
    x = Conv1D(units, kernel_size=3, padding='causal',
               activation='relu')(x)                      # (B, T, units)

    # ── 3. GRN on temporal features ──────────────────────────────────────
    x = grn_block(x, units, dropout_rate)

    # ── 4. Sinusoidal positional encoding ────────────────────────────────
    positions = np.arange(seq_len).reshape(-1, 1).astype(np.float32)
    dims      = np.arange(0, units, 2).astype(np.float32)
    angles    = positions / np.power(10000.0, dims / units)
    sin_enc   = np.sin(angles)
    cos_enc   = np.cos(angles)
    if units % 2 == 0:
        pos_enc = np.concatenate([sin_enc, cos_enc], axis=-1)
    else:
        pos_enc = np.concatenate([sin_enc, cos_enc[:, :-1]], axis=-1)
    pos_enc = pos_enc[np.newaxis, :, :]                   # (1, T, units)

    x = x + pos_enc.astype(np.float32)                   # broadcast add

    # ── 5. Stacked multi-head self-attention ─────────────────────────────
    key_dim = max(1, units // num_heads)
    for i in range(num_attn_layers):
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name=f'attn_{i}'
        )(x, x)
        attn = Dropout(dropout_rate)(attn)
        x    = LayerNormalization(epsilon=1e-6, name=f'attn_norm_{i}')(x + attn)
        x    = grn_block(x, units, dropout_rate)

    # ── 6. Aggregate temporal dimension ──────────────────────────────────
    x = GlobalAveragePooling1D()(x)                       # (B, units)

    # ── 7. Classification head ────────────────────────────────────────────
    x = grn_block(x, units // 2, dropout_rate)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax',
                    name='class_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='TFT_Classifier')


def train_tft(X_train, y_train, X_val, y_val, epochs=200):
    """
    Train the TFT classifier.
    y_train/y_val: integer class labels (argmax of 5-day fwd returns).
    """
    # Re-apply seed immediately before model build for full reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    seq_len      = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes  = len(np.unique(y_train))

    model = build_tft_model(
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        units=64,
        num_heads=4,
        num_attn_layers=2,
        dropout_rate=0.15
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=0
    )

    return model, history


def predict_tft(model, X_test):
    """Returns softmax probability array shape (N, num_classes)."""
    return model.predict(X_test, verbose=0)
