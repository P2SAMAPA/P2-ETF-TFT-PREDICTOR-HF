"""
Temporal Fusion Transformer (TFT-inspired) in Keras/TensorFlow.

Components:
  - Gated Residual Network (GRN)
  - Variable Selection Network (VSN)
  - Multi-head Self-Attention
  - Classification head (cross-entropy, softmax)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    Multiply, Add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


def gated_residual_network(x, units, dropout_rate=0.1, time_dist=True):
    residual = x
    def td(layer): return tf.keras.layers.TimeDistributed(layer) if time_dist else layer

    h = td(Dense(units, kernel_regularizer=l2(1e-4)))(x)
    h = tf.keras.layers.ELU()(h)
    h = Dropout(dropout_rate)(h)
    h = td(Dense(units, kernel_regularizer=l2(1e-4)))(h)
    gate = td(Dense(units, activation='sigmoid'))(x)
    h = Multiply()([h, gate])
    if residual.shape[-1] != units:
        residual = td(Dense(units, use_bias=False))(residual)
    out = Add()([h, residual])
    return LayerNormalization(epsilon=1e-6)(out)


def variable_selection_network(inputs, num_features, units, dropout_rate=0.1):
    feature_outputs = []
    for i in range(num_features):
        feat = tf.keras.layers.Lambda(
            lambda t, idx=i: tf.expand_dims(t[:, :, idx], axis=-1)
        )(inputs)
        feature_outputs.append(
            gated_residual_network(feat, units, dropout_rate, time_dist=True)
        )
    stacked = tf.stack(feature_outputs, axis=2)
    w = gated_residual_network(inputs, num_features, dropout_rate, time_dist=True)
    w = tf.keras.layers.TimeDistributed(Dense(num_features, activation='softmax'))(w)
    w_exp = tf.expand_dims(w, axis=-1)
    return tf.reduce_sum(stacked * w_exp, axis=2)


def build_tft_model(seq_len, num_features, num_outputs,
                    d_model=64, num_heads=4, num_layers=2,
                    dropout_rate=0.15, ff_mult=2):
    d_model = (d_model // num_heads) * num_heads
    inputs = Input(shape=(seq_len, num_features), name='seq_input')

    x = variable_selection_network(inputs, num_features, d_model, dropout_rate)
    x = gated_residual_network(x, d_model, dropout_rate, time_dist=True)

    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = tf.keras.layers.Embedding(
        input_dim=seq_len, output_dim=d_model, name='pos_emb'
    )(positions)
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = x + pos_emb

    for i in range(num_layers):
        attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            dropout=dropout_rate, name=f'attn_{i}'
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn)
        ff = gated_residual_network(x, d_model * ff_mult, dropout_rate, time_dist=True)
        ff = gated_residual_network(ff, d_model, dropout_rate, time_dist=True)
        x = LayerNormalization(epsilon=1e-6)(x + ff)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = gated_residual_network(x, d_model, dropout_rate, time_dist=False)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_outputs, activation='softmax', name='etf_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='TFT_ETF_Classifier')


def train_tft(X_train, y_train, X_val, y_val, epochs=200,
              d_model=64, num_heads=4, num_layers=2, dropout_rate=0.15):
    """
    Train TFT classifier.
    y_train/y_val: integer class labels (0-4, argmax of 5-day fwd returns)
    Loss: sparse_categorical_crossentropy  (correct for classification)
    LR  : cosine decay with warm restarts
    """
    seq_len      = X_train.shape[1]
    num_features = X_train.shape[2]
    num_outputs  = int(np.max(y_train)) + 1

    model = build_tft_model(
        seq_len=seq_len, num_features=num_features, num_outputs=num_outputs,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-4, first_decay_steps=500,
        t_mul=2.0, m_mul=0.9, alpha=1e-5
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64,
        callbacks=callbacks, verbose=1, shuffle=True
    )
    return model, history


def predict_tft(model, X_test):
    """
    Run inference with the TFT model.
    Returns softmax probability array of shape (N, num_classes).
    """
    return model.predict(X_test, verbose=0)
