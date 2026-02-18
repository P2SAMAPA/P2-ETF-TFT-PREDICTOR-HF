"""
Model architectures: Transformer, Random Forest, XGBoost
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    """Adds positional information to input sequences for Transformer"""
    
    def __init__(self, max_seq_len=100, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
    
    def build(self, input_shape):
        seq_len = input_shape[1]
        d_model = input_shape[2]
        
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * 
                         -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        pos_encoding[:, 0::2] = np.sin(position * div_term)[:, :len(range(0, d_model, 2))]
        
        if d_model > 1:
            cos_values = np.cos(position * div_term)
            odd_positions = range(1, d_model, 2)
            pos_encoding[:, 1::2] = cos_values[:, :len(odd_positions)]
        
        self.pos_encoding = self.add_weight(
            name='positional_encoding',
            shape=(1, seq_len, d_model),
            initializer=tf.keras.initializers.Constant(pos_encoding),
            trainable=False
        )
        
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        return inputs + self.pos_encoding
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'max_seq_len': self.max_seq_len})
        return config


def directional_loss(y_true, y_pred):
    """Custom loss that penalizes incorrect direction predictions"""
    abs_error = tf.abs(y_true - y_pred)
    signs_match = tf.cast(tf.math.sign(y_true) == tf.math.sign(y_pred), tf.float32)
    penalty = tf.where(signs_match > 0.5, abs_error, abs_error * 2.0)
    return tf.reduce_mean(penalty)


def build_transformer_model(input_shape, num_outputs, num_heads=2, ff_dim=64, 
                            num_layers=1, dropout_rate=0.2):
    """Build a pure Transformer architecture"""
    inputs = Input(shape=input_shape)
    x = PositionalEncoding()(inputs)
    
    for _ in range(num_layers):
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[1] // num_heads,
            dropout=dropout_rate
        )(x, x)
        
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.01))(x)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(input_shape[1], kernel_regularizer=l2(0.01))(ff_output)
        
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_outputs)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def train_transformer(X_train, y_train, X_val, y_val, epochs=100):
    """Train Transformer model"""
    model = build_transformer_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_outputs=y_train.shape[1]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=directional_loss,
        metrics=['mae']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history


def train_ensemble(X_train, y_train, X_val, y_val):
    """Train Random Forest + XGBoost ensemble"""
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mlogloss'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return rf_model, xgb_model


def predict_ensemble(rf_model, xgb_model, X_test):
    """Make predictions with ensemble"""
    rf_probs = rf_model.predict_proba(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)
    
    ensemble_probs = (rf_probs + xgb_probs) / 2
    preds = np.argmax(ensemble_probs, axis=1)
    
    return preds
