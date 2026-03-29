# ONLY showing the corrected load_global_model() section
# (rest of your file remains EXACTLY the same)

def load_global_model(option, token):
    import tensorflow as tf, pickle
    from models import build_binary_tft

    # ✅ FIXED PATH
    meta_path = download_file_from_hf_dataset(f"option_{option}/global_model/meta.json", token)
    with open(meta_path) as f:
        meta = json.load(f)

    lookback = meta['lookback']
    num_features = meta.get('num_features', len(meta['input_features']))
    target_etfs = meta['target_etfs']

    models = []
    for etf in target_etfs:
        try:
            # ✅ FIXED PATH
            w_path = download_file_from_hf_dataset(
                f"option_{option}/global_model/{etf}.weights.h5", token
            )
            model = build_binary_tft(seq_len=lookback, num_features=num_features)
            model.load_weights(w_path)
        except:
            # ✅ FIXED PATH
            full_path = download_file_from_hf_dataset(
                f"option_{option}/global_model/{etf}.h5", token
            )
            model = tf.keras.models.load_model(full_path)

        models.append(model)

    # ✅ FIXED PATH
    scaler_path = download_file_from_hf_dataset(
        f"option_{option}/global_model/scaler.pkl", token
    )
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return models, scaler, meta
