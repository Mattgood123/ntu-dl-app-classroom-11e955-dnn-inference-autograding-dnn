import numpy as np
import json
import os

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward Pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)


# === Train and convert to JSON + NPZ ===
if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.datasets import fashion_mnist
    import tensorflow as tf

    print("Step 1: Training model...")
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)
    model.save("fashion_model.h5")
    print("Model saved as fashion_model.h5")

    print("Step 2: Converting model to JSON and NPZ...")
    model_json = []

    weights_np = {}
    for idx, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        layer_name = f"layer_{idx}"
        if layer_type == "Flatten":
            model_json.append({
                "name": layer_name,
                "type": "Flatten",
                "config": {},
                "weights": []
            })
        elif layer_type == "Dense":
            w, b = layer.get_weights()
            wname = f"W{idx}"
            bname = f"b{idx}"
            weights_np[wname] = w
            weights_np[bname] = b
            model_json.append({
                "name": layer_name,
                "type": "Dense",
                "config": {"activation": layer.activation.__name__},
                "weights": [wname, bname]
            })

    os.makedirs("model", exist_ok=True)
    with open("model/fashion_mnist.json", "w") as f:
        json.dump(model_json, f, indent=2)
    np.savez("model/fashion_mnist.npz", **weights_np)
    print("Conversion complete: model/fashion_mnist.json + .npz saved.")
