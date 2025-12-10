# train.py
import os
import argparse
import numpy as np
import tensorflow as tf

# Use non-GUI backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import build_rnn_model, GRUCell, MGUCell
from utils import prepare_notmnist_dataset

# -----------------------------
# Train one trial
# -----------------------------
def train_one_trial(cell_class, trial_id, save_root, units=128, num_layers=1,
                    epochs=20, batch_size=128, lr=0.001, dropout=0.0, seed=1234):
    
    # Set random seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Prepare dataset
    ds_train, ds_val, (seq_len, input_dim) = prepare_notmnist_dataset(
        data_dir='notmnist_data', img_size=(28,28), batch_size=batch_size, seed=seed+trial_id)

    # Build model
    model = build_rnn_model(cell_class, units=units, num_layers=num_layers,
                            sequence_length=seq_len, input_dim=input_dim, num_classes=10, dropout=dropout)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    hist = model.fit(ds_train, validation_data=ds_val, epochs=epochs)

    # Save model
    model_dir = os.path.join(save_root, f"{cell_class.__name__}_units{units}_layers{num_layers}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{cell_class.__name__}_trial{trial_id+1}.keras")
    model.save(model_path)

    # Plot training vs validation
    plt.figure()
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.title(f'{cell_class.__name__} Trial {trial_id+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(model_dir, f"{cell_class.__name__}_trial{trial_id+1}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    val_acc = hist.history['val_accuracy'][-1]
    return val_acc, hist

# -----------------------------
# Run multiple experiments
# -----------------------------
def run_experiments(cell_name='GRU', units=128, num_layers=1, trials=3, epochs=20, batch_size=128, lr=0.001, dropout=0.0):
    cell_map = {'GRU': GRUCell, 'MGU': MGUCell}
    if cell_name not in cell_map:
        raise ValueError(f"Unknown cell type: {cell_name}. Use 'GRU' or 'MGU'.")

    cell_class = cell_map[cell_name]
    save_root = 'models'

    val_accuracies = []

    for t in range(trials):
        print(f"=== Trial {t+1}/{trials} for {cell_name} ===")
        val_acc, _ = train_one_trial(cell_class, t, save_root,
                                     units=units, num_layers=num_layers,
                                     epochs=epochs, batch_size=batch_size, lr=lr, dropout=dropout)
        val_accuracies.append(val_acc)
        print(f"Trial {t+1} Validation Accuracy: {val_acc:.4f}\n")

    mean_acc = np.mean(val_accuracies)
    print(f"Average Validation Accuracy over {trials} trials for {cell_name}: {mean_acc:.4f}")


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=str, default='GRU', help="Cell type: 'GRU' or 'MGU'")
    parser.add_argument('--units', type=int, default=128, help="Number of hidden units")
    parser.add_argument('--layers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('--trials', type=int, default=3, help="Number of trials")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    args = parser.parse_args()

    run_experiments(cell_name=args.cell, units=args.units, num_layers=args.layers,
                    trials=args.trials, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, dropout=args.dropout)
