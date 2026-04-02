"""
Script per visualizzare train loss e validation loss da checkpoint history JSON.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_losses(checkpoint_file, save_path=None, show=True):
    """
    Carica i dati di training da un file checkpoint JSON e plotta train/val loss.
    
    Args:
        checkpoint_file (str or Path): Percorso al file checkpoint JSON
        save_path (str or Path, optional): Percorso dove salvare il plot. Se None, non salva.
        show (bool): Se True, mostra il plot interattivo
    """
    checkpoint_file = Path(checkpoint_file)
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"File non trovato: {checkpoint_file}")
    
    # Carica i dati
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    # Estrae le epoche e le loss
    history = data.get('history', [])
    if not history:
        raise ValueError("Nessun dato di history trovato nel file JSON")
    
    epochs = [item['epoch'] for item in history]
    train_losses = [item['train_loss'] for item in history]
    val_losses = [item['val_loss'] for item in history]
    
    # Crea il plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salva se richiesto
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato in: {save_path}")
    
    # Mostra se richiesto
    if show:
        plt.show()
    
    return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotta train/val loss da checkpoint history JSON")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='output/models/checkpoint_history_fold0.json',
        help='Percorso al file checkpoint JSON (default: output/models/checkpoint_history_fold0.json)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Percorso dove salvare il plot. Se non specificato, non salva il file.'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Se specificato, non mostra il plot interattivo'
    )
    
    args = parser.parse_args()
    
    plot_losses(
        checkpoint_file=args.checkpoint,
        save_path=args.save,
        show=not args.no_show
    )
