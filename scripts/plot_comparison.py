import matplotlib.pyplot as plt

# Dati
metriche = ["Accuracy", "Bal. Accuracy", "F1-Score", "Precision", "Recall", "Specificity"]
azioni_binning = [91.34, 87.16, 67.35, 57.23, 81.82, 92.51]
array_concat = [87.29, 83.44, 57.40, 45.24, 78.51, 88.36]
storytelling = [92.61, 89.69, 71.72, 61.54, 85.95, 93.42]

# Setup della figura
plt.figure(figsize=(10, 5))

# Plot dei punti (il parametro 'o', 's', '^' definisce la forma del punto)
plt.plot(metriche, azioni_binning, 'o', markersize=10, label='Azioni + Binning', color='#1f77b4')
plt.plot(metriche, array_concat, 's', markersize=10, label='Array Concatenati', color='#ff7f0e')
plt.plot(metriche, storytelling, '^', markersize=10, label='Storytelling', color='#2ca02c')

# Personalizzazione
plt.title("Confronto Metriche per Tecniche di Embedding", fontsize=14, pad=15)
plt.ylabel("Percentuale (%)", fontsize=12)
plt.ylim(40, 100) # Imposta l'asse Y per far risaltare le differenze
plt.grid(True, linestyle='--', alpha=0.5) # Griglia leggera per aiutare l'occhio
plt.legend(loc='lower right')

# Layout e salvataggio
plt.tight_layout()
plt.savefig("confronto_metriche_punti.png", dpi=300) # dpi=300 garantisce qualità tipografica
#plt.show()