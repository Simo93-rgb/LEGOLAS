# Visualization Normalization Strategy

## 📊 Problema identificato e risolto (21 Oct 2025)

### **Bug originale**

Il codice commentava "normalizzazione SEPARATA per classe", ma implementava normalizzazione **globale**:

```python
# ❌ BUG (prima):
max_c0 = max(scores_c0)  # es. 0.85
max_c1 = max(scores_c1)  # es. 0.42
max_display = max(max_c0, max_c1)  # 0.85 (max globale!)

scores_c0_norm = [s / 0.85 for s in scores_c0]  # ✅ Max = 1.0
scores_c1_norm = [s / 0.85 for s in scores_c1]  # ❌ Max = 0.49
```

**Risultato visivo**: Classe 0 raggiunge 1.0, Classe 1 si ferma a ~0.5

---

### **Fix implementato**

Normalizzazione **veramente separata** per classe:

```python
# ✅ FIX (dopo):
max_c0 = max(scores_c0)  # es. 0.85
max_c1 = max(scores_c1)  # es. 0.42

scores_c0_norm = [s / max_c0 for s in scores_c0]  # Max = 1.0
scores_c1_norm = [s / max_c1 for s in scores_c1]  # Max = 1.0
```

**Risultato visivo**: Entrambe le classi raggiungono 1.0 al loro massimo

---

## 🎯 Rationale: Perché normalizzazione separata?

### **Caso d'uso**: Confronto importanza relativa **dentro ogni classe**

L'obiettivo è vedere:
- **Quali parole/azioni sono più importanti per Classe 0** (relativo al massimo di C0)
- **Quali parole/azioni sono più importanti per Classe 1** (relativo al massimo di C1)

Non vogliamo confrontare magnitudini assolute tra classi, ma **ranking relativo**.

### **Esempio pratico**

Supponiamo:

```python
# Attribution scores (raw)
Classe 0: "ricovero": 0.85, "dimissione": 0.70, "terapia": 0.50
Classe 1: "ricovero": 0.42, "dimissione": 0.35, "terapia": 0.20
```

#### **Con normalizzazione globale (BUG):**
```
Classe 0: "ricovero": 1.0, "dimissione": 0.82, "terapia": 0.59
Classe 1: "ricovero": 0.49, "dimissione": 0.41, "terapia": 0.24
```
❌ **Problema**: Sembra che Classe 1 abbia valori "piccoli", ma è solo per scale diverse

#### **Con normalizzazione separata (FIX):**
```
Classe 0: "ricovero": 1.0, "dimissione": 0.82, "terapia": 0.59
Classe 1: "ricovero": 1.0, "dimissione": 0.83, "terapia": 0.48
```
✅ **Beneficio**: Mostra che "ricovero" è UGUALMENTE importante (max in entrambe), e "dimissione" ha importanza relativa simile in entrambe le classi

---

## 📈 Tipi di visualizzazione

### **1. Heatmap** (normalizzazione globale)
- File: `plot_attention_heatmap()`, `plot_clinical_actions_heatmap()`
- Strategia: **Normalizzazione globale** (max tra tutte le classi)
- Razionale: Confrontare magnitudini assolute tra classi
- Utile per: Vedere quale classe ha attributions più forti in generale

### **2. Comparison Histogram** (normalizzazione separata)
- File: `plot_class_comparison()`, `plot_clinical_actions_comparison()`
- Strategia: **Normalizzazione separata** (ogni classe al proprio max)
- Razionale: Confrontare importanza relativa **dentro** ogni classe
- Utile per: Vedere ranking interno a ciascuna classe

---

## 🔧 Codice modificato

### **File**: `src/explainability/visualization.py`

#### **Funzioni modificate:**
1. `plot_class_comparison()` (linee ~178-183)
2. `plot_clinical_actions_comparison()` (linee ~479-485)

#### **Cambiamenti:**

```python
# PRIMA (bug):
max_display = max(max_c0, max_c1)
scores_c0_norm = [s / max_display for s in scores_c0]
scores_c1_norm = [s / max_display for s in scores_c1]

# DOPO (fix):
scores_c0_norm = [s / max_c0 for s in scores_c0]
scores_c1_norm = [s / max_c1 for s in scores_c1]
```

#### **Aggiornamenti titoli:**

```python
# PRIMA:
'Normalized to max value in display = 1.0'

# DOPO:
'Each class normalized to its own max = 1.0'
```

---

## ✅ Testing

Per verificare il fix:

```bash
# Rigenera visualizzazioni
uv run python src/explainability/extract_explainability.py \
  --n_samples 50 \
  --adaptive_steps \
  --use_ensemble

# Check plot
ls -lh output/explainability/histogram_*
```

**Verifica visiva attesa:**
- Entrambe le classi raggiungono 1.0 nelle loro barre più alte
- Il confronto mostra importanza **relativa** dentro ogni classe
- Le parole con ranking simile in entrambe le classi hanno altezze simili

---

## 📚 Riferimenti

- Issue discussa: 21 Oct 2025
- Fix commit: [link quando disponibile]
- Related: `docs/EXPLAINABILITY_GUIDE.md`
