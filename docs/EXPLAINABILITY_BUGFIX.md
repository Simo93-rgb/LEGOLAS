# Explainability Bug Fix - Embedding Layer

## Problema

Errore durante esecuzione Integrated Gradients:
```
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following 
scalar types: Long, Int; but got torch.cuda.FloatTensor instead
```

### Causa Root
Captum's Integrated Gradients **interpola linearmente** tra baseline e input durante i suoi steps:

```python
# Captum fa internamente:
interpolated_input = baseline + (alpha * (input - baseline))
# Se input = input_ids (Long), interpolated diventa Float!
```

Ma l'embedding layer di BERT/Transformer si aspetta **Long tensor** (indices), non Float.

## Soluzione

Usare **embeddings come input** invece di input_ids:

### Prima (❌ Non funziona)
```python
# Passa input_ids direttamente
attributions = ig.attribute(
    inputs=input_ids,  # Long tensor
    baselines=baseline_ids,  # Long tensor
    target=target_class
)
# IG interpola → Float tensor → ERRORE nell'embedding layer
```

### Dopo (✅ Funziona)
```python
# Converti prima in embeddings
embeddings = model.embeddings.word_embeddings(input_ids)  # Float tensor
baseline_embeddings = model.embeddings.word_embeddings(baseline_ids)

attributions = ig.attribute(
    inputs=embeddings,  # Float tensor
    baselines=baseline_embeddings,  # Float tensor
    target=target_class
)
# IG interpola embeddings → Float tensor → OK!
```

## Modifiche Implementate

### 1. Forward Function (`_create_forward_func`)
**Vecchio**: Prendeva `input_ids` (Long)
```python
def forward_func(input_ids, attention_mask=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs
```

**Nuovo**: Prende `embeddings` (Float) e salta l'embedding layer
```python
def forward_func(embeddings, attention_mask=None):
    # Aggiungi position embeddings
    position_embeddings = base_model.embeddings.position_embeddings(position_ids)
    embeddings = embeddings + position_embeddings
    
    # Token type embeddings
    token_type_embeddings = base_model.embeddings.token_type_embeddings(token_type_ids)
    embeddings = embeddings + token_type_embeddings
    
    # LayerNorm + Dropout
    embeddings = base_model.embeddings.LayerNorm(embeddings)
    embeddings = base_model.embeddings.dropout(embeddings)
    
    # Encoder
    encoder_outputs = base_model.encoder(embeddings, attention_mask=extended_attention_mask)
    
    # Classification head
    pooled_output = encoder_outputs[0][:, 0, :]
    logits = self.model.classificationHead(pooled_output)
    return logits
```

### 2. Explain Text (`explain_text`)
**Vecchio**: Passava `input_ids`
```python
input_ids = encoding['input_ids'].to(device)
baseline = torch.zeros_like(input_ids)

attributions = ig.attribute(
    inputs=input_ids,  # ❌ Long tensor
    baselines=baseline,
    ...
)
```

**Nuovo**: Converte in embeddings prima
```python
input_ids = encoding['input_ids'].to(device)

# Converti in embeddings
embeddings = base_model.embeddings.word_embeddings(input_ids)
baseline_ids = torch.zeros_like(input_ids)
baseline = base_model.embeddings.word_embeddings(baseline_ids)

attributions = ig.attribute(
    inputs=embeddings,  # ✅ Float tensor
    baselines=baseline,
    ...
)

# Somma su dimensione embedding
attr_scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
```

### 3. Rimosso `torch.no_grad()`
Captum ha **bisogno dei gradienti** per IG, quindi rimosso il context manager.

## Vantaggi Approccio Embedding

1. ✅ **Compatibile con Captum**: IG può interpolare embeddings continuamente
2. ✅ **Semanticamente corretto**: Interpolazione nello spazio embedding è più sensata
3. ✅ **Evita discretizzazione**: Non serve arrotondare interpolazioni a token IDs
4. ✅ **Standard practice**: È il metodo raccomandato da Captum per NLP

## Riferimenti

- Captum Tutorials: https://captum.ai/tutorials/Bert_SQUAD_Interpret
- Integrated Gradients Paper: https://arxiv.org/abs/1703.01365
- Interpretable Embeddings: https://captum.ai/api/layer.html#interpretable-embedding-base

## Test

Dopo la fix, l'esecuzione dovrebbe completare senza errori:
```bash
python -m src.explainability.extract_explainability \
    --model clinical-modernbert \
    --format narrativo \
    --n_samples 50 \
    --device cuda
```

Output atteso:
- ✅ Loading model + data
- ✅ Generating predictions
- ✅ Analyzing with IG (con progress bar)
- ✅ Extracting top words
- ✅ Creating visualizations
- ✅ Saving results
