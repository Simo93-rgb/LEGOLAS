"""
Integrated Gradients Explainer per modelli BERT
Implementa l'estrazione di attribution scores usando Integrated Gradients
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from captum.attr import IntegratedGradients
from tqdm import tqdm


class IntegratedGradientsExplainer:
    """
    Classe per estrarre attribution scores usando Integrated Gradients.

    Adattata per supportare modelli time-aware (TimeAwareBertClassifier),
    dove il forward accetta anche `time_deltas` e la fusione avviene
    su hidden states. In tal caso, gli IG vengono computati sugli
    embeddings dei token (word embeddings) e il forward wrapper del
    modello riproduce la pipeline di embedding+encoder+pooling prima
    della testa di classificazione.
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: Modello BERT addestrato
            tokenizer: Tokenizer corrispondente
            device: Device per computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Crea forward function per Captum (adatta BERT/Roberta e TimeAwareBertClassifier)
        forward_func = self._create_forward_func()
        
        # Inizializza Integrated Gradients
        self.ig = IntegratedGradients(forward_func)
        
    def _create_forward_func(self):
        """
        Crea wrapper del modello compatibile con Captum.
        Usa embeddings come input invece di input_ids per evitare problemi
        con l'interpolazione IG.
        Supporta:
        - Modelli con attributo `longformer` e `output_layer` (es. LongFormerMultiClassificationHeads)
        - `TimeAwareBertClassifier` con attributo `base` e `classifier`
        """
        def forward_func(embeddings, attention_mask=None, time_deltas: torch.Tensor = None):
            # Rileva tipo modello
            if hasattr(self.model, 'longformer') and hasattr(self.model, 'output_layer'):
                base_model = self.model.longformer
                # Costruisci embeddings completi (aggiungi posizioni e token_type se presenti)
                if hasattr(base_model.embeddings, 'position_embeddings'):
                    seq_length = embeddings.size(1)
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=embeddings.device)
                    position_ids = position_ids.unsqueeze(0).expand(embeddings.size(0), seq_length)
                    position_embeddings = base_model.embeddings.position_embeddings(position_ids)
                    embeddings = embeddings + position_embeddings
                if hasattr(base_model.embeddings, 'token_type_embeddings'):
                    token_type_ids = torch.zeros(embeddings.size()[:-1], dtype=torch.long, device=embeddings.device)
                    token_type_embeddings = base_model.embeddings.token_type_embeddings(token_type_ids)
                    embeddings = embeddings + token_type_embeddings

                # LayerNorm + Dropout
                embeddings = base_model.embeddings.LayerNorm(embeddings)
                embeddings = base_model.embeddings.dropout(embeddings)

                # Attention mask
                if attention_mask is None:
                    attention_mask = torch.ones(embeddings.size()[:-1], device=embeddings.device)
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=embeddings.dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embeddings.dtype).min

                # Encoder
                encoder_outputs = base_model.encoder(embeddings, attention_mask=extended_attention_mask)
                sequence_output = encoder_outputs[0]

                # Pooling
                if hasattr(base_model, 'pooler') and base_model.pooler is not None:
                    pooled_output = base_model.pooler(sequence_output)
                else:
                    pooled_output = sequence_output[:, 0, :]

                # Classification head
                logits = self.model.output_layer(pooled_output)
                return logits

            # TimeAwareBertClassifier path
            if hasattr(self.model, 'base') and hasattr(self.model, 'classifier'):
                base_model = self.model.base
                # Embeddings complete
                if hasattr(base_model.embeddings, 'position_embeddings'):
                    seq_length = embeddings.size(1)
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=embeddings.device)
                    position_ids = position_ids.unsqueeze(0).expand(embeddings.size(0), seq_length)
                    position_embeddings = base_model.embeddings.position_embeddings(position_ids)
                    embeddings = embeddings + position_embeddings
                if hasattr(base_model.embeddings, 'token_type_embeddings'):
                    token_type_ids = torch.zeros(embeddings.size()[:-1], dtype=torch.long, device=embeddings.device)
                    token_type_embeddings = base_model.embeddings.token_type_embeddings(token_type_ids)
                    embeddings = embeddings + token_type_embeddings

                # LayerNorm + Dropout
                embeddings = base_model.embeddings.LayerNorm(embeddings)
                embeddings = base_model.embeddings.dropout(embeddings)

                if attention_mask is None:
                    attention_mask = torch.ones(embeddings.size()[:-1], device=embeddings.device)
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=embeddings.dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embeddings.dtype).min

                # Passa attraverso encoder
                encoder_outputs = base_model.encoder(embeddings, attention_mask=extended_attention_mask)
                token_hidden = encoder_outputs[0]  # (B, T, H)

                # Time embedding e fusione (se forniti i time_deltas)
                if time_deltas is not None and hasattr(self.model, 'time_emb'):
                    t_emb = self.model.time_emb(time_deltas)  # (B, T, D)
                    if getattr(self.model, 'time_proj', None) is not None:
                        t_emb = self.model.time_proj(t_emb)  # (B, T, H)
                    fused = token_hidden + t_emb
                else:
                    fused = token_hidden

                # Pooling coerente con modello
                if getattr(self.model, 'use_cls_pooling', True) and getattr(base_model.config, 'cls_token_id', None) is not None:
                    pooled = fused[:, 0, :]
                else:
                    mask = attention_mask.unsqueeze(-1).float()
                    summed = (fused * mask).sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1.0)
                    pooled = summed / lengths

                logits = self.model.classifier(torch.nn.functional.dropout(pooled, p=self.model.dropout.p, training=False))
                return logits

            raise ValueError("Modello non supportato per explainability: expected LongFormerMultiClassificationHeads or TimeAwareBertClassifier")

        return forward_func
    
    def explain_text(
        self,
        text: str,
        target_class: int,
        n_steps: int = 50,
        internal_batch_size: int = 4
    ) -> Tuple[List[str], np.ndarray]:
        """
        Calcola attribution scores per un singolo testo
        
        Args:
            text: Testo da analizzare
            target_class: Classe target per l'attribution (0 o 1)
            n_steps: Numero di step per Integrated Gradients
            internal_batch_size: Batch size interno per IG
            
        Returns:
            tokens: Lista di token
            attributions: Array di attribution scores per token
        """
        # Tokenizza
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Converti input_ids in embeddings
        base_model = getattr(self.model, 'longformer', None) or getattr(self.model, 'base', None)
        if base_model is None:
            raise ValueError("Modello non supportato per explainability: missing base/longformer")
        embeddings = base_model.embeddings.word_embeddings(input_ids)
        
        # Baseline: embeddings di padding token
        baseline_ids = torch.zeros_like(input_ids)
        baseline = base_model.embeddings.word_embeddings(baseline_ids)
        
        # Calcola Integrated Gradients sugli embeddings
        # Per modelli time-aware, non abbiamo i time_deltas qui perché partiamo da testo.
        # L'extract_explainability per sequenze fornirà time_deltas e chiamerà una funzione dedicata.
        attributions = self.ig.attribute(
            inputs=embeddings,
            baselines=baseline,
            method='riemann_trapezoid',
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )
        
        # Somma attributions su dimensione embedding per ottenere score per token
        # attributions shape: (1, seq_len, embedding_dim)
        attr_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        
        # Decodifica tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        
        return tokens, attr_scores

    def explain_sequence(
        self,
        actions: List[str],
        time_deltas: List[float],
        target_class: int,
        n_steps: int = 50,
        internal_batch_size: int = 32,
        max_length: int = 512
    ) -> Dict:
        """
        Calcola attribution per una sequenza di azioni con time_deltas.

        Args:
            actions: Lista di azioni (stringhe)
            time_deltas: Lista di delta temporali (float) per ogni azione/sub-token
            target_class: Classe target
            n_steps: Passi IG
            internal_batch_size: Batch interno IG
            max_length: Lunghezza massima

        Returns:
            Dict con keys: text, tokens, token_attributions, word_attributions
        """
        # Tokenizza ogni azione e propaga time_deltas come in SequenceDataset
        ids = []
        times = []
        attention_mask = []
        cls_id = getattr(self.tokenizer, 'cls_token_id', None)
        sep_id = getattr(self.tokenizer, 'sep_token_id', None)

        if cls_id is not None:
            ids.append(cls_id)
            times.append(0.0)

        for act, d in zip(actions, time_deltas):
            tok_ids = self.tokenizer.encode(act, add_special_tokens=False)
            ids.extend(tok_ids)
            times.extend([float(d)] * len(tok_ids))
            if sep_id is not None:
                ids.append(sep_id)
                times.append(0.0)

        if len(ids) > max_length:
            ids = ids[:max_length]
            times = times[:max_length]

        attention_mask = [1] * len(ids)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=self.device).unsqueeze(0)
        time_deltas_t = torch.tensor(times, dtype=torch.float, device=self.device).unsqueeze(0)

        # Embeddings
        base_model = getattr(self.model, 'base', None) or getattr(self.model, 'longformer', None)
        if base_model is None:
            raise ValueError("Modello non supportato per explainability: missing base/longformer")
        embeddings = base_model.embeddings.word_embeddings(input_ids)
        baseline_ids = torch.zeros_like(input_ids)
        baseline = base_model.embeddings.word_embeddings(baseline_ids)

        # IG con forward che accetta anche time_deltas
        attributions = self.ig.attribute(
            inputs=embeddings,
            baselines=baseline,
            method='riemann_trapezoid',
            additional_forward_args=(attention_mask_t, time_deltas_t),
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )

        attr_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        word_attributions = self.aggregate_subword_attributions(tokens, attr_scores)

        return {
            'text': ' '.join(actions),
            'tokens': tokens,
            'token_attributions': attr_scores.tolist(),
            'word_attributions': word_attributions,
        }

    def build_closed_forward_fn(
        self,
        attention_mask: torch.Tensor,
        time_deltas: torch.Tensor = None
    ):
        """
        Costruisce una forward_fn(input_embeds) che chiude su attention_mask e time_deltas.
        Usata da ig_completeness per la strategia adaptive_steps, dove la IG viene
        chiamata più volte con steps crescenti sullo stesso input.

        Args:
            attention_mask: (1, seq_len) attention mask già su device
            time_deltas: (1, seq_len) time deltas già su device (opzionale)

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: forward_fn compatibile con
            compute_ig_with_completeness_check
        """
        model = self.model

        if hasattr(model, 'base') and hasattr(model, 'classifier'):
            base_model = model.base

            def forward_fn(input_embeds):
                inp = input_embeds
                if hasattr(base_model.embeddings, 'position_embeddings'):
                    seq_len = inp.size(1)
                    pos_ids = torch.arange(seq_len, dtype=torch.long, device=inp.device)
                    pos_ids = pos_ids.unsqueeze(0).expand(inp.size(0), seq_len)
                    inp = inp + base_model.embeddings.position_embeddings(pos_ids)
                if hasattr(base_model.embeddings, 'token_type_embeddings'):
                    tt_ids = torch.zeros(inp.size()[:-1], dtype=torch.long, device=inp.device)
                    inp = inp + base_model.embeddings.token_type_embeddings(tt_ids)
                inp = base_model.embeddings.LayerNorm(inp)
                inp = base_model.embeddings.dropout(inp)

                ext_mask = attention_mask[:, None, None, :].to(dtype=inp.dtype)
                ext_mask = (1.0 - ext_mask) * torch.finfo(inp.dtype).min
                enc_out = base_model.encoder(inp, attention_mask=ext_mask)
                token_hidden = enc_out[0]

                if time_deltas is not None and hasattr(model, 'time_emb'):
                    t_emb = model.time_emb(time_deltas)
                    if getattr(model, 'time_proj', None) is not None:
                        t_emb = model.time_proj(t_emb)
                    fused = token_hidden + t_emb
                else:
                    fused = token_hidden

                if getattr(model, 'use_cls_pooling', True):
                    pooled = fused[:, 0, :]
                else:
                    mask_f = attention_mask.unsqueeze(-1).float()
                    pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

                import torch.nn.functional as F
                logits = model.classifier(F.dropout(pooled, p=0.0, training=False))
                return logits

            return forward_fn

        if hasattr(model, 'longformer') and hasattr(model, 'output_layer'):
            base_model = model.longformer

            def forward_fn(input_embeds):
                inp = input_embeds
                if hasattr(base_model.embeddings, 'position_embeddings'):
                    seq_len = inp.size(1)
                    pos_ids = torch.arange(seq_len, dtype=torch.long, device=inp.device)
                    pos_ids = pos_ids.unsqueeze(0).expand(inp.size(0), seq_len)
                    inp = inp + base_model.embeddings.position_embeddings(pos_ids)
                if hasattr(base_model.embeddings, 'token_type_embeddings'):
                    tt_ids = torch.zeros(inp.size()[:-1], dtype=torch.long, device=inp.device)
                    inp = inp + base_model.embeddings.token_type_embeddings(tt_ids)
                inp = base_model.embeddings.LayerNorm(inp)
                inp = base_model.embeddings.dropout(inp)

                ext_mask = attention_mask[:, None, None, :].to(dtype=inp.dtype)
                ext_mask = (1.0 - ext_mask) * torch.finfo(inp.dtype).min
                enc_out = base_model.encoder(inp, attention_mask=ext_mask)
                seq_out = enc_out[0]

                if hasattr(base_model, 'pooler') and base_model.pooler is not None:
                    pooled = base_model.pooler(seq_out)
                else:
                    pooled = seq_out[:, 0, :]

                return model.output_layer(pooled)

            return forward_fn

        raise ValueError("Modello non supportato per build_closed_forward_fn: expected TimeAwareBertClassifier or LongFormerMultiClassificationHeads")

    def aggregate_subword_attributions(
        self,
        tokens: List[str],
        attributions: np.ndarray
    ) -> Dict[str, float]:
        """
        Aggrega attribution scores da sub-word tokens a parole complete
        
        Strategia:
        - Per ogni parola, somma gli score di tutti i suoi sub-token
        - Gestisce token BERT (##prefix)
        
        Args:
            tokens: Lista di token dal tokenizer
            attributions: Attribution scores per token
            
        Returns:
            Dict mapping parola -> attribution score aggregato
        """
        word_attributions = {}
        current_word = ""
        current_score = 0.0
        
        for token, score in zip(tokens, attributions):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Check se è un sub-token (inizia con ##)
            if token.startswith('##'):
                # Continua parola corrente
                current_word += token[2:]  # Rimuovi ##
                current_score += score
            else:
                # Salva parola precedente se esiste
                if current_word:
                    word_attributions[current_word] = current_score
                
                # Inizia nuova parola
                current_word = token
                current_score = score
        
        # Salva ultima parola
        if current_word:
            word_attributions[current_word] = current_score
        
        return word_attributions
    
    def explain_batch(
        self,
        texts: List[str],
        labels: List[int],
        predicted_classes: List[int],
        batch_size: int = 8,
        n_steps: int = 50
    ) -> List[Dict]:
        """
        Spiega un batch di testi
        
        Args:
            texts: Lista di testi
            labels: Label vere
            predicted_classes: Classi predette dal modello
            batch_size: Batch size per processing
            n_steps: Step per Integrated Gradients
            
        Returns:
            Lista di dizionari con risultati per ogni testo
        """
        results = []
        
        print(f"\n🔍 Analyzing {len(texts)} texts with Integrated Gradients...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_preds = predicted_classes[i:i+batch_size]
            
            for text, label, pred in zip(batch_texts, batch_labels, batch_preds):
                # Spiega rispetto alla classe predetta
                tokens, attributions = self.explain_text(
                    text=text,
                    target_class=pred,
                    n_steps=n_steps
                )
                
                # Aggrega sub-words
                word_attributions = self.aggregate_subword_attributions(
                    tokens, attributions
                )
                
                results.append({
                    'text': text,
                    'true_label': label,
                    'predicted_label': pred,
                    'tokens': tokens,
                    'token_attributions': attributions.tolist(),
                    'word_attributions': word_attributions
                })
        
        return results
    
    def extract_top_words(
        self,
        results: List[Dict],
        top_k: int = 25,
        by_class: bool = True
    ) -> Dict:
        """
        Estrae le top-K parole più importanti.

        Se i risultati contengono 'word_attr_class_0' e 'word_attr_class_1'
        (modalità dual-class), aggrega su TUTTI i campioni usando le attributzioni
        per-classe esplicite — nessuno zero spurio.

        Altrimenti, fallback: separa campioni per predicted_label (legacy).

        Args:
            results: Risultati da explain_batch / extract loop
            top_k: Numero di parole da estrarre
            by_class: Se True, estrae separatamente per classe

        Returns:
            Dict con top words aggregate
        """
        if by_class:
            # --- Dual-class mode: every sample has explicit per-class attributions ---
            has_dual = results and 'word_attr_class_0' in results[0] and 'word_attr_class_1' in results[0]
            if has_dual:
                class_0_words: dict = {}
                class_1_words: dict = {}
                for result in results:
                    for word, score in result['word_attr_class_0'].items():
                        class_0_words.setdefault(word, []).append(abs(score))
                    for word, score in result['word_attr_class_1'].items():
                        class_1_words.setdefault(word, []).append(abs(score))
            else:
                # Legacy: split samples by predicted_label
                class_0_words = {}
                class_1_words = {}
                for result in results:
                    target_dict = class_0_words if result['predicted_label'] == 0 else class_1_words
                    for word, score in result['word_attributions'].items():
                        target_dict.setdefault(word, []).append(abs(score))

            class_0_avg = {word: np.mean(scores) for word, scores in class_0_words.items()}
            class_1_avg = {word: np.mean(scores) for word, scores in class_1_words.items()}

            top_class_0 = sorted(class_0_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_class_1 = sorted(class_1_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]

            return {
                'class_0': dict(top_class_0),
                'class_1': dict(top_class_1)
            }
        else:
            # Aggregato globale
            all_words: dict = {}
            for result in results:
                for word, score in result['word_attributions'].items():
                    all_words.setdefault(word, []).append(abs(score))

            avg_scores = {word: np.mean(scores) for word, scores in all_words.items()}
            top_words = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            return {'all': dict(top_words)}


if __name__ == "__main__":
    print("Integrated Gradients Explainer module")
    print("Usage: from src.explainability import IntegratedGradientsExplainer")
