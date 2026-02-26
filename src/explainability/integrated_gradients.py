"""
Integrated Gradients Explainer per modelli BERT
Implementa l'estrazione di attribution scores usando Integrated Gradients
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from tqdm import tqdm


class IntegratedGradientsExplainer:
    """
    Classe per estrarre attribution scores usando Integrated Gradients
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
        
        # Crea forward function per Captum
        forward_func = self._create_forward_func()
        
        # Inizializza Integrated Gradients
        self.ig = IntegratedGradients(forward_func)
        
    def _create_forward_func(self):
        """
        Crea wrapper del modello compatibile con Captum
        Usa embeddings come input invece di input_ids per evitare problemi con interpolazione
        """
        def forward_func(embeddings, attention_mask=None):
            """
            Forward pass wrapper per Captum
            
            Args:
                embeddings: Tensor di embeddings (batch_size, seq_len, embedding_dim)
                attention_mask: Attention mask (batch_size, seq_len)
            """
            # Ottieni base model (longformer/bert)
            base_model = self.model.longformer
            
            # Passa embeddings direttamente all'encoder (salta embedding layer)
            # Per BERT-like models: usa il metodo che prende embeddings
            if hasattr(base_model, 'encoder'):
                # Aggiungi position embeddings e token_type_embeddings se necessario
                if hasattr(base_model.embeddings, 'position_embeddings'):
                    seq_length = embeddings.size(1)
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=embeddings.device)
                    position_ids = position_ids.unsqueeze(0).expand_as(embeddings[:, :, 0].long())
                    position_embeddings = base_model.embeddings.position_embeddings(position_ids)
                    embeddings = embeddings + position_embeddings
                
                if hasattr(base_model.embeddings, 'token_type_embeddings'):
                    token_type_ids = torch.zeros(embeddings.size()[:-1], dtype=torch.long, device=embeddings.device)
                    token_type_embeddings = base_model.embeddings.token_type_embeddings(token_type_ids)
                    embeddings = embeddings + token_type_embeddings
                
                # LayerNorm + Dropout
                embeddings = base_model.embeddings.LayerNorm(embeddings)
                embeddings = base_model.embeddings.dropout(embeddings)
                
                # Passa attraverso encoder
                if attention_mask is None:
                    attention_mask = torch.ones(embeddings.size()[:-1], device=embeddings.device)
                
                # Extended attention mask per encoder
                extended_attention_mask = attention_mask[:, None, None, :]
                extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embeddings.dtype).min
                
                encoder_outputs = base_model.encoder(
                    embeddings,
                    attention_mask=extended_attention_mask,
                )
                
                sequence_output = encoder_outputs[0]
                
                # Usa pooler se disponibile, altrimenti CLS token
                if hasattr(base_model, 'pooler') and base_model.pooler is not None:
                    pooled_output = base_model.pooler(sequence_output)
                else:
                    pooled_output = sequence_output[:, 0, :]  # CLS token
            else:
                raise ValueError("Modello non supportato per explainability")
            
            # Usa output_layer del nostro modello (non classificationHead)
            logits = self.model.output_layer(pooled_output)
            
            return logits
        
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
        base_model = self.model.longformer
        embeddings = base_model.embeddings.word_embeddings(input_ids)
        
        # Baseline: embeddings di padding token
        baseline_ids = torch.zeros_like(input_ids)
        baseline = base_model.embeddings.word_embeddings(baseline_ids)
        
        # Calcola Integrated Gradients sugli embeddings
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
        Estrae le top-K parole più importanti
        
        Args:
            results: Risultati da explain_batch
            top_k: Numero di parole da estrarre
            by_class: Se True, estrae separatamente per classe
            
        Returns:
            Dict con top words aggregate
        """
        if by_class:
            # Separa per classe predetta
            class_0_words = {}
            class_1_words = {}
            
            for result in results:
                target_dict = class_0_words if result['predicted_label'] == 0 else class_1_words
                
                for word, score in result['word_attributions'].items():
                    if word not in target_dict:
                        target_dict[word] = []
                    target_dict[word].append(abs(score))  # Usa valore assoluto
            
            # Calcola media per classe
            class_0_avg = {
                word: np.mean(scores) 
                for word, scores in class_0_words.items()
            }
            class_1_avg = {
                word: np.mean(scores)
                for word, scores in class_1_words.items()
            }
            
            # Ordina e prendi top-K
            top_class_0 = sorted(
                class_0_avg.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            top_class_1 = sorted(
                class_1_avg.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            return {
                'class_0': dict(top_class_0),
                'class_1': dict(top_class_1)
            }
        else:
            # Aggregato globale
            all_words = {}
            for result in results:
                for word, score in result['word_attributions'].items():
                    if word not in all_words:
                        all_words[word] = []
                    all_words[word].append(abs(score))
            
            avg_scores = {
                word: np.mean(scores)
                for word, scores in all_words.items()
            }
            
            top_words = sorted(
                avg_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            return {'all': dict(top_words)}


if __name__ == "__main__":
    print("Integrated Gradients Explainer module")
    print("Usage: from src.explainability import IntegratedGradientsExplainer")
