"""
Clinical Action Aggregator
Aggrega attribution scores a livello di azioni cliniche (frasi complete)
Implementa la strategia Fase B del documento XAI.md
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class ClinicalActionAggregator:
    """
    Aggrega attribution scores da singole parole a azioni cliniche complete
    """
    
    def __init__(self, clinical_actions: List[str] = None):
        """
        Args:
            clinical_actions: Lista di azioni cliniche da cercare
                             Se None, usa un set predefinito
        """
        if clinical_actions is None:
            # Dizionario predefinito di azioni cliniche comuni
            self.clinical_actions = self._default_clinical_actions()
        else:
            self.clinical_actions = clinical_actions
        
        # Ordina per lunghezza decrescente per matching corretto
        # (match prima "Psychiatric follow-up visit" poi "Psychiatric visit")
        self.clinical_actions = sorted(
            self.clinical_actions,
            key=lambda x: len(x.split()),
            reverse=True
        )
    
    def _default_clinical_actions(self) -> List[str]:
        """
        Carica azioni cliniche dal file di traduzione
        Usa le traduzioni inglesi dal translation_cache.json
        """
        translation_file = Path(__file__).parent.parent.parent / "data" / "translation_cache.json"
        
        if not translation_file.exists():
            # Fallback: usa un set minimo predefinito
            print(f"⚠️  Translation file not found: {translation_file}")
            print(f"   Using minimal fallback clinical actions")
            return [
                "Psychiatric follow-up visit",
                "Psychiatric visit",
                "Emergency surgical visit",
                "Lower abdomen ultrasound",
                "Upper abdomen CT",
                "Brain MRI without and with contrast",
                "Full spine X-ray",
            ]
        
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            # Estrai valori (traduzioni inglesi)
            clinical_actions = list(translations.values())
            
            print(f"✅ Loaded {len(clinical_actions)} clinical actions from translation file")
            
            return clinical_actions
            
        except Exception as e:
            print(f"⚠️  Error loading translation file: {e}")
            print(f"   Using minimal fallback clinical actions")
            return [
                "Psychiatric follow-up visit",
                "Psychiatric visit",
                "Emergency surgical visit",
            ]
    
    def find_actions_in_text(
        self,
        text: str,
        word_attributions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Trova azioni cliniche nel testo e calcola loro attribution score
        
        Args:
            text: Testo della storia clinica
            word_attributions: Dict parola -> attribution score
            
        Returns:
            Dict azione_clinica -> attribution score totale
        """
        text_lower = text.lower()
        action_scores = {}
        
        # Trova ogni azione nel testo
        for action in self.clinical_actions:
            action_lower = action.lower()
            
            # Cerca l'azione nel testo
            if action_lower in text_lower:
                # Calcola score come somma degli score delle parole componenti
                words_in_action = action.split()
                score = 0.0
                words_found = 0
                
                for word in words_in_action:
                    word_lower = word.lower()
                    
                    # Cerca la parola nelle word_attributions
                    # (case-insensitive matching)
                    for attr_word, attr_score in word_attributions.items():
                        if attr_word.lower() == word_lower:
                            score += abs(attr_score)
                            words_found += 1
                            break
                
                # Salva solo se abbiamo trovato almeno metà delle parole
                if words_found >= len(words_in_action) / 2:
                    action_scores[action] = score
        
        return action_scores
    
    def aggregate_across_dataset(
        self,
        results: List[Dict],
        by_class: bool = True
    ) -> Dict:
        """
        Aggrega action attributions su tutto il dataset
        
        Args:
            results: Lista risultati da IntegratedGradientsExplainer.explain_batch
            by_class: Se True, separa per classe
            
        Returns:
            Dict con statistiche aggregate per azione
        """
        if by_class:
            class_0_actions = {}
            class_1_actions = {}
            
            for result in results:
                # Trova azioni in questo testo
                action_scores = self.find_actions_in_text(
                    result['text'],
                    result['word_attributions']
                )
                
                # Aggiungi a classe appropriata
                target_dict = (class_0_actions if result['predicted_label'] == 0 
                              else class_1_actions)
                
                for action, score in action_scores.items():
                    if action not in target_dict:
                        target_dict[action] = {
                            'scores': [],
                            'count': 0
                        }
                    target_dict[action]['scores'].append(score)
                    target_dict[action]['count'] += 1
            
            # Calcola statistiche
            def compute_stats(action_dict):
                stats = {}
                for action, data in action_dict.items():
                    if data['scores']:
                        stats[action] = {
                            'mean_score': np.mean(data['scores']),
                            'std_score': np.std(data['scores']),
                            'count': data['count'],
                            'total_score': np.sum(data['scores'])
                        }
                return stats
            
            return {
                'class_0': compute_stats(class_0_actions),
                'class_1': compute_stats(class_1_actions)
            }
        
        else:
            all_actions = {}
            
            for result in results:
                action_scores = self.find_actions_in_text(
                    result['text'],
                    result['word_attributions']
                )
                
                for action, score in action_scores.items():
                    if action not in all_actions:
                        all_actions[action] = {
                            'scores': [],
                            'count': 0
                        }
                    all_actions[action]['scores'].append(score)
                    all_actions[action]['count'] += 1
            
            # Statistiche
            stats = {}
            for action, data in all_actions.items():
                if data['scores']:
                    stats[action] = {
                        'mean_score': np.mean(data['scores']),
                        'std_score': np.std(data['scores']),
                        'count': data['count'],
                        'total_score': np.sum(data['scores'])
                    }
            
            return {'all': stats}
    
    def get_top_actions(
        self,
        aggregated_results: Dict,
        top_k: int = 25,
        sort_by: str = 'mean_score'
    ) -> Dict:
        """
        Estrae top-K azioni più importanti
        
        Args:
            aggregated_results: Output di aggregate_across_dataset
            top_k: Numero di azioni da estrarre
            sort_by: Metrica per ordinamento ('mean_score', 'count', 'total_score')
            
        Returns:
            Dict con top-K azioni per classe
        """
        result = {}
        
        for class_key, actions in aggregated_results.items():
            if not actions:
                result[class_key] = {}
                continue
            
            # Ordina per metrica scelta
            sorted_actions = sorted(
                actions.items(),
                key=lambda x: x[1][sort_by],
                reverse=True
            )[:top_k]
            
            result[class_key] = dict(sorted_actions)
        
        return result


if __name__ == "__main__":
    print("Clinical Action Aggregator module")
    
    # Test esempio
    aggregator = ClinicalActionAggregator()
    print(f"\n📋 Default clinical actions: {len(aggregator.clinical_actions)}")
    print("\nExamples:")
    for action in aggregator.clinical_actions[:5]:
        print(f"  - {action}")
