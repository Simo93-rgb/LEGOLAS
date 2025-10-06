"""
Clinical Token Mapper per GALADRIEL.

Strategia REPLACE: sostituisce procedure cliniche complete con token atomici
Esempio: "Pacemaker carrier follow-up visit" → "[CLIN_001]"
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ClinicalTokenMapper:
    """
    Mapper per convertire procedure cliniche in token atomici.
    
    Utilizza translation_cache.json per creare mapping da procedure complete
    a token atomici, migliorando la coesione dell'attention analysis in XAI.
    """
    
    def __init__(self, translation_cache_path: str = "data/translation_cache.json"):
        """
        Inizializza il mapper clinico.
        
        STRATEGIA SEMPLICE: Crea un token per OGNI traduzione inglese nel translation_cache.
        Nessun filtro - mapping 1:1 tra procedure e token atomici.
        
        Args:
            translation_cache_path: Percorso al file translation_cache.json
        """
        self.translation_cache_path = Path(translation_cache_path)
        
        # Mappings
        self.procedure_to_token: Dict[str, str] = {}
        self.token_to_procedure: Dict[str, str] = {}
        self.italian_to_token: Dict[str, str] = {}
        
        # Statistiche
        self.total_procedures: int = 0
        
        # Inizializza mappings
        self._load_clinical_procedures()
        self._create_token_mappings()
        
        logger.info(f"🏥 ClinicalTokenMapper inizializzato:")
        logger.info(f"   📊 Procedure totali: {self.total_procedures}")
        logger.info(f"   �️  Token atomici creati: {len(self.procedure_to_token)}")
        logger.info(f"   📈 Coverage: 100% (no filtering)")
    
    def _load_clinical_procedures(self) -> None:
        """Carica TUTTE le procedure cliniche dal translation_cache.json - nessun filtro."""
        
        if not self.translation_cache_path.exists():
            logger.error(f"❌ File translation_cache non trovato: {self.translation_cache_path}")
            return
        
        try:
            with open(self.translation_cache_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
            
            self.total_procedures = len(translation_data)
            logger.info(f"📥 Caricati {self.total_procedures} mappings dal translation_cache")
            
            # NESSUN FILTRO: Prendi TUTTO dal translation_cache
            self.clinical_procedures = {}
            
            for italian_text, english_text in translation_data.items():
                # Accetta TUTTE le procedure dal translation_cache
                self.clinical_procedures[english_text] = italian_text
            
            logger.info(f"✅ Mapping completo: {len(self.clinical_procedures)} procedure cliniche")
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento translation_cache: {e}")
            self.clinical_procedures = {}
    
    def _create_token_mappings(self) -> None:
        """Crea i mappings da procedure a token atomici."""
        
        if not self.clinical_procedures:
            logger.warning("⚠️  Nessuna procedura clinica da mappare")
            return
        
        # Ordina procedure per lunghezza (dal più lungo al più corto)
        # Importante per evitare sostituzioni parziali
        sorted_procedures = sorted(
            self.clinical_procedures.keys(), 
            key=len, 
            reverse=True
        )
        
        # Crea token atomici sequenziali
        for idx, english_procedure in enumerate(sorted_procedures, 1):
            token_id = f"[CLIN_{idx:03d}]"
            italian_procedure = self.clinical_procedures[english_procedure]
            
            # Mappings bidirezionali
            self.procedure_to_token[english_procedure] = token_id
            self.token_to_procedure[token_id] = english_procedure
            self.italian_to_token[italian_procedure] = token_id
        
        logger.info(f"✅ Creati {len(self.procedure_to_token)} token atomici")
        
        # NUOVO: Salva automaticamente il mapping su file
        self._save_token_mappings()
        
        # Log delle prime 5 procedure per debug
        for i, (procedure, token) in enumerate(list(self.procedure_to_token.items())[:5]):
            logger.info(f"   {token}: {procedure[:50]}{'...' if len(procedure) > 50 else ''}")
    
    def replace_procedures_with_tokens(self, text: str) -> str:
        """
        Sostituisce procedure cliniche nel testo con token atomici.
        
        Args:
            text: Testo originale della storia clinica
            
        Returns:
            Testo con procedure sostituite da token atomici
        """
        if not self.procedure_to_token:
            return text
        
        processed_text = text
        replacements_made = []
        
        # Sostituisci dal più lungo al più corto per evitare sostituzioni parziali
        for procedure, token in self.procedure_to_token.items():
            if procedure in processed_text:
                processed_text = processed_text.replace(procedure, token)
                replacements_made.append((procedure, token))
        
        # Log delle sostituzioni per debug
        if replacements_made:
            logger.debug(f"🔄 Sostituzioni effettuate in testo (case_id non disponibile):")
            for procedure, token in replacements_made[:3]:  # Prime 3
                logger.debug(f"   '{procedure}' → '{token}'")
            if len(replacements_made) > 3:
                logger.debug(f"   ... e altre {len(replacements_made) - 3} sostituzioni")
        
        return processed_text
    
    def get_token_interpretation(self, token: str) -> Optional[str]:
        """
        Ottieni l'interpretazione leggibile di un token atomico.
        
        Args:
            token: Token atomico (es. "[CLIN_001]")
            
        Returns:
            Procedura clinica originale o None se non trovata
        """
        return self.token_to_procedure.get(token)
    
    def get_all_tokens(self) -> List[str]:
        """Restituisce lista di tutti i token atomici creati."""
        return list(self.token_to_procedure.keys())
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """
        Restituisce riassunto dei mappings creati.
        
        Returns:
            Dizionario con statistiche e esempi
        """
        if not self.procedure_to_token:
            return {
                'total_tokens': 0,
                'examples': [],
                'stats': {
                    'total_procedures_in_cache': self.total_procedures,
                    'coverage': '0%'
                }
            }
        
        # Esempi dei primi 5 mappings
        examples = []
        for i, (procedure, token) in enumerate(list(self.procedure_to_token.items())[:5]):
            examples.append({
                'token': token,
                'procedure': procedure,
                'italian': self.clinical_procedures.get(procedure, 'N/A'),
                'length': len(procedure)
            })
        
        return {
            'total_tokens': len(self.procedure_to_token),
            'examples': examples,
            'stats': {
                'total_procedures_in_cache': self.total_procedures,
                'coverage': f"{len(self.procedure_to_token)}/{self.total_procedures} (100%)",
                'longest_procedure': max(self.procedure_to_token.keys(), key=len) if self.procedure_to_token else '',
                'shortest_procedure': min(self.procedure_to_token.keys(), key=len) if self.procedure_to_token else ''
            }
        }
    
    def save_mappings(self, output_path: str) -> None:
        """
        Salva i mappings su file per riferimento futuro.
        
        Args:
            output_path: Percorso file di output
        """
        mapping_data = {
            'procedure_to_token': self.procedure_to_token,
            'token_to_procedure': self.token_to_procedure,
            'italian_to_token': self.italian_to_token,
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_procedures_in_cache': self.total_procedures,
                'total_unique_procedures': len(self.clinical_procedures),
                'total_tokens_created': len(self.procedure_to_token),
                'coverage': '100% (no filtering)',
                'translation_cache_source': str(self.translation_cache_path),
                'mapping_strategy': 'REPLACE - ogni procedura inglese → token atomico [CLIN_XXX]',
                'sorting_method': 'lunghezza decrescente (evita sostituzioni parziali)'
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Mappings salvati in: {output_file}")

    def _save_token_mappings(self) -> None:
        """Salva automaticamente i mappings nel file standard."""
        output_path = "data/clinical_token_mappings.json"
        self.save_mappings(output_path)


def create_clinical_token_mapper(**kwargs) -> ClinicalTokenMapper:
    """
    Factory function per creare ClinicalTokenMapper con configurazione default.
    
    Args:
        **kwargs: Parametri per ClinicalTokenMapper
        
    Returns:
        Istanza configurata di ClinicalTokenMapper
    """
    return ClinicalTokenMapper(**kwargs)


# Test e demo
if __name__ == "__main__":
    # Demo del mapper
    print("🏥 GALADRIEL - Clinical Token Mapper Demo")
    print("=" * 50)
    
    # Crea mapper
    mapper = ClinicalTokenMapper()
    
    # Mostra riassunto
    summary = mapper.get_mapping_summary()
    print(f"\n📊 Riassunto Mappings:")
    print(f"   🏷️  Token totali: {summary['total_tokens']}")
    print(f"   📋 Procedure filtrate: {summary['stats']['filtered_procedures']} / {summary['stats']['total_procedures_in_cache']}")
    
    print(f"\n🔍 Esempi Token:")
    for example in summary['examples']:
        print(f"   {example['token']}: {example['procedure']}")
    
    # Test sostituzione
    test_text = """
    Patient underwent CT of the skull without/with contrast medium.
    Then had a Pacemaker carrier follow-up visit.
    Final step was Infectious disease consultation.
    """
    
    print(f"\n🧪 Test Sostituzione:")
    print(f"Testo originale: {test_text.strip()}")
    
    processed_text = mapper.replace_procedures_with_tokens(test_text)
    print(f"Testo processato: {processed_text.strip()}")
    
    # Salva mappings
    mapper.save_mappings("data/clinical_token_mappings.json")
    
    print(f"\n✅ Demo completata!")
