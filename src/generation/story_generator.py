"""
Generatore di storie basato su template per GALADRIEL.
Converte le tracce di pazienti in narrazioni testuali usando template predefiniti.
Supporta token clinici atomici per migliorare l'analisi XAI.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import os
import logging

from utils.types import PatientTrace, Event, PatientStory, ClassificationTarget
from src.data.clinical_token_mapper import ClinicalTokenMapper

logger = logging.getLogger(__name__)


class StoryGenerator:
    """Generatore di storie narrative per tracce di pazienti usando template."""

    def __init__(self, format_style: str = "narrative", enable_clinical_tokens: bool = False):
        """
        Inizializza il generatore con i template e le mappature delle attività.
        
        Args:
            format_style: Stile di formattazione ("bullet_points" o "narrative")
            enable_clinical_tokens: Se abilitare i token clinici atomici (SPERIMENTALE)
        """
        self.format_style = format_style
        self.enable_clinical_tokens = enable_clinical_tokens
        self.activity_templates = self._load_activity_templates()
        
        # Carica vocabolario medico (traduzioni inglesi) dalla cartella data
        self.medical_vocabulary = self._load_berting_vocabulary()
        self.activity_mapping = self._create_activity_mapping()
        
        # Carica translation cache per mappatura italiano→inglese
        self.translation_cache = self._load_translation_cache()
        
        if not self.translation_cache or not self.medical_vocabulary or not self.activity_mapping:
            print("⚠️  Attenzione: Mancano dati di traduzione o vocabolario medico!")
            raise ValueError("Dati di traduzione o vocabolario medico non caricati correttamente.")

        # NUOVO: Inizializza clinical token mapper se abilitato
        self.clinical_mapper: Optional[ClinicalTokenMapper] = None
        if self.enable_clinical_tokens:
            try:
                self.clinical_mapper = ClinicalTokenMapper()
                mapper_summary = self.clinical_mapper.get_mapping_summary()
                logger.info(f"🏥 Clinical Token Mapper abilitato: {mapper_summary['total_tokens']} token atomici")
            except Exception as e:
                logger.warning(f"⚠️  Errore inizializzazione Clinical Token Mapper: {e}")
                self.enable_clinical_tokens = False
        
        # LEAKAGE REMOVED: Non includere più l'esito finale 
        # Originale causa data leakage:
        # self.outcome_mapping = {
        #     ClassificationTarget.DISCHARGED: "dimesso a casa",
        #     ClassificationTarget.ADMITTED: "ricoverato in ospedale"
        # }
        # 
        # Il modello deve imparare dall'ontologia medica, non da frasi esplicite!

    def _load_berting_vocabulary(self) -> Dict[str, str]:
        """Carica il dizionario delle traduzioni IT->EN dalla cartella data."""
        # Path corretto: src/generation/ -> radice progetto -> data/translation_cache.json
        translation_path = Path(__file__).parent.parent.parent / "data" / "translation_cache.json"
        
        if not translation_path.exists():
            print(f"⚠️  Dizionario traduzioni non trovato: {translation_path}")
            return {}
            
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                translation_dict = json.load(f)
            # Estrai solo i valori (traduzioni inglesi) per il vocabolario
            english_activities = {}
            for idx, english_activity in enumerate(translation_dict.values()):
                english_activities[str(idx)] = english_activity
            print(f"✅ Caricato dizionario traduzioni: {len(english_activities)} attività mediche inglesi")
            return english_activities
        except Exception as e:
            print(f"⚠️  Errore caricamento dizionario traduzioni: {e}")
            return {}
    
    def _create_activity_mapping(self) -> Dict[str, str]:
        """Crea mappatura da attività italiane a terminologia BERTing (inglese)."""
        if not self.medical_vocabulary:
            return {}
            
        # Crea mappatura case-insensitive per matching
        mapping = {}
        for idx, english_activity in self.medical_vocabulary.items():
            # Aggiungi sia versione originale che lowercase
            mapping[english_activity.lower()] = english_activity
            
        return mapping
    
    def _standardize_activity_name(self, activity: str) -> str:
        """
        Traduce un'attività italiana in inglese usando translation_cache.
        
        Args:
            activity: Nome attività in italiano dal dataset XES
            
        Returns:
            Nome attività tradotto in inglese
        """
        # Traduzione diretta IT->EN usando translation_cache
        if activity in self.translation_cache:
            english_activity = self.translation_cache[activity]
            print(f"Traduzione: '{activity}' -> '{english_activity}'")
            return english_activity
        
        # Se non trovato, prova variazioni di case
        for italian_key, english_value in self.translation_cache.items():
            if activity.upper() == italian_key.upper():
                print(f"Traduzione (case-insensitive): '{activity}' -> '{english_value}'")
                return english_value
                
        # Se non trovato, restituisce l'originale con warning
        # print(f"⚠️  Traduzione non trovata per: '{activity}'")
        return activity
    
    def _load_translation_cache(self) -> Dict[str, str]:
        """Load translation cache for direct Italian→English mapping."""
        # Path corretto: src/generation/ -> radice progetto -> data/translation_cache.json
        cache_path = Path(__file__).parent.parent.parent / "data" / "translation_cache.json"
        
        if not cache_path.exists():
            print(f"⚠️  Translation cache not found: {cache_path}")
            return {}
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"✅ Loaded translation cache: {len(cache)} Italian→English mappings")
            return cache
        except Exception as e:
            print(f"⚠️  Error loading translation cache: {e}")
            return {}
    
    def _translate_to_english(self, italian_activity: str) -> str:
        """
        Translate Italian activity to English using translation cache.
        
        Args:
            italian_activity: Activity name in Italian
            
        Returns:
            English activity name or original if not found
        """
        if not self.translation_cache:
            return italian_activity
            
        # Direct lookup in translation cache
        english_activity = self.translation_cache.get(italian_activity.upper())
        if english_activity:
            return english_activity
            
        # If not found, try case variations
        for italian_key, english_value in self.translation_cache.items():
            if italian_activity.upper() == italian_key.upper():
                return english_value
                
        # If still not found, return original
        return italian_activity

    def _translate_activity_if_needed(self, activity: str) -> str:
        """
        Traduce l'attività dall'italiano all'inglese se presente nella cache.
        
        Args:
            activity: Nome dell'attività (possibilmente in italiano)
            
        Returns:
            Nome dell'attività tradotto in inglese o originale se non trovato
        """
        if activity in self.translation_cache:
            translated = self.translation_cache[activity]
            print(f"Traduzione automatica: '{activity}' -> '{translated}'")
            return translated
        return activity

    def _load_activity_templates(self) -> Dict[str, str]:
        """
        Load templates for different categories of clinical activities (English).
        
        Returns:
            Dictionary with templates for each activity category
        """
        return {
            # Medical visits
            "VISIT": "The patient underwent {activity}",
            "FIRST VISIT": "The patient had {activity}",
            "VISIT CONTROL": "{activity} was performed",
            "VISIT ANESTHESIOLOGICAL": "The patient underwent {activity}",
            
            # Radiography and imaging
            "RX": "{activity} was performed",
            "TC": "{activity} was performed",
            "RM": "{activity} was performed",
            "ULTRASOUND": "{activity} was performed",
            "ANGIO": "{activity} was performed",
            "SCINTIGRAPHY": "{activity} was performed",
            "MAMMOGRAPHY": "{activity} was performed",
            
            # Invasive procedures
            "BIOPSY": "{activity} was performed",
            "AGOBIOPSY": "{activity} was performed",
            "ENDOSCOPY": "{activity} was performed",
            "COLONOSCOPY": "{activity} was performed",
            "GASTROSCOPY": "{activity} was performed",
            "EGDS": "{activity} was performed",
            "BRONCHOSCOPY": "{activity} was performed",
            "CYSTOSCOPY": "{activity} was performed",
            
            # Treatments and therapies
            "RADIOTHERAPY": "The patient received {activity}",
            "HEMODIALYSIS": "{activity} was performed",
            "DIALYSIS": "{activity} was performed",
            "HEMOFILTRATION": "{activity} was performed",
            "THERAPY": "The patient received {activity}",
            "REHABILITATION": "{activity} was performed",
            "COUNSELING": "{activity} was provided",
            
            # Surgical procedures
            "POSITIONING": "{activity} was performed",
            "CATHETERIZATION": "{activity} was performed",
            "DRAINAGE": "{activity} was performed",
            "VERTEBROPLASTY": "{activity} was performed",
            "EMBOLIZATION": "{activity} was performed",
            "STENT": "{activity} was positioned",
            
            # Laboratory tests and monitoring
            "ELECTROCARDIOGRAM": "{activity} was performed",
            "HEMOGASANALYSIS": "{activity} was performed",
            "SPIROMETRY": "{activity} was performed",
            "TEST": "{activity} was performed",
            "EVOKED POTENTIALS": "{activity} was performed",
            "ELECTROENCEPHALOGRAM": "{activity} was performed",
            
            # Monitoring and controls
            "CONTROL": "{activity} was performed",
            "MONITORING": "{activity} was performed",
            "EVALUATION": "{activity} was performed",
            "CONSULTATION": "{activity} was provided",
            "ADVICE": "{activity} was provided",
            
            # Operating room
            "TSRM IN OPERATING ROOM": "Multislice tomosynthesis (TSRM) was performed in operating room",
            
            # Default for uncategorized activities
            "DEFAULT": "{activity} was performed"
        }

    def _categorize_activity(self, activity: str) -> str:
        """
        Categorizza un'attività clinica per scegliere il template appropriato.
        
        Args:
            activity: Nome dell'attività clinica
            
        Returns:
            Categoria dell'attività
        """
        activity_upper = activity.upper()
        
        # Controllo per corrispondenze esatte
        if activity_upper in self.activity_templates:
            return activity_upper
            
        # Controllo per parole chiave
        for category in self.activity_templates.keys():
            if category != "DEFAULT" and category in activity_upper:
                return category
                
        return "DEFAULT"

    def _format_activity_description(self, activity: str) -> str:
        """
        Formatta la descrizione di un'attività usando il template appropriato.
        
        Args:
            activity: Nome dell'attività clinica
            
        Returns:
            Descrizione formattata dell'attività (senza prefissi verbali)
        """
        # STANDARDIZZA attività secondo vocabolario BERTing PRIMA di formattare
        standardized_activity = self._standardize_activity_name(activity)
        
        category = self._categorize_activity(standardized_activity)
        template = self.activity_templates[category]
        
        # USA la terminologia standardizzata (mantenendo maiuscole/minuscole originali)
        description = template.format(activity=standardized_activity)
        
        # Remove common prefixes to have only activity description
        prefixes_to_remove = [
            " was performed",
            " was executed", 
            "The patient underwent ",
            "The patient had ",
            "The patient received ",
            " was provided",
            " was positioned"
        ]
        
        for prefix in prefixes_to_remove:
            if description.endswith(prefix):
                description = description[:-len(prefix)]
                break
            elif description.startswith(prefix):
                description = description[len(prefix):]
                break
                
        return description

    def _calculate_time_elapsed(self, start_time: datetime, current_time: datetime) -> int:
        """
        Calcola il tempo trascorso in secondi dall'inizio del caso.
        
        Args:
            start_time: Timestamp di inizio
            current_time: Timestamp corrente
            
        Returns:
            Secondi trascorsi
        """
        return int((current_time - start_time).total_seconds())

    def _create_patient_intro(self, trace: PatientTrace) -> str:
        """
        Create patient introduction with demographic information (English).
        
        Args:
            trace: Patient trace
            
        Returns:
            Patient introduction
        """
        intro_parts = []
        
        # Demographic information
        if trace.patient_gender:
            gender_en = "male" if trace.patient_gender == "M" else "female"
            if trace.patient_age:
                intro_parts.append(f"The patient, a {trace.patient_age}-year-old {gender_en},")
            else:
                intro_parts.append(f"The patient, {gender_en},")
        elif trace.patient_age:
            intro_parts.append(f"The patient, {trace.patient_age} years old,")
        else:
            intro_parts.append("The patient")
        
        # LEAKAGE PREVENTION: No longer include final outcome in introduction
        # Narrative format must not reveal classification!
        intro_parts.append("underwent a series of examinations and interventions during hospitalization.")
            
        return " ".join(intro_parts)

    def _format_event_group(self, event_group: List[Event], start_time: datetime) -> str:
        """
        Formatta un gruppo di eventi simultanei.
        
        Args:
            event_group: Gruppo di eventi simultanei
            start_time: Timestamp di inizio del caso
            
        Returns:
            Descrizione formattata del gruppo di eventi
        """
        time_elapsed = self._calculate_time_elapsed(start_time, event_group[0].timestamp)
        
        if len(event_group) == 1:
            # Evento singolo
            activity_desc = self._format_activity_description(event_group[0].activity)
            return f"{activity_desc} dopo {time_elapsed} secondi"
        else:
            # Eventi multipli simultanei
            activity_descriptions = [
                self._format_activity_description(event.activity) 
                for event in event_group
            ]
            
            if len(activity_descriptions) == 2:
                combined = " e ".join(activity_descriptions)
            else:
                combined = ", ".join(activity_descriptions[:-1]) + f", e {activity_descriptions[-1]}"
                
            return f"{combined} dopo {time_elapsed} secondi"

    def _group_simultaneous_events(self, sorted_events: List) -> List[List]:
        """
        Raggruppa eventi che hanno lo stesso timestamp.
        
        Args:
            sorted_events: Lista di eventi ordinati per timestamp
            
        Returns:
            Lista di gruppi di eventi simultanei
        """
        if not sorted_events:
            return []
            
        groups = []
        current_group = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            if event.timestamp == current_group[0].timestamp:
                # Stesso timestamp - aggiungi al gruppo corrente
                current_group.append(event)
            else:
                # Timestamp diverso - inizia nuovo gruppo
                groups.append(current_group)
                current_group = [event]
        
        # Aggiungi l'ultimo gruppo
        groups.append(current_group)
        
        return groups

    def _create_narrative_sequence(self, trace: PatientTrace) -> str:
        """
        Creates the narrative sequence of clinical events in bullet points format
        with timestamp grouping and "concurrent" indication.
        
        Args:
            trace: Patient trace
            
        Returns:
            Narrative sequence of events in bullet points format (English)
        """
        if not trace.events:
            return "No specific clinical events were recorded."
            
        # Sort events by timestamp
        sorted_events = sorted(trace.events, key=lambda x: x.timestamp)
        # start_time = sorted_events[0].timestamp  # non usato per delta relativo
        
        # Group events by timestamp
        event_groups = self._group_simultaneous_events(sorted_events)
        
        # Create narrative in bullet points format
        narrative_lines = []
        
        # Delta temporale relativo al gruppo precedente
        prev_time = event_groups[0][0].timestamp
        
        for idx, group in enumerate(event_groups):
            time_seconds = 0 if idx == 0 else int((group[0].timestamp - prev_time).total_seconds())
            
            if len(group) == 1:
                # Single event
                activity_desc = self._format_activity_description(group[0].activity)
                activity_desc = activity_desc[0].upper() + activity_desc[1:] if activity_desc else ""
                narrative_lines.append(f"- After {time_seconds} seconds: {activity_desc}")
            else:
                # Simultaneous events - format with sub-bullets
                narrative_lines.append(f"- After {time_seconds} seconds:")
                
                for i, event in enumerate(group):
                    activity_desc = self._format_activity_description(event.activity)
                    activity_desc = activity_desc[0].upper() + activity_desc[1:] if activity_desc else ""
                    
                    if i == 0:
                        # First action of the group
                        narrative_lines.append(f"    - {activity_desc}")
                    else:
                        # Subsequent actions with "(concurrent)"
                        narrative_lines.append(f"    - {activity_desc} (concurrent)")
            
            # Aggiorna tempo precedente al primo evento del gruppo corrente
            prev_time = group[0].timestamp
        
        return "\n".join(narrative_lines)

    def _create_narrative_sequence_narrative(self, trace: PatientTrace) -> str:
        """
        Creates the narrative sequence of clinical events in discursive format for BERT:
        paragraphs with complete sentences describing events and their temporality.
        
        Args:
            trace: Patient trace
            
        Returns:
            Narrative sequence of events in discursive format (English)
        """
        if not trace.events:
            return "No specific clinical events were recorded."
            
        # Sort events by timestamp
        sorted_events = sorted(trace.events, key=lambda x: x.timestamp)
        # start_time = sorted_events[0].timestamp  # non usato per delta relativo
        
        # Group events by timestamp
        event_groups = self._group_simultaneous_events(sorted_events)
        
        # Create discursive narrative
        narrative_paragraphs = []
        
        # Delta temporale relativo al gruppo precedente
        prev_time = event_groups[0][0].timestamp
        
        for idx, group in enumerate(event_groups):
            time_seconds = 0 if idx == 0 else int((group[0].timestamp - prev_time).total_seconds())
            
            if len(group) == 1:
                # Single event
                activity_desc = self._format_activity_description(group[0].activity)
                
                if time_seconds == 0:
                    # First event
                    paragraph = f"The {activity_desc} was performed at the beginning of hospitalization, after 0 seconds."
                else:
                    # Other single events (delta rispetto al precedente)
                    time_formatted = f"{time_seconds:,}"
                    paragraph = f"After {time_formatted} seconds, {activity_desc} was performed."
                    
                narrative_paragraphs.append(paragraph)
                
            else:
                # Simultaneous events
                time_formatted = f"{time_seconds:,}"
                activities = []
                
                for event in group:
                    activity_desc = self._format_activity_description(event.activity)
                    activities.append(activity_desc)
                
                if len(activities) == 2:
                    activities_text = f"{activities[0]} and {activities[1]}"
                elif len(activities) > 2:
                    activities_text = ", ".join(activities[:-1]) + f", and {activities[-1]}"
                else:
                    activities_text = activities[0]
                
                if len(group) > 1:
                    paragraph = f"After {time_formatted} seconds, the following examinations were performed simultaneously: {activities_text}."
                else:
                    paragraph = f"After {time_formatted} seconds, {activities_text} was performed."
                    
                narrative_paragraphs.append(paragraph)
            
            # Aggiorna tempo precedente al primo evento del gruppo corrente
            prev_time = group[0].timestamp
        
        return "\n\n".join(narrative_paragraphs)

    def generate_story(self, trace: PatientTrace) -> PatientStory:
        """
        Genera una storia narrativa per una traccia di paziente usando template.
        
        Args:
            trace: Traccia del paziente da convertire in storia
            
        Returns:
            PatientStory con la narrativa generata
        """
        # Crea l'introduzione del paziente
        intro = self._create_patient_intro(trace)
        
        # Scegli il metodo di narrazione in base al formato
        if self.format_style == "narrative":
            narrative = self._create_narrative_sequence_narrative(trace)
            # Per il formato narrativo, combina intro e narrativa con doppio a capo
            complete_story = f"{intro}\n\n{narrative}"
        else:
            # Formato bullet points (default)
            narrative = self._create_narrative_sequence(trace)
            complete_story = f"{intro}\n{narrative}"
        
        # NUOVO: Applica sostituzione con token clinici se abilitata
        if self.enable_clinical_tokens and self.clinical_mapper:
            original_story = complete_story
            complete_story = self.clinical_mapper.replace_procedures_with_tokens(complete_story)
            
            # Log sostituzione per debug (solo se effettivamente cambiato)
            if original_story != complete_story:
                logger.debug(f"🔄 Token clinici applicati per case_id {trace.case_id}")
        
        # LEAKAGE PREVENTION: Rimuovi completamente gli ending espliciti
        # Non aggiungere più conclusioni che rivelano la classificazione!
        # Il modello deve imparare dall'ontologia delle attività mediche.
        
        return PatientStory(
            case_id=trace.case_id,
            story_text=complete_story,
            language="en",  # Generazione in inglese
            classification=trace.classification,
            original_trace=trace
        )

    def generate_batch_stories(self, traces: List[PatientTrace]) -> List[PatientStory]:
        """
        Genera storie per un batch di tracce usando parallelizzazione.
        
        Args:
            traces: Lista di tracce di pazienti
            
        Returns:
            Lista di storie generate
        """
        if not traces:
            return []
        
        # Calcola numero di thread: max_threads() - 2, minimo 1
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count - 2)
        
        print(f"🚀 Generazione parallela di {len(traces)} storie usando {max_workers} thread")
        
        def generate_single_story(trace: PatientTrace) -> PatientStory:
            """Genera una singola storia con gestione errori."""
            try:
                return self.generate_story(trace)
            except Exception as e:
                print(f"⚠️  Errore nella generazione della storia per paziente {trace.case_id}: {e}")
                # Crea una storia di fallback
                return PatientStory(
                    case_id=trace.case_id,
                    story_text=f"Error generating narrative for patient {trace.case_id}.",
                    language="en",  # Fallback in inglese
                    classification=trace.classification,
                    original_trace=trace
                )
        
        # Usa ThreadPoolExecutor per parallelizzazione
        stories = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Sottometti tutti i task
            future_to_trace = {
                executor.submit(generate_single_story, trace): trace 
                for trace in traces
            }
            
            # Raccogli i risultati mantenendo l'ordine originale
            trace_to_story = {}
            for future in future_to_trace:
                trace = future_to_trace[future]
                try:
                    story = future.result()
                    trace_to_story[trace.case_id] = story
                except Exception as e:
                    print(f"⚠️  Errore nel thread per paziente {trace.case_id}: {e}")
                    # Fallback per errori del thread
                    trace_to_story[trace.case_id] = PatientStory(
                        case_id=trace.case_id,
                        story_text=f"Thread error for patient {trace.case_id}.",
                        language="en",
                        classification=trace.classification,
                        original_trace=trace
                    )
            
            # Ricostruisci lista mantenendo ordine originale
            stories = [trace_to_story[trace.case_id] for trace in traces]
        
        print(f"✅ Completata generazione parallela di {len(stories)} storie")
        return stories
