"""
XES file parser for the LEGOLAS pipeline.
Handles loading and processing of XES event log files into structured data.
"""

import pandas as pd
import pm4py
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from utils.types import Event, PatientTrace, ClassificationTarget


class XESParser:
    """
    Parser for XES files containing patient event logs.
    
    This class handles the conversion from XES format to structured
    PatientTrace objects suitable for the LEGOLAS pipeline.
    """
    
    def __init__(self, xes_file_path: str):
        """
        Initialize the XES parser.
        
        Args:
            xes_file_path: Path to the XES file to parse
            
        Raises:
            FileNotFoundError: If the XES file doesn't exist
        """
        self.xes_file_path = Path(xes_file_path)
        if not self.xes_file_path.exists():
            raise FileNotFoundError(f"XES file not found: {xes_file_path}")
        
        self._log = None
        self._dataframe = None
    
    def load_xes_file(self) -> Tuple[object, pd.DataFrame]:
        """
        Load the XES file and convert to DataFrame.
        
        Returns:
            Tuple containing the original log object and converted DataFrame
            
        Raises:
            Exception: If XES file cannot be loaded or parsed
        """
        try:
            self._log = pm4py.read_xes(str(self.xes_file_path))
            # Se il log è già un DataFrame, non riconvertire
            if isinstance(self._log, pd.DataFrame):
                self._dataframe = self._log.copy()
            else:
                self._dataframe = pm4py.convert_to_dataframe(self._log)
            # Rinomina colonne chiave per compatibilità con itertuples
            self._dataframe = self._dataframe.rename(columns={
                'case:concept:name': 'case_concept_name',
                'concept:name': 'concept_name',
                'time:timestamp': 'time_timestamp'
            })
            # Validate required columns (ora senza i due punti)
            required_cols = ['case_concept_name', 'concept_name', 'time_timestamp']
            missing_cols = [col for col in required_cols if col not in self._dataframe.columns]
            if missing_cols:
                raise ValueError(f"Required columns missing from XES: {missing_cols}")
            return self._log, self._dataframe
        except Exception as e:
            raise Exception(f"Failed to load XES file: {str(e)}")
    
    def extract_patient_traces(self) -> List[PatientTrace]:
        """
        Extract patient traces from the loaded XES data (ottimizzato: parallelo + itertuples).
        """
        if self._dataframe is None or self._log is None:
            raise RuntimeError("XES file must be loaded first. Call load_xes_file().")

        import concurrent.futures
        grouped = list(self._dataframe.groupby('case_concept_name'))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            traces = list(executor.map(lambda args: self._create_patient_trace(str(args[0]), args[1]), grouped))
        return traces
    
    def _create_patient_trace(self, case_id: str, group: pd.DataFrame) -> PatientTrace:
        """
        Create a PatientTrace from a group of events.
        
        Args:
            case_id: The case identifier
            group: DataFrame containing events for this case
            
        Returns:
            PatientTrace object
        """
        # Sort events by timestamp (colonna già rinominata)
        sorted_group = group.sort_values('time_timestamp').reset_index(drop=True)
        
        # Extract events
        events = self._extract_events_from_group(sorted_group)
        
        # Extract case-level metadata
        patient_metadata = self._extract_case_metadata(str(case_id))
        
        # Create PatientTrace
        return PatientTrace(
            case_id=str(case_id),
            events=events,
            classification=patient_metadata.get('classification'),
            patient_age=patient_metadata.get('age'),
            patient_gender=patient_metadata.get('gender'),
            additional_metadata=patient_metadata
        )
    
    def _extract_events_from_group(self, group: pd.DataFrame) -> List[Event]:
        """
        Extract Event objects from a DataFrame group (ottimizzato: usa itertuples).
        """
        events = []
        for row in group.itertuples(index=False):
            ts = getattr(row, 'time_timestamp')
            # Solo timestamp validi (datetime o pd.Timestamp)
            if isinstance(ts, (pd.Timestamp, datetime)):
                timestamp = ts
            else:
                try:
                    timestamp = pd.to_datetime(ts)
                except Exception:
                    continue  # Salta evento se timestamp non valido
            if not isinstance(timestamp, (pd.Timestamp, datetime)) or pd.isna(timestamp):
                continue  # Salta evento se timestamp non valido
            
            event = Event(
                activity=getattr(row, 'concept_name'),
                timestamp=timestamp,
                case_id=str(getattr(row, 'case_concept_name')),
                additional_attributes={}  # Salta attributi aggiuntivi per evitare errori di shadowing
            )
            events.append(event)
        return events
    
    def _extract_event_attributes(self, row) -> Dict[str, str]:
        """
        Extract additional attributes from an event row (robusto: lavora sempre su dict).
        """
        attributes = {}
        standard_cols = {'case_concept_name', 'concept_name', 'time_timestamp'}
        python_builtins = {
            'int', 'str', 'float', 'bool', 'complex', 'bytes', 'date', 'datetime', 'timedelta',
            'Timestamp', 'Timedelta', 'timedelta64', 'datetime64', 'complexfloating', 'floating', 'integer'
        }
        for col, value in row.items():
            if col in python_builtins:
                continue
            if self._should_include_attribute(col, value, standard_cols):
                attributes[col] = str(value)
        return attributes
    
    def _should_include_attribute(self, col: str, value: Any, standard_cols: set) -> bool:
        """
        Check if an attribute should be included (robusto: ignora errori di chiamabilità e pd.notna, filtra tipi problematici).
        """
        if col in standard_cols:
            return False
        # Escludi colonne e valori che ombreggiano tipi built-in o pandas problematici
        problematic_names = {
            'int', 'str', 'float', 'bool', 'complex', 'bytes', 'date', 'datetime', 'timedelta',
            'Timestamp', 'Timedelta', 'timedelta64', 'datetime64', 'complexfloating', 'floating', 'integer'
        }
        if col in problematic_names:
            return False
        if callable(value):
            return False
        try:
            return pd.notna(value)
        except Exception:
            return False
    
    def _extract_case_metadata(self, case_id: str) -> Dict[str, Any]:
        """
        Extract case-level metadata including classification and patient info.
        
        Args:
            case_id: The case identifier
            
        Returns:
            Dictionary containing case metadata
        """
        if self._dataframe is None:
            return {}
            
        metadata = {}
        case_data = self._dataframe[self._dataframe['case_concept_name'] == case_id]
        
        # Extract classification and demographics from DataFrame
        metadata = self._extract_from_dataframe(case_data, metadata)
        
        # Fallback: try to extract from original trace attributes
        if not metadata.get('classification'):
            metadata = self._extract_from_trace_attributes(case_id, metadata)
        
        return metadata
    
    def _extract_from_dataframe(self, case_data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from DataFrame case data.
        
        Args:
            case_data: DataFrame containing case data
            metadata: Existing metadata dictionary
            
        Returns:
            Updated metadata dictionary
        """
        # Extract classification
        metadata = self._extract_classification_from_df(case_data, metadata)
        
        # Extract demographics
        metadata = self._extract_demographics_from_df(case_data, metadata)
        
        return metadata
    
    def _extract_classification_from_df(self, case_data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract classification from DataFrame."""
        if 'case:class_ricovero_dimissioni' in case_data.columns:
            class_value = case_data['case:class_ricovero_dimissioni'].iloc[0]
            if pd.notna(class_value):
                metadata['classification'] = ClassificationTarget(int(class_value))
        return metadata
    
    def _extract_demographics_from_df(self, case_data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographics from DataFrame."""
        # Check for age range (Fascia_Eta)
        if 'case:Fascia_Eta' in case_data.columns:
            age_range = case_data['case:Fascia_Eta'].iloc[0]
            if pd.notna(age_range):
                metadata['age_range'] = str(age_range)
                metadata['age'] = self._parse_age_from_range(age_range)
        # Legacy age field support
        if 'case:age' in case_data.columns:
            age = case_data['case:age'].iloc[0]
            if pd.notna(age):
                metadata['age'] = int(age)
        # Gender (if available)
        if 'case:gender' in case_data.columns:
            gender = case_data['case:gender'].iloc[0]
            if pd.notna(gender):
                metadata['gender'] = str(gender)
        
        return metadata
    
    def _extract_from_trace_attributes(self, case_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from original trace attributes as fallback.
        
        Args:
            case_id: The case identifier
            metadata: Existing metadata dictionary
            
        Returns:
            Updated metadata dictionary
        """
        if self._log is None:
            return metadata
            
        for trace in self._log:
            if self._is_matching_trace(trace, case_id):
                metadata = self._extract_trace_attributes(trace, metadata)
                break
        
        return metadata
    
    def _is_matching_trace(self, trace: Any, case_id: str) -> bool:
        """Check if trace matches the given case_id."""
        if not hasattr(trace, 'attributes'):
            return False
        trace_case_id = trace.attributes.get('concept:name', '')
        return str(trace_case_id) == str(case_id)
    
    def _extract_trace_attributes(self, trace: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from a trace object."""
        # Extract classification
        class_value = trace.attributes.get('class_ricovero_dimissioni')
        if class_value is not None:
            metadata['classification'] = ClassificationTarget(int(class_value))
        
        # Extract other attributes
        for key, value in trace.attributes.items():
            if key not in {'concept:name', 'class_ricovero_dimissioni'}:
                metadata[key] = value
        
        return metadata
    
    def _parse_age_from_range(self, age_range: str) -> Optional[int]:
        """
        Parse numeric age from age range string.
        
        Args:
            age_range: Age range string (e.g., "55-59", "65+", "18-24")
            
        Returns:
            Approximate numeric age (middle of range) or None if cannot parse
        """
        if not age_range or pd.isna(age_range):
            return None
        
        age_str = str(age_range).strip()
        
        # Handle range format like "55-59"
        if '-' in age_str:
            try:
                parts = age_str.split('-')
                if len(parts) == 2:
                    min_age = int(parts[0].strip())
                    max_age = int(parts[1].strip())
                    return (min_age + max_age) // 2  # Return middle of range
            except ValueError:
                pass
        
        # Handle "65+" format
        if '+' in age_str:
            try:
                base_age = int(age_str.replace('+', '').strip())
                return base_age + 5  # Add 5 years as approximation
            except ValueError:
                pass
        
        # Handle single number
        try:
            return int(age_str)
        except ValueError:
            return None
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics
            
        Raises:
            RuntimeError: If XES file hasn't been loaded yet
        """
        if self._dataframe is None:
            raise RuntimeError("XES file must be loaded first. Call load_xes_file().")
        
        # Count cases by classification
        classification_counts = {}
        if 'case:class_ricovero_dimissioni' in self._dataframe.columns:
            case_classifications = (
                self._dataframe[['case_concept_name', 'case:class_ricovero_dimissioni']]
                .drop_duplicates()
            )
            classification_counts = case_classifications['case:class_ricovero_dimissioni'].value_counts().to_dict()
        stats = {
            'total_events': len(self._dataframe),
            'total_cases': self._dataframe['case_concept_name'].nunique(),
            'unique_activities': self._dataframe['concept_name'].nunique(),
            'classification_distribution': classification_counts,
            'date_range': {
                'start': self._dataframe['time_timestamp'].min(),
                'end': self._dataframe['time_timestamp'].max()
            }
        }
        return stats
