import os
import yaml
import json
from typing import Dict, List, Any, Union


def read_yaml(filepath: str) -> Dict:
    """Read YAML file and return as dictionary."""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {filepath}: {e}")
            raise


def save_yaml(config: Dict, filepath: str) -> None:
    """Save dictionary as YAML file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        try:
            yaml.dump(config, f, default_flow_style=False)
        except yaml.YAMLError as e:
            print(f"Error saving YAML file {filepath}: {e}")
            raise


def read_json(filepath: str) -> Dict:
    """Read JSON file and return as dictionary."""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file {filepath}: {e}")
            raise


def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary as JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        try:
            json.dump(data, f, indent=2)
        except TypeError as e:
            print(f"Error saving JSON file {filepath}: {e}")
            raise


def read_manifest(filepath: str) -> List[Dict]:
    """Read ASR manifest file and return as list of dictionaries.
    
    Each line in the manifest is expected to be in the format:
    path_to_audio_file|transcription|speaker_id
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 2:
                item = {
                    'audio_filepath': parts[0],
                    'text': parts[1],
                }
                if len(parts) >= 3:
                    item['speaker_id'] = parts[2]
                data.append(item)
    
    return data


def write_manifest(data: List[Dict], filepath: str) -> None:
    """Write list of dictionaries to ASR manifest file.
    
    Each item in the list should have 'audio_filepath' and 'text' keys,
    and optionally a 'speaker_id' key.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            line = f"{item['audio_filepath']}|{item['text']}"
            if 'speaker_id' in item:
                line += f"|{item['speaker_id']}"
            f.write(line + '\n') 