#!/usr/bin/env python3
"""
Convert synthetic data files from current format to JSONL conversation format.

Current format:
{
  "text_samples": [
    "Text content here...",
    ...
  ]
}

Target format (JSONL - one JSON object per line):
{"prompt": [{"role": "user", "content": "Text content here..."}]}
{"prompt": [{"role": "user", "content": "Another text here..."}]}
...
"""

import json
import os
from pathlib import Path

def convert_file(input_path, output_path):
    """Convert a single synthetic data file to the new format."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # extract text samples
        text_samples = data.get('text_samples', [])
        
        # convert to new format and write as jsonl
        entry_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in text_samples:
                converted_entry = {
                    "prompt": [
                        {
                            "role": "user",
                            "content": text
                        }
                    ]
                }
                # write each entry as a separate line (jsonl format)
                f.write(json.dumps(converted_entry, ensure_ascii=False) + '\n')
                entry_count += 1
        
        print(f"Converted {input_path} -> {output_path} ({entry_count} entries)")
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

def main():
    """Convert all synthetic data files in the synthetic_data directory."""
    synthetic_data_dir = Path("synthetic_data")
    converted_dir = Path("synthetic_data_converted")
    
    # create output directory if it doesn't exist
    converted_dir.mkdir(exist_ok=True)
    
    # find all json files in synthetic_data directory
    json_files = list(synthetic_data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in synthetic_data directory")
        return
    
    print(f"Found {len(json_files)} files to convert:")
    for file_path in json_files:
        print(f"  {file_path}")
    
    print("\nConverting files...")
    
    # convert each file
    for input_file in json_files:
        # change extension to .jsonl
        output_filename = f"converted_{input_file.stem}.jsonl"
        output_file = converted_dir / output_filename
        convert_file(input_file, output_file)
    
    print(f"\nConversion complete! Converted files saved in {converted_dir}/")

if __name__ == "__main__":
    main() 