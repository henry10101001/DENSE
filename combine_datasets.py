import json
import os
from pathlib import Path

def combine_datasets():
    """combine all converted jsonl datasets into a single file"""
    
    # directory containing converted datasets
    input_dir = Path("synthetic_data_converted")
    output_file = "synthetic_data_converted/combined_dataset.jsonl"
    
    # find all jsonl files in the directory
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("no jsonl files found in synthetic_data_converted directory")
        return
    
    print(f"found {len(jsonl_files)} files to combine:")
    for file in jsonl_files:
        print(f"  - {file.name}")
    
    combined_data = []
    source_stats = {}
    
    # process each file
    for file_path in jsonl_files:
        print(f"\nprocessing {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    try:
                        data = json.loads(line)
                        # add source file info to track origin
                        data['source_file'] = file_path.name
                        combined_data.append(data)
                        count += 1
                    except json.JSONDecodeError as e:
                        print(f"  warning: skipping invalid json line in {file_path.name}: {e}")
        
        source_stats[file_path.name] = count
        print(f"  loaded {count} entries")
    
    # write combined dataset
    print(f"\nwriting combined dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # print summary
    total_entries = len(combined_data)
    print(f"\ncombination complete!")
    print(f"total entries: {total_entries}")
    print("\nbreakdown by source:")
    for source, count in source_stats.items():
        percentage = (count / total_entries) * 100
        print(f"  {source}: {count} entries ({percentage:.1f}%)")
    
    print(f"\ncombined dataset saved as: {output_file}")

if __name__ == "__main__":
    combine_datasets() 