import os
import re
import json
from datasets import Dataset
import pandas as pd

def parse_vrt_file(file_path):
    """
    Parse a VRT file and extract text content.
    
    Args:
        file_path: Path to the VRT file
        
    Returns:
        List of dictionaries with extracted text and metadata
    """
    print(f"Parsing {file_path}...")
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Split by document tags
    doc_pattern = re.compile(r'<text(.*?)>(.*?)</text>', re.DOTALL)
    doc_matches = doc_pattern.findall(content)
    
    for doc_attrs, doc_content in doc_matches:
        # Extract document metadata
        metadata = {}
        attr_pattern = re.compile(r'(\w+)="([^"]*)"')
        for attr, value in attr_pattern.findall(doc_attrs):
            metadata[attr] = value
        
        # Extract sentences/lines
        sentences = []
        current_sentence = []
        
        for line in doc_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Skip XML tags for sentences, paragraphs, etc.
            if line.startswith('<') and line.endswith('>'):
                # If we were building a sentence and hit a new tag, save the sentence
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
                continue
            
            # If it's a token line (not a tag), add to current sentence
            parts = line.split('\t')
            if len(parts) >= 1:
                # Take the first part as the token
                token = parts[0]
                # Skip if it's just a special character
                if token and not token.startswith('<'):
                    current_sentence.append(token)
        
        # Add the last sentence if there is one
        if current_sentence:
            sentences.append(' '.join(current_sentence))
        
        # Create a record
        if sentences:
            text = ' '.join(sentences)
            record = {
                'text': text,
                **metadata
            }
            records.append(record)
    
    print(f"Extracted {len(records)} documents from {file_path}")
    return records

def process_vrt_files(directory, output_dir="./processed_oracc"):
    """
    Process all VRT files in a directory and create a Hugging Face dataset.
    
    Args:
        directory: Directory containing VRT files
        output_dir: Directory to save the processed dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_records = []
    
    # Process each VRT file
    for filename in os.listdir(directory):
        if filename.endswith('.vrt'):
            file_path = os.path.join(directory, filename)
            records = parse_vrt_file(file_path)
            all_records.extend(records)
    
    print(f"Total extracted records: {len(all_records)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Create dataset
    dataset = Dataset.from_pandas(df)
    
    # Split dataset
    dataset_dict = dataset.train_test_split(test_size=0.1)
    
    # Save to disk
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    
    # Also save a sample as JSON for inspection
    sample_size = min(10, len(all_records))
    sample = all_records[:sample_size]
    with open(os.path.join(output_dir, 'sample.json'), 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    return dataset_dict

def create_instruction_dataset(records, output_dir="./oracc_instructions"):
    """
    Convert ORACC records to an instruction dataset format.
    
    Args:
        records: List of dictionaries with text and metadata
        output_dir: Directory to save the instruction dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    instructions = []
    
    # Create instruction templates based on the data
    for record in records:
        text = record.get('text', '')
        if not text:
            continue
            
        # Get metadata
        language = record.get('language', 'unknown')
        period = record.get('period', 'unknown')
        genre = record.get('genre', 'unknown')
        
        # Create various instruction examples
        
        # 1. Translation task
        if language.lower() in ['akkadian', 'sumerian']:
            instructions.append({
                "instruction": f"Translate the following {language} text to English:",
                "input": text,
                "output": "This is a cuneiform text from the " + period + " period, which translates as: [Translation would be here]"
            })
        
        # 2. Analysis task
        instructions.append({
            "instruction": f"Analyze the following {language} text from {period} period:",
            "input": text,
            "output": f"This text is written in {language} from the {period} period. It belongs to the {genre} genre. [More detailed analysis would follow here]"
        })
        
        # 3. Completion task
        words = text.split()
        if len(words) > 10:
            input_text = ' '.join(words[:len(words)//2])
            output_text = ' '.join(words[len(words)//2:])
            
            instructions.append({
                "instruction": f"Complete this {language} text fragment:",
                "input": input_text,
                "output": output_text
            })
    
    # Create dataset
    df = pd.DataFrame(instructions)
    dataset = Dataset.from_pandas(df)
    
    # Split dataset
    dataset_dict = dataset.train_test_split(test_size=0.1)
    
    # Save to disk
    dataset_dict.save_to_disk(output_dir)
    print(f"Instruction dataset saved to {output_dir}")
    
    # Also save a sample as JSON for inspection
    sample_size = min(10, len(instructions))
    sample = instructions[:sample_size]
    with open(os.path.join(output_dir, 'sample_instructions.json'), 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    return dataset_dict

if __name__ == "__main__":
    # Example usage:
    # 1. Process VRT files
    vrt_directory = "./oracc_files"  # Directory containing VRT files
    dataset = process_vrt_files(vrt_directory, "./processed_oracc")
    
    # 2. Create instruction dataset from the processed records
    create_instruction_dataset(dataset["train"], "./oracc_instructions")
    
    print("Dataset preparation completed!")