import argparse
import re
from pathlib import Path

def normalize_description(desc: str) -> str:
    """
    Normalizes a description string to ensure consistency by replacing
    typographer's quotes with standard straight quotes.
    """
    return desc.replace('’', "'").replace('‘', "'")

def load_required_descriptions(filepath: Path) -> set[str] | None:
    """
    Loads and normalizes all required descriptions from a text file.
    """
    if not filepath.is_file():
        print(f"Error: Source file '{filepath}' not found or is not a file.")
        return None
        
    print(f"Loading and normalizing required descriptions from '{filepath}'...")
    with open(filepath, 'r', encoding='utf-8') as f:
        descriptions = {normalize_description(line.strip()) for line in f if line.strip()}
    print(f"-> Found {len(descriptions)} unique, normalized descriptions.")
    return descriptions


def parse_code_into_blocks(filepath: Path, debug: bool = False) -> dict[str, str] | None:
    """
    Parses a Python file to extract code blocks. This version correctly
    handles unescaped single quotes within the description string.
    """
    if not filepath.is_file():
        print(f"Error: Python code file '{filepath}' not found or is not a file.")
        return None

    print(f"Parsing and normalizing code blocks from '{filepath}'...")
    
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return None

    processed_blocks = {}
    
    # Greedy regex to capture the full content of a single-quoted string on one line
    block_start_pattern = re.compile(r"description\s*=\s*'(.*)'")
    
    matches = list(block_start_pattern.finditer(content))
    if not matches:
        print("Warning: No lines matching \"description = '...'\" were found in the code file.")
        return {}

    if debug:
        print("\n[DEBUG] Extracted Descriptions:")
        
    for i, match in enumerate(matches):
        description = match.group(1)
        normalized_description = normalize_description(description)

        if debug:
            print(f"  - '{normalized_description}'")

        start_pos = match.start()
        end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(content)
        code_block = content[start_pos:end_pos].strip()
        
        processed_blocks[normalized_description] = code_block

    print(f"-> Parsed {len(processed_blocks)} code blocks.")
    return processed_blocks

def sanitize_filename(name: str) -> str:
    """Removes characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)


def main(descriptions_path: Path, code_path: Path, output_path: Path, debug: bool) -> None:
    """Main function to run the check and split process."""
    required = load_required_descriptions(descriptions_path)
    processed_blocks = parse_code_into_blocks(code_path, debug)

    if required is None or processed_blocks is None:
        print("\nAborting due to file loading errors.")
        return

    print("\n--- Verification Stage ---")
    processed_descriptions = set(processed_blocks.keys())
    missing_descriptions = required - processed_descriptions

    if missing_descriptions:
        print(f"Error: The check failed. Found {len(missing_descriptions)} required description(s) missing from the code file:")
        for i, desc in enumerate(sorted(list(missing_descriptions)), 1):
            print(f"  {i}. {desc}")
        print("\nAborting. No files will be written.")
        return

    print("Verification successful. All required descriptions are present in the code file.")

    print("\n--- Splitting and Saving Stage ---")
    output_path.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for desc_normalized, code_block in processed_blocks.items():
        if desc_normalized in required:
            # --- THIS IS THE MODIFIED LINE ---
            # Save the file with a .txt extension instead of .py
            safe_filename = sanitize_filename(desc_normalized) + ".txt"
            # -----------------------------------
            file_save_path = output_path / safe_filename
            file_save_path.write_text(code_block, encoding='utf-8')
            saved_count += 1

    print(f"\n--- Final Report ---")
    print(f"Successfully saved {saved_count} code blocks to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifies descriptions and splits a code file into individual scenario files."
    )
    parser.add_argument("descriptions_file", type=Path, help="Path to the descriptions text file.")
    parser.add_argument("code_file", type=Path, help="Path to the Python code file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the split files.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print all parsed descriptions.")
    
    args = parser.parse_args()
    main(args.descriptions_file, args.code_file, args.output_dir, args.debug)