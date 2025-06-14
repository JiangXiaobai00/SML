import json
import argparse

def extract_unique_entries(input_path: str, output_path: str):
    """
    Loads data from a JSON file, extracts all unique string entries from its
    list values, and writes them to a text file.

    Args:
        input_path (str): The path to the input JSON file.
        output_path (str): The path for the output TXT file.
    """
    print(f"[*] Reading JSON file: {input_path}")
    
    try:
        # Use a 'with' statement to ensure the file is properly closed after reading.
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[!] Error: File '{input_path}' not found. Please check the file path.")
        return
    except json.JSONDecodeError:
        print(f"[!] Error: The file '{input_path}' is not a valid JSON file.")
        return

    # Use a set to store all entries, which automatically handles duplicates.
    unique_entries_set = set()

    # The script expects the top-level JSON structure to be a dictionary.
    if not isinstance(data, dict):
        print(f"[!] Error: The top-level structure of the JSON file is not a dictionary.")
        return

    # Iterate over all values in the dictionary.
    for entry_list in data.values():
        # We expect each value to be a list of strings.
        if isinstance(entry_list, list):
            for entry in entry_list:
                # Ensure the item in the list is a string before processing.
                if isinstance(entry, str):
                    # Strip leading/trailing whitespace to clean up the data.
                    cleaned_entry = entry.strip()
                    # Add to the set only if the string is not empty after cleaning.
                    if cleaned_entry:
                        unique_entries_set.add(cleaned_entry)

    # Convert the set back to a list and sort it alphabetically for consistent output.
    unique_entries_list = sorted(list(unique_entries_set))

    print(f"[*] Found {len(unique_entries_list)} unique entries.")
    print(f"[*] Writing results to file: {output_path}")

    # Write the unique entries to the output file, one entry per line.
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in unique_entries_list:
                f.write(entry + '\n')
        print(f"[+] Successfully created {output_path}")
    except IOError as e:
        print(f"[!] Error: Could not write to file '{output_path}'. Reason: {e}")

# This block runs when the script is executed directly from the command line.
if __name__ == "__main__":
    # 1. Initialize the ArgumentParser
    parser = argparse.ArgumentParser(
        description="Extract unique string entries from a JSON file and save them to a TXT file."
    )

    # 2. Define command-line arguments
    # Positional argument: input_file (required)
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the input JSON file (e.g., data.json)."
    )
    
    # Optional argument: --output / -o
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="unique_entries.txt",
        help="Path for the output TXT file (default: unique_entries.txt)."
    )

    # 3. Parse the arguments provided by the user
    args = parser.parse_args()

    # 4. Call the main function with the parsed arguments
    extract_unique_entries(args.input_file, args.output)