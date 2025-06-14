import argparse
from pathlib import Path

def consolidate_error_descriptions(error_dir: Path, output_file: Path) -> None:
    """
    Scans a directory for error log files (.txt), extracts their filenames as
    descriptions, and writes them into a single consolidated output file.

    Args:
        error_dir: The directory containing the error log files.
        output_file: The path to the output file where the consolidated list
                     of descriptions will be saved.
    """
    # 1. Validate that the input directory exists
    if not error_dir.is_dir():
        print(f"Error: The specified error directory does not exist or is not a directory.")
        print(f"       Path: {error_dir}")
        return

    print(f"Scanning for error files in: {error_dir}")

    # 2. Find all .txt files in the directory and extract their names
    # The .stem attribute of a Path object gives the filename without the final extension.
    error_descriptions = sorted([p.stem for p in error_dir.glob('*.txt')])

    if not error_descriptions:
        print("No error files (.txt) found in the specified directory.")
        # You might still want to create an empty output file, which the code below will do.
        # Or you can return here if you prefer not to.

    print(f"Found {len(error_descriptions)} error(s). Consolidating into: {output_file}")

    # 3. Ensure the parent directory for the output file exists
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create parent directory for the output file.")
        print(f"       Path: {output_file.parent}")
        print(f"       Details: {e}")
        return

    # 4. Write the collected descriptions to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for desc in error_descriptions:
                f.write(desc + '\n')
    except Exception as e:
        print(f"Error: Could not write to the output file.")
        print(f"       Path: {output_file}")
        print(f"       Details: {e}")
        return

    print("Consolidation complete.")


if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Consolidates error descriptions from individual .txt log files into a single master list."
    )

    # Add positional arguments for the input directory and output file
    parser.add_argument(
        "error_directory",
        type=Path,
        help="The path to the directory containing the error log files (e.g., 'output/sm_predictions/exp100/results/errors')."
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="The full path for the consolidated output .txt file (e.g., 'consolidated_errors.txt')."
    )

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    consolidate_error_descriptions(args.error_directory, args.output_file)