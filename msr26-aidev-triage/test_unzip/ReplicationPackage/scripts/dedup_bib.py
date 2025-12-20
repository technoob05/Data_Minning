
import re
import sys

def deduplicate_bib(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by @Entry type (assuming standard formatting)
    # This regex looks for @word{ at start of line or after newline
    entries = re.split(r'(?m)^@', content)
    
    unique_entries = []
    seen_keys = set()
    
    # First chunk is usually preamble/comments before first entry
    preamble = entries[0]
    
    cleaned_entries = [preamble]

    for entry in entries[1:]:
        # Reconstruct the @
        full_entry = '@' + entry
        
        # Extract key: @Type{key,
        match = re.search(r'@[a-zA-Z]+\{([^,]+),', full_entry)
        if match:
            key = match.group(1).strip()
            if key in seen_keys:
                print(f"Skipping duplicate key: {key}")
                continue
            seen_keys.add(key)
            cleaned_entries.append(full_entry)
        else:
            # comments or malformed
            cleaned_entries.append(full_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(cleaned_entries))
    
    print(f"Deduplicated bibliography. Saved {len(seen_keys)} unique entries.")

if __name__ == "__main__":
    deduplicate_bib('d:/Work/DataMinning/Data_Minning/msr26-aidev-triage/paper/sample-base.bib', 'd:/Work/DataMinning/Data_Minning/msr26-aidev-triage/paper/sample-base.bib')
