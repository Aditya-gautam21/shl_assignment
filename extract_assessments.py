import os
import re
import json

def extract_assessments_from_md(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    assessments = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for a line that starts with '|' and has at least 3 '|' (so at least 2 columns)
        if line.strip().startswith('|') and line.count('|') >= 3:
            # This might be the header of a table
            header_line = line.strip()
            # Move to the next line to see if it's a separator
            i += 1
            if i >= len(lines):
                break
            sep_line = lines[i].strip()
            # Check if it's a separator line (only contains |, -, :, and spaces)
            if re.match(r'^[\s|:-]+$', sep_line):
                # Now we have a table, read the header
                header_cells = [cell.strip() for cell in header_line.split('|')[1:-1]]
                # Normalize header names: remove markdown, make lowercase, replace spaces with underscores
                header_names = []
                for cell in header_cells:
                    # Remove any markdown formatting like **, *, `, etc.
                    clean = re.sub(r'[\*`_]', '', cell).strip().lower()
                    # Replace spaces and hyphens with underscore
                    clean = re.sub(r'[\s\-]+', '_', clean)
                    header_names.append(clean)
                # Now read rows until we hit a line that doesn't start with '|'
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|'):
                    row_line = lines[i].strip()
                    row_cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                    if len(row_cells) == len(header_names):
                        row_dict = {}
                        for idx, header in enumerate(header_names):
                            row_dict[header] = row_cells[idx]
                        # Now extract the fields we need: name, test_type, url
                        # We expect the header to have been normalized, but we can be flexible.
                        # Let's map possible header names to our fields.
                        name = row_dict.get('name', '') or row_dict.get('assessment_name', '') or row_dict.get('assessment', '')
                        test_type = row_dict.get('test_type', '') or row_dict.get('type', '') or row_dict.get('testtype', '')
                        url = row_dict.get('url', '') or row_dict.get('link', '')
                        # If url is a markdown link, extract the URL
                        url_match = re.search(r'\]\((http[^)]+)\)', url)
                        if url_match:
                            url = url_match.group(1)
                        # If we have at least a name and a url, we consider it an assessment
                        if name and url:
                            assessment = {
                                'name': name,
                                'test_type': test_type,
                                'url': url,
                                'description': '',  # we don't have description in the table
                                'duration': row_dict.get('duration', '') or row_dict.get('time', ''),
                                'languages': row_dict.get('languages', '') or row_dict.get('language', '')
                            }
                            assessments.append(assessment)
                    i += 1
                # We are now at the line after the table, continue outer loop
                continue
            else:
                # Not a separator line, so not a table we can parse. Move on.
                i += 1
                continue
        else:
            i += 1
    return assessments

def main():
    sample_dir = './sample_conversations/GenAI_SampleConversations'
    all_assessments = []
    seen_urls = set()
    for filename in os.listdir(sample_dir):
        if filename.endswith('.md'):
            filepath = os.path.join(sample_dir, filename)
            assessments = extract_assessments_from_md(filepath)
            for a in assessments:
                if a['url'] and a['url'] not in seen_urls:
                    seen_urls.add(a['url'])
                    all_assessments.append(a)
    print(f"Extracted {len(all_assessments)} unique assessments")
    # Save to shl_catalog.json
    with open('shl_catalog.json', 'w', encoding='utf-8') as f:
        json.dump(all_assessments, f, indent=2, ensure_ascii=False)
    print("Saved to shl_catalog.json")

if __name__ == '__main__':
    main()