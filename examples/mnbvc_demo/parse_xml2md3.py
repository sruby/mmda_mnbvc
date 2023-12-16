import xml.etree.ElementTree as ET

def parse_author(author_element):
    forename = author_element.find('.//forename').text
    surname = author_element.find('.//surname').text
    return f"{forename} {surname}\n"

def parse_formula(formula_element):
    return f"$$\n{formula_element.text.strip()}\n$$\n"

def parse_table(table_element):
    markdown = ""
    for row in table_element.findall('.//row'):
        cells = row.findall('.//cell')
        markdown += '| ' + ' | '.join(cell.text for cell in cells) + ' |\n'
    header_sep = '| ' + ' | '.join('---' for _ in cells) + ' |\n'
    return header_sep + markdown + '\n'

def parse_figure(figure_element):
    label = figure_element.find('.//label').text
    fig_desc = figure_element.find('.//figDesc').text
    # Assuming the figure element contains a URL attribute or similar
    figure_url = figure_element.get('url', '')
    return f"![Figure {label}]({figure_url})\n*{fig_desc}*\n"

def parse_reference(reference_element):
    # Assuming the reference element contains a target attribute
    ref_target = reference_element.get('target', '')
    return f"[{reference_element.text.strip()}]({ref_target})\n"

# Parse the XML document
tree = ET.parse('e5910c027af0ee9c1901c57f6579d903aedee7f4.xml')
root = tree.getroot()

# Define a Markdown document
markdown_document = "# Title of the Document\n\n"

# Process the XML elements
for elem in root.iter():
    if elem.tag.endswith('titleStmt'):
        for title in elem.findall('.//title'):
            markdown_document += f"## {title.text.strip()}\n\n"
    elif elem.tag.endswith('author'):
        markdown_document += parse_author(elem)
    elif elem.tag.endswith('abstract'):
        markdown_document += "### Abstract\n\n"
        markdown_document += elem.find('.//p').text.strip() + '\n\n'
    elif elem.tag.endswith('body'):
        markdown_document += "### Body\n\n"
        for p in elem.findall('.//p'):
            markdown_document += p.text.strip() + '\n\n'
    elif elem.tag.endswith('formula'):
        markdown_document += parse_formula(elem)
    elif elem.tag.endswith('figure'):
        markdown_document += parse_figure(elem)
    elif elem.tag.endswith('table'):
        markdown_document += parse_table(elem)
    elif elem.tag.endswith('ref'):
        markdown_document += parse_reference(elem)

# Save or print the Markdown document
with open('document.md', 'w', encoding='utf-8') as md_file:
    md_file.write(markdown_document)