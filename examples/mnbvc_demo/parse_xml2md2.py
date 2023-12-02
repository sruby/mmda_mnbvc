import xml.etree.ElementTree as ET

# Define the namespaces used in the XML document
namespaces = {
    'tei': 'http://www.tei-c.org/ns/1.0',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'xlink': 'http://www.w3.org/1999/xlink'
    # Add other namespaces if needed
}

def convert(xml_content):
    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Extract and format the title
    title = root.find('.//tei:titleStmt/tei:title', namespaces=namespaces).text
    md = f'# {title}\n'

    # Extract and format the authors' names
    authors = root.findall('./tei:teiHeader//tei:author/tei:persName', namespaces=namespaces)
    for author in authors:
        forename = author.find('tei:forename', namespaces=namespaces)
        if forename is not None:
            forename = forename.text
        surname = author.find('tei:surname', namespaces=namespaces)

        if surname is not None:
            surname = surname.text
        md += f'- {forename} {surname}\n'
    md += '\n'

    # Extract and format the abstract
    abstract = root.find('.//tei:abstract', namespaces=namespaces)
    md += f'## Abstract\n'
    abstract_s_list = abstract.findall('.//tei:s', namespaces=namespaces)
    for abstract_s in abstract_s_list:
        md += f'{abstract_s.text}'

    # extract and format the body
    body_div_list = root.findall('./tei:text/tei:body/tei:div', namespaces=namespaces)
    for body_div in body_div_list:
        div_head = body_div.find('./tei:head', namespaces=namespaces)
        head_n = div_head.get('n')
        head_text = div_head.text
        if head_n is not None:
            head_text = head_n + ' '+ head_text
        md += f'\n## {head_text}\n'

        body_dev_p_list = body_div.findall('./tei:p', namespaces=namespaces)
        for body_div_p in body_dev_p_list:
            body_div_p_s_list = body_div_p.findall('tei:s', namespaces=namespaces)
            for body_div_p_s in body_div_p_s_list :
                md += f'{body_div_p_s.text}'
            md += '\n'
    return md

# parse
XML_PATH = './e5910c027af0ee9c1901c57f6579d903aedee7f4.xml'
xml = open(XML_PATH, encoding='utf-8').read()
mdContent = convert(xml)
with open('output2.md', 'w', encoding='utf-8') as f:
        f.write(mdContent)
