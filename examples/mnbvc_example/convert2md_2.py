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
    # .//tei:titleStmt/tei:title: This is an XPath expression.
    # The . represents the current node (which is the root in this case).
    # The  means select nodes in the document from the current node that match the selection no matter where
    # they are. tei:titleStmt/tei:title means select all title elements in the titleStmt namespace that are children of the current node.
    # //表示后代节点，/表示子节点
    title = root.find('.//tei:titleStmt/tei:title', namespaces=namespaces).text
    md = f'# {title}\n\n'

    # Extract and format the authors' names
    authors = root.findall('.//tei:fileDesc//tei:author/tei:persName', namespaces=namespaces)
    for author in authors:
        forename = author.find('tei:forename', namespaces=namespaces).text
        surname = author.find('tei:surname', namespaces=namespaces).text
        md += f'- {forename} {surname}\n'
    md += '\n'

    # Extract and format the abstract
    abstract = root.find('.//tei:abstract', namespaces=namespaces)
    abstract = root.find('.//tei:abstract', namespaces=namespaces)
    # Find all sentence elements within the abstract
    sentences = abstract.findall('.//tei:s', namespaces=namespaces)
    md += f'## Abstract\n'
    md += ''.join([sentence.text for sentence in sentences if sentence.text is not None])

    # extract and format body
    bodyNode = root.find('./tei:text/tei:body', namespaces=namespaces)
    bodyItems = bodyNode.findall('./tei:div', namespaces=namespaces)
    for bodyItem in bodyItems:
        head = bodyItem.find('./tei:head', namespaces=namespaces)
        attr_number = head.attrib.get('n')
        sentences = bodyItem.findall('.//tei:s', namespaces=namespaces)
        if attr_number is None:
            md += f'\n## {head.text}\n'
        else:
            md += f'\n## {attr_number} {head.text}\n'

        md += ''.join([sentence.text for sentence in sentences if sentence.text is not None])

    return md


XML_PATH = '../../tests/fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4.xml'

xml = open(XML_PATH, encoding='utf-8').read()
mdContent = convert(xml)

with open('output2.md', 'w', encoding='utf-8') as f:
    f.write(mdContent)
