import xml.etree.ElementTree as et
from collections import defaultdict

XML_PATH = '../../tests/fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4.xml'

xml = open(XML_PATH, encoding='utf-8').read()

xml_root = et.fromstring(xml)

# Open markdown file
with open('output.md', 'w', encoding='utf-8') as f:
    # Iterate over all elements in the XML tree
    for elem in xml_root.iter():
        # If element has text, write it to the file
        if elem.text:
            f.write(elem.text + '\n')