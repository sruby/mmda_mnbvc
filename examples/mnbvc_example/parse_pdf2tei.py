# %load_ext autoreload
# %autoreload 2

PDF_PATH = '../../tests/fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4.pdf'
from mmda.parsers import PDFPlumberParser
from mmda.types import Document
# PDF to text
pdf_plumber = PDFPlumberParser()
doc: Document = pdf_plumber.parse(input_pdf_path=PDF_PATH)
doc.fields

from mmda.parsers.grobid_augment_existing_document_parser import GrobidAugmentExistingDocumentParser
parser = GrobidAugmentExistingDocumentParser(config_path='../../src/mmda/parsers/grobid.config', check_server=True)

doc = parser.parse(PDF_PATH, doc, ".")