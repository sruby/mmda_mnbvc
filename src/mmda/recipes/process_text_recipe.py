"""
Miracleyin
MNBVC Mo

"""

import logging

logger = logging.getLogger(__name__)

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.sklearn_predictors.svm_word_predictor import SVMWordPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.recipes.recipe import Recipe
from mmda.types import *


class ProcessTextRecipe(Recipe):
    """
    basic pure text PDF Recipe
    for /Producer contains "ProcessText Group"
    stage 1: pure text, but span, bbox annotate
    """
    def __init__(
        self,
        svm_word_predictor_path: str = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/mmda/models/svm_word_predictor.tar.gz",
    ):
        logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()
        self.word_predictor = SVMWordPredictor.from_path(svm_word_predictor_path)
        logger.info("Finished instantiating recipe")

    def from_path(self, pdfpath: str) -> Document:
        logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdfpath) # get token rows pages

        logger.info("Predicting words...")
        words = self.word_predictor.predict(document=doc)
        doc.annotate(words=words)

        return doc


    
