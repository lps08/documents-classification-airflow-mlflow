import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import re

class PDF2Text(object):
    def __init__(self, pdf_path:str, num_pages:int=None, ocr_config:str = '--oem 1 --psm 3') -> None:
        self.ocr_config = ocr_config
        self.pdf_path = pdf_path
        self.image_quality = 500
        self.num_pages = num_pages

    def _preprocess_page(self, img:np.array):
        img = np.uint8(img)
        img = cv2.resize(img, None, fx=1.2, fy=1.2)

        # increasing constract
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        img = self._closing_operation(img, 2)

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = self._closing_operation(img, 2)

        return img

    def _closing_operation(self, img:np.array, iter=1):
        kernel = np.ones((1, 1), np.uint8)
        page = cv2.erode(img, kernel, iterations=iter)
        return cv2.dilate(page, kernel, iterations=iter)
    
    def _filter_ocr_results(self, ocr_output:dict, confiance:int=50):
        confs = np.array(ocr_output['conf'])
        words = np.array(ocr_output['text'])
        words_filtered = [words[i] for i, conf in enumerate(confs) if conf > confiance and not re.match(r'\W+', words[i])]

        return ' '.join(words_filtered)

    def get_pages(self):
        preprocessed_pages = []

        # checking if the number of pages has setted by the user
        if self.num_pages != None:
            pages = convert_from_path(self.pdf_path, self.image_quality, first_page=1, last_page=self.num_pages, thread_count=4, grayscale=True, strict=True)
        else:
            pages = convert_from_path(self.pdf_path, self.image_quality, thread_count=4, grayscale=True, strict=True)

        for page in pages:
            preprocessed_pages.append(self._preprocess_page(page))

        return preprocessed_pages

    def get_texts(self) -> str:
        pages = self.get_pages()
        ocr_outputs = [pytesseract.image_to_data(page, config=self.ocr_config, lang='por', output_type=pytesseract.Output.DICT) for page in pages]
        pages_texts = [self._filter_ocr_results(ocr_out) for ocr_out in ocr_outputs]

        return ' '.join(pages_texts)