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
        """
        Preprocess an input image of a page for OCR (Optical Character Recognition).

        Parameters:
        img (np.array): A NumPy array representing the input image.

        Returns:
        np.array: A preprocessed image suitable for OCR.

        Example:
        ```
        # Assuming 'image' is a NumPy array representing the input image
        preprocessed_image = self._preprocess_page(image)
        # Perform OCR on 'preprocessed_image'...
        ```
        This method takes an input image of a page, applies a series of preprocessing operations, and returns a
        preprocessed image suitable for Optical Character Recognition (OCR). The preprocessing includes resizing,
        contrast adjustment, thresholding, and morphological operations to enhance the text for better OCR results.
        """
        img = np.uint8(img)
        img = cv2.resize(img, None, fx=1.2, fy=1.2)

        # increasing constract
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        img = self._closing_operation(img, 2)

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = self._closing_operation(img, 2)

        return img

    def _closing_operation(self, img:np.array, iter=1):
        """
        Apply a morphological closing operation to an input image.

        Parameters:
        img (np.array): A NumPy array representing the input image.
        iterations (int, optional): The number of iterations for the closing operation. Default is 1.

        Returns:
        np.array: The image after the closing operation.

        Example:
        ```
        # Assuming 'image' is a NumPy array representing the input image
        closed_image = self._closing_operation(image, iterations=2)
        # Perform further processing on 'closed_image'...
        ```
        This method applies a morphological closing operation to the input image, which involves eroding and dilating the
        image with a square-shaped kernel. The number of iterations for the operation can be specified using the 'iterations'
        parameter, with the default being 1.
        """
        kernel = np.ones((1, 1), np.uint8)
        page = cv2.erode(img, kernel, iterations=iter)
        return cv2.dilate(page, kernel, iterations=iter)
    
    def _filter_ocr_results(self, ocr_output:dict, confiance:int=50):
        """
        Filter OCR (Optical Character Recognition) results based on confidence and content.

        Parameters:
        ocr_output (dict): A dictionary containing OCR results with 'text' and 'conf' keys.
        confidence (int, optional): The confidence threshold for retaining OCR results. Default is 50.

        Returns:
        str: The filtered and concatenated OCR results.

        Example:
        ```
        # Assuming 'ocr_results' is a dictionary with OCR results
        filtered_text = self._filter_ocr_results(ocr_results, confidence=60)
        # Process the filtered text...
        ```
        This method filters OCR results based on confidence scores and content. It retains OCR results with confidence
        scores higher than the specified 'confidence' threshold and removes results that consist of only non-word characters.
        The filtered results are concatenated into a single string.
        """
        confs = np.array(ocr_output['conf'])
        words = np.array(ocr_output['text'])
        words_filtered = [words[i] for i, conf in enumerate(confs) if conf > confiance and not re.match(r'\W+', words[i])]

        return ' '.join(words_filtered)

    def get_pages(self):
        """
        Extract and preprocess pages from a PDF document.

        Returns:
        list: A list of preprocessed page images.

        Example:
        ```
        # Assuming 'pdf_processor' is an instance of a PDF processing class
        preprocessed_pages = pdf_processor.get_pages()
        for page in preprocessed_pages:
            # Perform OCR or further processing on each page...
        ```
        This method extracts pages from a PDF document and preprocesses each page image. The preprocessing includes resizing,
        contrast adjustment, thresholding, and morphological operations to enhance the text for better OCR results.
        The preprocessed pages are returned as a list of images.
        """
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
        """
        Extract and process text from pages of a PDF document.

        Returns:
        str: A concatenated string of text extracted from the PDF pages.

        Example:
        ```
        # Assuming 'pdf_processor' is an instance of a PDF processing class
        extracted_text = pdf_processor.get_texts()
        # Process the extracted text...
        ```
        This method extracts text from the pages of a PDF document. It preprocesses each page image and performs OCR
        (Optical Character Recognition) to recognize the text. The recognized text is filtered based on confidence and content,
        and the results from all pages are concatenated into a single string.
        """
        pages = self.get_pages()
        ocr_outputs = [pytesseract.image_to_data(page, config=self.ocr_config, lang='por', output_type=pytesseract.Output.DICT) for page in pages]
        pages_texts = [self._filter_ocr_results(ocr_out) for ocr_out in ocr_outputs]

        return ' '.join(pages_texts)