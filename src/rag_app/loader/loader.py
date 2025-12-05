import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List

import fitz  # type: ignore
from pdf2image import convert_from_path
from unstructured.partition.pdf import partition_pdf


@dataclass(slots=True)
class PDFImage:
    img_base64: str
    ext: str  # z.B. "png", "jpeg", "jpg"
    page_number: int

    @property
    def image_url(self) -> str:
        # ext normalisieren für MIME-Type
        ext = self.ext.lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{self.img_base64}"


@dataclass(slots=True)
class PDFText:
    text: str
    page_number: int


@dataclass(slots=True)
class PDFTable:
    text: str | None
    html: str | None
    page_number: int | None


def load_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Load PDF metadata as dict.
    """
    with fitz.open(pdf_path) as doc:
        meta = dict(doc.metadata or {})
        meta.setdefault("page_count", doc.page_count)
        meta.setdefault("is_encrypted", doc.is_encrypted)
    return meta


def load_texts_from_pdf(pdf_path: str) -> List[PDFText]:
    """
    Load every PDF page as one Text object.

    Returns:
        List[Text]: Eine Liste von Text-Objekten, je Seite ein Eintrag.
                    page_number ist als String und 1-basiert ("1", "2", ...).
    """
    pages: List[PDFText] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(
                PDFText(
                    text=text.strip(),
                    page_number=page_index,
                )
            )

    return pages


def load_imgs_from_pdf(pdf_path: str) -> List[PDFImage]:
    images: List[PDFImage] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")  # z.B. "png", "jpeg", "jpg"
                b64_str = base64.b64encode(image_bytes).decode("ascii")

                images.append(
                    PDFImage(
                        img_base64=b64_str,
                        ext=ext,
                        page_number=page_index,
                    )
                )

    return images



def load_page_imgs_from_pdf(pdf_path: str) -> list[PDFImage]:
    """
    Load every PDF page as an image (Base64-kodiert) und liefere eine Liste von PDFImage.
    Jede Seite wird als PNG in Graustufen gerendert.
    """
    imgs = convert_from_path(pdf_path, dpi=120)

    imgs_gray = [img.convert("L") for img in imgs]

    pdf_imgs: list[PDFImage] = []

    for page_number, img in enumerate(imgs_gray, start=1):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        pdf_imgs.append(
            PDFImage(
                img_base64=img_base64,
                ext="png",
                page_number=page_number,
            )
        )

    return pdf_imgs

def load_tables_from_pdf(pdf_path: str) -> list[PDFTable]:
    "detectron2_onnx: used for  document layout"
    "tesseract: for ocr"

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=False,
        infer_table_structure=True,
        pdf_infer_table_structure=True,
        languages=["eng", "deu"],
    )

    tables = [e for e in elements if e.category == "Table"]

    pdf_tables = []
    for t in tables:
        pdf_table = PDFTable(t.text, t.metadata.text_as_html, t.metadata.page_number)

        pdf_tables.append(pdf_table)

    return pdf_tables


