import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict

import fitz  # type: ignore
from pdf2image import convert_from_path


@dataclass(slots=True)
class PDFImage:
    img_base64: str
    ext: str
    page_number: int

    @property
    def image_url(self) -> str:
        # ext normalisieren für MIME-Type
        ext = self.ext.lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{self.img_base64}"


def load_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Load PDF metadata as dict.
    """
    with fitz.open(pdf_path) as doc:
        meta = dict(doc.metadata or {})
        meta.setdefault("page_count", doc.page_count)
        meta.setdefault("is_encrypted", doc.is_encrypted)
    return meta


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
