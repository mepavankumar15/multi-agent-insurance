import os
import io
import base64
import json
import re
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None

def is_scanned_pdf(file_path: str) -> bool:
    """
    Check if a PDF is a scanned document (image-based) with little to no embedded text.
    Reads the first two pages. If total extracted characters per page is very low, returns True.
    """
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:2]
            if not pages:
                return True # Empty PDF, treat as scanned
            for page in pages:
                text = page.extract_text()
                if text is not None and len(text.strip()) > 20:
                    return False # Found a decent amount of text
        return True # Found very little text on the first two pages
    except Exception as e:
        print(f"Error checking if PDF is scanned: {e}")
        return True # Default to treating as scanned if error occurs


def extract_text_from_pdf(file_path: str) -> str:
    """Extract embedded text from a PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:8]:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"pdfplumber extract failed: {e}")
        return ""


def extract_fields_from_text(text: str) -> dict:
    """Parse common receipt fields from plain text."""
    if not text or not text.strip():
        return {}

    patient_name = None
    m = re.search(r"(?:^|\n)\s*Name:\s*([A-Za-z][A-Za-z\s.'-]{1,60})", text, re.IGNORECASE)
    if m:
        patient_name = m.group(1).strip().split("\n")[0][:80]
        if patient_name.lower() in {"details", "patient details"}:
            patient_name = None
    if not patient_name:
        m = re.search(
            r"(?:claimant|patient name)[:\s-]*([A-Za-z][A-Za-z\s.'-]{1,60})",
            text,
            re.IGNORECASE,
        )
        if m:
            patient_name = m.group(1).strip().split("\n")[0][:80]

    policy_number = None
    m = re.search(r"(POL[-\s]?\d{4,8})", text, re.IGNORECASE)
    if m:
        policy_number = m.group(1).upper().replace(" ", "")

    treatment_date = None
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    if m:
        treatment_date = m.group(1)

    amount = None
    m = re.search(
        r"(?:total|amount|charge|fee|hk\$|\$)[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            amount = float(m.group(1).replace(",", ""))
        except Exception:
            pass

    diagnosis = None
    m = re.search(
        r"(?:diagnosis|treatment|description)[:\s-]*(.+?)(?:\n\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        diagnosis = m.group(1).strip()[:300]

    return {
        "patient_name": patient_name,
        "policy_number": policy_number,
        "treatment_date": treatment_date,
        "diagnosis_or_treatment_description": diagnosis,
        "amount_charged": amount,
        "method": "pdf_text",
        "raw_ocr_text": text[:2000],
    }


def pdf_to_images(file_path: str) -> list:
    """
    Convert a PDF file to a list of PIL Images using PyMuPDF (fitz).
    Uses high DPI (~400) + preprocessing for best OCR quality on scanned receipts.
    """
    if Image is None:
        raise ImportError("Pillow (PIL) is not installed.")
        
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is not installed. Please pip install PyMuPDF.")

    images = []
    try:
        doc = fitz.open(file_path)
        # zoom=5.0 → ~360 DPI, good for medical receipts
        zoom_matrix = fitz.Matrix(5.0, 5.0)

        for page in doc:
            pix = page.get_pixmap(matrix=zoom_matrix)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img.load()

            # Basic preprocessing for OCR
            if img.mode != "L":
                img = img.convert("L")
            # Auto-contrast + slight sharpen
            from PIL import ImageEnhance, ImageFilter, ImageOps
            img = ImageOps.autocontrast(img)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.8)
            img = img.filter(ImageFilter.SHARPEN)

            images.append(img)
        doc.close()
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    return images

def encode_image_base64(img) -> str:
    """Convert PIL image to base64 jpeg string."""
    buffered = io.BytesIO()
    # Convert to RGB if necessary (e.g., if it has an alpha channel)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def _fallback_ocr_extraction(images: list) -> dict:
    """Tesseract OCR fallback with structured field parsing.
    Automatically detects common Windows Tesseract installation paths.
    """
    try:
        import pytesseract
    except ImportError:
        return {
            "method": "no_ocr_available",
            "error": "pytesseract not installed. Run: pip install pytesseract pillow"
        }

    # Auto-detect Tesseract on Windows (common locations)
    import os, shutil
    if os.name == "nt":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            shutil.which("tesseract") or "",
        ]
        for p in possible_paths:
            if p and os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

    extracted_text = ""
    for idx, img in enumerate(images):
        try:
            # Preprocess for better OCR on scanned receipts
            if img.mode != "L":
                img = img.convert("L")  # grayscale
            # Increase contrast
            from PIL import ImageEnhance, ImageOps
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = ImageOps.autocontrast(img)

            ocr_text = pytesseract.image_to_string(img, config="--psm 6")
            extracted_text += ocr_text + "\n"
        except Exception as e:
            print(f"Tesseract error on page {idx}: {e}")

    if not extracted_text.strip():
        return {"method": "tesseract_no_text", "raw_ocr_text": ""}

    # Lenient regex parsing
    import re
    text = extracted_text

    patient_name = None
    for pattern in [
        r"(?:patient|name|claimant)[:\s-]*([A-Za-z][A-Za-z\s]+?)(?:\n|  |$)",
        r"([A-Z][a-z]+\s+[A-Z][a-z]+)",  # any two capitalized words
    ]:
        m = re.search(pattern, text)
        if m:
            patient_name = m.group(1).strip()
            break

    provider_name = None
    m = re.search(r"(?:hospital|clinic|provider|dr\.?|doctor|medical)[:\s-]*([A-Za-z][A-Za-z\s&]+?)(?:\n|  |$)", text, re.IGNORECASE)
    if m:
        provider_name = m.group(1).strip()

    treatment_date = None
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    if m:
        treatment_date = m.group(1)

    amount = None
    m = re.search(r"(?:total|amount|charge|fee|hk\$|\$)[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)", text, re.IGNORECASE)
    if m:
        try:
            amount = float(m.group(1).replace(",", ""))
        except:
            pass

    diagnosis = None
    m = re.search(r"(?:diagnosis|treatment|description|reason|for)[:\s-]*(.+?)(?:\n\n|\n  |$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        diagnosis = m.group(1).strip()[:300]

    result = {
        "patient_name": patient_name,
        "provider_name": provider_name,
        "treatment_date": treatment_date,
        "diagnosis_or_treatment_description": diagnosis,
        "amount_charged": amount,
        "currency": "HKD",
        "has_official_stamp_or_signature": False,
        "raw_ocr_text": extracted_text,
        "method": "tesseract_ocr"
    }

    # If nothing structured was found, still return the raw text so user sees progress
    if not any(v for k, v in result.items() if k != "raw_ocr_text" and v):
        result["note"] = "OCR read text but could not parse fields. Check raw_ocr_text below."

    return result

def extract_generic_receipt(images: list) -> dict:
    """
    Generic (non-insurance) receipt extractor for SROIE-style receipts.
    Uses the same Grok vision path + Tesseract fallback.
    """
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        # Silent fallback (no spam)
        return _fallback_ocr_extraction(images)

    try:
        from langchain_xai import ChatXAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        print("langchain_xai not found. Falling back to OCR.")
        return _fallback_ocr_extraction(images)

    # Try the official vision model names in order
    vision_models = ["grok-2-vision-1212", "grok-2-vision", "grok-vision-beta"]
    llm = None
    for model_name in vision_models:
        try:
            llm = ChatXAI(model=model_name, temperature=0, max_tokens=800, xai_api_key=api_key)
            break
        except Exception:
            continue
    if llm is None:
        print("[Vision] No valid Grok vision model found")
        return _fallback_ocr_extraction(images)

    system_prompt = (
        "Extract the following fields from this receipt image as JSON only: "
        "company, address, date (YYYY-MM-DD if possible), total (number only). "
        "If a field is missing or unclear use null. Respond with ONLY the JSON object."
    )

    best = {}
    for img in images:
        try:
            b64 = encode_image_base64(img)
            msg = HumanMessage(content=[
                {"type": "text", "text": "Extract the receipt details."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ])
            resp = llm.invoke([SystemMessage(content=system_prompt), msg])
            cleaned = re.sub(r"```(?:json)?\s*", "", resp.content).strip().rstrip("`")
            data = json.loads(cleaned)
            # keep the first successful non-empty result
            if any(v for v in data.values() if v is not None):
                best = data
                break
        except Exception as e:
            print(f"Generic vision error on page: {e}")

    if best:
        best["method"] = "grok_vision_generic"
        return best
    return _fallback_ocr_extraction(images)

def extract_receipt_via_vision(images: list) -> dict:
    """
    Extract structured receipt data using Grok Vision via direct HTTP.
    Uses grok-4.20-0309-non-reasoning which supports image input.
    """
    # Load from .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        return _fallback_ocr_extraction(images)

    import requests

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = (
        "You are reading a scanned medical receipt or claim document. "
        "Extract the following fields as JSON: "
        "patient_name, provider_name (hospital/clinic), treatment_date (YYYY-MM-DD), "
        "diagnosis_or_treatment_description, amount_charged (number), "
        "currency, has_official_stamp_or_signature (bool). "
        "If a field is not visible or legible, use null. "
        "Respond with ONLY the JSON object, with no markdown fences or extra text."
    )

    best_extraction = {}
    max_fields_found = -1

    for i, img in enumerate(images):
        try:
            base64_image = encode_image_base64(img)

            payload = {
                "model": "grok-4.20-0309-non-reasoning",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the receipt details from this image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 600,
                "temperature": 0
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                print(f"[Vision] HTTP {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Clean JSON
            cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
            result = json.loads(cleaned)

            fields_found = sum(1 for v in result.values() if v is not None and str(v).strip() != "")
            if fields_found > max_fields_found:
                max_fields_found = fields_found
                best_extraction = result

        except Exception as e:
            print(f"[Vision] Page {i} extraction failed: {e}")

    if best_extraction:
        best_extraction["method"] = "grok_vision_direct"
        return best_extraction
    else:
        return _fallback_ocr_extraction(images)
