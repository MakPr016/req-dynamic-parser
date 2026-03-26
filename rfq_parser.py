import pdfplumber
import os
import json
import io
import re
from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-2.5-pro"

_client = None

PLACEHOLDER_PATTERNS = ["click or tap", "click here", "enter text", "type here"]
SKIP_DESCS = {
    "total", "subtotal", "grand total", "amount", "description", "item description",
    "transportation price", "insurance price", "installation price", "training price",
    "other charges (specify)", "other charges", "total price",
    "total final and all-inclusive price",
}

DESC_RE = re.compile(r'(description|specifications|commodity|item\s*name|item\s*desc)')
QTY_RE  = re.compile(r'(qty|quant|quantity|total\s*qty|total\s*quantity)')
SR_RE   = re.compile(r'\b(sr|item\s*no|pos\.?)\b|^no\.?$')
UNIT_RE = re.compile(r'(unit|uom|pack\s*size|measure)')


# ---------------------------------------------------------------------------
# CATEGORY DEFINITIONS (ported from old parser)
# Ordered by specificity. Whole-word boundary matching is applied.
# ---------------------------------------------------------------------------
CATEGORY_DEFINITIONS = {
    "Pharmaceuticals & Biologics": [
        "tablet", "tab", "capsule", "cap", "syrup", "suspension", "susp", "injection", "inj", "vial", "ampoule", "amp",
        "drops", "gtt", "inhaler", "vaccine", "insulin", "dose", "drug", "medication", "ointment", "cream", "gel",
        "lotion", "suppository", "supp", "antibiotic", "antiviral", "analgesic", "anesthetic", "hormone", "steroid",
        "vitamin", "mineral", "supplement", "lozenge", "patch", "solution", "powder for suspension", "elixir", "serum",
        "antitoxin",
    ],
    "Surgical Products": [
        "scalpel", "forceps", "retractor", "clamp", "suture", "stapler", "surgical mesh", "hemostatic", "sealant",
        "surgical drape", "surgical gown", "laparoscopic", "robotic surgery", "electrosurgical", "surgical laser",
        "surgical blade", "trocar", "surgical clip", "surgical scissor", "needle holder",
    ],
    "Orthopedic & Spine": [
        "orthopedic", "spine", "joint replacement", "trauma fixation", "bone plate", "bone screw",
        "intramedullary rod", "bone nail", "spinal implant", "spinal fusion", "bone graft", "orthopedic brace",
        "cast", "arthroscopy", "fixator", "prosthesis", "bone drill", "bone saw",
    ],
    "Cardiovascular Products": [
        "cardiac stent", "pacemaker", "defibrillator", "icd", "heart valve", "vascular graft", "cardiac catheter",
        "guidewire", "cardiac balloon", "ablation", "coronary", "angioplasty", "introducer sheath",
    ],
    "Medical Imaging Equipment": [
        "mri", "ct scanner", "x-ray", "ultrasound", "mammography", "fluoroscopy", "pet scanner", "c-arm",
        "medical imaging", "transducer", "x-ray film", "contrast media", "lead apron",
    ],
    "Diagnostic Products": [
        "diagnostic", "test kit", "glucose test", "reagent", "immunoassay", "chemistry analyzer", "hematology",
        "microbiology", "culture media", "pregnancy test", "covid", "rapid test", "urinalysis", "penlight",
        "specula", "otoscope", "ophthalmoscope", "lancet", "glucometer strips", "test strip",
    ],
    "Patient Monitoring Equipment": [
        "vital signs", "ecg", "ekg", "pulse oximeter", "blood pressure monitor", "sphygmomanometer",
        "medical thermometer", "capnography", "fetal monitor", "telemetry", "spo2 sensor", "bp cuff",
        "temperature probe",
    ],
    "Respiratory & Anesthesia": [
        "ventilator", "anesthesia machine", "oxygen concentrator", "nebulizer", "cpap", "bipap", "respiratory",
        "endotracheal", "tracheostomy", "spirometer", "oxygen mask", "breathing circuit", "nasal cannula",
        "resuscitator", "laryngoscope",
    ],
    "Infusion & Vascular Access": [
        "infusion pump", "syringe pump", "iv set", "iv catheter", "venous", "picc", "iv port",
        "dialysis catheter", "administration set", "extension set", "stopcock", "giving set", "saline",
        "dextrose", "ringer", "sodium chloride", "water for injection",
    ],
    "Wound Care & Tissue Management": [
        "wound dressing", "bandage", "gauze", "medical tape", "plaster", "adhesive", "wound foam", "alginate",
        "hydrocolloid", "compression bandage", "ostomy", "skin substitute", "negative pressure",
    ],
    "Dialysis & Renal Care": [
        "hemodialysis", "peritoneal", "dialyzer", "blood line", "fistula needle", "dialysis concentrate",
        "bicarbonate",
    ],
    "Ophthalmic Products": [
        "intraocular", "intraocular lens", "phaco", "vitrectomy", "lasik", "contact lens", "viscoelastic",
        "ophthalmic solution", "eye drops",
    ],
    "Dental Products": [
        "dental implant", "orthodontic", "dental bracket", "dental wire", "dental drill", "dental handpiece",
        "dental cement", "dental composite", "amalgam", "impression material", "teeth whitening", "dental chair",
    ],
    "Neurology & Neurosurgery": [
        "neurostimulation", "spinal cord stimulator", "neuro coil", "flow diverter", "cranial", "shunt",
        "neuro electrode", "eeg", "emg",
    ],
    "Laboratory Equipment & Supplies": [
        "microscope", "lab centrifuge", "incubator", "autoclave", "pipette", "glassware", "test tube",
        "petri dish", "flask", "beaker", "microscope slide", "cover glass", "fume hood", "biosafety cabinet",
    ],
    "Personal Protective Equipment (PPE)": [
        "ppe", "n95", "face shield", "safety eyewear", "goggles", "protective apron", "shoe cover",
        "head cover", "coverall", "isolation gown", "hazmat", "surgical mask",
    ],
    "Sterilization & Disinfection": [
        "sterilization", "disinfectant", "antiseptic", "povidone", "iodine", "chlorhexidine", "alcohol swab",
        "hand sanitizer", "medical soap", "enzymatic cleaner", "detergent", "washer disinfector", "sterilizer",
        "sterilization indicator",
    ],
    "Hospital Furniture & Equipment": [
        "hospital bed", "examination table", "stretcher", "medical trolley", "medical cart", "medical cabinet",
        "bedside locker", "overbed table", "iv pole", "wheelchair",
    ],
    "Rehabilitation & Physical Therapy": [
        "rehabilitation", "physiotherapy", "walker", "walking cane", "crutch", "exercise band", "traction",
        "electrotherapy", "massage table", "orthosis",
    ],
    "Home Healthcare Products": [
        "home care", "blood glucose meter", "hearing aid", "mobility aid", "bathroom safety", "commode",
    ],
    "Emergency & Trauma Care": [
        "emergency kit", "trauma kit", "first aid", "aed", "defibrillator", "manual resuscitator",
        "suction unit", "immobilizer", "cervical collar", "splint", "tourniquet", "crash cart",
    ],
    "Maternal & Neonatal Care": [
        "maternal", "neonatal", "infant incubator", "infant warmer", "phototherapy", "breast pump",
        "obstetric", "birthing bed", "fetal doppler", "umbilical",
    ],
    "Urology Products": [
        "urology", "foley catheter", "urine bag", "urinary drainage", "ureteral stent", "stone basket",
    ],
    "Gastroenterology & Endoscopy": [
        "endoscope", "gastroscope", "colonoscope", "biopsy forceps", "polypectomy snare", "gastric balloon",
        "ercp",
    ],
    "Oncology Products": [
        "oncology", "chemotherapy", "radiotherapy", "brachytherapy", "port-a-cath", "cancer diagnostic",
    ],
    "Pain Management": [
        "pain management", "pca pump", "epidural", "nerve block", "tens unit",
    ],
    "Sleep Medicine": [
        "sleep apnea", "cpap mask", "bipap mask", "sleep tubing", "polysomnography",
    ],
    "Telemedicine & Digital Health": [
        "telemedicine", "telehealth", "remote monitor", "medical software", "health app",
    ],
    "Blood Management": [
        "blood bag", "blood transfusion", "blood bank", "blood warmer", "apheresis",
    ],
    "Mortuary & Pathology": [
        "mortuary", "autopsy", "body bag", "morgue fridge", "dissection table", "microtome",
        "tissue processor",
    ],
    "Environmental Control": [
        "medical gas", "medical vacuum", "medical air plant", "gas manifold", "gas outlet", "gas alarm",
    ],
    "Mobility & Accessibility": [
        "patient lift", "patient hoist", "wheelchair ramp", "stair lift", "transfer board",
    ],
    "Bariatric Products": [
        "bariatric bed", "bariatric wheelchair", "heavy duty scale",
    ],
    "Medical Textiles": [
        "hospital linen", "bed sheet", "pillow case", "medical blanket", "towel", "privacy curtain",
        "medical uniform", "scrub suit", "lab coat",
    ],
    "Infection Control Products": [
        "waste bin", "sharps container", "biohazard bag", "spill kit", "air purifier",
    ],
    "Medical Gases & Cryogenics": [
        "gas cylinder", "oxygen regulator", "flowmeter", "liquid oxygen", "nitrogen tank",
    ],
    "Nutrition & Feeding": [
        "enteral feeding", "clinical nutrition", "nasogastric tube", "feeding pump", "feeding set", "peg tube",
    ],
    "Specimen Collection & Transport": [
        "specimen container", "sample collection", "transport media", "transport swab", "urine container",
        "stool container", "cool box", "transport bag",
    ],
    "Medical Software & IT": [
        "emr", "ehr", "pacs", "ris", "lis", "his", "hospital information system",
    ],
    "Aesthetics & Dermatology": [
        "dermatology", "aesthetic laser", "ipl", "dermal filler", "botulinum", "botox", "chemical peel",
        "microdermabrasion",
    ],
    # Catch-all — must remain last
    "Medical Supplies & Consumables": [
        "syringe", "needle", "glove", "examination glove", "disposable", "consumable", "cotton wool",
        "alcohol prep", "urinal", "bedpan", "underpad", "tongue depressor", "applicator",
        "lubricant jelly", "cannula",
    ],
}


def determine_item_category(description: str, unit: str = "") -> str:
    """
    Returns the best-matching category for a line item using whole-word regex
    matching against CATEGORY_DEFINITIONS.  Falls back to
    'Medical Supplies & Consumables' if nothing matches.
    """
    text = (description + " " + unit).lower()
    for category, keywords in CATEGORY_DEFINITIONS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                return category
    return "Medical Supplies & Consumables"


# ---------------------------------------------------------------------------
# Remaining helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _get_genai_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not configured")
        _client = genai.Client(api_key=api_key)
    return _client


def _clean(cell):
    return str(cell).replace("\n", " ").strip() if cell else ""


def _is_placeholder(text):
    t = text.lower()
    return any(p in t for p in PLACEHOLDER_PATTERNS)


def _parse_qty(s):
    q = re.sub(r"[^\d.]", "", s)
    if not q:
        return 0
    try:
        v = float(q)
        return int(v) if v.is_integer() else v
    except Exception:
        return 0


def _detect_header(table):
    for r_i, row in enumerate(table[:6]):
        cells = [_clean(c).lower() for c in row]
        flat = " ".join(cells)
        if not (DESC_RE.search(flat) and (QTY_RE.search(flat) or UNIT_RE.search(flat))):
            continue
        idx = {"sr": -1, "desc": -1, "unit": -1, "qty": -1}
        for c_i, h in enumerate(cells):
            if not h:
                continue
            if SR_RE.search(h) and idx["sr"] == -1:
                idx["sr"] = c_i
            elif DESC_RE.search(h) and idx["desc"] == -1:
                idx["desc"] = c_i
            elif QTY_RE.search(h) and idx["qty"] == -1:
                idx["qty"] = c_i
            elif UNIT_RE.search(h) and idx["unit"] == -1:
                idx["unit"] = c_i
        if idx["desc"] != -1:
            return r_i, idx, len(row)
    return -1, None, 0


def _remap_by_data_row(idx_map, table, header_idx):
    sample = next(
        (r for r in table[header_idx + 1:] if any(c is not None for c in r)),
        None
    )
    if not sample:
        return idx_map

    non_none = [i for i, c in enumerate(sample) if c is not None]
    if len(non_none) < 2:
        return idx_map

    remapped = {
        "sr":   non_none[0]  if len(non_none) > 0 else -1,
        "desc": non_none[1]  if len(non_none) > 1 else -1,
        "unit": non_none[-2] if len(non_none) > 2 else -1,
        "qty":  non_none[-1] if len(non_none) > 1 else -1,
    }
    return remapped


def _looks_like_item_continuation(table):
    hits = 0
    for row in table[:8]:
        non_empty = [_clean(c) for c in row if c is not None and _clean(c)]
        if len(non_empty) >= 2 and re.match(r'^\d+\.?$', non_empty[0]) and len(non_empty[1]) > 3:
            hits += 1
    return hits >= 2


def _extract_rows(rows, idx_map, num_cols, seen_srs, items):
    def _parse_description_parts(raw_desc):
        text = raw_desc.strip()
        if not text:
            return "", "", ""

        # Pull dosage-like fragments such as "156 Mg/5ml" or "500 mg".
        dosage_match = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|mg/ml|mcg/ml|g/ml)\b(?:\s*/\s*\d+(?:\.\d+)?\s*ml)?",
            text,
            flags=re.IGNORECASE,
        )
        dosage = dosage_match.group(0) if dosage_match else ""

        # Common dosage forms that appear in descriptions.
        form_match = re.search(
            r"\b(tablet|tab|capsule|cap|suspension|syrup|injection|inj|vial|ampoule|amp|drops|inhaler|ointment|cream|gel|lotion|suppository|supp|solution|powder|elixir|serum)\b",
            text,
            flags=re.IGNORECASE,
        )
        form = form_match.group(0) if form_match else ""

        cleaned = text
        for fragment in [dosage, form]:
            if fragment:
                cleaned = re.sub(re.escape(fragment), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.-")

        return cleaned, dosage, form

    def _parse_pack_from_unit(raw_unit):
        text = raw_unit.strip()
        if not text:
            return "", 0, ""

        # Match patterns like "Pack of 20 Tablet" or "Box of 100".
        pack_match = re.search(r"\b(pack|box|bottle|bag|tube|vial|ampoule|amp|ea|each|single unit)\b", text, flags=re.IGNORECASE)
        unit_type = pack_match.group(0) if pack_match else ""

        qty_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        pack_size = 0
        if qty_match:
            try:
                pack_size_val = float(qty_match.group(1))
                pack_size = int(pack_size_val) if pack_size_val.is_integer() else pack_size_val
            except Exception:
                pack_size = 0

        pack_unit = ""
        trailing = text
        if qty_match:
            trailing = text[qty_match.end():]
        if trailing:
            m = re.search(r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b", trailing)
            if m:
                pack_unit = m.group(1).strip()

        return unit_type.title() if unit_type else "", pack_size, pack_unit.title() if pack_unit else ""

    for row in rows:
        row_clean = [_clean(c) for c in row]
        row_clean = (row_clean + [""] * num_cols)[:num_cols]

        if not any(row_clean):
            continue
        if any(_is_placeholder(c) for c in row_clean):
            continue

        sr_val = None
        if idx_map["sr"] != -1 and idx_map["sr"] < len(row_clean):
            m = re.search(r'\d+', row_clean[idx_map["sr"]])
            if m:
                sr_val = int(m.group())
        if sr_val is None:
            non_empty = [c for c in row_clean if c]
            if non_empty and re.match(r'^\d+\.?$', non_empty[0]):
                sr_val = int(re.sub(r'\D', '', non_empty[0]))

        desc = ""
        if idx_map["desc"] != -1 and idx_map["desc"] < len(row_clean):
            desc = row_clean[idx_map["desc"]]
        if not desc:
            for c in row_clean:
                if c and not re.match(r'^[\d.,]+$', c) and not _is_placeholder(c):
                    desc = c
                    break

        desc = desc.strip()
        if not desc or len(desc) < 3 or desc.lower() in SKIP_DESCS or _is_placeholder(desc):
            continue

        unit_val = ""
        if idx_map["unit"] != -1 and idx_map["unit"] < len(row_clean):
            unit_val = row_clean[idx_map["unit"]]

        qty_val = 0
        if idx_map["qty"] != -1 and idx_map["qty"] < len(row_clean):
            qty_val = _parse_qty(row_clean[idx_map["qty"]])

        key = sr_val if sr_val is not None else desc
        if key in seen_srs:
            continue
        seen_srs.add(key)

        clean_desc, dosage, form = _parse_description_parts(desc)
        unit_type, pack_size, pack_unit = _parse_pack_from_unit(unit_val)

        # --- NEW: classify the item ---
        category = determine_item_category(clean_desc or desc, unit_val)

        items.append({
            "sr": sr_val if sr_val is not None else len(items) + 1,
            "description": clean_desc or desc,
            "dosage": dosage,
            "form": form.title() if form else "",
            "pack_size": pack_size,
            "pack_unit": pack_unit,
            "unit": unit_type,
            "qty": qty_val,
            "unit_price": None,
            "total_price": None,
            "brand": "",
            "expiry_date": "",
            "remarks": "",
            "category": category,       # ← new field
        })


def extract_line_items(pdf_bytes):
    items = []
    seen_srs = set()
    active_schema = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if len(table) < 2:
                    continue

                h_idx, idx_map, num_cols = _detect_header(table)

                if h_idx != -1 and idx_map and idx_map["desc"] != -1:
                    remapped = _remap_by_data_row(idx_map, table, h_idx)
                    active_schema = {"idx": remapped, "num_cols": num_cols}
                    _extract_rows(table[h_idx + 1:], remapped, num_cols, seen_srs, items)
                    continue

                if active_schema and _looks_like_item_continuation(table):
                    actual_cols = max(len(r) for r in table)
                    sample = next((r for r in table if any(c is not None for c in r)), None)
                    none_ratio = sum(1 for c in (sample or []) if c is None) / max(len(sample or [1]), 1)

                    if none_ratio > 0.4:
                        non_none = [i for i, c in enumerate(sample) if c is not None]
                        remapped = {
                            "sr":   non_none[0]  if len(non_none) > 0 else -1,
                            "desc": non_none[1]  if len(non_none) > 1 else -1,
                            "unit": non_none[-2] if len(non_none) > 2 else -1,
                            "qty":  non_none[-1] if len(non_none) > 1 else -1,
                        }
                    else:
                        remapped = {"sr": 0, "desc": 1, "unit": 2, "qty": 3}

                    _extract_rows(table, remapped, actual_cols, seen_srs, items)

    return items


def _extract_line_items_from_llm(full_text):
    system_prompt = (
        "You are an expert at parsing RFQ documents. Extract ALL line items / schedule of requirements from the text. "
        "Return a JSON array only. Each object must have exactly these keys: "
        '{"sr": integer, "description": "string", "unit": "string or empty string", "qty": number or 0, '
        '"unit_price": null, "total_price": null, "brand": "", "expiry_date": "", "remarks": "", "category": "string"}. '
        "For 'category', classify each item into the most appropriate medical supply category "
        "(e.g. 'Pharmaceuticals & Biologics', 'Surgical Products', 'Diagnostic Products', etc.). "
        "If no line items are found, return []. RETURN JSON ARRAY ONLY, no markdown, no preamble."
    )
    try:
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_text[:30000],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        result = json.loads(response.text)
        if isinstance(result, list):
            # Apply local rule-based categorisation as a safety net in case
            # the LLM returns an empty or generic category string.
            for item in result:
                if not item.get("category") or item["category"] in ("string", ""):
                    item["category"] = determine_item_category(
                        item.get("description", ""),
                        item.get("unit", ""),
                    )
            return result
        return []
    except Exception:
        return []


def parse_rfq_pdf(pdf_bytes):
    full_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        pages_to_read = range(total_pages) if total_pages <= 10 else (
            list(range(5)) + list(range(total_pages - 5, total_pages))
        )
        for p_idx in pages_to_read:
            text = pdf.pages[p_idx].extract_text()
            if text:
                full_text += f"\n--- Page {p_idx + 1} ---\n{text}"

    system_prompt = """You are an expert RFQ Parser. Extract data from the RFQ text into the exact JSON structure below.

    JSON OUTPUT STRUCTURE:
    {
      "title": "string",
      "description": "string",
      "sections": [
         "Quotation Submission",
         "Vendor Information",
         "Declaration of Conformity",
         "Schedule of Requirements",
         "Technical & Financial Offer",
         "Compliance & Delivery"
      ],
      "fields": [
        {
          "id": "snake_case_id",
          "label": "Human Readable Label",
          "type": "file" | "text" | "number" | "date" | "dropdown" | "checkbox" | "email" | "phone" | "textarea",
          "section": "Quotation Submission" | "Vendor Information" | "Declaration of Conformity" | "Schedule of Requirements" | "Technical & Financial Offer" | "Compliance & Delivery",
          "required": boolean,
          "default_value": null,
          "placeholder": "Helpful hint",
          "options": ["Option1", "Option2"],
          "validation": {"min": null, "max": null, "pattern": null}
        }
      ]
    }
    """

    try:
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_text[:30000],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt + "\nRETURN JSON ONLY.",
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        llm_data = json.loads(response.text)
    except Exception:
        llm_data = {"title": "Error Parsing", "description": "", "sections": [], "fields": []}

    line_items = extract_line_items(pdf_bytes)

    valid_items = [
        item for item in line_items
        if item.get("description") and not _is_placeholder(item["description"])
    ]

    if not valid_items:
        valid_items = _extract_line_items_from_llm(full_text)

    return {
        "title": llm_data.get("title", "RFQ Document"),
        "description": llm_data.get("description", ""),
        "sections": llm_data.get("sections", []),
        "line_items": valid_items,
        "fields": llm_data.get("fields", []),
    }