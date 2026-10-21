"""Feature definitions and the domain mappings (model families, towns to districts)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CAT = ["brand", "model_family", "gear", "fuel_type", "condition", "body_type", "origin_country", "district"]
NUM = ["age", "mileage_km", "km_per_year", "engine_cc", "power_options"]
FEATURES = CAT + NUM

# Raw files that each hold a single model: the file name is the most reliable model label.
FILE_MODEL = {
    "honda_city_edited": "City", "honda_civic_with_country": "Civic", "honda_cr-v_edited": "CR-V",
    "honda_fit_with_country": "Fit", "honda_grace_edited_cleaned": "Grace", "honda_vezel_edited": "Vezel",
    "nissan_clipper_with_country": "Clipper", "nissan_sunny_with_country": "Sunny",
    "suzuki_every_with_country": "Every", "suzuki_fronx_with_country": "Fronx", "suzuki_swift_with_country": "Swift",
    "suzuki_wagon_r_with_country": "Wagon R", "toyota_allion_with_country": "Allion",
    "toyota_aqua_with_country": "Aqua", "toyota_chr_with_country": "C-HR", "toyota_corolla_with_country": "Corolla",
    "toyota_premio_with_country": "Premio", "toyota_prius_with_country": "Prius", "toyota_raize": "Raize",
    "toyota_starlet_with_country": "Starlet", "toyota_vitz_edited_cleaned": "Vitz", "toyota_yaris": "Yaris",
    "volvo_edited_cleaned": "Volvo (all)", "jaguar_edited": "Jaguar (all)",
}


def family_from_text(source: str, text) -> str | None:
    """Map the seller's free-text model name to a model family, for brand-level raw files."""
    t = str(text).lower().strip()
    first = t.split()[0] if t else ""
    if source.startswith("audi"):
        m = re.match(r"(a[1-8])", first)
        return m.group(1).upper() if m else None
    if source.startswith("bmw"):
        if first.startswith("i8"):
            return "i8"
        m = re.match(r"([1-7])\d", first)
        return f"{m.group(1)} Series" if m else None
    if source.startswith("mercedes"):
        if first.startswith("cla"):
            return "CLA"
        if first.startswith("w210"):
            return "E-Class"
        m = re.match(r"([ces])\d", first)
        return f"{m.group(1).upper()}-Class" if m else None
    if source.startswith("mazda"):
        return {"familia": "Familia", "demio": "Demio", "2": "Demio", "axela": "Axela", "3": "Axela",
                "6": "Atenza", "atenza": "Atenza", "323": "323", "flair": "Flair", "rx-8": "RX-8"}.get(first)
    if source.startswith("mitsubishi"):
        return {"lancer": "Lancer", "ek": "eK", "galant": "Galant", "attrage": "Attrage", "mirage": "Mirage",
                "minica": "Minica"}.get(first)
    if source.startswith("nissan_cleaned"):
        return {"march": "March", "dayz": "Dayz", "roox": "Dayz", "bluebird": "Bluebird", "pulsar": "Pulsar",
                "tiida": "Tiida", "cefiro": "Cefiro"}.get(first)
    return None


_D = {
    "Colombo": "Avissawella Battaramulla Boralesgamuwa Dehiwala-Mount-Lavinia Hanwella Homagama Kaduwela Kesbewa "
               "Kolonnawa Kotikawatta Kottawa Kotte Maharagama Malabe Moratuwa Mulleriyawa Nawala Nugegoda Padukka "
               "Pannipitiya Piliyandala Rajagiriya Colombo",
    "Gampaha": "Gampaha Daluguma Divulapitiya Dompe Ganemulla Ja-Ela Kadawatha Kandana Katunayake Kelaniya "
               "Kiribathgoda Minuwangoda Mirigama Negombo Nittambuwa Ragama Veyangoda Wattala Welisara Peliyagoda",
    "Kalutara": "Kalutara Bandaragama Beruwala Horana horana Matugama Panadura",
    "Kandy": "Kandy Gampola Kadugannawa Nawalapitiya Wattegama Harispattuwa",
    "Matale": "Matale Dambulla", "Nuwara Eliya": "Nuwara-Eliya Hatton",
    "Galle": "Galle Ambalangoda Bentota Elpitiya Hikkaduwa Koggala", "Matara": "Matara Weligama",
    "Hambantota": "Hambantota Tangalle Middeniya",
    "Kurunegala": "Kurunegala Ambanpola Kuliyapitiya Pannala Polgahawela",
    "Puttalam": "Puttalam Chilaw Dankotuwa Marawila Nattandiya Wennappuwa",
    "Anuradhapura": "Anuradapura", "Polonnaruwa": "Polonnaruwa",
    "Badulla": "Badulla Bandarawela Haputale Mahiyanganaya Welimada", "Monaragala": "Moneragala Kataragama",
    "Ratnapura": "Ratnapura Balangoda Eheliyagoda Embilipitiya",
    "Kegalle": "Kegalle Dehiowita Mawanella Ruwanwella Warakapola",
    "Jaffna": "Jaffna", "Kilinochchi": "Kilinochchi", "Mannar": "Mannar", "Vavuniya": "Vavuniya",
    "Batticaloa": "Batticaloa Eravur", "Ampara": "Ampara Kalmunai Sammanthurai", "Trincomalee": "Trincomalee",
}
# 112 towns -> 24 districts (local knowledge; no listings came from Mullaitivu)
TOWN_DISTRICT = {town: district for district, towns in _D.items() for town in towns.split()}


def make_row(values: dict, ref_year: int) -> pd.DataFrame:
    """Turn user-entered car details into the exact one-row feature frame the model expects."""
    age = max(ref_year - int(values["year"]), 0)
    mileage = float(values["mileage_km"])
    engine = values.get("engine_cc")
    row = {**{c: str(values[c]) for c in CAT},
           "age": age, "mileage_km": mileage, "km_per_year": mileage / max(age, 1),
           "engine_cc": float(engine) if engine else np.nan,
           "power_options": float(values.get("power_options", 3))}
    return pd.DataFrame([row])[FEATURES]
