#!/usr/bin/env python3
"""Costruisce feature evento orarie per il dataset di modellazione Dozza.

Gli eventi curati restano la fonte principale; le configurazioni opzionali
aggiungono feed CSV/JSON leggibili automaticamente. Gli output sono orari e
allineabili su `timestamp`.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np
import pandas as pd


DOZZA_LAT = 44.3597
DOZZA_LON = 11.6286

DEFAULT_CATEGORIES = [
    "art_culture",
    "food_wine",
    "sport_motor",
    "fair_congress",
    "music_show",
    "market_festival",
    "religious_civic",
    "other",
]

DEFAULT_CITY_LOCATIONS = [
    {"city": "Dozza", "lat": 44.3597, "lon": 11.6286, "is_capoluogo": 0},
    {"city": "Imola", "lat": 44.3530, "lon": 11.7148, "is_capoluogo": 0},
    {"city": "Borgo Tossignano", "lat": 44.2764, "lon": 11.5898, "is_capoluogo": 0},
    {"city": "Riolo Terme", "lat": 44.2758, "lon": 11.7270, "is_capoluogo": 0},
    {"city": "Bologna", "lat": 44.4949, "lon": 11.3426, "is_capoluogo": 1},
    {"city": "Ravenna", "lat": 44.4184, "lon": 12.2035, "is_capoluogo": 1},
    {"city": "Forli", "lat": 44.2227, "lon": 12.0407, "is_capoluogo": 1},
    {"city": "Cesena", "lat": 44.1391, "lon": 12.2431, "is_capoluogo": 0},
    {"city": "Faenza", "lat": 44.2857, "lon": 11.8833, "is_capoluogo": 0},
    {"city": "Ferrara", "lat": 44.8381, "lon": 11.6198, "is_capoluogo": 1},
    {"city": "Modena", "lat": 44.6471, "lon": 10.9252, "is_capoluogo": 1},
    {"city": "Rimini", "lat": 44.0678, "lon": 12.5695, "is_capoluogo": 1},
    {"city": "Reggio Emilia", "lat": 44.6983, "lon": 10.6312, "is_capoluogo": 1},
    {"city": "Parma", "lat": 44.8015, "lon": 10.3279, "is_capoluogo": 1},
]


@dataclass
class SourceConfig:
    source_name: str
    url: str
    fmt: str
    city: str | None = None
    category: str | None = None
    scale: int | None = None
    confidence: float = 0.6
    name_field: str = "name"
    start_field: str = "start_datetime"
    end_field: str = "end_datetime"
    city_field: str = "city"
    category_field: str = "category"
    scale_field: str = "scale"
    lat_field: str = "lat"
    lon_field: str = "lon"
    source_url_field: str = "source_url"
    min_scale: int | None = None
    record_path: str = ""
    pagination: str = ""
    page_size: int = 100
    max_pages: int = 20
    page_param: str = "page"
    offset_param: str = "offset"
    limit_param: str = "limit"


MONTHS = {
    "january": 1,
    "jan": 1,
    "gennaio": 1,
    "gen": 1,
    "february": 2,
    "feb": 2,
    "febbraio": 2,
    "march": 3,
    "mar": 3,
    "marzo": 3,
    "april": 4,
    "apr": 4,
    "aprile": 4,
    "may": 5,
    "maggio": 5,
    "june": 6,
    "jun": 6,
    "giugno": 6,
    "july": 7,
    "jul": 7,
    "luglio": 7,
    "august": 8,
    "aug": 8,
    "agosto": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "semptember": 9,
    "settembre": 9,
    "set": 9,
    "ottobre": 10,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "novembre": 11,
    "december": 12,
    "dec": 12,
    "dicembre": 12,
}


def slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def normalize_event_dedup_name(value: Any) -> str:
    text = normalize_name(value)
    text = re.sub(r"\b(?:19|20)\d{2}\b", "", text)
    text = re.sub(r"\b(?:edition|edizione|annual|annuale)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fmt_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_Nessuna riga._"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    columns = [str(col) for col in view.columns]
    rows = [[fmt_value(value) for value in row] for row in view.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    suffix = ""
    if max_rows is not None and len(df) > max_rows:
        suffix = f"\n\n_Mostrate {max_rows} di {len(df)} righe._"
    return "\n".join([header, sep, *body]) + suffix


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "s"}


def clamp_scale(value: Any, default: int = 1) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return int(max(1, min(5, parsed)))


def parse_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(is_blank(item) for item in value)
    if isinstance(value, dict):
        return len(value) == 0
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def first_useful_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ["it", "label", "name", "title", "value", "date", "start", "end", "url", "href"]:
            if key in value and not is_blank(value[key]):
                return first_useful_value(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            candidate = first_useful_value(item)
            if not is_blank(candidate):
                return candidate
        return ""
    return value


def get_path(record: Any, path: str) -> Any:
    parts = [part.strip() for part in str(path or "").split(".") if part.strip()]
    if not parts:
        return None

    def resolve(current: Any, remaining: list[str]) -> Any:
        if current is None or not remaining:
            return current
        part = remaining[0]
        if isinstance(current, dict):
            if part in current:
                return resolve(current[part], remaining[1:])
            lower_map = {str(key).lower(): key for key in current}
            key = lower_map.get(part.lower())
            if key is not None:
                return resolve(current[key], remaining[1:])
            return None
        if isinstance(current, list):
            if part.isdigit():
                index = int(part)
                if index < len(current):
                    return resolve(current[index], remaining[1:])
                return None
            for item in current:
                candidate = resolve(item, remaining)
                if not is_blank(candidate):
                    return candidate
            return None
        return None

    return resolve(record, parts)


def pick_value(record: dict[str, Any], field_spec: str, default: Any = "") -> Any:
    for field in str(field_spec or "").split("|"):
        field = field.strip()
        if not field:
            continue
        candidate = first_useful_value(get_path(record, field))
        if not is_blank(candidate):
            return candidate
    return default


def set_query_params(url: str, **params: Any) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key, value in params.items():
        if key and value is not None:
            query[str(key)] = str(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def paginated_urls(config: SourceConfig) -> list[str]:
    pagination = slug(config.pagination)
    if pagination not in {"page", "offset", "opendatasoft_offset"}:
        return [config.url]
    page_size = max(1, config.page_size)
    max_pages = max(1, config.max_pages)
    urls = []
    for index in range(max_pages):
        if pagination == "page":
            urls.append(
                set_query_params(
                    config.url,
                    **{
                        config.page_param: index + 1,
                        config.limit_param: page_size,
                    },
                )
            )
        else:
            urls.append(
                set_query_params(
                    config.url,
                    **{
                        config.offset_param: index * page_size,
                        config.limit_param: page_size,
                    },
                )
            )
    return urls


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def xml_text(node: ElementTree.Element, names: set[str]) -> str:
    for child in node.iter():
        if child is node:
            continue
        if local_tag(child.tag) in names and child.text:
            return child.text.strip()
    return ""


def xml_link(node: ElementTree.Element) -> str:
    for child in node.iter():
        if child is node:
            continue
        if local_tag(child.tag) == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
            if child.text:
                return child.text.strip()
    return ""


def month_pattern() -> str:
    return "|".join(sorted((re.escape(month) for month in MONTHS), key=len, reverse=True))


def find_date_range_in_text(text: str) -> dict[str, Any] | None:
    clean = html.unescape(str(text or ""))
    clean = clean.replace("–", "-").replace("—", "-")
    clean = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", clean, flags=re.IGNORECASE)
    months = month_pattern()

    numeric_range = re.search(
        r"(?P<d1>\d{1,2})[./-](?P<m1>\d{1,2})[./-](?P<y1>\d{4})\s*-\s*"
        r"(?P<d2>\d{1,2})[./-](?P<m2>\d{1,2})[./-](?P<y2>\d{4})",
        clean,
    )
    if numeric_range:
        try:
            start = pd.Timestamp(
                year=int(numeric_range.group("y1")),
                month=int(numeric_range.group("m1")),
                day=int(numeric_range.group("d1")),
            )
            end = pd.Timestamp(
                year=int(numeric_range.group("y2")),
                month=int(numeric_range.group("m2")),
                day=int(numeric_range.group("d2")),
            )
        except ValueError:
            pass
        else:
            if end < start:
                end = start
            return {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "start_idx": numeric_range.start(),
                "end_idx": numeric_range.end(),
            }

    cross_month = re.search(
        rf"(?:dal\s+)?(?P<d1>\d{{1,2}})°?\s+(?:di\s+)?(?P<month1>{months})\s+"
        rf"(?:-|al)\s+(?P<d2>\d{{1,2}})°?\s+(?:di\s+)?(?P<month2>{months})\s+(?P<year>\d{{4}})",
        clean,
        flags=re.IGNORECASE,
    )
    if cross_month:
        month1 = MONTHS.get(cross_month.group("month1").lower())
        month2 = MONTHS.get(cross_month.group("month2").lower())
        if month1 is not None and month2 is not None:
            try:
                start = pd.Timestamp(
                    year=int(cross_month.group("year")),
                    month=month1,
                    day=int(cross_month.group("d1")),
                )
                end = pd.Timestamp(
                    year=int(cross_month.group("year")),
                    month=month2,
                    day=int(cross_month.group("d2")),
                )
            except ValueError:
                pass
            else:
                if end < start:
                    end = start
                return {
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "start_idx": cross_month.start(),
                    "end_idx": cross_month.end(),
                }

    patterns = [
        rf"dal\s+(?P<d1>\d{{1,2}})°?\s+al\s+(?P<d2>\d{{1,2}})°?\s+(?:di\s+)?(?P<month>{months})\s+(?P<year>\d{{4}})",
        rf"(?P<d1>\d{{1,2}})°?\s*(?:-\s*(?P<d2>\d{{1,2}})°?)?\s+(?:di\s+)?(?P<month>{months})\s+(?P<year>\d{{4}})",
        rf"(?P<month>{months})\s+(?P<d1>\d{{1,2}})°?(?:\s*-\s*(?P<d2>\d{{1,2}})°?)?,?\s+(?P<year>\d{{4}})",
        rf"(?P<year>\d{{4}})\s+(?P<month>{months})\s+(?P<d1>\d{{1,2}})°?(?:\s*-\s*(?P<d2>\d{{1,2}})°?)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if not match:
            continue
        month = MONTHS.get(match.group("month").lower())
        if month is None:
            continue
        year = int(match.group("year"))
        day_start = int(match.group("d1"))
        day_end = int(match.group("d2") or day_start)
        try:
            start = pd.Timestamp(year=year, month=month, day=day_start)
            end = pd.Timestamp(year=year, month=month, day=day_end)
        except ValueError:
            continue
        if end < start:
            end = start
        return {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "start_idx": match.start(),
            "end_idx": match.end(),
        }
    return None


def html_to_lines(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|article|section|span)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def parse_datetime(value: Any, analysis_tz: str, is_end: bool = False) -> pd.Timestamp:
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    text = str(value).strip()
    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    ts = pd.Timestamp(parsed)
    if date_only:
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59) if is_end else ts.normalize()
    if ts.tzinfo is not None:
        ts = ts.tz_convert(analysis_tz).tz_localize(None)
    return ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Costruisce feature orarie eventi per Dozza.")
    parser.add_argument("--manual-events", type=Path, default=Path("Data/Eventi/manual_events.csv"))
    parser.add_argument(
        "--additional-events-csv",
        type=Path,
        action="append",
        default=[],
        help="CSV eventi aggiuntivo con lo stesso schema di manual-events. Ripetibile.",
    )
    parser.add_argument("--city-locations", type=Path, default=Path("Data/Eventi/city_locations.csv"))
    parser.add_argument(
        "--source-config-csv",
        type=Path,
        help=(
            "CSV opzionale con fonti CSV/JSON scaricabili. Colonne: source_name,url,format,"
            "name_field,start_field,end_field,..."
        ),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help="CSV con colonna timestamp da usare per ricavare griglia oraria e, opzionalmente, join.",
    )
    parser.add_argument("--joined-output-csv", type=Path, help="Se impostato, salva reference-csv + feature eventi.")
    parser.add_argument(
        "--extra-joined-reference-csv",
        type=Path,
        action="append",
        default=[],
        help="Reference CSV aggiuntivo da arricchire con le stesse feature evento. Ripetibile.",
    )
    parser.add_argument(
        "--extra-joined-output-csv",
        type=Path,
        action="append",
        default=[],
        help="Output CSV per il reference extra corrispondente. Ripetibile.",
    )
    parser.add_argument("--start", help="Inizio griglia oraria, es. 2025-01-01. Ignorato se reference-csv presente.")
    parser.add_argument("--end", help="Fine griglia oraria, es. 2025-12-31. Ignorato se reference-csv presente.")
    parser.add_argument("--analysis-tz", default="Europe/Rome")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dozza_preprocess"))
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--dozza-min-scale", type=int, default=2)
    parser.add_argument("--imola-min-scale", type=int, default=3)
    parser.add_argument("--nearby-min-scale", type=int, default=3)
    parser.add_argument("--nearby-radius-km", type=float, default=15.0)
    parser.add_argument("--other-min-scale", type=int, default=4)
    parser.add_argument("--max-distance-km", type=float, default=120.0)
    parser.add_argument("--tau-local-km", type=float, default=20.0)
    parser.add_argument("--tau-regional-km", type=float, default=50.0)
    parser.add_argument("--keep-manual-below-threshold", action="store_true")
    parser.add_argument("--round-digits", type=int, default=6)
    return parser.parse_args()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return float(2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def load_city_locations(path: Path) -> pd.DataFrame:
    if path.exists():
        cities = pd.read_csv(path)
    else:
        cities = pd.DataFrame(DEFAULT_CITY_LOCATIONS)
    required = {"city", "lat", "lon"}
    missing = required - set(cities.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in city_locations: {sorted(missing)}")
    out = cities.copy()
    out["city"] = out["city"].astype(str).str.strip()
    out["city_slug"] = out["city"].map(slug)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    if "is_capoluogo" not in out:
        out["is_capoluogo"] = 0
    out["is_capoluogo"] = out["is_capoluogo"].map(truthy).astype(int)
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates("city_slug", keep="first")
    return out.reset_index(drop=True)


def infer_category(name: Any, fallback: str | None = None) -> str:
    fallback_slug = slug(fallback)
    if fallback_slug in DEFAULT_CATEGORIES:
        return fallback_slug
    text = normalize_name(f"{fallback or ''} {name or ''}")
    rules = [
        (
            "sport_motor",
            ["autodromo", "wec", "formula", "grand prix", "moto", "rally", "race", "gara", "sport", "velocita"],
        ),
        ("food_wine", ["vino", "wine", "enoteca", "degust", "cantin", "food", "sagra", "gastronom", "cucina"]),
        (
            "fair_congress",
            ["fiera", "expo", "congress", "congresso", "salone", "manifestazione fieristica", "conferenza"],
        ),
        ("music_show", ["concerto", "festival", "teatro", "spettacolo", "live", "show", "musica", "cinema", "danza"]),
        (
            "art_culture",
            ["biennale", "muro dipinto", "fantastika", "mostra", "museo", "arte", "cultura", "libri", "biblioteca"],
        ),
        ("market_festival", ["mercato", "festa", "notte", "palio"]),
        ("religious_civic", ["patrono", "relig", "commemorazione", "civica"]),
    ]
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def infer_scale(name: Any, city: str, category: str, source_name: str = "") -> int:
    text = normalize_name(f"{name} {city} {source_name}")
    city_slug = slug(city)
    if any(
        keyword in text
        for keyword in ["muro dipinto", "wec", "formula 1", "grand prix", "cosmoprof", "vasco rossi"]
    ):
        return 5
    if any(
        keyword in text
        for keyword in ["autodromo", "bolognafiere", "fiera internazionale", "concerto", "festival", "expo", "salone"]
    ):
        return 4 if city_slug not in {"dozza", "imola"} else 5
    if any(keyword in text for keyword in ["festa del vino", "fantastika"]):
        return 4
    base = {
        "fair_congress": 4,
        "sport_motor": 4,
        "food_wine": 3,
        "music_show": 3,
        "art_culture": 2,
        "market_festival": 2,
        "religious_civic": 2,
        "other": 1,
    }.get(category, 1)
    if city_slug == "dozza" and base < 4:
        base += 1
    if city_slug == "imola" and category == "sport_motor":
        base += 1
    return clamp_scale(base)


def empty_event_frame() -> pd.DataFrame:
    columns = [
        "event_id",
        "event_name",
        "city",
        "lat",
        "lon",
        "start_datetime",
        "end_datetime",
        "category",
        "scale",
        "scale_reason",
        "source",
        "source_url",
        "confidence",
        "is_manual",
    ]
    return pd.DataFrame(columns=columns)


def normalize_events(raw: pd.DataFrame, analysis_tz: str, source_name: str = "manual") -> pd.DataFrame:
    if raw.empty:
        return empty_event_frame()
    rename_map = {
        "name": "event_name",
        "title": "event_name",
        "start": "start_datetime",
        "end": "end_datetime",
        "url": "source_url",
    }
    frame = raw.rename(columns={key: value for key, value in rename_map.items() if key in raw.columns}).copy()
    if "event_name" not in frame:
        raise ValueError("Gli eventi devono avere una colonna event_name o name.")
    if "start_datetime" not in frame:
        raise ValueError("Gli eventi devono avere una colonna start_datetime o start.")
    if "end_datetime" not in frame:
        frame["end_datetime"] = frame["start_datetime"]
    if "city" not in frame:
        frame["city"] = "Dozza"
    if "category" not in frame:
        frame["category"] = ""
    if "scale" not in frame:
        frame["scale"] = np.nan
    if "confidence" not in frame:
        frame["confidence"] = 1.0
    if "source" not in frame:
        frame["source"] = source_name
    if "source_url" not in frame:
        frame["source_url"] = ""
    if "scale_reason" not in frame:
        frame["scale_reason"] = ""
    if "is_manual" not in frame:
        frame["is_manual"] = 0
    if "lat" not in frame:
        frame["lat"] = np.nan
    if "lon" not in frame:
        frame["lon"] = np.nan

    out = pd.DataFrame()
    out["event_name"] = frame["event_name"].astype(str).str.strip()
    out["city"] = frame["city"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    out["start_datetime"] = frame["start_datetime"].map(lambda value: parse_datetime(value, analysis_tz, is_end=False))
    out["end_datetime"] = frame["end_datetime"].map(lambda value: parse_datetime(value, analysis_tz, is_end=True))
    out["category"] = [
        infer_category(name, fallback=category)
        for name, category in zip(out["event_name"], frame["category"])
    ]
    out["scale"] = [
        clamp_scale(scale, default=infer_scale(name, city, category, source))
        for scale, name, city, category, source in zip(
            frame["scale"], out["event_name"], out["city"], out["category"], frame["source"]
        )
    ]
    out["scale_reason"] = frame["scale_reason"].fillna("").astype(str)
    out["source"] = frame["source"].fillna(source_name).astype(str)
    out["source_url"] = frame["source_url"].fillna("").astype(str)
    out["confidence"] = frame["confidence"].map(lambda value: max(0.0, min(1.0, parse_float(value, 1.0))))
    out["is_manual"] = frame["is_manual"].map(truthy).astype(int)

    invalid_end = out["end_datetime"].isna() | (out["end_datetime"] < out["start_datetime"])
    out.loc[invalid_end, "end_datetime"] = out.loc[invalid_end, "start_datetime"] + pd.Timedelta(hours=23, minutes=59)
    out = out.dropna(subset=["event_name", "city", "start_datetime", "end_datetime"])
    out = out[out["event_name"].str.len() > 0].copy()
    out["event_name_norm"] = out["event_name"].map(normalize_name)
    out["city_slug"] = out["city"].map(slug)
    out["start_date"] = out["start_datetime"].dt.date.astype(str)
    out["end_date"] = out["end_datetime"].dt.date.astype(str)
    out["event_id"] = [
        slug(f"{city}_{start}_{name}")[:120]
        for city, start, name in zip(out["city"], out["start_date"], out["event_name"])
    ]
    return out.reset_index(drop=True)


def load_manual_events(path: Path, analysis_tz: str) -> pd.DataFrame:
    if not path.exists():
        return empty_event_frame()
    raw = pd.read_csv(path)
    raw["is_manual"] = 1
    if "source" not in raw:
        raw["source"] = "manual"
    return normalize_events(raw, analysis_tz=analysis_tz, source_name="manual")


def load_additional_events(paths: list[Path], analysis_tz: str) -> pd.DataFrame:
    parts = []
    for path in paths:
        if not path.exists():
            print(f"[WARN] CSV eventi aggiuntivo non trovato, skip: {path}")
            continue
        raw = pd.read_csv(path)
        if "is_manual" not in raw:
            raw["is_manual"] = 1
        if "source" not in raw:
            raw["source"] = path.stem
        parts.append(normalize_events(raw, analysis_tz=analysis_tz, source_name=path.stem))
    if not parts:
        return empty_event_frame()
    return pd.concat(parts, ignore_index=True)


def read_source_configs(path: Path | None) -> list[SourceConfig]:
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path).fillna("")
    configs: list[SourceConfig] = []
    for row in frame.to_dict(orient="records"):
        if not row.get("url"):
            continue
        configs.append(
            SourceConfig(
                source_name=str(row.get("source_name") or row.get("source") or "source").strip(),
                url=str(row["url"]).strip(),
                fmt=str(row.get("format") or row.get("fmt") or "csv").strip().lower(),
                city=str(row.get("city")).strip() or None,
                category=str(row.get("category")).strip() or None,
                scale=clamp_scale(row["scale"]) if str(row.get("scale", "")).strip() else None,
                confidence=parse_float(row.get("confidence"), 0.6),
                name_field=str(row.get("name_field") or "name").strip(),
                start_field=str(row.get("start_field") or "start_datetime").strip(),
                end_field=str(row.get("end_field") or "end_datetime").strip(),
                city_field=str(row.get("city_field") or "city").strip(),
                category_field=str(row.get("category_field") or "category").strip(),
                scale_field=str(row.get("scale_field") or "scale").strip(),
                lat_field=str(row.get("lat_field") or "lat").strip(),
                lon_field=str(row.get("lon_field") or "lon").strip(),
                source_url_field=str(row.get("source_url_field") or "source_url").strip(),
                min_scale=clamp_scale(row["min_scale"]) if str(row.get("min_scale", "")).strip() else None,
                record_path=str(row.get("record_path") or "").strip(),
                pagination=str(row.get("pagination") or "").strip().lower(),
                page_size=parse_int(row.get("page_size"), 100),
                max_pages=parse_int(row.get("max_pages"), 20),
                page_param=str(row.get("page_param") or "page").strip(),
                offset_param=str(row.get("offset_param") or "offset").strip(),
                limit_param=str(row.get("limit_param") or "limit").strip(),
            )
        )
    return configs


def fetch_url(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": "RetryDozzaEventBuilder/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def normalize_opendatasoft_record(item: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item.get("fields"), dict):
        row = item["fields"].copy()
        if "recordid" in item:
            row["recordid"] = item["recordid"]
        return row
    return item


def flatten_json_records(payload: Any, record_path: str = "") -> list[dict[str, Any]]:
    if record_path:
        candidate = get_path(payload, record_path)
        if isinstance(candidate, list):
            return [normalize_opendatasoft_record(item) for item in candidate if isinstance(item, dict)]
        if isinstance(candidate, dict):
            nested = flatten_json_records(candidate)
            return nested or [candidate]
        return []
    if isinstance(payload, list):
        return [normalize_opendatasoft_record(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("results"), list):
        return [normalize_opendatasoft_record(item) for item in payload["results"] if isinstance(item, dict)]
    if isinstance(payload.get("records"), list):
        return [normalize_opendatasoft_record(item) for item in payload["records"] if isinstance(item, dict)]
    for key in ["data", "items", "events"]:
        if isinstance(payload.get(key), list):
            return [normalize_opendatasoft_record(item) for item in payload[key] if isinstance(item, dict)]
    return []


def parse_rss_records(payload: bytes) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(payload)
    item_nodes = [node for node in root.iter() if local_tag(node.tag) in {"item", "entry"}]
    records = []
    for item in item_nodes:
        title = xml_text(item, {"title"})
        description = xml_text(item, {"description", "summary", "content"})
        published = xml_text(item, {"pubdate", "published", "updated"})
        link = xml_link(item)
        extracted = find_date_range_in_text(f"{title} {description}") or {}
        records.append(
            {
                "title": title,
                "description": description,
                "link": link,
                "pubDate": published,
                "published": published,
                "extracted_start": extracted.get("start", ""),
                "extracted_end": extracted.get("end", ""),
            }
        )
    return records


def jsonld_type_is_event(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower().endswith("event") or value.lower() == "event"
    if isinstance(value, list):
        return any(jsonld_type_is_event(item) for item in value)
    return False


def collect_jsonld_events(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(collect_jsonld_events(item))
        return rows
    if not isinstance(value, dict):
        return []
    rows = []
    if jsonld_type_is_event(value.get("@type")):
        location = first_useful_value(value.get("location", {}))
        address = get_path(value, "location.address") or {}
        geo = get_path(value, "location.geo") or {}
        rows.append(
            {
                "title": first_useful_value(value.get("name", "")),
                "startDate": first_useful_value(value.get("startDate", "")),
                "endDate": first_useful_value(value.get("endDate", "")),
                "category": first_useful_value(value.get("eventAttendanceMode", "")),
                "city": first_useful_value(get_path(address, "addressLocality") or ""),
                "lat": first_useful_value(get_path(geo, "latitude") or ""),
                "lon": first_useful_value(get_path(geo, "longitude") or ""),
                "url": first_useful_value(value.get("url", "")),
                "location": location,
                "description": first_useful_value(value.get("description", "")),
            }
        )
    for key in ["@graph", "itemListElement", "mainEntity", "events", "event"]:
        if key in value:
            rows.extend(collect_jsonld_events(value[key]))
    return rows


def parse_jsonld_html_records(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?is)<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        text,
    ):
        raw_json = html.unescape(match.group(1)).strip()
        if not raw_json:
            continue
        try:
            records.extend(collect_jsonld_events(json.loads(raw_json)))
        except json.JSONDecodeError:
            continue
    return records


def parse_text_html_records(payload: bytes) -> list[dict[str, Any]]:
    generic_titles = {
        "all events",
        "featured events",
        "past events",
        "discover now",
        "tutti gli eventi",
        "eventi",
        "in evidenza",
    }
    lines = html_to_lines(payload)
    records = []
    for index, line in enumerate(lines):
        date_info = find_date_range_in_text(line)
        if not date_info:
            continue
        title = f"{line[:date_info['start_idx']]} {line[date_info['end_idx']:]}".strip(" -|")
        if not title or normalize_name(title) in generic_titles or len(title) < 4:
            for previous in reversed(lines[max(0, index - 4) : index]):
                if find_date_range_in_text(previous):
                    continue
                previous_norm = normalize_name(previous)
                if previous_norm not in generic_titles and len(previous) >= 4:
                    title = previous
                    break
        if title and normalize_name(title) not in generic_titles:
            records.append(
                {
                    "title": title,
                    "extracted_start": date_info["start"],
                    "extracted_end": date_info["end"],
                }
            )
    return records


def parse_html_records(payload: bytes) -> list[dict[str, Any]]:
    records = parse_jsonld_html_records(payload)
    if records:
        return records
    return parse_text_html_records(payload)


def records_from_payload(payload: bytes, config: SourceConfig) -> list[dict[str, Any]]:
    if config.fmt == "csv":
        return pd.read_csv(io.BytesIO(payload)).to_dict(orient="records")
    elif config.fmt in {"json", "json_records", "opendatasoft"}:
        return flatten_json_records(json.loads(payload.decode("utf-8")), record_path=config.record_path)
    elif config.fmt in {"rss", "atom", "xml"}:
        return parse_rss_records(payload)
    elif config.fmt in {"html", "html_schemaorg"}:
        return parse_html_records(payload)
    else:
        raise ValueError(f"Formato fonte non supportato: {config.fmt}")


def load_source_records(config: SourceConfig, timeout: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for url in paginated_urls(config):
        payload = fetch_url(url, timeout=timeout)
        page_records = records_from_payload(payload, config)
        if not page_records:
            if config.pagination:
                break
            continue
        records.extend(page_records)
        if not config.pagination or len(page_records) < max(1, config.page_size):
            break
    return records


def normalize_source_record(record: dict[str, Any], config: SourceConfig) -> dict[str, Any]:
    event_name = pick_value(record, config.name_field)
    if is_blank(event_name):
        event_name = pick_value(record, "title|name|titolo|nome|description|label|value")
    start_datetime = pick_value(record, config.start_field)
    if is_blank(start_datetime):
        start_datetime = pick_value(record, "startDate|start_datetime|start|date_start|data_inizio|date|extracted_start")
    end_datetime = pick_value(record, config.end_field, "")
    if is_blank(end_datetime):
        end_datetime = pick_value(record, "endDate|end_datetime|end|date_end|data_fine|extracted_end", start_datetime)
    return {
        "event_name": event_name,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "city": pick_value(record, config.city_field, config.city or ""),
        "category": pick_value(record, config.category_field, config.category or ""),
        "scale": pick_value(
            record,
            config.scale_field,
            config.scale if config.scale is not None else np.nan,
        ),
        "lat": pick_value(record, config.lat_field, np.nan),
        "lon": pick_value(record, config.lon_field, np.nan),
        "source_url": pick_value(record, config.source_url_field, config.url),
        "source": config.source_name,
        "confidence": config.confidence,
        "is_manual": 0,
    }


def load_configured_source(config: SourceConfig, analysis_tz: str, timeout: int) -> pd.DataFrame:
    records = load_source_records(config, timeout=timeout)
    if not records:
        return empty_event_frame()

    normalized = pd.DataFrame([normalize_source_record(record, config) for record in records])
    if config.city:
        normalized["city"] = normalized["city"].replace("", config.city)
    if config.category:
        normalized["category"] = normalized["category"].replace("", config.category)
    if config.scale is not None:
        normalized["scale"] = normalized["scale"].fillna(config.scale).replace("", config.scale)
    out = normalize_events(normalized, analysis_tz=analysis_tz, source_name=config.source_name)
    if config.min_scale is not None and not out.empty:
        out = out[out["scale"].ge(config.min_scale)].copy()
    return out.reset_index(drop=True)


def load_configured_sources(configs: list[SourceConfig], analysis_tz: str, timeout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    status_rows = []
    for config in configs:
        try:
            frame = load_configured_source(config, analysis_tz=analysis_tz, timeout=timeout)
            parts.append(frame)
            status_rows.append(
                {
                    "source": config.source_name,
                    "url": config.url,
                    "status": "ok",
                    "rows": len(frame),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - formati sorgente esterni.
            status_rows.append(
                {
                    "source": config.source_name,
                    "url": config.url,
                    "status": "error",
                    "rows": 0,
                    "error": str(exc),
                }
            )
    if parts:
        events = pd.concat(parts, ignore_index=True)
    else:
        events = empty_event_frame()
    status = pd.DataFrame(status_rows)
    if status.empty:
        status = pd.DataFrame(columns=["source", "url", "status", "rows", "error"])
    return events, status


def enrich_with_locations(events: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    out["city_slug"] = out["city"].map(slug)
    if "lat" not in out:
        out["lat"] = np.nan
    if "lon" not in out:
        out["lon"] = np.nan
    merged = out.merge(
        cities[["city_slug", "lat", "lon", "is_capoluogo"]],
        on="city_slug",
        how="left",
        suffixes=("", "_city"),
    )
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce").fillna(merged["lat_city"])
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce").fillna(merged["lon_city"])
    merged["is_capoluogo"] = merged["is_capoluogo"].fillna(0).astype(int)
    merged = merged.drop(columns=[col for col in ["lat_city", "lon_city"] if col in merged])
    merged["distance_km"] = [
        haversine_km(DOZZA_LAT, DOZZA_LON, lat, lon) if not pd.isna(lat) and not pd.isna(lon) else np.nan
        for lat, lon in zip(merged["lat"], merged["lon"])
    ]
    return merged


def deduplicate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    frame = events.copy()
    frame["event_name_norm"] = frame["event_name"].map(normalize_name)
    frame["event_name_dedup"] = frame["event_name"].map(normalize_event_dedup_name)
    frame["start_date"] = frame["start_datetime"].dt.date.astype(str)
    frame["end_date"] = frame["end_datetime"].dt.date.astype(str)
    frame = frame.sort_values(
        ["is_manual", "confidence", "scale", "source"],
        ascending=[False, False, False, True],
    )
    group_cols = ["event_name_dedup", "city_slug", "start_date", "end_date"]
    rows = []
    for _key, group in frame.groupby(group_cols, dropna=False, sort=False):
        first = group.iloc[0].copy()
        first["scale"] = int(first["scale"])
        first["confidence"] = float(first["confidence"])
        first["start_datetime"] = group["start_datetime"].min()
        first["end_datetime"] = group["end_datetime"].max()
        first["source"] = "; ".join(sorted(set(group["source"].dropna().astype(str))))
        urls = [url for url in group["source_url"].dropna().astype(str).unique() if url]
        first["source_url"] = " | ".join(urls[:5])
        first["is_manual"] = int(group["is_manual"].max())
        rows.append(first)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["event_id"] = [
        slug(f"{city}_{start}_{name}")[:120]
        for city, start, name in zip(out["city"], out["start_date"], out["event_name"])
    ]
    return out


def threshold_for_event(row: pd.Series, args: argparse.Namespace) -> int:
    city_slug = str(row.get("city_slug", ""))
    if city_slug == "dozza":
        return args.dozza_min_scale
    if city_slug == "imola":
        return args.imola_min_scale
    distance = parse_float(row.get("distance_km"), np.nan)
    if not pd.isna(distance) and distance <= args.nearby_radius_km:
        return args.nearby_min_scale
    return args.other_min_scale


def filter_events(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    frame = events.copy()
    keep_rows = []
    for _, row in frame.iterrows():
        if pd.isna(row.get("distance_km")) or float(row["distance_km"]) > args.max_distance_km:
            keep_rows.append(False)
            continue
        threshold = threshold_for_event(row, args)
        keep = int(row["scale"]) >= threshold
        if args.keep_manual_below_threshold and int(row.get("is_manual", 0)) == 1:
            keep = True
        keep_rows.append(keep)
    return frame.loc[keep_rows].sort_values(["start_datetime", "city", "scale"], ascending=[True, True, False]).reset_index(drop=True)


def filter_events_to_grid(events: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    if events.empty or grid.empty:
        return events.copy()
    start = pd.Timestamp(grid.min())
    end = pd.Timestamp(grid.max())
    frame = events.copy()
    keep = (frame["end_datetime"] >= start) & (frame["start_datetime"] <= end)
    return frame.loc[keep].reset_index(drop=True)


def infer_grid(args: argparse.Namespace) -> pd.DatetimeIndex:
    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv, usecols=["timestamp"])
        timestamps = pd.to_datetime(ref["timestamp"], errors="coerce").dropna().dt.floor("h")
        if timestamps.empty:
            raise ValueError("reference-csv non contiene timestamp validi.")
        start = timestamps.min()
        end = timestamps.max()
    else:
        if not args.start or not args.end:
            raise ValueError("Specificare --reference-csv oppure --start e --end.")
        start = parse_datetime(args.start, args.analysis_tz, is_end=False).floor("h")
        end = parse_datetime(args.end, args.analysis_tz, is_end=True).floor("h")
    return pd.date_range(start=start, end=end, freq="h")


def city_group(row: pd.Series) -> str:
    city_slug = str(row.get("city_slug", ""))
    if city_slug in {"dozza", "imola", "bologna"}:
        return city_slug
    if int(row.get("is_capoluogo", 0)) == 1:
        return "capoluoghi"
    return "other"


def distance_weight(row: pd.Series, args: argparse.Namespace) -> float:
    distance = parse_float(row.get("distance_km"), np.nan)
    if pd.isna(distance):
        return 0.0
    tau = args.tau_local_km if str(row.get("city_slug", "")) in {"dozza", "imola"} else args.tau_regional_km
    if tau <= 0:
        return 1.0
    return float(math.exp(-distance / tau))


def build_hourly_features(events: pd.DataFrame, grid: pd.DatetimeIndex, args: argparse.Namespace) -> pd.DataFrame:
    features = pd.DataFrame({"timestamp": grid})
    n = len(features)
    base_cols = {
        "event_active_any": np.zeros(n, dtype=int),
        "event_count_active": np.zeros(n, dtype=float),
        "event_scale_max": np.zeros(n, dtype=float),
        "event_scale_sum": np.zeros(n, dtype=float),
        "event_intensity_sum": np.zeros(n, dtype=float),
        "event_intensity_max": np.zeros(n, dtype=float),
        "event_pre_24h_intensity": np.zeros(n, dtype=float),
        "event_pre_6h_intensity": np.zeros(n, dtype=float),
        "event_post_12h_intensity": np.zeros(n, dtype=float),
        "event_within_10km_count": np.zeros(n, dtype=float),
        "event_within_30km_count": np.zeros(n, dtype=float),
        "event_within_80km_count": np.zeros(n, dtype=float),
    }
    city_groups = ["dozza", "imola", "bologna", "capoluoghi", "other"]
    for group in city_groups:
        base_cols[f"event_{group}_active"] = np.zeros(n, dtype=int)
        base_cols[f"event_{group}_intensity"] = np.zeros(n, dtype=float)
        base_cols[f"event_{group}_scale_max"] = np.zeros(n, dtype=float)
    for category in DEFAULT_CATEGORIES:
        base_cols[f"event_{category}_active"] = np.zeros(n, dtype=int)
        base_cols[f"event_{category}_intensity"] = np.zeros(n, dtype=float)

    timestamps = features["timestamp"]
    if not events.empty:
        for _, event in events.iterrows():
            start = pd.Timestamp(event["start_datetime"]).floor("h")
            end = pd.Timestamp(event["end_datetime"]).floor("h")
            scale = float(event["scale"])
            intensity = scale * distance_weight(event, args) * float(event.get("confidence", 1.0))
            category = str(event.get("category") or "other")
            if category not in DEFAULT_CATEGORIES:
                category = "other"
            group = city_group(event)
            distance = parse_float(event.get("distance_km"), np.nan)

            active = (timestamps >= start) & (timestamps <= end)
            if active.any():
                idx = active.to_numpy()
                base_cols["event_active_any"][idx] = 1
                base_cols["event_count_active"][idx] += 1
                base_cols["event_scale_sum"][idx] += scale
                base_cols["event_scale_max"][idx] = np.maximum(base_cols["event_scale_max"][idx], scale)
                base_cols["event_intensity_sum"][idx] += intensity
                base_cols["event_intensity_max"][idx] = np.maximum(base_cols["event_intensity_max"][idx], intensity)
                base_cols[f"event_{group}_active"][idx] = 1
                base_cols[f"event_{group}_intensity"][idx] += intensity
                base_cols[f"event_{group}_scale_max"][idx] = np.maximum(base_cols[f"event_{group}_scale_max"][idx], scale)
                base_cols[f"event_{category}_active"][idx] = 1
                base_cols[f"event_{category}_intensity"][idx] += intensity
                if not pd.isna(distance):
                    if distance <= 10:
                        base_cols["event_within_10km_count"][idx] += 1
                    if distance <= 30:
                        base_cols["event_within_30km_count"][idx] += 1
                    if distance <= 80:
                        base_cols["event_within_80km_count"][idx] += 1

            pre24 = (timestamps < start) & (timestamps >= start - pd.Timedelta(hours=24))
            if pre24.any():
                hours_to_start = (start - timestamps[pre24]).dt.total_seconds().to_numpy() / 3600.0
                base_cols["event_pre_24h_intensity"][pre24.to_numpy()] += intensity * np.exp(-hours_to_start / 24.0)
            pre6 = (timestamps < start) & (timestamps >= start - pd.Timedelta(hours=6))
            if pre6.any():
                hours_to_start = (start - timestamps[pre6]).dt.total_seconds().to_numpy() / 3600.0
                base_cols["event_pre_6h_intensity"][pre6.to_numpy()] += intensity * np.exp(-hours_to_start / 6.0)
            post12 = (timestamps > end) & (timestamps <= end + pd.Timedelta(hours=12))
            if post12.any():
                hours_since_end = (timestamps[post12] - end).dt.total_seconds().to_numpy() / 3600.0
                base_cols["event_post_12h_intensity"][post12.to_numpy()] += intensity * np.exp(-hours_since_end / 12.0)

    for col, values in base_cols.items():
        features[col] = values
    features["event_weekend_event_active"] = (
        features["event_active_any"].eq(1) & features["timestamp"].dt.dayofweek.isin([5, 6])
    ).astype(int)

    float_cols = features.select_dtypes(include=["float"]).columns
    features[float_cols] = features[float_cols].round(args.round_digits)
    return features


def write_joined_reference(reference_csv: Path, features: pd.DataFrame, output_csv: Path) -> None:
    reference = pd.read_csv(reference_csv)
    if "timestamp" not in reference:
        raise ValueError("reference-csv deve contenere una colonna timestamp.")
    reference = reference.copy()
    reference["timestamp"] = pd.to_datetime(reference["timestamp"], errors="coerce").dt.floor("h")
    joined = reference.merge(features, on="timestamp", how="left")
    event_cols = [col for col in features.columns if col != "timestamp"]
    joined[event_cols] = joined[event_cols].fillna(0)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_csv, index=False)


def write_report(
    output_dir: Path,
    raw_events: pd.DataFrame,
    clean_events: pd.DataFrame,
    hourly: pd.DataFrame,
    source_status: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    if clean_events.empty:
        by_city = pd.DataFrame()
        by_category = pd.DataFrame()
    else:
        by_city = (
            clean_events.groupby("city", as_index=False)
            .agg(events=("event_id", "count"), scale_max=("scale", "max"), distance_min_km=("distance_km", "min"))
            .sort_values(["events", "scale_max"], ascending=[False, False])
        )
        by_category = (
            clean_events.groupby("category", as_index=False)
            .agg(events=("event_id", "count"), scale_mean=("scale", "mean"), scale_max=("scale", "max"))
            .sort_values(["events", "scale_max"], ascending=[False, False])
        )

    active_hours = int(hourly["event_active_any"].sum()) if "event_active_any" in hourly else 0
    report = f"""# Report feature evento

## Configurazione

- Manual events: `{args.manual_events}`
- Additional events CSV: `{", ".join(str(path) for path in args.additional_events_csv) if args.additional_events_csv else ""}`
- City locations: `{args.city_locations}`
- Source config: `{args.source_config_csv or ""}`
- Reference CSV: `{args.reference_csv or ""}`
- Timezone analisi: `{args.analysis_tz}`
- Min scale Dozza/Imola/nearby/altri: {args.dozza_min_scale}/{args.imola_min_scale}/{args.nearby_min_scale}/{args.other_min_scale}
- Nearby radius km: {args.nearby_radius_km}
- Max distance km: {args.max_distance_km}

## Sintesi

- Eventi raw: {len(raw_events)}
- Eventi puliti e filtrati: {len(clean_events)}
- Ore generate: {len(hourly)}
- Ore con almeno un evento attivo: {active_hours}

## Fonti automatiche

{markdown_table(source_status, max_rows=50)}

## Eventi per citta

{markdown_table(by_city, max_rows=50)}

## Eventi per categoria

{markdown_table(by_category, max_rows=50)}

## Eventi puliti

{markdown_table(clean_events[[
    "event_name",
    "city",
    "start_datetime",
    "end_datetime",
    "category",
    "scale",
    "distance_km",
    "source",
    "scale_reason",
]], max_rows=80) if not clean_events.empty else "_Nessun evento._"}

## Output

- `events_raw.csv`
- `events_clean.csv`
- `event_hourly_features.csv`
- `event_source_status.csv`
"""
    (output_dir / "event_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.extra_joined_reference_csv) != len(args.extra_joined_output_csv):
        raise ValueError(
            "--extra-joined-reference-csv e --extra-joined-output-csv devono avere lo stesso numero di valori."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cities = load_city_locations(args.city_locations)
    manual_events = load_manual_events(args.manual_events, analysis_tz=args.analysis_tz)
    additional_events = load_additional_events(args.additional_events_csv, analysis_tz=args.analysis_tz)
    configs = read_source_configs(args.source_config_csv)
    configured_events, source_status = load_configured_sources(
        configs,
        analysis_tz=args.analysis_tz,
        timeout=args.request_timeout,
    )

    event_parts = [frame for frame in [manual_events, additional_events, configured_events] if not frame.empty]
    raw_events = pd.concat(event_parts, ignore_index=True) if event_parts else empty_event_frame()
    raw_events = enrich_with_locations(raw_events, cities)
    raw_events = deduplicate_events(raw_events)
    grid = infer_grid(args)
    clean_events = filter_events(raw_events, args)
    clean_events = filter_events_to_grid(clean_events, grid)
    hourly = build_hourly_features(clean_events, grid, args)

    raw_events.to_csv(args.output_dir / "events_raw.csv", index=False)
    clean_events.to_csv(args.output_dir / "events_clean.csv", index=False)
    hourly.to_csv(args.output_dir / "event_hourly_features.csv", index=False)
    source_status.to_csv(args.output_dir / "event_source_status.csv", index=False)
    write_report(args.output_dir, raw_events, clean_events, hourly, source_status, args)

    if args.joined_output_csv:
        if not args.reference_csv:
            raise ValueError("--joined-output-csv richiede --reference-csv.")
        write_joined_reference(args.reference_csv, hourly, args.joined_output_csv)
    for reference_csv, output_csv in zip(args.extra_joined_reference_csv, args.extra_joined_output_csv):
        if not reference_csv.exists() or reference_csv.stat().st_size == 0:
            print(f"[WARN] Reference CSV extra non trovato o vuoto, skip: {reference_csv}")
            continue
        write_joined_reference(reference_csv, hourly, output_csv)

    print(f"[OK] Eventi raw: {len(raw_events)}")
    print(f"[OK] Eventi puliti: {len(clean_events)}")
    print(f"[OK] Feature orarie: {args.output_dir / 'event_hourly_features.csv'}")
    if args.joined_output_csv:
        print(f"[OK] Dataset joined: {args.joined_output_csv}")
    for output_csv in args.extra_joined_output_csv:
        if output_csv.exists():
            print(f"[OK] Dataset joined extra: {output_csv}")
    print(f"[OK] Report: {args.output_dir / 'event_report.md'}")


if __name__ == "__main__":
    main()
