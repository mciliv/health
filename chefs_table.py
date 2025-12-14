#!/usr/bin/env python3
"""Chef's Table Finder (terminal app).

Find restaurants near you that match "chef's table".

Data sources:
- Yelp (best results) if you export YELP_API_KEY
- Otherwise OpenStreetMap via Overpass (best-effort keyword matching)

Examples:
  python chefs_table.py --near "Austin, TX" --radius-km 8
  python chefs_table.py --lat 47.6062 --lon -122.3321 --radius-km 5
  YELP_API_KEY=... python chefs_table.py --near "Chicago, IL" --source yelp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

try:
    import requests
except ImportError:  # pragma: no cover
    print(
        "Missing dependency: requests\n\n"
        "Install it with:\n"
        "  python -m pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    raise


OSM_NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
IP_GEO_URL = "https://ipapi.co/json/"
YELP_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"

USER_AGENT = "chefs-table-finder/1.0 (terminal app; contact: local)"


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    distance_km: float
    source: str
    score: int = 0
    address: str | None = None
    phone: str | None = None
    url: str | None = None
    rating: float | None = None
    reviews: int | None = None


class AppError(RuntimeError):
    pass


def _http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return s


def haversine_km(a: Point, b: Point) -> float:
    # Great-circle distance between two lat/lon points.
    r = 6371.0088
    lat1 = math.radians(a.lat)
    lon1 = math.radians(a.lon)
    lat2 = math.radians(b.lat)
    lon2 = math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def format_km(km: float) -> str:
    if km < 1:
        return f"{km * 1000:.0f} m"
    return f"{km:.2f} km"


def shorten(s: str, width: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return textwrap.shorten(s, width=width, placeholder="…")


def get_ip_location(session: requests.Session, timeout_s: float = 10) -> tuple[Point, str]:
    try:
        r = session.get(IP_GEO_URL, timeout=timeout_s)
    except requests.RequestException as e:
        raise AppError(f"IP geolocation failed: {e}") from e

    if r.status_code != 200:
        raise AppError(f"IP geolocation failed: HTTP {r.status_code}")

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise AppError("IP geolocation returned invalid JSON") from e

    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is None or lon is None:
        raise AppError("IP geolocation did not return latitude/longitude")

    city = data.get("city")
    region = data.get("region")
    country = data.get("country_name")
    label_parts = [p for p in [city, region, country] if p]
    label = ", ".join(label_parts) if label_parts else "(approximate location)"
    return Point(float(lat), float(lon)), label


def geocode(session: requests.Session, query: str, timeout_s: float = 15) -> tuple[Point, str]:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
    }

    try:
        r = session.get(OSM_NOMINATIM_SEARCH_URL, params=params, timeout=timeout_s)
    except requests.RequestException as e:
        raise AppError(f"Geocoding failed: {e}") from e

    if r.status_code != 200:
        raise AppError(f"Geocoding failed: HTTP {r.status_code}")

    try:
        items = r.json()
    except json.JSONDecodeError as e:
        raise AppError("Geocoding returned invalid JSON") from e

    if not items:
        raise AppError(f"No results found for: {query!r}")

    item = items[0]
    try:
        p = Point(float(item["lat"]), float(item["lon"]))
    except (KeyError, ValueError) as e:
        raise AppError("Geocoding result missing lat/lon") from e

    label = str(item.get("display_name") or query)
    return p, label


CHEFS_TABLE_RE = re.compile(r"\bchef\s*'?s\s*table\b|\bchefs\s*table\b", re.IGNORECASE)


def osm_score(tags: dict[str, str]) -> int:
    # Heuristic: OSM rarely has explicit "chef's table" metadata.
    # We score based on keyword matches in name/description/website/etc.
    haystacks: list[str] = []
    for k in ("name", "description", "note", "website", "contact:website", "operator", "brand"):
        v = tags.get(k)
        if v:
            haystacks.append(v)

    cuisine = tags.get("cuisine")
    if cuisine:
        haystacks.append(cuisine.replace("_", " ").replace(";", " "))

    text = " \n ".join(haystacks)
    score = 0
    if CHEFS_TABLE_RE.search(text):
        score += 12

    lowered = text.lower()
    if "tasting" in lowered or "tasting menu" in lowered:
        score += 3
    if "omakase" in lowered:
        score += 3
    if "counter" in lowered and "chef" in lowered:
        score += 2
    if "reservation" in lowered and "chef" in lowered:
        score += 2

    return score


def overpass_query_for_restaurants(radius_m: int, p: Point) -> str:
    # Using around() against all element types; return tags and center for ways/relations.
    return (
        "[out:json][timeout:25];\n"
        "(\n"
        f'  node["amenity"="restaurant"](around:{radius_m},{p.lat},{p.lon});\n'
        f'  way["amenity"="restaurant"](around:{radius_m},{p.lat},{p.lon});\n'
        f'  relation["amenity"="restaurant"](around:{radius_m},{p.lat},{p.lon});\n'
        ");\n"
        "out center tags;\n"
    )


def search_osm(session: requests.Session, origin: Point, radius_km: float, limit: int) -> list[Place]:
    radius_m = max(200, int(radius_km * 1000))
    query = overpass_query_for_restaurants(radius_m=radius_m, p=origin)

    try:
        r = session.post(OSM_OVERPASS_URL, data=query.encode("utf-8"), timeout=45)
    except requests.RequestException as e:
        raise AppError(f"OpenStreetMap (Overpass) request failed: {e}") from e

    if r.status_code != 200:
        raise AppError(f"OpenStreetMap (Overpass) failed: HTTP {r.status_code}")

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise AppError("OpenStreetMap (Overpass) returned invalid JSON") from e

    elements = data.get("elements")
    if not isinstance(elements, list):
        raise AppError("OpenStreetMap (Overpass) response missing elements")

    results: list[Place] = []
    for el in elements:
        if not isinstance(el, dict):
            continue

        tags = el.get("tags")
        if not isinstance(tags, dict):
            continue

        name = tags.get("name")
        if not name:
            continue

        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            center = el.get("center")
            if isinstance(center, dict):
                lat = center.get("lat")
                lon = center.get("lon")

        if lat is None or lon is None:
            continue

        p = Point(float(lat), float(lon))
        dist = haversine_km(origin, p)

        score = osm_score({str(k): str(v) for k, v in tags.items()})

        # Best-effort address
        addr_bits = []
        for k in ("addr:housenumber", "addr:street", "addr:city"):
            v = tags.get(k)
            if v:
                addr_bits.append(v)
        address = ", ".join(addr_bits) if addr_bits else None

        results.append(
            Place(
                name=str(name),
                lat=p.lat,
                lon=p.lon,
                distance_km=dist,
                source="osm",
                score=score,
                address=address,
                phone=str(tags.get("phone") or tags.get("contact:phone"))
                if (tags.get("phone") or tags.get("contact:phone"))
                else None,
                url=str(tags.get("website") or tags.get("contact:website"))
                if (tags.get("website") or tags.get("contact:website"))
                else None,
            )
        )

    # Sort: score first (desc), then distance.
    results.sort(key=lambda x: (-x.score, x.distance_km, x.name.lower()))

    # If everything scores 0, fall back to nearest restaurants.
    if results and all(p.score == 0 for p in results):
        results.sort(key=lambda x: (x.distance_km, x.name.lower()))

    return results[:limit]


def yelp_radius_m(radius_km: float) -> int:
    # Yelp max radius is 40000m.
    return max(200, min(40000, int(radius_km * 1000)))


def search_yelp(
    session: requests.Session,
    origin: Point,
    radius_km: float,
    limit: int,
    api_key: str,
) -> list[Place]:
    headers = {"Authorization": f"Bearer {api_key}"}

    params = {
        "term": "chef's table",
        "latitude": origin.lat,
        "longitude": origin.lon,
        "radius": yelp_radius_m(radius_km),
        "limit": max(1, min(50, int(limit))),
        "sort_by": "distance",
    }

    try:
        r = session.get(YELP_SEARCH_URL, params=params, headers=headers, timeout=20)
    except requests.RequestException as e:
        raise AppError(f"Yelp request failed: {e}") from e

    if r.status_code == 401:
        raise AppError("Yelp API key rejected (401). Check YELP_API_KEY.")

    if r.status_code != 200:
        raise AppError(f"Yelp search failed: HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise AppError("Yelp returned invalid JSON") from e

    biz = data.get("businesses")
    if not isinstance(biz, list):
        raise AppError("Yelp response missing businesses")

    out: list[Place] = []
    for b in biz:
        if not isinstance(b, dict):
            continue
        name = b.get("name")
        coords = b.get("coordinates")
        if not name or not isinstance(coords, dict):
            continue
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        if lat is None or lon is None:
            continue

        p = Point(float(lat), float(lon))
        dist = haversine_km(origin, p)

        loc = b.get("location") if isinstance(b.get("location"), dict) else {}
        addr = None
        if isinstance(loc, dict):
            display = loc.get("display_address")
            if isinstance(display, list):
                addr = ", ".join(str(x) for x in display if x)

        out.append(
            Place(
                name=str(name),
                lat=p.lat,
                lon=p.lon,
                distance_km=dist,
                source="yelp",
                address=addr,
                phone=str(b.get("display_phone")) if b.get("display_phone") else None,
                url=str(b.get("url")) if b.get("url") else None,
                rating=float(b["rating"]) if isinstance(b.get("rating"), (int, float)) else None,
                reviews=int(b["review_count"]) if isinstance(b.get("review_count"), int) else None,
                score=0,
            )
        )

    out.sort(key=lambda x: (x.distance_km, x.name.lower()))
    return out[:limit]


def print_results(origin_label: str, origin: Point, radius_km: float, places: list[Place]) -> None:
    print(f"Search center: {origin_label}")
    print(f"Coordinates: {origin.lat:.5f}, {origin.lon:.5f} | Radius: {radius_km:.1f} km")
    print()

    if not places:
        print("No results found.")
        return

    # Decide columns based on source.
    show_yelp = any(p.source == "yelp" for p in places)
    show_osm = any(p.source == "osm" for p in places)

    headers = ["#", "Distance", "Name"]
    if show_yelp:
        headers += ["Rating", "Reviews"]
    if show_osm:
        headers += ["Score"]
    headers += ["Address", "Phone", "URL"]

    rows: list[list[str]] = []
    for i, p in enumerate(places, start=1):
        row = [
            str(i),
            format_km(p.distance_km),
            shorten(p.name, 38),
        ]
        if show_yelp:
            row += [
                f"{p.rating:.1f}" if p.rating is not None else "-",
                str(p.reviews) if p.reviews is not None else "-",
            ]
        if show_osm:
            row += [str(p.score) if p.source == "osm" else "-"]

        row += [
            shorten(p.address or "-", 34),
            shorten(p.phone or "-", 18),
            shorten(p.url or "-", 40),
        ]
        rows.append(row)

    widths = [len(h) for h in headers]
    for r in rows:
        for idx, cell in enumerate(r):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(r: Iterable[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))

    if show_osm and all(p.source == "osm" and p.score == 0 for p in places):
        print(
            "\nNote: OpenStreetMap results rarely include explicit 'chef\'s table' metadata. "
            "For more accurate matches, set YELP_API_KEY and run with --source yelp."
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chefs_table.py",
        description="Find chef's table restaurants near you (terminal app).",
    )

    loc = p.add_argument_group("Location")
    loc.add_argument(
        "--near",
        help="Address/city to search near (uses OSM Nominatim geocoding)",
    )
    loc.add_argument("--lat", type=float, help="Latitude")
    loc.add_argument("--lon", type=float, help="Longitude")
    loc.add_argument(
        "--use-ip",
        action="store_true",
        help="Use approximate IP geolocation as the search center",
    )

    srch = p.add_argument_group("Search")
    srch.add_argument(
        "--radius-km",
        type=float,
        default=5.0,
        help="Search radius in kilometers (default: 5)",
    )
    srch.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Max results to show (default: 15)",
    )
    srch.add_argument(
        "--source",
        choices=["auto", "yelp", "osm"],
        default="auto",
        help="Data source to use (default: auto)",
    )

    return p.parse_args(argv)


def resolve_origin(
    session: requests.Session, args: argparse.Namespace
) -> tuple[Point, str]:
    lat = args.lat
    lon = args.lon

    if lat is not None or lon is not None:
        if lat is None or lon is None:
            raise AppError("Both --lat and --lon are required when specifying coordinates")
        return Point(float(lat), float(lon)), "(custom coordinates)"

    if args.near:
        return geocode(session, args.near)

    if args.use_ip:
        return get_ip_location(session)

    # Interactive fallback
    try:
        print("No location provided.")
        ans = input("Use approximate IP location? [Y/n]: ").strip().lower()
    except EOFError:
        ans = ""

    if ans in ("", "y", "yes"):
        return get_ip_location(session)

    try:
        q = input("Enter a city/address to search near: ").strip()
    except EOFError:
        q = ""

    if not q:
        raise AppError("No location provided.")

    return geocode(session, q)


def choose_source(args: argparse.Namespace) -> Literal["yelp", "osm"]:
    api_key = os.getenv("YELP_API_KEY")

    if args.source == "yelp":
        if not api_key:
            raise AppError("--source yelp selected but YELP_API_KEY is not set")
        return "yelp"

    if args.source == "osm":
        return "osm"

    # auto
    if api_key:
        return "yelp"
    return "osm"


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.radius_km <= 0:
        raise AppError("--radius-km must be > 0")
    if args.limit <= 0:
        raise AppError("--limit must be > 0")

    session = _http_session()

    origin, origin_label = resolve_origin(session, args)

    source = choose_source(args)

    start = time.time()
    if source == "yelp":
        places = search_yelp(
            session,
            origin=origin,
            radius_km=float(args.radius_km),
            limit=int(args.limit),
            api_key=os.environ["YELP_API_KEY"],
        )
    else:
        places = search_osm(
            session,
            origin=origin,
            radius_km=float(args.radius_km),
            limit=int(args.limit),
        )

    elapsed_ms = int((time.time() - start) * 1000)
    print_results(origin_label=origin_label, origin=origin, radius_km=float(args.radius_km), places=places)
    print(f"\nSource: {source} | Results: {len(places)} | Took: {elapsed_ms} ms")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except AppError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
