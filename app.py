import streamlit as st
import pandas as pd
import math
import os
import csv
import base64
import uuid
import hashlib
from datetime import datetime

# ============================================================
# FILES
# ============================================================
DATA_FILE = "farm_data.csv"
ACCOUNTS_FILE = "accounts.csv"
MESSAGES_FILE = "messages.csv"
POSTS_FILE = "posts.csv"
LIKES_FILE = "likes.csv"
PROFILE_PIC_DIR = "profile_pics"
POST_IMAGE_DIR = "post_images"

os.makedirs(PROFILE_PIC_DIR, exist_ok=True)
os.makedirs(POST_IMAGE_DIR, exist_ok=True)

APP_NAME = "AgriEntropy"
CITY_NAME = "San Pedro"
PROVINCE_NAME = "Laguna"

# ============================================================
# OFFICIAL BARANGAY REFERENCE LIST
# ============================================================
SAN_PEDRO_BARANGAYS = [
    "Bagong Silang", "Calendola", "Chrysanthemum", "Cuyab", "Estrella",
    "Fatima", "G.S.I.S.", "Landayan", "Langgam", "Laram", "Magsaysay",
    "Maharlika", "Narra", "Nueva", "Pacita 1", "Pacita 2", "Poblacion",
    "Riverside", "Rosario", "Sampaguita Village", "San Antonio",
    "San Lorenzo Ruiz", "San Roque", "San Vicente", "Santo Nino",
    "United Bayanihan", "United Better Living",
]
BARANGAY_PLACEHOLDER = "Select your barangay"
GOV_LOCATION_PLACEHOLDER = "City Hall (all barangays)"

MAX_CROPS = 50


# ============================================================
# TEXT HELPERS
# ============================================================
def clean_text(value):
    return "" if value is None else str(value).strip()


def normalize_name(value):
    return clean_text(value).casefold()


def safe_filename(name):
    return "".join(c if c.isalnum() else "_" for c in clean_text(name))


def initials(name):
    parts = [p for p in clean_text(name).split(" ") if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def img_to_data_uri(path):
    if not path or not os.path.isfile(path):
        return None
    ext = "png" if path.lower().endswith("png") else "jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{encoded}"


# ============================================================
# ACCOUNT SECURITY (PIN)
# ============================================================
def hash_pin(pin, name):
    salted = f"{normalize_name(name)}::{clean_text(pin)}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def verify_pin(name, pin, stored_hash):
    if not stored_hash:
        return False
    return hash_pin(pin, name) == stored_hash


def valid_pin_format(pin):
    pin = clean_text(pin)
    return pin.isdigit() and 4 <= len(pin) <= 6


# ============================================================
# DIVERSITY CALCULATIONS
# ============================================================
def calculate_diversity(crop_names, crop_areas):
    valid_crops = [(n, a) for n, a in zip(crop_names, crop_areas) if a and a > 0]
    total_area = sum(a for _, a in valid_crops)

    if total_area == 0 or not valid_crops:
        return None

    proportions = [a / total_area for _, a in valid_crops]
    entropy = -sum(p * math.log(p) for p in proportions if p > 0)
    num_valid_crops = len(valid_crops)
    effective_crops = math.exp(entropy) if entropy > 0 else 1.0

    max_entropy = math.log(num_valid_crops) if num_valid_crops > 1 else None
    variety_score = (entropy / max_entropy) * 100 if max_entropy and max_entropy > 0 else 0.0

    crop_breakdown = []
    for (n, a), p in zip(valid_crops, proportions):
        remaining_pct = ((total_area - a) / total_area) * 100
        crop_breakdown.append({
            "name": n, "area": a, "share_pct": p * 100,
            "remaining_if_lost_pct": remaining_pct,
        })
    crop_breakdown.sort(key=lambda c: c["share_pct"], reverse=True)

    return {
        "variety_score": variety_score, "effective_crops": effective_crops,
        "num_crops": num_valid_crops, "valid_crops": valid_crops, "entropy": entropy,
        "total_area": total_area, "crop_breakdown": crop_breakdown,
    }


def verdict_for_score(score):
    if score >= 70:
        return "Strong spread", "Land is spread evenly across crops. A shock to one crop is unlikely to sink the whole farm."
    elif score >= 40:
        return "Getting there", "A reasonable mix, but one or two crops still dominate the land."
    return "Needs more variety", "Most of the land depends on very few crops. Spreading out would lower the risk."


def entropy_from_pairs(name_value_pairs):
    """
    Raw Shannon entropy H (nats), after merging repeated crop names.
    Crop names are matched case-/whitespace-insensitively so "Rice",
    "rice", "RICE" all count as ONE category instead of several.
    """
    totals = {}
    for n, v in name_value_pairs:
        if v and v > 0:
            key = normalize_name(n)
            if not key:
                continue
            totals[key] = totals.get(key, 0) + v
    total = sum(totals.values())
    if total == 0:
        return None
    props = [v / total for v in totals.values()]
    return -sum(p * math.log(p) for p in props if p > 0)


def compute_scale_diversity(scope_df):
    if scope_df is None or scope_df.empty:
        return None

    site_entropies = []
    for _, group in scope_df.groupby("farmer_name"):
        h = entropy_from_pairs(list(zip(group["crop"], group["area"])))
        if h is not None:
            site_entropies.append(h)

    if not site_entropies:
        return None

    alpha = sum(site_entropies) / len(site_entropies)
    gamma = entropy_from_pairs(list(zip(scope_df["crop"], scope_df["area"])))
    beta = (gamma / alpha) if (alpha and alpha > 0 and gamma is not None) else None

    return {"alpha": alpha, "gamma": gamma, "beta": beta, "num_sites": len(site_entropies)}


def aggregate_crop_pairs(scope_df):
    """
    Combine every farm's crop entries in scope_df into one pooled
    (crop name, total area) list -- the same name-normalization used
    for gamma diversity, so "Rice"/"rice"/"RICE" merge into one crop
    instead of being double-counted.
    """
    totals = {}
    display_names = {}
    for _, row in scope_df.iterrows():
        key = normalize_name(row["crop"])
        area = row["area"]
        if not key or not area or area <= 0:
            continue
        totals[key] = totals.get(key, 0) + area
        display_names.setdefault(key, clean_text(row["crop"]))
    names = [display_names[k] for k in totals]
    areas = [totals[k] for k in totals]
    return names, areas


def verdict_for_beta(beta):
    if beta is None:
        return "Not enough data yet", "Add more farms to compare variety across the community."
    if beta >= 1.5:
        return "Highly varied farms", "Different farms are growing quite different crops from one another -- a good sign for the city's overall food supply."
    elif beta >= 1.15:
        return "Some variety across farms", "Farms share a fair number of crops, with some real differences from one barangay to another."
    return "Farms grow similar crops", "Most farms are growing nearly the same mix. Encouraging different crops in different areas could help spread the risk citywide."


def community_overview_card(beta, num_sites):
    label, sub = verdict_for_beta(beta)
    beta_display = f"{beta:.2f}x" if beta is not None else "\u2014"
    farms_word = "farm" if num_sites == 1 else "farms"
    st.markdown(f"""
    <div class="gauge-card">
        <div class="eyebrow" style="color:var(--gold-300);">Community diversity overview &middot; {num_sites} {farms_word} reporting</div>
        <div class="gauge-top">
            <div class="gauge-number">{beta_display}</div>
            <div class="gauge-verdict">{label}</div>
        </div>
        <div class="gauge-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# STORAGE: ACCOUNTS
# ============================================================
ACCOUNT_COLUMNS = ["name", "role", "barangay", "pin_hash"]


def load_accounts():
    if not os.path.isfile(ACCOUNTS_FILE):
        return pd.DataFrame(columns=ACCOUNT_COLUMNS)
    try:
        df = pd.read_csv(ACCOUNTS_FILE)
    except Exception:
        return pd.DataFrame(columns=ACCOUNT_COLUMNS)
    for col in ACCOUNT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ACCOUNT_COLUMNS].fillna("")


def account_exists(name):
    accounts = load_accounts()
    if accounts.empty:
        return False
    return (accounts["name"].astype(str).map(normalize_name) == normalize_name(name)).any()


def get_account(name):
    accounts = load_accounts()
    if accounts.empty:
        return None
    matches = accounts[accounts["name"].astype(str).map(normalize_name) == normalize_name(name)]
    return None if matches.empty else matches.iloc[0].to_dict()


def register_account(name, role, barangay, pin):
    name, role, barangay = clean_text(name), clean_text(role), clean_text(barangay)
    if not name:
        return False, "Enter your full name."
    if account_exists(name):
        return False, "That name is already registered. Try signing in instead."
    if not valid_pin_format(pin):
        return False, "Choose a 4-6 digit PIN (numbers only)."
    accounts = load_accounts()
    new_row = pd.DataFrame([{
        "name": name, "role": role, "barangay": barangay,
        "pin_hash": hash_pin(pin, name),
    }])
    accounts = pd.concat([accounts, new_row], ignore_index=True)
    accounts.to_csv(ACCOUNTS_FILE, index=False)
    return True, "Account created."


# ============================================================
# STORAGE: FARM / CROP DATA
# ============================================================
FARM_COLUMNS = ["farmer_name", "barangay", "crop", "area"]


def load_all_farms():
    if not os.path.isfile(DATA_FILE):
        return pd.DataFrame(columns=FARM_COLUMNS)
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame(columns=FARM_COLUMNS)
    for col in FARM_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("farmer_name", "barangay", "crop") else 0.0
    return df[FARM_COLUMNS]


def save_farm_data(farmer_name, barangay, valid_crops):
    all_farms = load_all_farms()
    key = normalize_name(farmer_name)
    if not all_farms.empty:
        all_farms = all_farms[all_farms["farmer_name"].astype(str).map(normalize_name) != key]
    new_rows = pd.DataFrame(
        [{"farmer_name": farmer_name, "barangay": barangay, "crop": n, "area": a} for n, a in valid_crops]
    )
    updated = pd.concat([all_farms, new_rows], ignore_index=True)
    updated.to_csv(DATA_FILE, index=False)


def load_farmer_crops(farmer_name):
    all_farms = load_all_farms()
    if all_farms.empty:
        return []
    key = normalize_name(farmer_name)
    mine = all_farms[all_farms["farmer_name"].astype(str).map(normalize_name) == key]
    return list(zip(mine["crop"], mine["area"])) if not mine.empty else []


def delete_farm_data(farmer_name):
    all_farms = load_all_farms()
    key = normalize_name(farmer_name)
    all_farms = all_farms[all_farms["farmer_name"].astype(str).map(normalize_name) != key]
    all_farms.to_csv(DATA_FILE, index=False)


# ============================================================
# STORAGE: MESSAGES
# ============================================================
MESSAGE_COLUMNS = ["from_name", "to_name", "message", "timestamp"]


def send_message(from_name, to_name, message):
    file_exists = os.path.isfile(MESSAGES_FILE)
    with open(MESSAGES_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(MESSAGE_COLUMNS)
        writer.writerow([from_name, to_name, message, datetime.now().strftime("%Y-%m-%d %H:%M")])


def load_conversation(user_a, user_b):
    if not os.path.isfile(MESSAGES_FILE):
        return pd.DataFrame(columns=MESSAGE_COLUMNS)
    try:
        df = pd.read_csv(MESSAGES_FILE)
    except Exception:
        return pd.DataFrame(columns=MESSAGE_COLUMNS)
    for col in MESSAGE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[MESSAGE_COLUMNS]
    return df[
        ((df["from_name"] == user_a) & (df["to_name"] == user_b)) |
        ((df["from_name"] == user_b) & (df["to_name"] == user_a))
    ]


def unread_partners_count(current_user):
    if not os.path.isfile(MESSAGES_FILE):
        return 0
    try:
        df = pd.read_csv(MESSAGES_FILE)
    except Exception:
        return 0
    if "to_name" not in df.columns or "from_name" not in df.columns:
        return 0
    return df[df["to_name"] == current_user]["from_name"].nunique()


# ============================================================
# STORAGE: POSTS + LIKES
# ============================================================
POST_COLUMNS = ["post_id", "author", "barangay", "caption", "image_path", "timestamp"]


def save_post(author, barangay, caption, image_path):
    post_id = uuid.uuid4().hex[:10]
    file_exists = os.path.isfile(POSTS_FILE)
    with open(POSTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(POST_COLUMNS)
        writer.writerow([post_id, author, barangay, caption, image_path or "",
                          datetime.now().strftime("%Y-%m-%d %H:%M")])
    return post_id


def load_posts(author_filter=None, barangay_filter=None):
    if not os.path.isfile(POSTS_FILE):
        df = pd.DataFrame(columns=POST_COLUMNS)
    else:
        try:
            df = pd.read_csv(POSTS_FILE, dtype={"image_path": str})
        except Exception:
            df = pd.DataFrame(columns=POST_COLUMNS)
        for col in POST_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[POST_COLUMNS].fillna("")

    if author_filter:
        df = df[df["author"] == author_filter]
    if barangay_filter and barangay_filter != GOV_LOCATION_PLACEHOLDER:
        df = df[df["barangay"] == barangay_filter]
    return df.sort_values("timestamp", ascending=False)


def seed_posts_if_empty():
    if not os.path.isfile(POSTS_FILE):
        save_post("Aling Nena's Farm", "San Antonio",
                   "Harvested kamote and okra this week. Still hoping to find talong seedlings to trade.", "")
        save_post("Kuya Bert's Farm", "Landayan",
                   "Looking for mungbean seeds. Anyone have extra to share or trade?", "")


def load_likes():
    if not os.path.isfile(LIKES_FILE):
        return pd.DataFrame(columns=["post_id", "liker"])
    try:
        df = pd.read_csv(LIKES_FILE)
    except Exception:
        return pd.DataFrame(columns=["post_id", "liker"])
    for col in ["post_id", "liker"]:
        if col not in df.columns:
            df[col] = ""
    return df[["post_id", "liker"]]


def toggle_like(post_id, user):
    likes = load_likes()
    already = not likes[(likes["post_id"] == post_id) & (likes["liker"] == user)].empty
    if already:
        likes = likes[~((likes["post_id"] == post_id) & (likes["liker"] == user))]
        likes.to_csv(LIKES_FILE, index=False)
    else:
        file_exists = os.path.isfile(LIKES_FILE)
        with open(LIKES_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["post_id", "liker"])
            writer.writerow([post_id, user])


def like_count(likes_df, post_id):
    return len(likes_df[likes_df["post_id"] == post_id])


def user_liked(likes_df, post_id, user):
    return not likes_df[(likes_df["post_id"] == post_id) & (likes_df["liker"] == user)].empty


# ============================================================
# STORAGE: CITY PRODUCTION RECORDS (Exp 3 -- real Agri Office data)
# ------------------------------------------------------------
# This is a SEPARATE dataset from farm_data.csv. Farm_data.csv is
# per-farm recorded AREA, entered by individual farmers under My
# Farm. This file is the City Agriculture Office's own city-level,
# per-commodity PRODUCTION records (in metric tons) -- not tied to
# any specific farm, garden, or barangay. It exists so the
# single-crop-loss resilience simulation (R_j = (T - y_j) / T x 100)
# can run on the real recorded tonnage, not area used as a stand-in.
# ============================================================
PRODUCTION_FILE = "production_records.csv"
PRODUCTION_COLUMNS = ["year", "crop", "production_mt"]


def load_production_records():
    if not os.path.isfile(PRODUCTION_FILE):
        return pd.DataFrame(columns=PRODUCTION_COLUMNS)
    try:
        df = pd.read_csv(PRODUCTION_FILE)
    except Exception:
        return pd.DataFrame(columns=PRODUCTION_COLUMNS)
    for col in PRODUCTION_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("year", "crop") else 0.0
    df["production_mt"] = pd.to_numeric(df["production_mt"], errors="coerce").fillna(0.0)
    return df[PRODUCTION_COLUMNS]


def save_production_record(year, crop, production_mt):
    year, crop = clean_text(year), clean_text(crop)
    if not year or not crop:
        return False, "Enter both a year/period and a crop name."
    if production_mt is None or production_mt <= 0:
        return False, "Production must be greater than zero."
    df = load_production_records()
    key_crop = normalize_name(crop)
    if not df.empty:
        mask = (df["year"].astype(str) == year) & (df["crop"].astype(str).map(normalize_name) == key_crop)
        df = df[~mask]
    new_row = pd.DataFrame([{"year": year, "crop": crop, "production_mt": production_mt}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PRODUCTION_FILE, index=False)
    return True, "Record saved."


def delete_production_record(year, crop):
    df = load_production_records()
    key_crop = normalize_name(crop)
    df = df[~((df["year"].astype(str) == clean_text(year)) & (df["crop"].astype(str).map(normalize_name) == key_crop))]
    df.to_csv(PRODUCTION_FILE, index=False)


def seed_production_if_empty():
    """
    Seeds the file with the San Pedro City Agriculture Office's actual
    verified 2025 and 2026 (Jan-Jul) per-crop production totals -- the
    same figures already hand-extracted and cross-checked against the
    source spreadsheet for Exp 3 (2025 T = 4.6289 Mt across 17 crops;
    2026 partial-year T = 0.6300 Mt across 15 crops, Aug-Dec not yet
    recorded). These are real recorded values, not simulated ones.
    """
    if os.path.isfile(PRODUCTION_FILE):
        return
    records_2025 = [
        ("Okra", 1.1131), ("Kangkong (upland)", 1.0682), ("Lettuce", 0.7665),
        ("String beans", 0.4270), ("Eggplant", 0.2708), ("Hot Pepper/Siling Labuyo", 0.2085),
        ("Siling Panigang", 0.2000), ("Pechay", 0.1580), ("Mustard", 0.1013),
        ("Ampalaya", 0.0950), ("Tomato", 0.0633), ("Cucumber", 0.0372),
        ("Patola", 0.0370), ("Squash", 0.0350), ("Upo", 0.0270),
        ("Sweet Potato", 0.0110), ("Gabi", 0.0100),
    ]
    records_2026 = [
        ("Eggplant", 0.1828), ("Lettuce", 0.1183), ("Pechay", 0.0746),
        ("Kangkong (upland)", 0.0689), ("Mustard", 0.0500), ("Cucumber", 0.0274),
        ("Tomato", 0.0205), ("Ampalaya", 0.0203), ("Okra", 0.0201),
        ("String beans", 0.0173), ("Upo", 0.0160), ("Patola", 0.0053),
        ("Hot Pepper/Siling Labuyo", 0.0050), ("Radish", 0.0025), ("Siling Panigang", 0.0010),
    ]
    rows = [{"year": "2025", "crop": c, "production_mt": v} for c, v in records_2025]
    rows += [{"year": "2026 (Jan-Jul)", "crop": c, "production_mt": v} for c, v in records_2026]
    pd.DataFrame(rows, columns=PRODUCTION_COLUMNS).to_csv(PRODUCTION_FILE, index=False)


def compute_production_resilience(year):
    """
    City-level, per-commodity single-crop-loss simulation, per the
    manuscript's Exp 3 formula: R_j = (T - y_j) / T x 100, where T is
    total recorded production for the chosen year/period and y_j is
    one crop's own recorded production. This is city-wide, not tied
    to any specific farm or barangay.
    """
    df = load_production_records()
    df = df[df["year"].astype(str) == clean_text(year)]
    if df.empty:
        return None
    total = df["production_mt"].sum()
    if total <= 0:
        return None
    rows = []
    for _, r in df.sort_values("production_mt", ascending=False).iterrows():
        y_j = float(r["production_mt"])
        rows.append({
            "crop": r["crop"], "production_mt": y_j,
            "share_pct": (y_j / total) * 100,
            "remaining_pct": ((total - y_j) / total) * 100,
        })
    return {"year": year, "total": total, "rows": rows}


# ============================================================
# PROFILE / POST IMAGES
# ============================================================
def save_profile_picture(name, uploaded_file):
    path = os.path.join(PROFILE_PIC_DIR, f"{safe_filename(name)}.png")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def get_profile_picture_path(name):
    path = os.path.join(PROFILE_PIC_DIR, f"{safe_filename(name)}.png")
    return path if os.path.isfile(path) else None


def save_post_image(uploaded_file):
    filename = f"{uuid.uuid4().hex[:10]}.png"
    path = os.path.join(POST_IMAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ============================================================
# APP SETUP
# ============================================================
seed_posts_if_empty()
seed_production_if_empty()
st.set_page_config(page_title=APP_NAME, layout="centered", initial_sidebar_state="collapsed")

defaults = {
    "auth_stage": "welcome",
    "pending_role": None,
    "username": "",
    "role": "",
    "barangay": "",
    "auth_mode": "Sign in",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SESSION HELPERS
# ------------------------------------------------------------
# NOTE ON A REMOVED FEATURE: earlier drafts tried to keep an account
# signed in across browser refreshes by writing the username into the
# page's URL (?u=...) and reading it back on load. That was removed --
# it let anyone who had (or guessed) that URL sign in as that account
# with NO PIN check at all, which defeated the whole point of adding
# PINs. Streamlit's own session state already keeps a signed-in user
# logged in for as long as their browser tab stays open (which is all
# a live demo needs); it just won't survive a manual page refresh.
# That's the correct, secure trade-off, so remember_login/forget_login
# below are now intentionally empty -- kept only so the rest of the
# code calling them doesn't need to change.
# ============================================================
def remember_login(name, role, barangay):
    pass


def forget_login():
    pass


def reset_transient_state():
    """Wipe every piece of per-account screen/calculator state so a new
    sign-in or a freshly created account never inherits leftover data
    left behind in this browser tab by whoever used this device or
    session before."""
    keep = set(defaults.keys())
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]


def log_out():
    forget_login()
    reset_transient_state()
    for key, value in defaults.items():
        st.session_state[key] = value


# ============================================================
# VISUAL IDENTITY
# ============================================================
def terrace_svg(band_colors, height=210, sun=True, sun_color="#E8A33D"):
    n = len(band_colors)
    step_h = height / n
    paths = []
    for i, color in enumerate(band_colors):
        top = i * step_h
        wobble = 14 if i % 2 == 0 else 22
        paths.append(
            f'<path d="M0,{top+step_h:.0f} C 200,{top+step_h-wobble:.0f} 420,{top+step_h+wobble:.0f} '
            f'700,{top+step_h-wobble/2:.0f} S 1100,{top+step_h+wobble:.0f} 1200,{top+step_h:.0f} '
            f'L1200,{height} L0,{height} Z" fill="{color}" opacity="{0.55 + i*0.09:.2f}"/>'
        )
    sun_el = f'<circle cx="1080" cy="46" r="30" fill="{sun_color}" opacity="0.9"/>' if sun else ""
    return (
        f'<svg viewBox="0 0 1200 {height}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:100%;">'
        f'{sun_el}{"".join(paths)}</svg>'
    )


def role_icon_svg(kind):
    if kind == "farmer":
        return '''<svg viewBox="0 0 64 64" width="34" height="34" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 6c-8 6-8 16-8 16h16s0-10-8-16z" fill="#E8A33D"/>
            <ellipse cx="32" cy="24" rx="17" ry="5" fill="#2F5233"/>
            <path d="M20 24 L23 46 h18 L44 24 Z" fill="#5C8A52"/>
            <path d="M26 30 v16 M32 28 v18 M38 30 v16" stroke="#2F5233" stroke-width="2" fill="none"/>
        </svg>'''
    return '''<svg viewBox="0 0 64 64" width="34" height="34" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="28" width="44" height="24" fill="#3E7E90"/>
        <path d="M8 28 L32 12 L56 28 Z" fill="#2F5D6B"/>
        <rect x="16" y="34" width="5" height="18" fill="#EAF3F5"/>
        <rect x="26" y="34" width="5" height="18" fill="#EAF3F5"/>
        <rect x="36" y="34" width="5" height="18" fill="#EAF3F5"/>
        <rect x="46" y="34" width="5" height="18" fill="#EAF3F5"/>
        <rect x="8" y="52" width="48" height="4" fill="#1F3A24"/>
    </svg>'''


def phone_status_bar():
    st.markdown("""
    <div class="status-bar">
        <span>9:41</span>
        <span class="status-icons">
            <svg width="16" height="11" viewBox="0 0 16 11" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M1 8L1 8C4 4 12 4 15 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none"/>
                <path d="M4 9.5C6 7.5 10 7.5 12 9.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none"/>
                <circle cx="8" cy="10.5" r="1" fill="currentColor"/>
            </svg>
            <svg width="20" height="11" viewBox="0 0 20 11" xmlns="http://www.w3.org/2000/svg">
                <rect x="1" y="1" width="16" height="9" rx="2" stroke="currentColor" stroke-width="1.2" fill="none"/>
                <rect x="2.5" y="2.5" width="12" height="6" rx="1" fill="currentColor"/>
                <rect x="18" y="3.5" width="1.6" height="4" rx="0.8" fill="currentColor"/>
            </svg>
        </span>
    </div>
    """, unsafe_allow_html=True)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Work+Sans:wght@400;500;600;700&family=DM+Mono:wght@500&display=swap');

    :root{
        --palay-900:#1F3A24; --palay-700:#2F5233; --palay-500:#5C8A52; --palay-300:#9CC28C;
        --gold-500:#E8A33D; --gold-300:#F3C983;
        --soil-700:#6B4226; --soil-500:#8C5A3C;
        --sky-500:#3E7E90;
        --husk-50:#FCFAF2; --husk-100:#F6F1DF; --ink:#26301F;
        --desk:#1B1712;
    }

    html {
        overflow-y: scroll !important;
        overflow-x: hidden !important;
    }

    html, body, [class*="css"]  { font-family:'Work Sans', sans-serif; color:var(--ink); }
    h1,h2,h3, .display-font { font-family:'Baloo 2', sans-serif; }
    .eyebrow { font-family:'DM Mono', monospace; letter-spacing:.14em; text-transform:uppercase;
               font-size:.72rem; color:var(--palay-700); opacity:.8; }

    p, span, label, div, li, td, th,
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    [data-testid="stTable"] * ,
    .stRadio label, .stRadio p, .stRadio span,
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: var(--ink) !important;
    }

    textarea,
    input[type="text"],
    input[type="number"],
    input[type="search"],
    input[type="password"],
    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid #E7E0C4 !important;
    }
    textarea::placeholder, input::placeholder {
        color: #9AA391 !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"],
    section[data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1.5px dashed #C9C09A !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: var(--ink) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--gold-500) !important;
        color: var(--palay-900) !important;
        border: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        color: var(--palay-900) !important;
    }

    .hero-title, .hero-sub, .hero-brand,
    .auth-side, .auth-side *,
    .gauge-card, .gauge-card *,
    .profile-score-badge, .profile-score-badge *,
    .chat-bubble-me,
    .app-header .brand {
        color: #fff !important;
    }
    .hero-brand, .hero-sub, .auth-side .eyebrow,
    .gauge-card .eyebrow, .gauge-verdict,
    .profile-score-badge .eyebrow {
        color: var(--gold-300) !important;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
        background: var(--desk) !important;
    }
    .stAppViewContainer, .stMain { display:flex; justify-content:center; }

    .block-container {
        padding: 44px 1.3rem 14px 1.3rem !important;
        width: min(393px, 92vw) !important;
        max-width: min(393px, 92vw) !important;
        height: min(852px, 90vh);
        margin: 26px auto !important;
        background: var(--husk-50);
        border-radius: 46px;
        box-shadow: 0 40px 70px rgba(0,0,0,.5), 0 0 0 9px #100D09, 0 0 0 12px #322B22;
        position: relative;
        overflow-y: auto;
        overflow-x: hidden;
    }
    .block-container::before {
        content:"";
        position:absolute; top:12px; left:50%; transform:translateX(-50%);
        width:84px; height:22px; background:#100D09; border-radius:14px; z-index:80;
    }
    .block-container::after { display: none; }
    .status-bar { display:flex; justify-content:space-between; align-items:center;
                  font-family:'DM Mono',monospace; font-size:.78rem; font-weight:600;
                  color:var(--ink); padding: 2px 4px 10px 4px; }
    .status-icons { display:flex; align-items:center; gap:6px; color:var(--ink); }

    @media (max-width: 480px) {
        .block-container {
            max-width: 100% !important;
            width: 100%;
            height: auto;
            min-height: 100vh;
            margin: 0 !important;
            border-radius: 0;
            box-shadow: none;
            padding: 20px 1rem 26px 1rem !important;
            overflow-y: visible;
        }
        .block-container::before,
        .block-container::after {
            display: none;
        }
        .status-bar { display: none; }
        html {
            background: var(--husk-50) !important;
        }
        .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] {
            margin: 10px -1rem -26px -1rem !important;
            border-radius: 0;
            padding-bottom: calc(.7rem + env(safe-area-inset-bottom, 0px));
        }
    }

    #MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }

    .bleed { margin-left: -1.15rem; margin-right: -1.15rem; }

    .hero-shell { position:relative; overflow:hidden; background:var(--palay-900);
                  padding: 2.2rem 1.3rem 190px 1.3rem; margin-bottom:0; border-radius: 0 0 28px 28px; }
    .hero-inner { position:relative; z-index:2; text-align:center; }
    .hero-shell .terrace-layer { position:absolute; left:0; right:0; bottom:0; height:170px; z-index:1; }
    .hero-brand { font-size:.85rem; letter-spacing:.22em; text-transform:uppercase;
                  font-family:'DM Mono',monospace; opacity:.9; margin-bottom:.6rem; }
    .hero-title { font-size:2.3rem; font-weight:800; line-height:1.05; margin:0 0 .7rem 0; }
    .hero-sub { font-size:1rem; margin:0 auto; font-weight:500; }

    .role-card { background:var(--husk-50); border-radius:22px; padding:1.5rem 1.3rem 1.1rem 1.3rem;
                 box-shadow: 0 14px 30px rgba(31,58,36,.14); border-top:6px solid var(--palay-500);
                 height:100%; }
    .role-card.sky { border-top-color:var(--sky-500); }
    .role-icon { width:58px; height:58px; border-radius:16px; display:flex; align-items:center;
                 justify-content:center; background:rgba(92,138,82,.14); margin-bottom:.7rem; }
    .role-card.sky .role-icon { background:rgba(62,126,144,.14); }
    .role-card h3 { margin:.2rem 0 .5rem 0; font-size:1.25rem; }
    .role-card p { color:#54604A !important; font-size:.9rem; line-height:1.5; margin-bottom:0; }

    .auth-side { background:var(--palay-700); border-radius:22px; padding:1.8rem 1.5rem;
                 position:relative; overflow:hidden; min-height:220px; }
    .auth-side .terrace-layer { position:absolute; left:0; right:0; bottom:0; height:110px; z-index:0; }
    .auth-side-content { position:relative; z-index:1; }
    .auth-side h2 { font-size:1.5rem; margin-bottom:.5rem; }
    .auth-side p { font-size:.92rem; line-height:1.5; }
    .auth-card { background:var(--husk-50); border-radius:22px; padding:1.6rem 1.5rem 1.2rem 1.5rem;
                 box-shadow: 0 10px 26px rgba(31,58,36,.10); }

    .field-note { font-size:.82rem; color:#6B7861 !important; margin-top:-.6rem; margin-bottom:.8rem; }
    .banner { border-radius:14px; padding:.75rem 1rem; font-size:.9rem; margin:.4rem 0 1rem 0; }
    .banner.warn { background:#FBECD2; color:#7A4A12 !important; border:1px solid #EFCB8C; }
    .banner.warn * { color:#7A4A12 !important; }
    .banner.ok { background:#E3EEDD; color:var(--palay-900) !important; border:1px solid var(--palay-300); }
    .banner.ok * { color:var(--palay-900) !important; }

    div[role="radiogroup"] {
        gap:.35rem; background:var(--husk-100); padding:.32rem; border-radius:999px;
        display:inline-flex; border:1px solid #E7E0C4; flex-wrap:wrap;
    }
    div[role="radiogroup"] label {
        background:transparent; border-radius:999px !important; padding:.4rem .9rem !important;
        margin:0 !important; transition:.15s;
    }
    div[role="radiogroup"] label * { color: var(--ink) !important; list-style:none !important; }
    div[role="radiogroup"] li::marker { content:"" !important; }
    div[role="radiogroup"] label:has(input:checked) { background:var(--palay-700); }
    div[role="radiogroup"] label:has(input:checked) * { color:#fff !important; }
    div[role="radiogroup"] input { display:none; }

    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] {
        position: sticky;
        bottom: 0;
        z-index: 90;
        gap: 0;
        width: 100%;
        margin: 10px -1.3rem -14px -1.3rem !important;
        padding: .5rem .2rem calc(.4rem + env(safe-area-inset-bottom, 0px)) .2rem;
        background: var(--palay-900);
        border-top: 1px solid var(--palay-700);
        box-shadow: 0 -8px 20px rgba(0,0,0,.22);
        border-radius: 0 0 40px 40px;
        display: flex;
        flex-wrap: nowrap;
        align-items: stretch;
        justify-content: space-between;
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] > * {
        flex: 1 1 0;
        min-width: 64px;
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] label {
        width: 100%;
        height: 100%;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 2px;
        text-align: center;
        background: transparent;
        border-radius: 14px !important;
        padding: .35rem .05rem !important;
        margin: 0 !important;
        transition: .15s;
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] label * {
        color: #B9C4AF !important;
        font-weight: 600 !important;
        font-size: .62rem !important;
        line-height: 1.3 !important;
        white-space: pre-line !important;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        hyphens: none !important;
        text-align: center !important;
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(232,163,61,.14);
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * {
        color: var(--gold-500) !important;
        font-weight: 800 !important;
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] input {
        display: none;
    }
    .bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] label:last-child::before {
        margin-bottom: 1px;
    }

    div.stButton > button { background:var(--gold-500); color:var(--palay-900) !important; border:none;
                             border-radius:12px; font-weight:700; padding:.6rem 1rem; }
    div.stButton > button p { color:var(--palay-900) !important; }
    div.stButton > button:hover { background:var(--gold-300); color:var(--palay-900) !important; }
    div.stButton > button[kind="secondary"] { background:transparent; color:var(--palay-700) !important;
                             border:1.5px solid var(--palay-300); }
    div.stButton > button[kind="secondary"] p { color:var(--palay-700) !important; }

    .app-header { display:flex; justify-content:space-between; align-items:center; padding: .5rem 0 .4rem 0; }
    .app-header .brand { font-family:'Baloo 2',sans-serif; font-weight:800; font-size:1.4rem; }
    .app-header .who { font-size:.8rem; color:#6B7861 !important; text-align:right; }
    .app-header .who * { color:#6B7861 !important; }
    .app-header .who b { color:var(--ink) !important; }

    .post-card { background:#fff; border:1px solid #ECE6CE; border-radius:18px; padding:1rem 1.1rem;
                 margin-bottom:.6rem; box-shadow:0 2px 10px rgba(31,58,36,.05); }
    .post-head { display:flex; align-items:center; gap:.7rem; margin-bottom:.6rem; }
    .avatar { width:42px; height:42px; border-radius:50%; object-fit:cover; }
    .avatar-fallback { width:42px; height:42px; border-radius:50%; background:var(--palay-500); color:#fff !important;
                        display:flex; align-items:center; justify-content:center; font-weight:700; font-family:'Baloo 2'; }
    .avatar-fallback * { color:#fff !important; }
    .post-author { font-weight:700; font-size:.95rem; }
    .post-meta { font-size:.76rem; color:#8A9481 !important; }
    .post-caption { font-size:.92rem; line-height:1.5; margin:.3rem 0 .6rem 0; }
    .post-image { border-radius:12px; max-width:100%; margin-bottom:.5rem; }

    .gauge-card { background:var(--palay-700); border-radius:22px; padding:1.6rem 1.6rem 1.3rem 1.6rem;
                  margin-bottom:1rem; }
    .gauge-top { display:flex; justify-content:space-between; align-items:baseline; }
    .gauge-number { font-family:'Baloo 2'; font-size:2.6rem; font-weight:800; }
    .gauge-verdict { font-weight:700; font-size:1rem; }
    .gauge-track { background:rgba(255,255,255,.18); border-radius:999px; height:14px; margin:.7rem 0 .5rem 0; overflow:hidden; }
    .gauge-fill { background:linear-gradient(90deg, var(--gold-500), var(--gold-300)); height:100%; border-radius:999px; }
    .gauge-sub { font-size:.85rem; color:#DCE7D3 !important; }

    .stat-pill { background:#fff; border:1px solid #ECE6CE; border-radius:16px; padding:.8rem .6rem; text-align:center; }
    .stat-pill .num { font-family:'Baloo 2'; font-size:1.25rem; font-weight:700; color:var(--palay-700) !important; }
    .stat-pill .lbl { font-size:.68rem; color:#8A9481 !important; line-height:1.25; }

    .chat-bubble-me { background:var(--palay-500); padding:.5rem .9rem; border-radius:14px 14px 2px 14px;
                       display:inline-block; float:right; clear:both; margin:.25rem 0; max-width:75%; }
    .chat-bubble-me * { color:#fff !important; }
    .chat-bubble-them { background:var(--husk-100); color:var(--ink) !important; padding:.5rem .9rem; border-radius:14px 14px 14px 2px;
                         display:inline-block; float:left; clear:both; margin:.25rem 0; max-width:75%; }
    .chat-bubble-them * { color:var(--ink) !important; }

    .profile-top { display:flex; align-items:flex-start; justify-content:space-between; gap:.8rem; margin:.4rem 0 .7rem 0; }
    .profile-score-badge { background:var(--palay-700); border-radius:18px; padding:.8rem 1rem;
                            text-align:center; min-width:118px; }
    .profile-score-badge .eyebrow { font-size:.6rem; }
    .profile-score-num { font-family:'Baloo 2'; font-size:1.5rem; font-weight:800; margin-top:.15rem; }
    .profile-name { font-family:'Baloo 2'; font-size:1.4rem; font-weight:700; margin-top:.2rem; }
    .profile-location { color:#6B7861 !important; font-size:.88rem; margin-bottom:.9rem; }

    hr { border-color:#E7E0C4; }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ============================================================
# SHARED UI PIECES
# ============================================================
def render_avatar_html(name, size=42):
    uri = img_to_data_uri(get_profile_picture_path(name))
    if uri:
        return f'<img src="{uri}" class="avatar" style="width:{size}px;height:{size}px;">'
    return (f'<div class="avatar-fallback" style="width:{size}px;height:{size}px;'
            f'font-size:{size*0.4:.0f}px;">{initials(name)}</div>')


def inject_profile_tab_avatar_css(current_user):
    uri = img_to_data_uri(get_profile_picture_path(current_user))
    if uri:
        icon_css = (
            'content:""; display:block; width:20px; height:20px; border-radius:50%; '
            f'background-image:url(\'{uri}\'); background-size:cover; background-position:center;'
        )
    else:
        icon_css = (
            f'content:"{initials(current_user)}"; display:flex; align-items:center; justify-content:center; '
            'width:20px; height:20px; border-radius:50%; background:var(--palay-500); '
            'color:#fff; font-size:9px; font-weight:800;'
        )
    st.markdown(
        f'<style>.bottom-nav-marker ~ div[data-testid="stRadio"] div[role="radiogroup"] '
        f'label:last-child::before {{ {icon_css} }}</style>',
        unsafe_allow_html=True,
    )


def app_header(current_user, role_label):
    st.markdown(f"""
    <div class="app-header">
        <div class="brand">{APP_NAME}</div>
        <div class="who">Signed in as <b>{current_user}</b><br>{role_label}</div>
    </div>
    <hr style="margin:0 0 1rem 0;">
    """, unsafe_allow_html=True)


def diversity_gauge(score, verdict, verdict_sub):
    width = max(3, min(100, score))
    st.markdown(f"""
    <div class="gauge-card">
        <div class="eyebrow" style="color:var(--gold-300);">Farm diversity score</div>
        <div class="gauge-top">
            <div class="gauge-number">{score:.0f}%</div>
            <div class="gauge-verdict">{verdict}</div>
        </div>
        <div class="gauge-track"><div class="gauge-fill" style="width:{width:.0f}%;"></div></div>
        <div class="gauge-sub">{verdict_sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_feed(current_user, editable_composer=True, barangay="", clickable_profiles=False):
    if editable_composer:
        with st.expander("Add a post"):
            new_caption = st.text_area("What's happening on your farm?", key="new_post_text")
            new_image = st.file_uploader("Add a photo (optional)", type=["png", "jpg", "jpeg"], key="new_post_image")
            if st.button("Post", key="submit_post"):
                if new_caption.strip():
                    image_path = save_post_image(new_image) if new_image is not None else ""
                    save_post(current_user, barangay, new_caption.strip(), image_path)
                    st.rerun()
                else:
                    st.markdown('<div class="banner warn">Write something before posting.</div>', unsafe_allow_html=True)

    posts = load_posts()
    likes_df = load_likes()

    if posts.empty:
        st.caption("No posts yet.")
        return

    for _, post in posts.iterrows():
        avatar_html = render_avatar_html(post["author"])
        image_html = ""
        if post["image_path"] and os.path.isfile(post["image_path"]):
            uri = img_to_data_uri(post["image_path"])
            if uri:
                image_html = f'<img src="{uri}" class="post-image">'

        st.markdown(f"""
        <div class="post-card">
            <div class="post-head">
                {avatar_html}
                <div>
                    <div class="post-author">{post['author']}</div>
                    <div class="post-meta">{post['barangay']} &middot; {post['timestamp']}</div>
                </div>
            </div>
            <div class="post-caption">{post['caption']}</div>
            {image_html}
        </div>
        """, unsafe_allow_html=True)

        n_likes = like_count(likes_df, post["post_id"])
        liked = user_liked(likes_df, post["post_id"], current_user)
        label = f"{'Liked' if liked else 'Like'} ({n_likes})"

        if clickable_profiles and post["author"] != current_user:
            c1, c2 = st.columns([1, 1.3])
            with c1:
                if st.button(label, key=f"like_{post['post_id']}"):
                    toggle_like(post["post_id"], current_user)
                    st.rerun()
            with c2:
                if st.button(f"View {post['author']}'s farm", key=f"viewprofile_{post['post_id']}"):
                    st.session_state.viewing_profile = post["author"]
                    st.rerun()
        else:
            if st.button(label, key=f"like_{post['post_id']}"):
                toggle_like(post["post_id"], current_user)
                st.rerun()


def render_messages_section(current_user):
    st.markdown("#### Messages")
    accounts = load_accounts()
    other_accounts = accounts[accounts["name"] != current_user]["name"].tolist()

    if not other_accounts:
        st.caption("No other accounts to message yet.")
        return

    jump_target = st.session_state.pop("jump_to_chat_with", None)
    if jump_target and jump_target in other_accounts:
        st.session_state["chat_with_select"] = jump_target

    chosen_default = st.session_state.get("chat_with_select", other_accounts[0])
    if chosen_default not in other_accounts:
        chosen_default = other_accounts[0]
    chat_with = st.selectbox("Chat with", other_accounts,
                              index=other_accounts.index(chosen_default), key="chat_with_select")
    convo = load_conversation(current_user, chat_with)

    with st.container(border=True, height=230):
        if convo.empty:
            st.caption("No messages yet. Say hello.")
        for _, row in convo.iterrows():
            bubble_class = "chat-bubble-me" if row["from_name"] == current_user else "chat-bubble-them"
            st.markdown(f'<div class="{bubble_class}">{row["message"]}</div>', unsafe_allow_html=True)

    new_msg = st.text_input("Type a message", key=f"chat_input_{chat_with}", label_visibility="collapsed",
                             placeholder="Type a message")
    if st.button("Send", key=f"send_{chat_with}"):
        if new_msg.strip():
            send_message(current_user, chat_with, new_msg.strip())
            st.rerun()


def render_search(barangay_key):
    st.markdown("#### Search")
    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input("Search by name", placeholder="Search farmers, offices, or accounts", label_visibility="collapsed")
    with c2:
        barangay_pick = st.selectbox("Barangay", ["All barangays"] + SAN_PEDRO_BARANGAYS, key=barangay_key)

    accounts = load_accounts()
    if accounts.empty:
        st.caption("No registered accounts yet.")
        return

    results = accounts.copy()
    if query:
        results = results[results["name"].astype(str).str.contains(query, case=False, na=False)]
    if barangay_pick != "All barangays":
        results = results[results["barangay"] == barangay_pick]

    if results.empty:
        st.caption("No matches found.")
        return

    for _, row in results.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 6])
            with cols[0]:
                st.markdown(render_avatar_html(row["name"], size=36), unsafe_allow_html=True)
            with cols[1]:
                location = row["barangay"] if clean_text(row["barangay"]) else "San Pedro City Hall"
                st.markdown(f"**{row['name']}** &middot; {row['role']} &middot; {location}", unsafe_allow_html=True)


def render_profile_picture_uploader(current_user, upload_key):
    pic_path = get_profile_picture_path(current_user)
    col_pic, col_upload = st.columns([1, 2])
    with col_pic:
        if pic_path:
            st.image(pic_path, width=110)
        else:
            st.markdown(render_avatar_html(current_user, size=90), unsafe_allow_html=True)
    with col_upload:
        uploaded_pic = st.file_uploader("Update profile picture", type=["png", "jpg", "jpeg"], key=upload_key)
        if uploaded_pic is not None:
            save_profile_picture(current_user, uploaded_pic)
            st.rerun()


def render_farmer_public_profile(profile_name):
    if st.button("\u2190 Back", key="back_from_profile"):
        st.session_state.pop("viewing_profile", None)
        st.rerun()

    account = get_account(profile_name)
    barangay = clean_text(account["barangay"]) if account else ""

    crops = load_farmer_crops(profile_name)
    result = calculate_diversity([c for c, _ in crops], [a for _, a in crops]) if crops else None
    score_display = f"{result['variety_score']:.0f}%" if result else "Not yet calculated"

    st.markdown(f"""
    <div class="profile-top">
        {render_avatar_html(profile_name, size=88)}
        <div class="profile-score-badge">
            <div class="eyebrow">Whole-farm diversity</div>
            <div class="profile-score-num">{score_display}</div>
        </div>
    </div>
    <div class="profile-name">{profile_name}</div>
    <div class="profile-location">{barangay or 'San Pedro, Laguna'}</div>
    """, unsafe_allow_html=True)

    if st.button("\U0001F4AC Message", use_container_width=True, key="msg_from_profile"):
        st.session_state.jump_to_chat_with = profile_name
        st.session_state.pop("viewing_profile", None)
        st.session_state.gov_nav = "Profile"
        st.rerun()

    st.divider()
    st.markdown("###### Posts")
    my_posts = load_posts(author_filter=profile_name)
    if my_posts.empty:
        st.caption("No posts yet.")
    else:
        cols = st.columns(3)
        for i, (_, post) in enumerate(my_posts.iterrows()):
            with cols[i % 3]:
                if post["image_path"] and os.path.isfile(post["image_path"]):
                    st.image(post["image_path"], use_container_width=True)
                else:
                    st.caption(post["caption"][:60])


# ============================================================
# SCREEN 1 - WELCOME (hero, role choice)
# ============================================================
def screen_welcome():
    st.markdown(
        f"""
        <div class="bleed">
          <div class="hero-shell">
            <div class="hero-inner">
                <div class="hero-brand">{CITY_NAME} &middot; {PROVINCE_NAME}</div>
                <div class="hero-title">{APP_NAME}</div>
                <div class="hero-sub">A shared home for {CITY_NAME}'s farms and the office that supports them &mdash;
                track what's growing, see how spread out your harvest is, and stay in touch with your barangay.</div>
            </div>
            <div class="terrace-layer">{terrace_svg(["#5C8A52","#3F6B45","#2F5233","#1F3A24"])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="eyebrow" style="text-align:center;">Are you a...</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(f"""
        <div class="role-card">
            <div class="role-icon">{role_icon_svg('farmer')}</div>
            <h3>Farmer</h3>
            <p>Log your crops, see your diversity score, post updates, and message
            other growers or your barangay office.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Continue as a farmer", use_container_width=True, key="pick_farmer"):
            st.session_state.pending_role = "Farmer"
            st.session_state.auth_stage = "auth"
            st.rerun()
    with col2:
        st.markdown(f"""
        <div class="role-card sky">
            <div class="role-icon">{role_icon_svg('gov')}</div>
            <h3>Government office</h3>
            <p>Follow crop diversity across San Pedro's barangays and stay
            reachable to the farmers you serve.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Continue as an office", use_container_width=True, key="pick_gov"):
            st.session_state.pending_role = "Government"
            st.session_state.auth_stage = "auth"
            st.rerun()


# ============================================================
# SCREEN 2 - SIGN IN / CREATE ACCOUNT
# ============================================================
def screen_auth():
    role = st.session_state.pending_role
    is_farmer = role == "Farmer"

    st.markdown(
        f"""
        <div class="auth-side">
            <div class="terrace-layer">{terrace_svg(["#5C8A52","#2F5233"], height=110, sun=False)}</div>
            <div class="auth-side-content">
                <div class="eyebrow" style="color:var(--gold-300);">{role} account</div>
                <h2>{"Welcome back to the field" if is_farmer else "Keep an eye on the harvest"}</h2>
                <p>{"Your crops and posts are saved under your name and PIN, so signing in again brings everything right back." if is_farmer
                    else "Track diversity scores and farm activity across all 27 barangays of San Pedro."}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Back", key="auth_back"):
        st.session_state.auth_stage = "welcome"
        st.rerun()

    st.write("")
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Sign in", "Create account"], horizontal=True,
                     label_visibility="collapsed", key="auth_mode")

    if mode == "Sign in":
        signin_name = st.text_input("Full name", key="signin_name")
        signin_pin = st.text_input("PIN", type="password", key="signin_pin", max_chars=6)

        if st.button("Sign in", use_container_width=True):
            account = get_account(signin_name)
            if (
                account is not None
                and clean_text(account.get("role")) == role
                and verify_pin(signin_name, signin_pin, account.get("pin_hash", ""))
            ):
                reset_transient_state()
                st.session_state.username = account["name"]
                st.session_state.role = account["role"]
                st.session_state.barangay = account["barangay"]
                st.session_state.auth_stage = "app"
                remember_login(account["name"], account["role"], account["barangay"])
                st.rerun()
            else:
                st.markdown('<div class="banner warn">Name or PIN is incorrect.</div>', unsafe_allow_html=True)
    else:
        name = st.text_input("Full name")
        barangay = ""
        if is_farmer:
            barangay = st.selectbox("Barangay", [BARANGAY_PLACEHOLDER] + SAN_PEDRO_BARANGAYS)
            st.markdown(
                '<div class="field-note">Picked from the official list of San Pedro barangays, so it can\'t be misspelled.</div>',
                unsafe_allow_html=True)
        else:
            barangay = st.selectbox("Office location (optional)",
                                     [GOV_LOCATION_PLACEHOLDER] + SAN_PEDRO_BARANGAYS)

        pin = st.text_input("Choose a 4-6 digit PIN", type="password", key="create_pin", max_chars=6)
        pin_confirm = st.text_input("Confirm PIN", type="password", key="create_pin_confirm", max_chars=6)
        st.markdown(
            '<div class="field-note">This PIN keeps your account private to you -- you\'ll need it to sign '
            'back in, even on a different phone or computer.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Create account", use_container_width=True):
            if is_farmer and barangay == BARANGAY_PLACEHOLDER:
                st.markdown('<div class="banner warn">Select your barangay before continuing.</div>',
                            unsafe_allow_html=True)
            elif not valid_pin_format(pin):
                st.markdown('<div class="banner warn">Choose a 4-6 digit PIN (numbers only).</div>',
                            unsafe_allow_html=True)
            elif pin != pin_confirm:
                st.markdown('<div class="banner warn">PINs do not match.</div>', unsafe_allow_html=True)
            else:
                stored_barangay = "" if barangay in (BARANGAY_PLACEHOLDER, GOV_LOCATION_PLACEHOLDER) else barangay
                ok, message = register_account(name, role, stored_barangay, pin)
                if ok:
                    reset_transient_state()
                    st.session_state.username = clean_text(name)
                    st.session_state.role = role
                    st.session_state.barangay = stored_barangay
                    st.session_state.auth_stage = "app"
                    remember_login(clean_text(name), role, stored_barangay)
                    st.rerun()
                else:
                    st.markdown(f'<div class="banner warn">{message}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SCREEN 3a - FARMER APP
# ============================================================
FARMER_NAV_OPTIONS = ["\U0001F3E0\nHome", "\U0001F50D\nSearch", "\U0001F331\nMy Farm", "Profile"]
GOV_NAV_OPTIONS = ["\U0001F3E0\nHome", "\U0001F50D\nSearch", "Profile"]


def screen_farmer_app():
    app_header(st.session_state.username, f"Farmer &middot; {st.session_state.barangay or 'San Pedro'}")

    page = st.session_state.get("farmer_nav", FARMER_NAV_OPTIONS[0])

    if page == FARMER_NAV_OPTIONS[0]:
        st.markdown("#### Nearby farms")
        render_feed(st.session_state.username, editable_composer=True, barangay=st.session_state.barangay)

    elif page == FARMER_NAV_OPTIONS[1]:
        render_search("search_barangay_farmer")

    elif page == FARMER_NAV_OPTIONS[2]:
        render_my_farm(st.session_state.username, st.session_state.barangay)

    elif page == FARMER_NAV_OPTIONS[3]:
        render_profile_page(is_farmer=True)

    inject_profile_tab_avatar_css(st.session_state.username)
    st.markdown('<div class="bottom-nav-marker"></div>', unsafe_allow_html=True)
    st.radio("Navigate", FARMER_NAV_OPTIONS, horizontal=True,
             label_visibility="collapsed", key="farmer_nav")


def render_my_farm(farmer_name, barangay):
    st.markdown("#### My farm")
    saved_crops = load_farmer_crops(farmer_name)
    default_count = max(len(saved_crops), 1)

    if "last_result" not in st.session_state and saved_crops:
        auto_result = calculate_diversity([c for c, _ in saved_crops], [a for _, a in saved_crops])
        if auto_result:
            st.session_state.last_result = auto_result

    num_crops = st.number_input("How many different crops do you grow?", min_value=1, max_value=MAX_CROPS,
                                 value=default_count, step=1)

    crop_names, crop_areas = [], []
    for i in range(int(num_crops)):
        default_name = saved_crops[i][0] if i < len(saved_crops) else ""
        default_area = float(saved_crops[i][1]) if i < len(saved_crops) else 0.0
        c1, c2 = st.columns(2)
        with c1:
            name_c = st.text_input(f"Crop {i+1} name", value=default_name, key=f"name_{i}")
        with c2:
            area = st.number_input("Land used (sq. m)", min_value=0.0, value=default_area, key=f"area_{i}")
        crop_names.append(name_c if name_c else f"Crop {i+1}")
        crop_areas.append(area)

    col_calc, col_reset = st.columns([2, 1])
    with col_calc:
        calc_clicked = st.button("Calculate my farm's diversity", use_container_width=True)
    with col_reset:
        if st.button("Clear saved data", use_container_width=True):
            delete_farm_data(farmer_name)
            st.session_state.pop("last_result", None)
            st.rerun()

    if calc_clicked:
        result = calculate_diversity(crop_names, crop_areas)
        if result is None:
            st.markdown('<div class="banner warn">Enter some land area for at least one crop.</div>', unsafe_allow_html=True)
        else:
            st.session_state.last_result = result

    if "last_result" in st.session_state:
        result = st.session_state.last_result
        score = result["variety_score"]
        verdict, verdict_sub = verdict_for_score(score)
        diversity_gauge(score, verdict, verdict_sub)

        m1, m2, m3 = st.columns(3)
        for col, num, lbl in [
            (m1, f"{result['total_area']:.0f} m2", "Total farm area"),
            (m2, f"{result['effective_crops']:.1f}", "Effective crops"),
            (m3, f"{result['num_crops']}", "Crops grown"),
        ]:
            with col:
                st.markdown(f'<div class="stat-pill"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                             unsafe_allow_html=True)

        st.write("")
        st.markdown("###### Crops on this farm")
        table_data = [{"Crop": c["name"], "Land area (sq.m)": round(c["area"], 1),
                        "Share of farm": f"{c['share_pct']:.1f}%"} for c in result["crop_breakdown"]]
        st.table(table_data)

        st.markdown("###### If one crop fails")
        for c in result["crop_breakdown"]:
            st.write(f"If **{c['name']}** fails, {c['remaining_if_lost_pct']:.0f}% of production is still standing.")

        if st.button("Save this data", use_container_width=True):
            save_farm_data(farmer_name, barangay, result["valid_crops"])
            st.markdown('<div class="banner ok">Saved. This will pre-fill next time you open My Farm.</div>',
                        unsafe_allow_html=True)


def render_profile_page(is_farmer):
    st.markdown("#### Profile")
    render_profile_picture_uploader(st.session_state.username, "pic_upload")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Name**  \n{st.session_state.username}")
    with c2:
        current_barangay = st.session_state.barangay or "Not set"
        st.markdown(f"**Barangay**  \n{current_barangay}")
        new_barangay = st.selectbox(
            "Change barangay", [current_barangay] + [b for b in SAN_PEDRO_BARANGAYS if b != current_barangay],
            label_visibility="collapsed", key="profile_barangay_change",
        )
        if new_barangay != current_barangay and st.button("Update barangay", key="update_barangay_btn"):
            accounts = load_accounts()
            accounts.loc[accounts["name"] == st.session_state.username, "barangay"] = new_barangay
            accounts.to_csv(ACCOUNTS_FILE, index=False)
            st.session_state.barangay = new_barangay
            remember_login(st.session_state.username, st.session_state.role, new_barangay)
            st.rerun()

    if is_farmer:
        st.divider()
        st.markdown("#### My posts")
        my_posts = load_posts(author_filter=st.session_state.username)
        if my_posts.empty:
            st.caption("No posts yet.")
        else:
            cols = st.columns(3)
            for i, (_, post) in enumerate(my_posts.iterrows()):
                with cols[i % 3]:
                    if post["image_path"] and os.path.isfile(post["image_path"]):
                        st.image(post["image_path"], use_container_width=True)
                    else:
                        st.caption(post["caption"][:70])

    st.divider()
    render_messages_section(st.session_state.username)

    if is_farmer:
        st.divider()
        st.markdown("#### Data and privacy")
        all_farms = load_all_farms()
        my_data = all_farms[all_farms["farmer_name"] == st.session_state.username] if not all_farms.empty else all_farms
        if not my_data.empty:
            csv_bytes = my_data.to_csv(index=False).encode("utf-8")
            st.download_button("Request a copy of my data", data=csv_bytes,
                                file_name="my_farm_data.csv", mime="text/csv")
        else:
            st.caption("No saved farm data yet.")

    st.divider()
    if st.button("Sign out", key="logout_btn", use_container_width=True):
        log_out()
        st.rerun()


# ============================================================
# SCREEN 3b - GOVERNMENT APP
# ============================================================
def screen_gov_app():
    app_header(st.session_state.username, f"Government office &middot; {st.session_state.barangay or 'City Hall'}")

    if st.session_state.get("viewing_profile"):
        render_farmer_public_profile(st.session_state.viewing_profile)
        return

    page = st.session_state.get("gov_nav", GOV_NAV_OPTIONS[0])
    all_farms = load_all_farms()

    if page == GOV_NAV_OPTIONS[0]:
        st.markdown(f"#### {CITY_NAME} City overview")

        if all_farms.empty:
            st.caption("No farm data submitted yet. Once farmers save their data, it appears here.")
        else:
            barangays_present = sorted(b for b in all_farms["barangay"].dropna().unique().tolist() if clean_text(b))
            scale_choice = st.selectbox("View diversity for", ["City-wide (all barangays)"] + barangays_present,
                                         key="gov_scale_choice")
            scope_df = all_farms if scale_choice.startswith("City-wide") else all_farms[all_farms["barangay"] == scale_choice]

            scale_stats = compute_scale_diversity(scope_df)

            if scale_stats is None:
                st.caption("Not enough data yet for this view.")
            else:
                community_overview_card(scale_stats["beta"], scale_stats["num_sites"])

                with st.expander("See the detailed breakdown (Alpha, Beta, Gamma)"):
                    beta_display = f"{scale_stats['beta']:.2f}" if scale_stats["beta"] is not None else "\u2014"
                    m1, m2, m3 = st.columns(3)
                    for col, num, lbl in [
                        (m1, f"{scale_stats['alpha']:.2f}", "Alpha (\u03b1) \u2014 typical single-farm variety"),
                        (m2, f"{scale_stats['gamma']:.2f}", "Gamma (\u03b3) \u2014 whole-area variety, combined"),
                        (m3, beta_display, "Beta (\u03b2) \u2014 how different farms are from each other"),
                    ]:
                        with col:
                            st.markdown(f'<div class="stat-pill"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                                         unsafe_allow_html=True)
                    st.caption("Alpha = average diversity within a single farm. Gamma = diversity of all listed "
                               "farms combined, as one area. Beta = Gamma \u00f7 Alpha \u2014 how much more varied "
                               "farms are from each other than they are individually (near 1 means every farm "
                               "grows the same mix; higher means farms differ more).")

                agg_names, agg_areas = aggregate_crop_pairs(scope_df)
                agg_result = calculate_diversity(agg_names, agg_areas)
                if agg_result:
                    with st.expander(f"See aggregated production resilience \u2014 {scale_choice}"):
                        st.caption("If a single crop failed across every recorded farm in this view, how much "
                                   "of the combined recorded area would still be standing:")
                        resilience_rows = [
                            {"Crop": c["name"], "Combined area (sq.m)": round(c["area"], 1),
                             "Share of total": f"{c['share_pct']:.1f}%",
                             "If this crop fails": f"{c['remaining_if_lost_pct']:.0f}% remains"}
                            for c in agg_result["crop_breakdown"]
                        ]
                        st.table(resilience_rows)
                        st.caption("Based on recorded farm area entered under My Farm, the same stand-in used "
                                   "for every other resilience figure in this app -- not the City Agriculture "
                                   "Office's separate production-tonnage records.")

                st.write("")
                scores = []
                for farm_name, group in scope_df.groupby("farmer_name"):
                    result = calculate_diversity(list(group["crop"]), list(group["area"]))
                    if result:
                        scores.append({
                            "Farm": farm_name, "Barangay": group["barangay"].iloc[0],
                            "Score": result["variety_score"], "Effective crops": result["effective_crops"],
                        })

                if scores:
                    low_variety = [s["Farm"] for s in scores if s["Score"] < 40]
                    if low_variety:
                        st.markdown(
                            f'<div class="banner warn">Farms that may need support: {", ".join(low_variety)}</div>',
                            unsafe_allow_html=True)

                    with st.expander(f"\U0001F4C4 Open full farm data sheet ({len(scores)} farms)"):
                        st.table([{"Farm": s["Farm"], "Barangay": s["Barangay"],
                                  "Diversity score": f"{s['Score']:.0f}%",
                                  "Effective crops": f"{s['Effective crops']:.1f}"} for s in scores])
                        scores_csv = pd.DataFrame(scores).to_csv(index=False).encode("utf-8")
                        st.download_button("\u2b07\ufe0f Download this data sheet", data=scores_csv,
                                            file_name="san_pedro_registered_farms.csv", mime="text/csv",
                                            key="download_scores_sheet")

        st.divider()

        with st.expander("\U0001F4C8 City production records (Agri Office)"):
            st.caption("City-wide, per-commodity production -- not tied to any specific farm or barangay. "
                       "Used for the single-crop-loss simulation: if one crop's recorded production were lost, "
                       "how much of the city's total recorded production would remain.")
            prod_df = load_production_records()
            years_present = sorted(prod_df["year"].dropna().unique().tolist(), reverse=True) if not prod_df.empty else []

            if not years_present:
                st.caption("No production records yet. Add one below.")
            else:
                prod_year = st.selectbox("Year / period", years_present, key="gov_prod_year")
                if "(Jan-Jul)" in prod_year or "Jan-Jul" in prod_year:
                    st.markdown('<div class="banner warn">This period only covers Jan-Jul -- Aug-Dec has not been '
                                'recorded yet. Do not compare this total directly against a full-year total.</div>',
                                unsafe_allow_html=True)

                prod_stats = compute_production_resilience(prod_year)
                if prod_stats:
                    st.markdown(f'<div class="stat-pill"><div class="num">{prod_stats["total"]:.4f} Mt</div>'
                                f'<div class="lbl">Total recorded production (T), {prod_year}</div></div>',
                                unsafe_allow_html=True)
                    st.write("")
                    resilience_table = [
                        {"Crop": r["crop"], "Production, y_j (Mt)": round(r["production_mt"], 4),
                         "Share of T": f"{r['share_pct']:.1f}%",
                         "If lost, R_j (% of T retained)": f"{r['remaining_pct']:.2f}%"}
                        for r in prod_stats["rows"]
                    ]
                    st.table(resilience_table)
                    st.caption("R_j = (T \u2212 y_j) \u00f7 T \u00d7 100. Lower R_j means that crop makes up a larger "
                               "share of citywide recorded production, so losing it would retain less of the total.")

            st.markdown("###### Add or update a production record")
            pr_col1, pr_col2 = st.columns(2)
            with pr_col1:
                pr_year_choice = st.selectbox("Year / period", years_present + ["New year/period..."],
                                               key="pr_year_choice")
                pr_year = (st.text_input("New year/period label (e.g. 2027)", key="pr_year_new")
                           if pr_year_choice == "New year/period..." else pr_year_choice)
            with pr_col2:
                pr_crop = st.text_input("Crop name", key="pr_crop_name")
            pr_value = st.number_input("Production (metric tons)", min_value=0.0, step=0.001,
                                        format="%.4f", key="pr_value")
            if st.button("Save production record", key="save_prod_record"):
                ok, msg = save_production_record(pr_year, pr_crop, pr_value)
                if ok:
                    st.rerun()
                else:
                    st.markdown(f'<div class="banner warn">{msg}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Community feed")
        render_feed(st.session_state.username, editable_composer=False, clickable_profiles=True)

    elif page == GOV_NAV_OPTIONS[1]:
        render_search("search_barangay_gov")

    elif page == GOV_NAV_OPTIONS[2]:
        render_profile_page(is_farmer=False)
        st.divider()
        st.markdown("#### Reports")
        if not all_farms.empty:
            csv_bytes = all_farms.to_csv(index=False).encode("utf-8")
            st.download_button("Download full city report", data=csv_bytes,
                                file_name="san_pedro_farm_report.csv", mime="text/csv")
        else:
            st.caption("No data to report yet.")

    inject_profile_tab_avatar_css(st.session_state.username)
    st.markdown('<div class="bottom-nav-marker"></div>', unsafe_allow_html=True)
    st.radio("Navigate", GOV_NAV_OPTIONS, horizontal=True,
             label_visibility="collapsed", key="gov_nav")


# ============================================================
# ROUTER
# ============================================================
phone_status_bar()

if st.session_state.auth_stage == "welcome":
    screen_welcome()
elif st.session_state.auth_stage == "auth":
    screen_auth()
elif st.session_state.role == "Farmer":
    screen_farmer_app()
elif st.session_state.role == "Government":
    screen_gov_app()
else:
    log_out()
    st.rerun()
