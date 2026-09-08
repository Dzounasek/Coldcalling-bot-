"""
Cold Call Prep Ninja
=====================
A lightweight Streamlit tool that helps a salesperson quickly prep for a
cold call to an e-shop by:
  1. Generating a direct link to the Meta (Facebook) Ads Library, filtered
     by the shop's domain, so the rep can see if the shop is running ads.
  2. Crawling the shop's homepage + a handful of common "contact" style
     subpages to scrape any Czech/Slovak phone numbers it can find.
  3. Extracting the Czech company registration number (IČO) from the same
     pages, looking up the official company name via the ARES API, and
     linking straight to the Justice.cz public register for owner/director
     ("jednatel") details.

Designed to run on Streamlit Community Cloud: no heavy dependencies,
generous timeouts, and every network call is wrapped so a single bad
website (or a flaky government API) can never crash the whole app.

Layout is intentionally linear (single column, top to bottom) so it's
comfortable on small/older monitors with no horizontal scrolling.
"""

import re
import time
from urllib.parse import urlparse, urljoin, quote

import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
# No "layout=wide" here on purpose - default (centered) layout keeps
# everything in a single readable column on small screens.
st.set_page_config(
    page_title="Cold Call Ninja",
    page_icon="📞",
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------

# A realistic desktop browser User-Agent. Many e-shops block requests that
# come in with the default python-requests UA string, so we pretend to be
# a normal Chrome/Windows visitor.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.8",
}

# How long we're willing to wait for any single page to respond.
# Cloud environments (and slow e-shops) can hang, so we keep this tight.
REQUEST_TIMEOUT = 7  # seconds

# Short timeout for the ARES government API - it's usually fast, and we
# don't want a slow/unavailable API to stall the whole app.
ARES_TIMEOUT = 5  # seconds

# Common paths on Czech/Slovak e-shops where phone numbers (and company
# details like IČO) tend to live, even when the homepage hides them.
SUBPAGES_TO_TRY = [
    "/kontakty",
    "/kontakt",
    "/o-nas",
    "/obchodni-podminky",
]

# Regex for Czech (+420) and Slovak (+421) phone numbers.
# Handles:
#   - optional international prefix: +420 / 00420 / +421 / 00421
#   - optional spaces (or no spaces at all) between digit groups
#   - the classic 9-digit CZ/SK national number, split as 3-3-3
PHONE_REGEX = re.compile(
    r"""
    (?:                                  # optional international prefix
        (?:\+|00)
        (?:420|421)
        [\s.-]?
    )?
    (?:                                  # the 9-digit national number,
        \d{3}[\s.-]?\d{3}[\s.-]?\d{3}    # written as 3-3-3 groups...
        |
        \d{9}                            # ...or with no separators at all
    )
    """,
    re.VERBOSE,
)

# Regex for the Czech IČO (Identifikační číslo osoby) - an 8-digit company
# registration number, typically labelled "IČ", "IČO", "IC", or "ICO"
# followed by an optional separator (:, ., -) and the 8 digits.
ICO_REGEX = re.compile(r"(?:I[CČ]O?)\s*[:.-]?\s*(\d{8})", re.IGNORECASE)

# ARES REST API endpoint template for looking up a company by its IČO.
ARES_API_URL = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{ico}"

# Czech Public Register (Justice.cz) search-by-IČO template - shows the
# company record including statutory representatives ("jednatel").
JUSTICE_URL_TEMPLATE = "https://or.justice.cz/ias/ui/rejstrik-$firma?ico={ico}"


# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def normalize_url(raw_input: str) -> str:
    """
    Ensure the user-provided URL has a proper schema.
    'eshop.cz' -> 'https://eshop.cz'
    'https://eshop.cz' -> unchanged
    """
    raw_input = raw_input.strip()
    if not raw_input.lower().startswith(("http://", "https://")):
        raw_input = "https://" + raw_input
    return raw_input


def extract_root_domain(url: str) -> str:
    """
    Parse a URL and return the bare domain, stripped of a leading 'www.'.
    e.g. 'https://www.alza.cz/foo' -> 'alza.cz'
    """
    netloc = urlparse(url).netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def build_facebook_ads_library_url(domain: str) -> str:
    """
    Build a link into the Meta Ads Library, pre-filled with the shop's
    domain as the search query, scoped to Czech ads.
    We don't scrape Facebook (it's not feasible without login/API access) -
    we just hand the rep a ready-to-click link.
    """
    base = "https://www.facebook.com/ads/library/"
    query = quote(domain)
    return f"{base}?active_status=all&ad_type=all&country=CZ&q={query}"


def extract_phone_numbers(text: str) -> set:
    """
    Run the phone regex over a blob of text and return a cleaned set of
    matches (whitespace collapsed, so duplicates that differ only in
    spacing get merged together).
    """
    found = set()
    for match in PHONE_REGEX.findall(text):
        cleaned = re.sub(r"\s+", " ", match).strip()
        # Guard against grabbing short junk (e.g. stray "123" from a date
        # or an ID) - a real CZ/SK number always has at least 9 digits.
        digit_count = len(re.sub(r"\D", "", cleaned))
        if digit_count >= 9:
            found.add(cleaned)
    return found


def extract_ico_numbers(text: str) -> set:
    """
    Run the IČO regex over a blob of text and return a de-duplicated set
    of 8-digit company registration numbers found.
    """
    found = set()
    for match in ICO_REGEX.findall(text):
        found.add(match.strip())
    return found


def fetch_page(url: str):
    """
    Fetch a single URL safely.
    Returns (html_text, error_message). Exactly one of the two will be None.
    Never raises - all exceptions are caught and turned into a friendly
    error string so the calling code (and the UI) can stay simple.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.text, None
        else:
            return None, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Connection failed (site may be blocking bots or is down)"
    except requests.exceptions.RequestException as exc:
        # Catch-all for anything else requests can throw
        # (too many redirects, invalid schema, SSL errors, etc.)
        return None, f"Request error: {exc}"


def get_visible_text(html: str) -> str:
    """
    Strip a raw HTML page down to visible text using BeautifulSoup, so the
    regexes aren't tripped up by phone/IČO-like strings buried in <script>
    tags, tracking pixels, or JSON blobs.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def scrape_site_for_contact_info(base_url: str, domain: str):
    """
    The core crawling engine.

    Visits the homepage plus a fixed list of common "contact" subpages,
    extracts visible text from each, and aggregates any phone numbers
    AND IČO company registration numbers found across all of them into
    de-duplicated sets.

    Returns:
        all_numbers (set): every unique phone number found
        all_icos (set): every unique 8-digit IČO found
        page_errors (list of tuples): (url, error_message) for pages that
                                       failed to load, so we can be
                                       transparent with the user
        pages_checked (int): how many pages were actually reachable
    """
    all_numbers = set()
    all_icos = set()
    page_errors = []
    pages_checked = 0

    # Build the full list of URLs to try: homepage first, then subpages.
    urls_to_try = [base_url] + [urljoin(base_url, path) for path in SUBPAGES_TO_TRY]

    for page_url in urls_to_try:
        html, error = fetch_page(page_url)

        if error:
            page_errors.append((page_url, error))
            continue

        pages_checked += 1
        visible_text = get_visible_text(html)
        all_numbers.update(extract_phone_numbers(visible_text))
        all_icos.update(extract_ico_numbers(visible_text))

        # Small, polite delay between requests so we don't hammer the
        # target site like a bot storm.
        time.sleep(0.3)

    return all_numbers, all_icos, page_errors, pages_checked


def lookup_company_name_ares(ico: str):
    """
    Query the official Czech ARES REST API for the company name behind
    a given IČO.

    Returns the company name string ("obchodniJmeno") on success, or
    None if the lookup fails for any reason (API down, timeout, IČO not
    found, unexpected response shape, etc). Never raises.
    """
    try:
        url = ARES_API_URL.format(ico=ico)
        response = requests.get(url, headers=HEADERS, timeout=ARES_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("obchodniJmeno")
        return None
    except requests.exceptions.RequestException:
        # ARES down, timed out, or unreachable - fail quietly, the UI
        # will just show the IČO without a company name.
        return None
    except ValueError:
        # response.json() failed to parse - treat the same as "not found".
        return None


def build_justice_url(ico: str) -> str:
    """
    Build a direct link into the Czech Public Register (Justice.cz) for
    the given IČO, where the rep can see statutory representatives
    ("jednatel") and other official filings.
    """
    return JUSTICE_URL_TEMPLATE.format(ico=ico)


# ----------------------------------------------------------------------------
# UI - HEADER
# ----------------------------------------------------------------------------

st.title("📞 Cold Call Prep Ninja")
st.caption("Quick-fire prep for outbound calls: ad activity, phone numbers, and company info, in one shot.")

user_input = st.text_input(
    "Enter the e-shop URL (e.g., alza.cz or https://www.alza.cz):",
    placeholder="alza.cz",
)

submitted = st.button("🚀 Prep this call", type="primary")

# Trigger on either pressing Enter in the text_input or clicking the button.
if user_input and submitted:
    st.divider()

    # --- URL PROCESSING -----------------------------------------------
    full_url = normalize_url(user_input)
    domain = extract_root_domain(full_url)

    if not domain:
        st.error("⚠️ That doesn't look like a valid URL. Please check it and try again.")
        st.stop()

    st.markdown(f"**Target domain:** `{domain}`")

    # --- SECTION 1: META ADS LIBRARY -----------------------------------
    st.subheader("1. Meta Ads Library")
    with st.spinner("Building Ads Library link..."):
        fb_url = build_facebook_ads_library_url(domain)

    st.markdown(
        f"""
        <a href="{fb_url}" target="_blank" style="
            display: inline-block;
            background-color: #1877F2;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            font-size: 16px;
        ">🔎 Check Facebook Ads Library for "{domain}"</a>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Opens in a new tab. Look for active ads to gauge marketing spend.")

    st.divider()

    # --- SECTION 2: PHONE NUMBERS + COMPANY INFO -----------------------
    st.subheader("2. Extracted Phone Numbers & Company Info")
    with st.spinner(f"Crawling {domain} and common contact pages..."):
        try:
            numbers, icos, errors, pages_ok = scrape_site_for_contact_info(full_url, domain)
        except Exception as exc:
            # Absolute last-resort safety net - should rarely trigger
            # since fetch_page() already catches network errors, but
            # this guarantees the app never crashes outright.
            st.error(f"❌ Unexpected error while scraping: {exc}")
            numbers, icos, errors, pages_ok = set(), set(), [], 0

    # --- Phone numbers ---
    if numbers:
        st.success(f"✅ Found {len(numbers)} unique phone number(s):")
        for number in sorted(numbers):
            st.success(f"📱 {number}")
    else:
        st.warning(
            "⚠️ No phone numbers found on the homepage or common contact pages. "
            "The site may hide them behind JavaScript, a contact form, or bot protection."
        )

    # --- IČO / company lookup ---
    if icos:
        st.markdown("**🏢 Company registration number(s) found:**")
        for ico in sorted(icos):
            # Look up the official company name via ARES - fails quietly
            # to None if the API is down or the IČO isn't recognized.
            with st.spinner(f"Looking up IČO {ico} in ARES..."):
                company_name = lookup_company_name_ares(ico)

            justice_url = build_justice_url(ico)

            if company_name:
                st.info(f"🏢 **IČO {ico}** — {company_name}")
            else:
                st.info(f"🏢 **IČO {ico}** — company name not found in ARES")

            st.markdown(
                f"[⚖️ Zobrazit jednatele ve Veřejném rejstříku (Justice.cz)]({justice_url})"
            )
    else:
        st.warning(
            "⚠️ No IČO (company registration number) found on the scanned pages. "
            "Try checking the site's Terms & Conditions page manually."
        )

    if pages_ok == 0:
        st.error(
            "❌ Could not reach the site at all — it may be blocking automated "
            "requests, be temporarily down, or the URL may be incorrect."
        )

    # Transparency: show which pages failed and why, without being alarming.
    if errors:
        with st.expander(f"ℹ️ {len(errors)} page(s) could not be checked"):
            for page_url, error_msg in errors:
                st.write(f"- `{page_url}` → {error_msg}")

elif submitted and not user_input:
    st.warning("👆 Please enter a URL first.")
