##
# @file src/schema.py
# @brief Single source of truth for the raw-log schema and the engineered
#        feature set.
#
# Kept separate from preprocess.py so that the streaming aggregator
# (src/aggregate.py) and the DataFrame-level pipeline (src/preprocess.py)
# can share these constants without a circular import.
##

import datetime as _dt

# ---------------------------------------------------------------------------
# Engineered feature set
# ---------------------------------------------------------------------------

## Behavioural feature columns produced per (user, day). Order is significant:
#  it is the slot order used by the streaming aggregator's counter vectors.
FEATURE_COLS = [
    "n_logon", "n_logoff", "n_afterhours_logon",
    "n_device_connect",
    "n_file_events", "n_file_to_removable",
    "n_http_events", "n_http_upload", "n_distinct_domains",
    "n_email_sent", "n_email_external", "n_email_attachments", "total_email_size",
]

N_FEATURES = len(FEATURE_COLS)

# Note: n_distinct_domains is the one slot that cannot be summed across
# shards. The aggregator leaves it at zero and reconstructs it during the
# merge from unioned per-(user, day) domain-id sets.

## Slot indices into a counter vector, by name.
SLOT = {name: i for i, name in enumerate(FEATURE_COLS)}

# ---------------------------------------------------------------------------
# Raw log schema (CERT r6.2)
# ---------------------------------------------------------------------------

## Column layout of each raw log file, as shipped in r6.2.
LOG_COLUMNS = {
    "logon":  ["id", "date", "user", "pc", "activity"],
    "device": ["id", "date", "user", "pc", "file_tree", "activity"],
    "file":   ["id", "date", "user", "pc", "filename", "activity",
               "to_removable_media", "from_removable_media", "content"],
    "http":   ["id", "date", "user", "pc", "url", "activity", "content"],
    "email":  ["id", "date", "user", "pc", "to", "cc", "bcc", "from",
               "activity", "size", "attachments", "content"],
}

LOG_KINDS = list(LOG_COLUMNS)

## Organisation-internal email domain. Recipients outside it count as
#  external. r6.2 models a fictional company, "DTAA".
INTERNAL_EMAIL_DOMAIN = "dtaa.com"

## Hours outside [AFTERHOURS_START, AFTERHOURS_END) count as after-hours.
AFTERHOURS_END   = 7    # activity before 07:00 is after-hours
AFTERHOURS_START = 19   # activity from 19:00 onward is after-hours

# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------
#
# Every timestamp in r6.2 is a fixed-width "MM/DD/YYYY HH:MM:SS" string.
# Because the width is fixed we can slice out the calendar day and the hour
# directly instead of running a general-purpose datetime parser over
# ~120 million rows -- this is the single biggest win in the ingest loop.

DATE_LEN = len("MM/DD/YYYY")


def day_ordinal(stamp: str, _cache: dict = {}) -> int:
    """
    @brief Convert an "MM/DD/YYYY ..." timestamp to a proleptic Gregorian
           day ordinal.

    Memoised on the 10-character date prefix. The dataset spans ~17 months,
    so the cache saturates at a few hundred entries after the first rows and
    every subsequent call is a plain dict hit.

    @param stamp  Raw timestamp string from a CERT log.
    @return       Integer day ordinal (see datetime.date.toordinal).
    """
    key = stamp[:DATE_LEN]
    ordinal = _cache.get(key)
    if ordinal is None:
        mm, dd, yyyy = key.split("/")
        ordinal = _dt.date(int(yyyy), int(mm), int(dd)).toordinal()
        _cache[key] = ordinal
    return ordinal


def hour_of(stamp: str) -> int:
    """
    @brief Extract the hour-of-day from an "MM/DD/YYYY HH:MM:SS" timestamp.

    @param stamp  Raw timestamp string from a CERT log.
    @return       Hour in [0, 23].
    """
    return (ord(stamp[11]) - 48) * 10 + (ord(stamp[12]) - 48)


def is_afterhours(stamp: str) -> bool:
    """@brief True if the timestamp falls outside normal working hours."""
    h = hour_of(stamp)
    return h < AFTERHOURS_END or h >= AFTERHOURS_START


def netloc_of(url: str) -> str:
    """
    @brief Extract the bare host from a URL, without urllib's overhead.

    Strips the scheme, any path/query, and a leading "www." so that
    "http://www.dropbox.com/home/x.aspx" and "https://dropbox.com/y"
    collapse to the same domain.

    @param url  Absolute URL from http.csv.
    @return     Lower-cased host, or "" if the URL is unparseable.
    """
    i = url.find("://")
    start = i + 3 if i >= 0 else 0
    j = url.find("/", start)
    host = url[start:j] if j >= 0 else url[start:]
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host
