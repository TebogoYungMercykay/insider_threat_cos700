##
# @file src/cohort.py
# @brief Reproducible construction of a user cohort (a training subset).
#
# Aggregation (src/aggregate.py) already removes the compute barrier: the
# full 4 000-user population reduces to a table small enough to train on
# directly. A cohort is therefore an *experimental control*, not a workaround
# — it exists so that development iterations are fast and so that the
# reported experiment is a fixed, citable, reproducible sample rather than
# "whatever fit in memory that day".
#
# Two properties make a cohort defensible in a write-up:
#
#   1. *No positives are ever discarded.* Every designated insider is
#      included unconditionally. r6.2 has only five, so dropping even one
#      would materially change the evaluation.
#
#   2. *The negative class is downsampled by role-stratified sampling.*
#      The per-user framework falls back to a role-group baseline for users
#      with too little history, so a cohort that lost whole roles would
#      degrade that fallback and understate the framework's accuracy.
#      Stratifying keeps every role represented in proportion.
#
# The selected user list and the parameters that produced it are written
# alongside the data so a run can be reproduced exactly.
##

import json
import os

import numpy as np
import pandas as pd


def _allocate(role_sizes: pd.Series, budget: int) -> dict:
    """
    @brief Split a sampling budget across roles, proportional to role size.

    Uses largest-remainder allocation so the parts sum exactly to the
    budget, then guarantees at least one user per role while any budget is
    still unspent — small roles (r6.2 has several with a single member)
    would otherwise vanish entirely and take their role-group baseline
    with them.

    @param role_sizes  Series {role: number of available benign users}.
    @param budget      Total benign users to select.
    @return            Dict {role: number to sample from that role}.
    """
    total = int(role_sizes.sum())
    budget = min(budget, total)
    if budget <= 0:
        return {r: 0 for r in role_sizes.index}

    exact = role_sizes / total * budget
    alloc = np.floor(exact).astype(int)
    alloc = pd.Series(alloc, index=role_sizes.index)

    # Guarantee representation for every role, spending from the remainder.
    spare = budget - int(alloc.sum())
    for role in role_sizes.index:
        if spare <= 0:
            break
        if alloc[role] == 0 and role_sizes[role] > 0:
            alloc[role] = 1
            spare -= 1

    # Hand out what is left by largest fractional remainder.
    if spare > 0:
        remainder = (exact - np.floor(exact)).sort_values(ascending=False)
        for role in remainder.index:
            if spare <= 0:
                break
            if alloc[role] < role_sizes[role]:
                alloc[role] += 1
                spare -= 1

    alloc = alloc.clip(upper=role_sizes)
    return {r: int(n) for r, n in alloc.items()}


def build_cohort(org_df: pd.DataFrame, insiders: list, n_users: int,
                 seed: int = 42) -> dict:
    """
    @brief Select a reproducible cohort of users to aggregate and train on.

    @param org_df    Org structure with 'user' and 'role' columns.
    @param insiders  Designated insider user ids; always included in full.
    @param n_users   Target cohort size, insiders included.
    @param seed      RNG seed, recorded in the manifest for reproducibility.
    @return          Manifest dict with the user list and its provenance.
    """
    org = org_df.drop_duplicates("user").copy()
    org["role"] = org["role"].fillna("Unknown").astype(str)

    known = set(org["user"])
    insiders = sorted(set(insiders))
    missing = [u for u in insiders if u not in known]

    benign = org[~org["user"].isin(insiders)]
    budget = max(0, n_users - len(insiders))

    role_sizes = benign.groupby("role").size().sort_values(ascending=False)
    alloc = _allocate(role_sizes, budget)

    rng = np.random.default_rng(seed)
    picked = []
    for role in role_sizes.index:
        take = alloc[role]
        if take <= 0:
            continue
        pool = np.sort(benign.loc[benign["role"] == role, "user"].to_numpy())
        idx = rng.choice(len(pool), size=take, replace=False)
        picked.extend(pool[np.sort(idx)].tolist())

    users = sorted(set(insiders) | set(picked))

    roles_in = org[org["user"].isin(users)].groupby("role").size()
    return {
        "users":             users,
        "n_users":           len(users),
        "n_insiders":        len(insiders),
        "insiders":          insiders,
        "insiders_missing_from_ldap": missing,
        "n_benign":          len(users) - len(insiders),
        "seed":              seed,
        "requested_n_users": n_users,
        "population_size":   int(len(org)),
        "strategy":          "all-insiders + role-stratified benign sample",
        "n_roles_covered":   int(len(roles_in)),
        "n_roles_total":     int(org["role"].nunique()),
        "role_counts":       {str(k): int(v) for k, v in roles_in.sort_values(ascending=False).items()},
    }


def save_cohort(manifest: dict, path: str) -> None:
    """
    @brief Write a cohort manifest to JSON.

    @param manifest  Output of build_cohort().
    @param path      Destination .json path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_cohort(path: str) -> dict:
    """
    @brief Read a previously saved cohort manifest.

    @param path  Path to the manifest JSON.
    @return      Manifest dict.
    """
    with open(path) as f:
        return json.load(f)
