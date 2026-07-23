"""Shared filename -> semantic axis map for the sptc-bench .tns pipeline.

Single source of truth for both `tns-to-sptc-coo.py` (which needs it for
the empty-file shape fallback) and `check-sptc-shapes.py` (which audits
that every axis with the same semantic label has the same extent across
files). Previously each script carried its own copy kept "in sync" only
by a comment — a filename added to one and not the other silently
degraded the other script's coverage (see code-review finding on
check-sptc-shapes.py:23).
"""
from __future__ import annotations

# Filename -> semantic axis label per column position.
# "osv" = a in rank-3 C / rank-2 t; "pno" = a in rank-4 C / rank-4 t.
AXIS_MAP: dict[str, list[str]] = {
    "g_i_1_i_2_Κ_1.tns":     ["occ", "occ", "ri"],
    "g_i_1_m_1_Κ_1.tns":     ["occ", "pao", "ri"],
    "g_m_1_m_2_Κ_1.tns":     ["pao", "pao", "ri"],
    "f_i_1_i_2.tns":          ["occ", "occ"],
    "f_i_1_m_1.tns":          ["occ", "pao"],
    "f_m_1_m_2.tns":          ["pao", "pao"],
    "s_m_1_m_2.tns":          ["pao", "pao"],
    "C_i_1_m_1_a_1.tns":      ["occ", "pao", "osv"],
    "C_i_1_i_2_m_1_a_1.tns":  ["occ", "occ", "pao", "pno"],
    "t_i_1_a_1.tns":          ["occ", "osv"],
    "t_i_1_i_2_a_1_a_2.tns":  ["occ", "occ", "pno", "pno"],
}
