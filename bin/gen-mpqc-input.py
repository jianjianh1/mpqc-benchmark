#!/usr/bin/env python3
"""Generate a per-molecule MPQC input JSON from a template.

Reads a template JSON (e.g. water_4-full.json), swaps two fields, and
writes the result. All other keys — basis, PNO thresholds, the 11
`exprs`, etc. — pass through untouched. Stdlib only.

Fields rewritten:
  molecule.file_name                                 -> --xyz (path,
                                                      resolved relative
                                                      to the output JSON
                                                      by MPQC)
  wfn.sequant.trace.selected.tns_outdir              -> --tns-outdir

The xyz path is stored as a relative path from the output JSON's
directory (MPQC resolves a relative `file_name` against the JSON's
dirname; see src/mpqc/chemistry/molecule/molecule.cpp:132-136 +
src/bin/mpqc/mpqc_init.cpp:519-526).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _set_nested(cfg: dict, path: list[str], value, template_name: str) -> None:
    """Walk `cfg` along `path` and set the leaf to `value`.

    Raises ValueError naming the template + the missing segment if any
    intermediate key is absent, so a wrong template produces an
    actionable message instead of a bare KeyError traceback.
    """
    node = cfg
    for i, key in enumerate(path[:-1]):
        if not isinstance(node, dict) or key not in node:
            traversed = ".".join(path[: i + 1])
            raise ValueError(
                f"template {template_name!r} is missing key '{traversed}' — "
                f"cannot set '{'.'.join(path)}'"
            )
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError(
            f"template {template_name!r}: '{'.'.join(path[:-1])}' is not an object"
        )
    node[path[-1]] = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", required=True, help="source JSON template")
    ap.add_argument(
        "--xyz",
        required=True,
        help="path to the xyz file (relative from --output's dir, or absolute)",
    )
    ap.add_argument(
        "--tns-outdir",
        required=True,
        help="absolute path where MPQC should write the .tns files",
    )
    ap.add_argument("--output", required=True, help="output JSON path")
    args = ap.parse_args()

    with open(args.template, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Convert --xyz to a path relative to the output JSON's directory
    # so MPQC's relative-`file_name` resolution (which uses the JSON's
    # dirname as the prefix) finds the file. This matches the
    # convention used by `water_4-full.json` ("water_4.xyz" colocated
    # with the JSON). A relative --xyz is interpreted relative to the
    # current working directory before relpath conversion.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    xyz_field = os.path.relpath(os.path.abspath(args.xyz), out_dir)

    try:
        _set_nested(cfg, ["molecule", "file_name"], xyz_field, args.template)
        _set_nested(
            cfg,
            ["wfn", "sequant", "trace", "selected", "tns_outdir"],
            args.tns_outdir,
            args.template,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps unicode μ̃ in the `exprs` list as-is
        # (matches the readability of water_4-full.json).
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  wrote {args.output} (file_name={xyz_field}, tns_outdir={args.tns_outdir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
