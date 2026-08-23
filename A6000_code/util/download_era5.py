# -*- coding: utf-8 -*-
"""
Full ERA5 download for the SDADiff training period (1993-2018), via Copernicus CDS.

Fixes in the original util/download_data.py:
  * year window / number of bins are configurable and bins never exceed years
  * resume support: existing files that pass a validity check are skipped
  * corruption detection: a file is only trusted if it opens with netCDF4 AND
    its time axis covers the requested years
  * disk-space check before each request
  * retry with backoff, request-queue friendly (one request at a time)

Expected output: 3 variables x 13 bins = 39 files named {u10,v10,msl}_{01..13}.nc
in <store>/ERA5/stormSurge_hourly_79_18/ (same layout as the original pipeline).

Usage (attach an external disk first, e.g. /Volumes/ERA5_DISK):
    python util/download_era5.py --store /Volumes/ERA5_DISK/Data2 \
        --start 1993 --end 2018 --bins 13 --var u10 --dry-run      # inspect plan
    python util/download_era5.py --store /Volumes/ERA5_DISK/Data2 \
        --start 1993 --end 2018 --bins 13                          # full run
"""
import os, sys, time, argparse, logging
import numpy as np
import cdsapi
import netCDF4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("era5")

VAR_MAP = {"u10": "10m_u_component_of_wind",
           "v10": "10m_v_component_of_wind",
           "msl": "mean_sea_level_pressure"}
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]

MIN_SIZE = 50 * 1024 * 1024  # a valid single-bin file is >> 50 MB


def verify_file(path, years):
    """True only if the file opens and its time axis covers all requested years."""
    try:
        if os.path.getsize(path) < MIN_SIZE:
            return False
        ds = netCDF4.Dataset(path)
        t = ds.variables["time"]
        n = len(t)
        ds.close()
        if n < 8000:  # hourly: >= ~8760 hours/year requested
            return False
        units = getattr(t, "units", "")
        if "hours since" in units:
            import datetime as dt
            base = dt.datetime.strptime(units.split("since ")[1].split()[0], "%Y-%m-%d")
            first = base + dt.timedelta(hours=int(t[0]))
            last = base + dt.timedelta(hours=int(t[-1]))
            have = set(range(first.year, last.year + 1))
            return have.issuperset(set(years))
        return True
    except Exception as e:
        log.warning("verify failed for %s: %s", path, e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="Data2", help="root data dir (attach external disk)")
    ap.add_argument("--start", type=int, default=1993)
    ap.add_argument("--end", type=int, default=2018)
    ap.add_argument("--bins", type=int, default=13)
    ap.add_argument("--vars", default="u10,v10,msl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    era5_dir = os.path.join(args.store, "ERA5", "stormSurge_hourly_79_18")

    years = np.arange(args.start, args.end + 1)
    if args.bins > len(years):
        raise SystemExit(f"--bins {args.bins} > years {len(years)}; reduce bins or widen range")
    intervals = np.array_split(years, args.bins)
    vars_ = [v for v in args.vars.split(",") if v in VAR_MAP]

    plan = [(v, i + 1, list(iv)) for v in vars_ for i, iv in enumerate(intervals)]
    log.info("plan: %d requests -> %s", len(plan), era5_dir)
    for v, idx, iv in plan:
        fname = os.path.join(era5_dir, f"{v}_{idx:02d}.nc")
        log.info("  %s  years %s  %s", v, iv, fname)
    if args.dry_run:
        log.info("dry run, exiting (no dirs created)")
        return
    os.makedirs(era5_dir, exist_ok=True)

    c = cdsapi.Client()
    for v, idx, iv in plan:
        fname = os.path.join(era5_dir, f"{v}_{idx:02d}.nc")
        if verify_file(fname, iv):
            log.info("skip (valid): %s", fname)
            continue
        if os.path.exists(fname):
            log.warning("remove invalid/partial: %s", fname)
            os.remove(fname)

        # disk space check (need ~ 3x the compressed request size as margin)
        free = os.statvfs(era5_dir).f_bavail * os.statvfs(era5_dir).f_frsize
        log.info("free disk: %.1f GB", free / 1e9)
        if free < 5e9:
            raise SystemExit("disk < 5 GB free, stopping")

        for attempt in range(1, args.retries + 1):
            try:
                log.info("fetch %s years=%s -> %s (attempt %d/%d)",
                         v, iv, fname, attempt, args.retries)
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": [VAR_MAP[v]],
                        "year": [str(y) for y in iv],
                        "month": ALL_MONTHS, "day": ALL_DAYS, "time": ALL_HOURS,
                    },
                    fname)
                if verify_file(fname, iv):
                    log.info("OK: %s", fname)
                    break
                log.warning("downloaded but verification failed, will retry")
            except Exception as e:
                log.warning("request failed: %s", e)
            if attempt < args.retries:
                time.sleep(30 * attempt)
        else:
            log.error("GIVING UP on %s after %d attempts", fname, args.retries)
    log.info("done. run util/build_cache_full.py after all files are present")


if __name__ == "__main__":
    main()
