"""HIMALAYAN EARTHPULSE command line.

  earthpulse regions                         list configured regions
  earthpulse add-region ID --name N --bbox W S E N
  earthpulse discover [--region ID]          query ASF for NISAR granules (cached in data/catalog.json)
  earthpulse resolve  [--region ID]          auto-detect stacks, models, reference area, masks
  earthpulse ingest   --region ID --product GUNW|GCOV|GOFF   remote AOI subsetting of selected stacks
  earthpulse run      gdm|cdm|bdm|epitn --region ID
  earthpulse export   [--region ID]          write web/public/data
  earthpulse all      --region ID            everything above for one region
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import typer

from earthpulse import config

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("earthpulse")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "earthaccess", "urllib3", "fsspec", "aiohttp", "rasterio", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _regions(region: Optional[str]):
    return [config.get_region(region)] if region else config.load_regions()


@app.command()
def regions() -> None:
    for r in config.load_regions():
        typer.echo(f"{r.id:20s} {r.priority:8s} {r.bbox}  {r.name}")


@app.command("add-region")
def add_region(region_id: str, name: str = typer.Option(...), bbox: tuple[float, float, float, float] = typer.Option(...),
               story: str = "", priority: str = "catalog") -> None:
    r = config.add_region(region_id, name, bbox, story, priority)
    typer.echo(f"added {r.id} {r.bbox}. Next: earthpulse resolve --region {r.id}")


@app.command()
def discover(region: Optional[str] = None, refresh: bool = True) -> None:
    _setup_logging()
    from earthpulse.catalog.discover import summarize
    from earthpulse.catalog.stacks import region_granules

    for r in _regions(region):
        gs = region_granules(r, refresh=refresh)
        s = summarize(gs)
        typer.echo(f"{r.id}: " + ", ".join(f"{p} {v['count']}" for p, v in s.items()))


@app.command()
def resolve(region: Optional[str] = None) -> None:
    _setup_logging()
    from earthpulse.sites.autodetect import resolve_region

    resolved = config.load_resolved()
    for r in _regions(region):
        resolved["regions"][r.id] = resolve_region(r)
        typer.echo(json.dumps({k: v for k, v in resolved["regions"][r.id].items()
                               if k in ("models", "stacks", "reference", "land_cover")}, indent=1, default=str))
    config.save_resolved(resolved)


@app.command()
def ingest(region: str = typer.Option(...), product: str = typer.Option(...), stack: Optional[str] = None,
           workers: int = 12, limit: int = 0, every: int = 1) -> None:
    _setup_logging()
    from earthpulse.catalog.stacks import region_granules, select_stacks
    from earthpulse.io.nisar import fetch, fetch_many

    r = config.get_region(region)
    gs = region_granules(r)
    stacks = select_stacks(gs, product)
    if stack:
        stacks = {k: v for k, v in stacks.items() if k == stack} or {
            stack: [g for g in gs if g.product == product and g.stack_key == stack]}
    for key, members in stacks.items():
        if every > 1:
            members = members[::every]
        if limit:
            members = members[:limit]
        log.info("%s %s stack %s: %d granules", r.id, product, key, len(members))
        if product != "SME2":  # acquisition geometry, noise floor and processing record once per stack
            fetch(r, product, members[0].url, members[0].granule_id, geometry=True)
        subs = fetch_many(r, [g.to_json() for g in members], workers=workers)
        log.info("stack %s: %d/%d subsets cached", key, len(subs), len(members))


@app.command()
def run(model: str, region: str = typer.Option(...)) -> None:
    _setup_logging()
    r = config.get_region(region)
    if model == "gdm":
        from earthpulse.models.gdm import run_gdm
        run_gdm(r)
    elif model == "cdm":
        from earthpulse.models.cdm import run_cdm
        run_cdm(r)
    elif model == "bdm":
        from earthpulse.models.bdm import run_bdm
        run_bdm(r)
    elif model == "epitn":
        from earthpulse.epitn.run import run_epitn
        run_epitn(r)
    else:
        raise typer.BadParameter("model must be gdm|cdm|bdm|epitn")


@app.command()
def verify(region: Optional[str] = None) -> None:
    """Verify every preprocessing stage on the real inputs (writes web/public/data/<region>/verification.json)."""
    _setup_logging()
    from earthpulse.validation.preprocess_check import verify_region

    for r in _regions(region):
        res = verify_region(r)
        typer.echo(f"{r.id}: {res['counts']}")
        for c in res["checks"]:
            typer.echo(f"  [{c['status']:4s}] {c['stage']:18s} {c['check']}: {c['value']}")


@app.command()
def hazards(region: str = typer.Option("joshimath")) -> None:
    """Phase 2: forcing -> slope instability (SIPM) -> Kalman engine (DEHFE) -> flood -> alerts, exported for the web."""
    _setup_logging()
    from earthpulse.export.hazards import run_hazards

    m = run_hazards(config.get_region(region))
    a = m["alert"]
    typer.echo(f"{region}: alert {a['name']} ({a['label']}) - " + "; ".join(a["reasons"]))


@app.command()
def export(region: Optional[str] = None) -> None:
    _setup_logging()
    from earthpulse.export.web import export_all

    export_all([r.id for r in _regions(region)])


@app.command("all")
def run_all(region: str = typer.Option(...)) -> None:
    _setup_logging()
    resolve(region)
    res = config.load_resolved()["regions"][region]
    products = {"gdm": "GUNW", "cdm": "GOFF", "bdm": "GCOV"}
    for m in res["models"]:
        ingest(region=region, product=products[m], stack=None, workers=12, limit=0, every=1)
        run(m, region)
    if len(res["models"]) >= 2:
        run("epitn", region)
    export(region)


if __name__ == "__main__":
    app()
