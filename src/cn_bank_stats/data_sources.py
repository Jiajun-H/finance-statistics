"""Data loading helpers for the banking statistics project.

The project keeps a small manual dataset so the course analysis is reproducible
offline. The PBC Excel downloader is optional and can be used to refresh or
extend the credit-balance data when the official page structure is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

PBC_CREDIT_PAGE_URL = (
    "https://www.pbc.gov.cn/diaochatongjisi/116219/116319/5225358/5225362/index.html"
)
PBC_DEFAULT_XLSX_URL = (
    "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111817274124837.xlsx"
)
NFRA_Q4_2024_URL = (
    "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1199327&itemId=915"
)
NFRA_Q2_2025_URL = (
    "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1221429"
    "&generaltype=0&itemId=915"
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    manual_dir: Path
    raw_dir: Path
    tables_dir: Path
    figures_dir: Path


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_project_paths(root: Path | None = None) -> ProjectPaths:
    project_root = Path(root) if root else default_project_root()
    return ProjectPaths(
        root=project_root,
        manual_dir=project_root / "data" / "manual",
        raw_dir=project_root / "data" / "raw",
        tables_dir=project_root / "outputs" / "tables",
        figures_dir=project_root / "outputs" / "figures",
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)


def _add_period_date(df: pd.DataFrame, period_col: str = "period") -> pd.DataFrame:
    out = df.copy()
    periods = pd.PeriodIndex(out[period_col].astype(str), freq="Q")
    out[period_col] = periods.astype(str)
    out["date"] = periods.to_timestamp(how="end").normalize()
    out["year"] = periods.year
    out["quarter"] = periods.quarter
    return out


def read_manual_credit_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"period", "loans_trillion", "deposits_trillion", "source_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manual credit file is missing columns: {sorted(missing)}")
    df = _add_period_date(df)
    df["data_source"] = "manual_credit_csv"
    return df.sort_values("date").reset_index(drop=True)


def read_nfra_indicators(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "period",
        "npl_ratio",
        "nim",
        "provision_coverage",
        "capital_adequacy",
        "liquidity_ratio",
        "source_url",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NFRA indicator file is missing columns: {sorted(missing)}")
    df = _add_period_date(df)
    df["data_source"] = "manual_nfra_csv"
    return df.sort_values("date").reset_index(drop=True)


def download_file(url: str, dest: Path, timeout: int = 30) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/126.0 Safari/537.36"
        )
    }
    with requests.get(url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return dest


def discover_pbc_excel_links(page_url: str = PBC_CREDIT_PAGE_URL) -> list[tuple[str, str]]:
    response = requests.get(page_url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "lxml")
    links: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().endswith((".xlsx", ".xls")):
            links.append((anchor.get_text(" ", strip=True), urljoin(page_url, href)))
    return links


def download_pbc_credit_excel(
    raw_dir: Path,
    page_url: str = PBC_CREDIT_PAGE_URL,
    fallback_url: str = PBC_DEFAULT_XLSX_URL,
) -> Path:
    try:
        links = discover_pbc_excel_links(page_url)
        selected = next(
            (url for title, url in links if "人民币信贷收支表" in title),
            links[0][1] if links else fallback_url,
        )
    except Exception:
        selected = fallback_url

    filename = Path(urlparse(selected).path).name or "pbc_credit.xlsx"
    return download_file(selected, raw_dir / filename)


def _parse_month_cell(value: object, previous_month: int | None = None) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("年", ".").replace("月", "")
    if not text:
        return None

    year: int | None = None
    month: int | None = None
    if "." in text:
        left, right = text.split(".", 1)
        if left.isdigit() and right:
            year = int(left)
            right = "".join(ch for ch in right if ch.isdigit())
            if not right:
                return None
            if len(right) == 1 and previous_month == 9 and right == "1":
                month = 10
            else:
                month = int(right[:2])
    elif "-" in text or "/" in text:
        sep = "-" if "-" in text else "/"
        parts = text.split(sep)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            year = int(parts[0])
            month = int(parts[1])

    if year is None or month is None or not 1900 <= year <= 2100 or not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _infer_month_columns(row: pd.Series) -> dict[int, pd.Timestamp]:
    previous_month: int | None = None
    parsed: dict[int, pd.Timestamp] = {}
    for col, value in row.items():
        date = _parse_month_cell(value, previous_month=previous_month)
        if date is not None:
            parsed[int(col)] = date
            previous_month = int(date.month)
    return parsed


def _row_contains_keyword(raw: pd.DataFrame, row_idx: int, keyword: str) -> bool:
    row_text = " ".join(str(x) for x in raw.iloc[row_idx].dropna().tolist())
    return keyword in row_text


def _extract_block_series(
    raw: pd.DataFrame,
    start_row: int,
    end_row: int,
    date_cols: dict[int, pd.Timestamp],
    keyword: str,
) -> pd.Series | None:
    candidate_rows = [
        row_idx
        for row_idx in range(start_row, end_row)
        if _row_contains_keyword(raw, row_idx, keyword)
    ]
    cols = list(date_cols)
    for row_idx in candidate_rows:
        for nearby in range(row_idx, min(row_idx + 3, end_row)):
            values = pd.to_numeric(raw.iloc[nearby, cols], errors="coerce")
            if values.notna().sum() >= max(3, len(cols) // 2):
                values.index = [date_cols[col] for col in cols]
                return values
    return None


def parse_pbc_credit_excel(path: Path) -> pd.DataFrame:
    """Parse PBC credit Excel files into quarter-end loans and deposits.

    The PBC workbook often repeats a 12-month table by printed block. This
    parser scans each block, finds rows containing ``各项存款`` and ``各项贷款``,
    and keeps the quarter-end month in each quarter.
    """

    excel = pd.ExcelFile(path)
    monthly_frames: list[pd.DataFrame] = []
    for sheet_name in excel.sheet_names:
        raw = pd.read_excel(excel, sheet_name=sheet_name, header=None, dtype=object)
        date_blocks: list[tuple[int, dict[int, pd.Timestamp]]] = []
        for row_idx in range(len(raw)):
            month_cols = _infer_month_columns(raw.iloc[row_idx])
            if len(month_cols) >= 4:
                date_blocks.append((row_idx, month_cols))

        for block_idx, (date_row, date_cols) in enumerate(date_blocks):
            block_end = (
                date_blocks[block_idx + 1][0] if block_idx + 1 < len(date_blocks) else len(raw)
            )
            deposits = _extract_block_series(raw, date_row + 1, block_end, date_cols, "各项存款")
            loans = _extract_block_series(raw, date_row + 1, block_end, date_cols, "各项贷款")
            if deposits is None or loans is None:
                continue
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(list(date_cols.values())),
                    "deposits_trillion": deposits.to_numpy(dtype=float) / 10000,
                    "loans_trillion": loans.to_numpy(dtype=float) / 10000,
                }
            )
            monthly_frames.append(frame)

    if not monthly_frames:
        raise ValueError(f"No credit series could be parsed from {path}")

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = monthly.dropna(subset=["loans_trillion", "deposits_trillion"])
    monthly = monthly.drop_duplicates("date", keep="last").sort_values("date")
    monthly["period"] = monthly["date"].dt.to_period("Q").astype(str)
    quarterly = monthly.sort_values("date").groupby("period", as_index=False).tail(1)
    quarterly = _add_period_date(quarterly[["period", "loans_trillion", "deposits_trillion"]])
    quarterly["source_url"] = PBC_CREDIT_PAGE_URL
    quarterly["source_note"] = f"parsed_from_pbc_excel:{path.name}"
    quarterly["data_source"] = "pbc_excel"
    return quarterly.sort_values("date").reset_index(drop=True)


def load_credit_data(paths: ProjectPaths, refresh_pbc: bool = False) -> pd.DataFrame:
    manual_path = paths.manual_dir / "pbc_credit_summary.csv"
    manual = read_manual_credit_summary(manual_path)
    if not refresh_pbc:
        return manual

    try:
        excel_path = download_pbc_credit_excel(paths.raw_dir)
        parsed = parse_pbc_credit_excel(excel_path)
    except Exception as exc:
        print(f"[warn] PBC Excel refresh failed; using manual CSV. Reason: {exc}")
        return manual

    combined = pd.concat([manual, parsed], ignore_index=True)
    combined = combined.sort_values(["date", "data_source"])
    combined = combined.drop_duplicates("period", keep="last")
    return combined.sort_values("date").reset_index(drop=True)


def load_nfra_data(paths: ProjectPaths) -> pd.DataFrame:
    return read_nfra_indicators(paths.manual_dir / "nfra_bank_indicators.csv")


def merge_quarterly_data(credit: pd.DataFrame, nfra: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["date", "year", "quarter", "data_source"]
    nfra_for_merge = nfra.drop(columns=[c for c in cols_to_drop if c in nfra.columns])
    merged = credit.merge(nfra_for_merge, on="period", how="left", suffixes=("", "_nfra"))
    return _add_period_date(merged).sort_values("date").reset_index(drop=True)
