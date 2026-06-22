#!/usr/bin/env python3
"""
疾病-基因关联数据自动获取与整合脚本
===================================================
从多个权威数据源下载并整合阿尔茨海默病（AD）和衰老（Aging）
的疾病-基因关联数据，生成 disease_gene_associations.csv。

数据源：
  AD基因：  AlzGene（金标准）+ DisGeNET + GenAge（交叉验证）
  Aging基因：GenAge（金标准）+ CellAge + DisGeNET

用法：
  python disease_gene_scraper.py                  # 默认运行
  python disease_gene_scraper.py --cache          # 使用缓存
  python disease_gene_scraper.py --force-download # 强制重新下载
  python disease_gene_scraper.py --help           # 显示帮助
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # L1/
CACHE_DIR = BASE_DIR / "disease_cache"
OUTPUT_DIR = BASE_DIR / "l1_results"
CORE_GENE_FILE = OUTPUT_DIR / "L1_gene_level_analysis.csv"
OUTPUT_FILE = OUTPUT_DIR / "disease_gene_associations.csv"

# 手动下载文件预期路径
MANUAL_ALZGENE_FILE = CACHE_DIR / "alzgene_manual.csv"
MANUAL_DISGENET_AD_FILE = CACHE_DIR / "disgenet_ad_manual.tsv"
MANUAL_DISGENET_AGING_FILE = CACHE_DIR / "disgenet_aging_manual.tsv"

# ---------------------------------------------------------------------------
# HTTP Session with retry
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}


def create_session(retries=3, backoff_factor=1.0):
    """创建带重试机制的 requests Session。"""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


# ===================================================================
# 数据源 1: GenAge — 人类衰老基因组资源 (金标准 Aging)
# ===================================================================
GENAGE_URL = "https://genomics.senescence.info/genes/"


def fetch_genage(session, cache_path):
    """
    从 GenAge 网页抓取人类衰老相关基因列表。
    若缓存存在则直接返回。
    """
    if cache_path.exists():
        log.info("  GenAge 缓存命中: %s", cache_path)
        return pd.read_csv(cache_path)

    log.info("  正在从 GenAge 抓取数据: %s", GENAGE_URL)
    try:
        resp = session.get(GENAGE_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("  GenAge 抓取失败: %s", e)
        return pd.DataFrame(columns=["gene", "symbol"])

    # 使用 pandas read_html 解析 HTML 表格
    try:
        tables = pd.read_html(resp.text)
        # GenAge 页面通常第一个大表格即基因列表
        df = None
        for t in tables:
            # 查找含有 gene symbol 列的表格
            cols_lower = [str(c).lower() for c in t.columns]
            if any("symbol" in c or "gene" in c for c in cols_lower):
                df = t
                break
        if df is None and tables:
            df = tables[0]  # fallback

        if df is not None:
            # 标准化列名
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if "symbol" in cl or "gene" in cl:
                    col_map[c] = "gene"
            if col_map:
                df = df.rename(columns=col_map)
                if "gene" in df.columns:
                    df = df[["gene"]].dropna().copy()
                    df["gene"] = df["gene"].astype(str).str.strip()
                    df = df[df["gene"] != ""]
                    df.to_csv(cache_path, index=False)
                    log.info("  GenAge: 获取 %d 个基因", len(df))
                    return df

        log.warning("  GenAge: 未能在页面中找到基因表格，可能需要手动下载。")
        log.warning("  请访问 %s 并手动下载 GenAge Human 列表。", GENAGE_URL)
        return pd.DataFrame(columns=["gene"])

    except Exception as e:
        log.warning("  GenAge HTML 解析失败: %s", e)
        traceback.print_exc()
        return pd.DataFrame(columns=["gene"])


# ===================================================================
# 数据源 2: CellAge — 细胞衰老标记基因
# ===================================================================
CELLAGE_URL = "https://genomics.senescence.info/cells/"


def fetch_cellage(session, cache_path):
    """
    从 CellAge 网页抓取细胞衰老标记基因列表。
    """
    if cache_path.exists():
        log.info("  CellAge 缓存命中: %s", cache_path)
        return pd.read_csv(cache_path)

    log.info("  正在从 CellAge 抓取数据: %s", CELLAGE_URL)
    try:
        resp = session.get(CELLAGE_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("  CellAge 抓取失败: %s", e)
        return pd.DataFrame(columns=["gene"])

    try:
        tables = pd.read_html(resp.text)
        df = None
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any("symbol" in c or "gene" in c for c in cols_lower):
                df = t
                break
        if df is None and tables:
            df = tables[-1]  # 通常最后一个表是数据表

        if df is not None:
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if "symbol" in cl or "gene" in c.lower():
                    col_map[c] = "gene"
            if col_map:
                df = df.rename(columns=col_map)
                if "gene" in df.columns:
                    df = df[["gene"]].dropna().copy()
                    df["gene"] = df["gene"].astype(str).str.strip()
                    df = df[df["gene"] != ""]
                    df.to_csv(cache_path, index=False)
                    log.info("  CellAge: 获取 %d 个基因", len(df))
                    return df

        log.warning("  CellAge: 未能在页面中找到基因表格。")
        return pd.DataFrame(columns=["gene"])

    except Exception as e:
        log.warning("  CellAge HTML 解析失败: %s", e)
        traceback.print_exc()
        return pd.DataFrame(columns=["gene"])


# ===================================================================
# 数据源 3: DisGeNET — 综合性疾病基因数据库 (REST API)
# ===================================================================
DISGENET_API = "https://www.disgenet.org/api"


def fetch_disgenet(session, disease_name, cache_path, api_key=None):
    """
    通过 DisGeNET REST API 检索疾病-基因关联。

    Parameters
    ----------
    disease_name : str
        疾病名称，如 "Alzheimer's Disease" 或 "Aging"
    cache_path : Path
        缓存文件路径
    api_key : str, optional
        DisGeNET API key。若未提供且无手动文件，则跳过。

    Returns
    -------
    pd.DataFrame with columns [gene, score, source]
    """
    if cache_path.exists():
        log.info("  DisGeNET (%s) 缓存命中: %s", disease_name, cache_path)
        return pd.read_csv(cache_path)

    # 检查手动下载文件
    manual_path = cache_path.with_suffix(".manual.tsv")
    if manual_path.exists():
        log.info("  DisGeNET (%s) 手动文件命中: %s", disease_name, manual_path)
        try:
            df = pd.read_csv(manual_path, sep="\t")
            # 尝试标准化列名
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if "gene" in cl or "symbol" in cl:
                    col_map[c] = "gene"
                if "score" in cl:
                    col_map[c] = "score"
            df = df.rename(columns=col_map)
            if "gene" not in df.columns:
                log.warning("  DisGeNET 手动文件缺少 gene 列")
                return pd.DataFrame(columns=["gene", "score"])
            if "score" not in df.columns:
                df["score"] = 0.5
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            log.warning("  DisGeNET 手动文件读取失败: %s", e)
            traceback.print_exc()
            return pd.DataFrame(columns=["gene", "score"])

    # 若无 API key，提示手动下载
    if api_key is None:
        log.warning("  DisGeNET: 未提供 API key，跳过 API 查询。")
        log.warning(
            "  请访问 https://www.disgenet.org/ 注册获取 API key，"
            "或手动下载 \"%s\" 的基因关联数据并保存至: %s",
            disease_name,
            manual_path,
        )
        return pd.DataFrame(columns=["gene", "score"])

    # API 查询
    log.info("  正在查询 DisGeNET API: %s", disease_name)
    params = {
        "disease": f'"{disease_name}"',
        "format": "tsv",
    }
    headers_api = dict(HEADERS)
    headers_api["Authorization"] = f"Bearer {api_key}"

    for attempt in range(3):
        try:
            resp = session.get(
                f"{DISGENET_API}/gda/disease/{disease_name}",
                params=params,
                headers=headers_api,
                timeout=60,
            )
            resp.raise_for_status()

            # 解析 TSV
            lines = resp.text.strip().split("\n")
            if len(lines) <= 1:
                log.warning("  DisGeNET (%s): 返回空结果", disease_name)
                return pd.DataFrame(columns=["gene", "score"])

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), sep="\t")
            # 标准化
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if "gene" in cl or "symbol" in cl:
                    col_map[c] = "gene"
                if "score" in cl:
                    col_map[c] = "score"
            df = df.rename(columns=col_map)

            # 筛选 score > 0.3
            if "score" in df.columns:
                df = df[df["score"] > 0.3]
            else:
                df["score"] = 0.5

            df = df[["gene", "score"]].dropna().copy()
            df["gene"] = df["gene"].astype(str).str.strip()
            df = df[df["gene"] != ""]

            df.to_csv(cache_path, index=False)
            log.info("  DisGeNET (%s): 获取 %d 个基因", disease_name, len(df))
            return df

        except requests.RequestException as e:
            log.warning("  DisGeNET API 请求失败 (attempt %d/3): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
        except Exception as e:
            log.warning("  DisGeNET 解析失败: %s", e)
            traceback.print_exc()
            break

    return pd.DataFrame(columns=["gene", "score"])


# ===================================================================
# 数据源 4: AlzGene — AD 专门数据库（金标准 AD）
# ===================================================================
ALZGENE_URL = "http://www.alzgene.org/"


def fetch_alzgene(session, cache_path):
    """
    尝试获取 AlzGene 数据。
    由于 AlzGene 网站较老，通常需要手动下载。

    若检测到手动下载文件则直接使用；
    否则输出明确下载指引。
    """
    # 检查手动下载文件
    manual_path = cache_path  # alzgene_manual.csv

    if manual_path.exists():
        log.info("  AlzGene 手动文件命中: %s", manual_path)
        try:
            df = pd.read_csv(manual_path)
            # 标准化列名
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if "gene" in cl or "symbol" in cl:
                    col_map[c] = "gene"
                if "p" in cl and "value" in cl or cl == "pval" or cl == "p":
                    col_map[c] = "pvalue"
            df = df.rename(columns=col_map)

            if "gene" not in df.columns:
                log.warning("  AlzGene 手动文件缺少 gene 列，尝试使用第一列。")
                df["gene"] = df.iloc[:, 0]

            df = df[["gene"]].dropna().copy()
            df["gene"] = df["gene"].astype(str).str.strip()
            df = df[df["gene"] != ""]
            log.info("  AlzGene: 获取 %d 个基因 (手动)", len(df))
            return df

        except Exception as e:
            log.warning("  AlzGene 手动文件读取失败: %s", e)
            traceback.print_exc()
            return pd.DataFrame(columns=["gene"])

    # 尝试自动抓取
    log.info("  尝试自动抓取 AlzGene: %s", ALZGENE_URL)
    log.info("  （注意：AlzGene 网站可能不支持自动下载）")

    try:
        resp = session.get(ALZGENE_URL, timeout=30)
        resp.raise_for_status()

        # 查找可能的下载链接
        from urllib.parse import urljoin

        # 尝试常见下载链接模式
        download_candidates = [
            "/download/",
            "/downloads/",
            "/data/",
            "/results/",
            "/locus/",
        ]

        found = False
        for dc in download_candidates:
            try:
                dl_resp = session.get(urljoin(ALZGENE_URL, dc), timeout=30)
                dl_resp.raise_for_status()

                # 查找 CSV/Excel 链接
                import re
                csv_links = re.findall(
                    r'href="([^"]+\.(?:csv|xlsx|xls|txt|tsv))"',
                    dl_resp.text,
                )
                for link in csv_links:
                    full_url = urljoin(urljoin(ALZGENE_URL, dc), link)
                    log.info("  发现可能的下载链接: %s", full_url)
                    found = True
                    break
                if found:
                    break
            except Exception as exc:
                log.warning("AlzGene 候选链接 %s 访问失败: %s", dc, exc)
                traceback.print_exc()
                continue

    except requests.RequestException as e:
        log.warning("  AlzGene 自动抓取失败: %s", e)

    # 输出手动下载指引
    log.warning("=" * 60)
    log.warning("  AlzGene 无法自动获取，请手动下载：")
    log.warning("  1. 访问 http://www.alzgene.org/")
    log.warning("  2. 下载完整基因列表（Top Results 或 Download 页面）")
    log.warning("  3. 将文件另存为 CSV 格式")
    log.warning("  4. 保存至: %s", manual_path)
    log.warning("  文件应至少包含一列基因符号（列名含 'gene' 或 'symbol'）")
    log.warning("=" * 60)
    return pd.DataFrame(columns=["gene"])


# ===================================================================
# 备选方案：从 NCBI Gene / 文献构建 AD 已知基因列表
# ===================================================================
def build_ad_fallback_genes():
    """
    当 AlzGene 和 DisGeNET 都无法获取时，使用文献中已知的
    AD 核心基因作为 fallback。

    来源：
      1. GWAS Catalog (Bellenguez 2022, Jansen 2019, Kunkle 2019)
      2. AlzGene Top Results (http://www.alzgene.org/)
      3. DisGeNET curated AD gene-disease associations (score > 0.3)
      4. AD pathway genes: amyloid processing, tau phosphorylation,
         neuroinflammation, oxidative stress, ferroptosis, autophagy,
         synaptic dysfunction, lipid metabolism
      5. Neurodegeneration Reviews (Long & Holtzman 2019, Heneka 2015)
    """
    ad_core_genes = [
        # ================================================================
        # 一、经典 AD GWAS 风险基因 (Bellenguez 2022 Nat Genet, 75 loci)
        # ================================================================
        "APP", "PSEN1", "PSEN2", "APOE", "TREM2", "BIN1",
        "CLU", "ABCA7", "CR1", "PICALM", "MS4A6A", "CD33",
        "EPHA1", "CD2AP", "SORL1", "BACE1", "BACE2", "ADAM10",
        "PLCG2", "ABI3", "TSPO", "CASS4", "PTK2B",
        "ZCWPW1", "CELF1", "NME8", "FERMT2", "SLC24A4",
        "INPP5D", "MEF2C", "HLA-DRB1", "HLA-DRB5",
        "ADAMTS1", "ADAMTS4", "APH1B", "SPPL2A",
        "SPI1", "SIGLEC11", "ALPK2", "ANKH", "APH1B",
        "BLNK", "CD2AP", "CHRNE", "CLNK", "CNTNAP2",
        "COBL", "CR2", "CTSH", "CUX1", "CYP27A1",
        "DGKQ", "DOC2A", "DPYSL2", "ECHDC3", "EED",
        "EGFR", "FAM180B", "FOXF1", "FYN", "GPC6",
        "GRN", "HESX1", "HS3ST1", "HSPA2", "ICA1",
        "ICA1L", "IDUA", "IL34", "IQCK", "JADE1",
        "KAT8", "KLF16", "LILRB2", "LINC00173", "LINGO2",
        "MAF", "MADD", "MAPT", "MINDY2", "MIR142HG",
        "MS4A2", "MS4A4A", "MS4A4E", "MS4A6E",
        "NCK2", "NDUFAF6", "NECTIN2", "NEK6", "NFATC2",
        "NTN5", "NUP160", "NYAP1", "OPCML", "OSTN",
        "PDCD6IP", "PDE1C", "PINX1", "PLCG2", "PLEKHA1",
        "PRDM7", "PRKD3", "PTK2B", "RABEP1", "RASGEF1C",
        "RBCK1", "RHOH", "RIN3", "RMND5B", "SCIMP",
        "SEC61G", "SHARPIN", "SLC10A2", "SLC6A17", "SNX1",
        "SNX32", "SORT1", "SPDYE3", "SPPL2A", "SPRED2",
        "STX6", "SUZ12", "TET2", "TMEM106B", "TMEM163",
        "TNFRSF13B", "TNK1", "TOMM40", "TPCN1", "TPPP",
        "TREML2", "TRIM47", "TSPAN14", "UMAD1", "USP6NL",
        "VSNL1", "WDR12", "WDR81", "WWOX", "ZBED9",
        "ZNF652", "GSK3B", "IDE", "LRP1", "NECTIN2",
        "ACE", "CHRNB2", "BDNF", "VEGF", "NOS3",
        "PRNP", "CST3", "SNCA", "FUS", "TARDBP",

        # ================================================================
        # 二、淀粉样蛋白 (Aβ) 代谢通路
        # ================================================================
        "APP", "BACE1", "BACE2", "PSEN1", "PSEN2", "ADAM10",
        "ADAM17", "APH1A", "APH1B", "PSENEN", "NCSTN",
        "IDE", "NEP", "MME", "ECE1", "ECE2", "ACE",
        "LRP1", "LRP2", "LRP8", "RAGE", "AGER",
        "CLU", "APOE", "APOJ", "TTR", "ALB",
        "CTSB", "CTSD", "CTSL", "CST3",

        # ================================================================
        # 三、Tau 蛋白磷酸化与 NFT 通路
        # ================================================================
        "MAPT", "GSK3B", "CDK5", "CDK5R1",
        "PPP3CA", "PPP3CB", "PPP3CC", "DUSP1", "DUSP6",
        "PIN1", "PP2A", "PPP2CA", "PPP2CB",
        "MARK1", "MARK2", "MARK3", "MARK4",
        "DYRK1A", "Cdk5", "ERK1", "ERK2",

        # ================================================================
        # 四、神经炎症 (Neuroinflammation)
        # ================================================================
        "IL1B", "IL1A", "IL6", "TNF", "TNFRSF1A",
        "IL18", "IL10", "IL4", "TGFB1", "TGFB2",
        "NFKB1", "NFKB2", "RELA", "RELB",
        "NLRP3", "NLRP1", "NLRC4", "AIM2", "PYCARD", "CASP1",
        "TLR2", "TLR3", "TLR4", "TLR7", "TLR9",
        "MYD88", "IRAK1", "IRAK4", "TRAF6",
        "TREM2", "TYROBP", "CD33", "SIGLEC3",
        "CCL2", "CCL3", "CCL4", "CCL5", "CCL8", "CCL11",
        "CXCL1", "CXCL2", "CXCL8", "CXCL10", "CXCL12",
        "CX3CL1", "CX3CR1", "CCR2", "CCR5",
        "ICAM1", "VCAM1", "SELE", "SELP", "SELPLG",
        "PTGS2", "NOS2", "NOS3", "HMOX1",
        "IFNG", "IFNGR1", "IFNGR2",
        "TSPO", "S100B", "S100A8", "S100A9", "S100A12",
        "HMGB1", "RAGE", "AGER",
        "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B",

        # ================================================================
        # 五、氧化应激与铁死亡 (Oxidative Stress & Ferroptosis)
        # ================================================================
        "GPX1", "GPX4", "SOD1", "SOD2", "CAT",
        "PRDX1", "PRDX2", "PRDX3", "PRDX6",
        "TXN", "TXN2", "TXNRD1", "TXNRD2",
        "NFE2L2", "KEAP1", "HMOX1", "NQO1",
        "GCLC", "GCLM", "GSR", "GSS",
        "ACSL4", "LPCAT3", "ALOX5", "ALOX12", "ALOX15",
        "SLC7A11", "SLC3A2", "TFRC", "FTH1", "FTL",
        "STEAP3", "NCOA4", "IREB2", "HAMP",
        "SAT1", "SAT2", "SMS",
        "VDAC1", "VDAC2", "VDAC3",
        "CHAC1", "DDIT3", "ATF3", "ATF4",
        "TRIB3", "SESN2", "HSPA5", "HSPD1",
        "SLC40A1", "CP", "TF", "TFR2",

        # ================================================================
        # 六、线粒体功能障碍 (Mitochondrial Dysfunction)
        # ================================================================
        "PPARGC1A", "TFAM", "NRF1", "NRF2",
        "CYCS", "BAX", "BAK1", "BCL2", "BCL2L1", "BAD",
        "BNIP3", "BNIP3L", "FUNDC1",
        "PINK1", "PRKN", "PARK2", "PARK7",
        "MFN1", "MFN2", "OPA1", "DNM1L",
        "UCP2", "UCP3", "UCP4", "UCP5",
        "SIRT3", "SIRT4", "SIRT5",
        "SOD2", "GPX1", "PRDX3",
        "TOMM40", "TOMM20", "TIMM23", "TIMM44",
        "ATP5A1", "ATP5B", "ATP5C1",
        "ND1", "ND4", "ND5", "ND6", "COX1", "COX2", "COX3",
        "CYTB", "ATP6", "ATP8",

        # ================================================================
        # 七、自噬-溶酶体通路 (Autophagy-Lysosome)
        # ================================================================
        "BECN1", "ATG3", "ATG5", "ATG7", "ATG12", "ATG16L1",
        "SQSTM1", "OPTN", "NBR1", "CALCOCO2",
        "LAMP1", "LAMP2", "CTSB", "CTSD", "CTSL",
        "GABARAP", "GABARAPL1", "GABARAPL2",
        "MAP1LC3A", "MAP1LC3B", "MAP1LC3C",
        "ULK1", "ULK2", "RB1CC1",
        "PIK3C3", "PIK3R4", "UVRAG",
        "TFEB", "TFE3", "MITF",
        "PSEN1", "PSEN2",

        # ================================================================
        # 八、细胞凋亡与程序性坏死 (Apoptosis & Necroptosis)
        # ================================================================
        "TP53", "TP53BP1", "MDM2", "MDM4",
        "BAX", "BAK1", "BCL2", "BCL2L1", "BAD", "BID",
        "CASP2", "CASP3", "CASP6", "CASP7", "CASP8", "CASP9",
        "APAF1", "DIABLO", "XIAP", "BIRC5",
        "RIPK1", "RIPK3", "MLKL",
        "FADD", "TRADD",
        "PARP1", "PARP2", "AIFM1", "ENDOG",
        "CYCS",

        # ================================================================
        # 九、DNA 损伤修复 (DNA Damage Repair)
        # ================================================================
        "ATM", "ATR", "PRKDC", "DNAPK",
        "BRCA1", "BRCA2", "RAD51", "RAD50", "MRE11", "NBN",
        "XRCC1", "XRCC4", "XRCC5", "XRCC6",
        "LIG1", "LIG3", "LIG4",
        "CHEK1", "CHEK2", "CHK1", "CHK2",
        "TP53", "TP53BP1", "MDC1", "53BP1",
        "H2AFX", "H2AX", "RPA1", "RPA2", "RPA3",
        "PCNA", "FANCD2", "BLM", "WRN", "RECQL",
        "ERCC1", "ERCC2", "ERCC4", "ERCC5", "ERCC6", "ERCC8",
        "MSH2", "MSH3", "MSH6", "MLH1", "PMS2",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",

        # ================================================================
        # 十、表观遗传调控 (Epigenetic Regulation in AD)
        # ================================================================
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6",
        "SIRT1", "SIRT2", "SIRT3", "SIRT6", "SIRT7",
        "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2", "TET3",
        "EZH2", "SUZ12", "EED",
        "EP300", "CREBBP", "KAT2A", "KAT2B",
        "BRD4", "BRD2", "BRD3",
        "CREB1", "CREB5",

        # ================================================================
        # 十一、细胞周期与衰老 (Cell Cycle & Senescence)
        # ================================================================
        "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
        "RB1", "RBL1", "RBL2",
        "E2F1", "E2F2", "E2F3", "E2F4",
        "CDK1", "CDK2", "CDK4", "CDK6",
        "CCNA2", "CCNB1", "CCND1", "CCNE1", "CCNE2",
        "MYC", "MAX", "MNT", "MXD1",
        "PLK1", "AURKA", "AURKB",
        "BUB1", "BUB1B", "BUB3", "MAD2L1",
        "CDC20", "CDC25A", "CDC25B", "CDC25C",
        "MKI67", "TOP2A", "PCNA",
        "HMGA1", "HMGA2",
        "LMNA", "LMNB1", "LMNB2",
        "TP53", "MDM2",

        # ================================================================
        # 十二、MAPK / PI3K-AKT / mTOR 信号与突触可塑性
        # ================================================================
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14", "MAPK11",
        "JUN", "FOS", "FOSB", "FOSL1", "FOSL2",
        "ATF2", "ATF3", "ATF4", "ATF6",
        "DDIT3", "EIF2A", "EIF2AK3", "ERN1",
        "AKT1", "AKT2", "AKT3",
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
        "PTEN", "MTOR", "RPTOR", "RPS6KB1", "EIF4EBP1",
        "TSC1", "TSC2", "RHEB",
        "PRKAA1", "PRKAA2", "PRKAB1", "PRKAG1",
        "FOXO1", "FOXO3", "FOXO4",
        "STAT3", "JAK2", "STAT1",
        "NFATC1", "NFATC2", "NFATC3", "NFATC4",
        "CREBBP", "EP300",
        "SYP", "SYN1", "SYN2", "SYN3",
        "DLG4", "PSD95", "GRIN1", "GRIN2A", "GRIN2B",
        "GRIA1", "GRIA2", "GRIA3", "GRIA4",
        "GABRA1", "GABRB2", "GABRG2",
        "SNAP25", "VAMP2", "STX1A", "STXBP1",
        "BDNF", "NTRK2", "NGF", "NGFR",

        # ================================================================
        # 十三、脂质代谢与胆固醇 (Lipid/Cholesterol in AD)
        # ================================================================
        "APOE", "CLU", "ABCA1", "ABCA7", "ABCG1",
        "LRP1", "LDLR", "VLDLR",
        "SREBF1", "SREBF2", "SCAP",
        "HMGCR", "CYP46A1", "CYP27A1",
        "LPL", "LIPC", "LCAT",
        "PLA2G4A", "PLA2G6", "PLD1", "PLD2",
        "LPIN1", "LPIN2", "LPIN3",
        "PNPLA2", "PLIN2", "PLIN3", "LIPE",
        "PPARA", "PPARG", "PPARD",
        "FABP3", "FABP5", "FABP7",
        "ACSL1", "ACSL3", "ACSL4", "ACSL5",

        # ================================================================
        # 十四、铜/铁/锌稳态 (Metal Homeostasis in AD)
        # ================================================================
        "ATP7A", "ATP7B", "CP", "SLC31A1", "SLC31A2",
        "MT1A", "MT1E", "MT1F", "MT1G", "MT1H",
        "MT1X", "MT2A", "MT3",
        "COMMD1", "CCS", "SOD1",
        "TFRC", "TF", "TFR2", "FTH1", "FTL",
        "FPN1", "SLC40A1", "HAMP", "HFE",
        "STEAP1", "STEAP2", "STEAP3", "STEAP4",
        "DMT1", "SLC11A2", "ZIP8", "SLC39A8",
        "ZIP14", "SLC39A14",
        "SLC30A1", "SLC30A3", "SLC30A10",

        # ================================================================
        # 十五、细胞外基质与血管 (ECM & Vascular in AD)
        # ================================================================
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2",
        "FN1", "FBN1", "ELN",
        "MMP1", "MMP2", "MMP3", "MMP7", "MMP9",
        "MMP10", "MMP12", "MMP13", "MMP14",
        "TIMP1", "TIMP2", "TIMP3", "TIMP4",
        "SERPINE1", "PLAT", "PLAU", "PLAUR",
        "ICAM1", "VCAM1", "SELE", "SELP",
        "VEGFA", "VEGFB", "VEGFC",
        "FLT1", "KDR", "FLT4",
        "HIF1A", "EPAS1", "ARNT",

        # ================================================================
        # 十六、AD 脑转录组失调基因 (AD brain transcriptomics)
        # ================================================================
        "GFAP", "AIF1", "IBA1", "ITGAM", "CD68",
        "CD4", "CD8A", "CD8B", "CD74",
        "HLA-A", "HLA-B", "HLA-C",
        "FCGR1A", "FCGR2A", "FCGR3A",
        "CSF1R", "CSF1", "CSF2", "CSF3",
        "CX3CR1", "P2RY12", "TMEM119", "OLR1",
        "AQP4", "GJA1", "GJB6",
        "MOG", "MBP", "PLP1", "MAG", "OMG",
        "OLIG1", "OLIG2", "SOX10", "NKX2-2",
        "RBFOX3", "NEUN", "DCX",
        "GAD1", "GAD2", "SLC17A6", "SLC17A7",
        "SLC32A1", "SLC6A1",
        "TH", "DDC", "SLC6A3",
        "CHAT", "ACHE", "BCHE", "SLC5A7",
        "TPH1", "TPH2", "SLC6A4",
        "HOMER1", "ARC", "EGR1", "NPAS4",
        "BDNF", "NTRK2", "CREB1",

        # ================================================================
        # 十七、AD 代谢失调 (Metabolic Dysregulation in AD)
        # ================================================================
        "INSR", "IRS1", "IRS2",
        "G6PC", "PCK1", "PCK2",
        "HK1", "HK2", "GCK",
        "PFKL", "PFKM", "PFKP",
        "PKM", "PKLR",
        "LDHA", "LDHB", "LDHC",
        "PDHA1", "PDHB",
        "CS", "IDH1", "IDH2", "IDH3A",
        "SDHA", "SDHB", "SDHC", "SDHD",
        "FH", "MDH1", "MDH2",
        "GLS", "GLUL", "GLUD1", "GLUD2",
        "SLC2A1", "SLC2A3", "SLC2A4",
        "MCT1", "SLC16A1", "MCT4", "SLC16A3",
        "CD38", "NAMPT", "NMNAT1", "NMNAT2", "NMNAT3",
        "CDO1", "GOT1", "GOT2", "GPT", "GPT2",
        "CBS", "CTH", "MTR", "MTRR", "MTHFR",
        "BHMT", "BHMT2", "DMGDH", "SARDH",

        # ================================================================
        # 十八、AD 泛素-蛋白酶体 (UPS & Protein Quality)
        # ================================================================
        "UBB", "UBC", "UBA52", "RPS27A",
        "UBA1", "UBA2", "UBA3", "UBA5", "UBA6", "UBA7",
        "UBE2A", "UBE2B", "UBE2D1", "UBE2D2", "UBE2D3",
        "UBE2E1", "UBE2L3", "UBE2N",
        "UBE3A", "UBE3B", "UBE3C",
        "PSMA1", "PSMB5", "PSMC1", "PSMD1",
        "PSME1", "PSME2",
        "HSP90AA1", "HSP90AB1", "HSP90B1",
        "HSPA1A", "HSPA1B", "HSPA4", "HSPA5", "HSPA8", "HSPA9",
        "HSPB1", "HSPB8",
        "HSF1", "HSF2",
        "DNAJA1", "DNAJA2", "DNAJB1", "DNAJB2",
        "DNAJB6", "DNAJC5", "DNAJC6",
        "BAG1", "BAG2", "BAG3", "BAG4", "BAG5",
        "CHIP", "STUB1",
        "VCP", "UBQLN1", "UBQLN2", "UBQLN4",
        "PSEN1", "PSEN2",
    ]
    # 去重
    ad_core_genes = list(dict.fromkeys(ad_core_genes))
    log.info("  使用文献 fallback AD 基因列表 (%d 个基因)", len(ad_core_genes))
    return pd.DataFrame({"gene": ad_core_genes})


# ===================================================================
# 备选方案：从 GenAge/CellAge 文献构建 Aging 已知基因列表
# ===================================================================
def build_aging_fallback_genes():
    """
    当 GenAge、CellAge 和 DisGeNET 都无法获取时，使用综合的
    衰老相关基因列表作为 fallback。

    来源：
      1. GenAge Human (https://genomics.senescence.info/genes/) — 307 基因
      2. CellAge (https://genomics.senescence.info/cells/) — 279 基因
      3. Lopez-Otin 2023 Hallmarks of Aging (Cell 186:243-278)
      4. SenMayo 基因集 (Saul 2022 Nat Commun)
      5. GO:0007568 Aging + GO:0090398 Cellular Senescence
      6. Reactome Aging pathway (R-HSA-2559582)
      7. SASP Atlas (senescence-associated secretory phenotype)
      8. KEGG Longevity regulating pathway (hsa04211)
      9. FRIDGE senescence gene set
      10. Ageing Clocks (Horvath, Hannum, PhenoAge, GrimAge)
    """
    aging_core_genes = [
        # ================================================================
        # 一、基因组不稳定性 (Genomic Instability)
        # ================================================================
        # DNA 损伤修复核心
        "TP53", "ATM", "ATR", "PRKDC", "DNAPK",
        "CHEK1", "CHEK2", "CHK1", "CHK2",
        "H2AFX", "H2AX", "MDC1", "TP53BP1", "53BP1",
        "RPA1", "RPA2", "RPA3", "PCNA",
        # 同源重组
        "BRCA1", "BRCA2", "RAD51", "RAD50", "MRE11", "NBN",
        "RAD52", "RAD54L", "XRCC2", "XRCC3",
        "PALB2", "BARD1", "BRIP1",
        # 非同源末端连接
        "XRCC4", "XRCC5", "XRCC6", "LIG4",
        # 碱基切除修复
        "XRCC1", "LIG1", "LIG3",
        "OGG1", "MUTYH", "NTHL1", "NEIL1", "NEIL2", "NEIL3",
        "APEX1", "APEX2", "POLB",
        # 核苷酸切除修复
        "ERCC1", "ERCC2", "ERCC3", "ERCC4", "ERCC5",
        "ERCC6", "ERCC8", "XPA", "XPC", "DDB1", "DDB2",
        "RAD23A", "RAD23B", "CETN2",
        # 错配修复
        "MSH2", "MSH3", "MSH6", "MLH1", "PMS1", "PMS2",
        # Fanconi anemia
        "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG",
        "FANCI", "FANCL", "FANCM",
        # RecQ 解旋酶
        "WRN", "BLM", "RECQL", "RECQL4", "RECQL5",
        # MCM 复合体
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        # 其他
        "PARP1", "PARP2",
        "TOP1", "TOP2A", "TOP2B",
        "RMI1", "RMI2", "TOP3A", "TOP3B",
        "SMUG1", "TDG", "UNG", "MPG",

        # ================================================================
        # 二、端粒磨损 (Telomere Attrition)
        # ================================================================
        "TERT", "TERC", "DKC1", "NOP10", "NHP2", "GAR1",
        "POT1", "TINF2", "ACD", "TERF1", "TERF2",
        "TPP1", "RAP1", "TERF2IP",
        "TEP1", "WRAP53", "TCAB1",
        "RTEL1", "CTC1", "STN1", "TEN1",
        "NAF1", "PARN", "OBFC1",
        "XRCC5", "XRCC6",  # Ku70/Ku80
        "PINX1",

        # ================================================================
        # 三、表观遗传改变 (Epigenetic Alterations)
        # ================================================================
        # DNA 甲基化
        "DNMT1", "DNMT3A", "DNMT3B", "DNMT3L",
        "TET1", "TET2", "TET3",
        "MBD1", "MBD2", "MBD3", "MBD4", "MECP2",
        "UHRF1", "UHRF2",
        # 组蛋白修饰
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6",
        "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11",
        "SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5",
        "SIRT6", "SIRT7",
        "EP300", "CREBBP", "KAT2A", "KAT2B",
        "KAT5", "KAT6A", "KAT6B", "KAT7", "KAT8",
        "EZH1", "EZH2", "SUZ12", "EED", "RBBP4", "RBBP7",
        "KDM1A", "KDM1B", "KDM2A", "KDM2B",
        "KDM3A", "KDM3B", "KDM4A", "KDM4B", "KDM4C",
        "KDM5A", "KDM5B", "KDM5C", "KDM5D",
        "KDM6A", "KDM6B", "KDM7A",
        # 染色质重塑
        "SMARCA2", "SMARCA4", "SMARCB1", "SMARCC1", "SMARCD1",
        "ARID1A", "ARID1B", "ARID2",
        "BRD2", "BRD3", "BRD4", "BRDT",
        "HMGA1", "HMGA2",
        "CBX1", "CBX3", "CBX5",  # HP1
        "H1F0", "H1FX", "HIST1H1A",

        # ================================================================
        # 四、蛋白质稳态丧失 (Loss of Proteostasis)
        # ================================================================
        # 热休克蛋白 / 分子伴侣
        "HSP90AA1", "HSP90AB1", "HSP90B1",
        "HSPA1A", "HSPA1B", "HSPA2", "HSPA4", "HSPA5",
        "HSPA8", "HSPA9", "HSPA12A", "HSPA14",
        "HSPB1", "HSPB2", "HSPB3", "HSPB6", "HSPB7", "HSPB8",
        "HSPD1", "HSPE1",
        "HSF1", "HSF2", "HSF4",
        "DNAJA1", "DNAJA2", "DNAJB1", "DNAJB2", "DNAJB6",
        "DNAJC1", "DNAJC3", "DNAJC5", "DNAJC6",
        "BAG1", "BAG2", "BAG3", "BAG4", "BAG5",
        "STUB1", "CHIP",
        "CCT1", "CCT2", "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8",
        # 泛素-蛋白酶体
        "UBB", "UBC", "UBA52", "RPS27A",
        "UBA1", "UBA2", "UBA3", "UBA5", "UBA6", "UBA7",
        "UBE2A", "UBE2B", "UBE2D1", "UBE2D2", "UBE2D3",
        "UBE2E1", "UBE2L3", "UBE2N",
        "UBE3A", "UBE3B", "UBE3C",
        "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
        "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
        "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
        "PSMD1", "PSMD2", "PSMD3", "PSMD4",
        "PSME1", "PSME2", "PSME3", "PSME4",
        "UBQLN1", "UBQLN2", "UBQLN4",
        "VCP", "NPLOC4", "UFD1",
        # 未折叠蛋白反应 (UPR)
        "XBP1", "ATF4", "ATF6", "DDIT3",
        "ERN1", "EIF2AK3", "EIF2A",
        "HSPA5", "DNAJC3", "ERN1",
        # 自噬-溶酶体
        "BECN1", "ATG3", "ATG5", "ATG7", "ATG10", "ATG12",
        "ATG13", "ATG14", "ATG16L1", "ATG101",
        "SQSTM1", "OPTN", "NBR1", "CALCOCO2", "TOLLIP",
        "LAMP1", "LAMP2",
        "CTSB", "CTSD", "CTSL",
        "GABARAP", "GABARAPL1", "GABARAPL2",
        "MAP1LC3A", "MAP1LC3B", "MAP1LC3C",
        "ULK1", "ULK2", "RB1CC1", "ATG13",
        "PIK3C3", "PIK3R4", "UVRAG", "RUBCN",
        "TFEB", "TFE3", "MITF", "ZKSCAN3",
        "WIPI1", "WIPI2", "AMBRA1",

        # ================================================================
        # 五、营养感知失调 (Deregulated Nutrient Sensing)
        # ================================================================
        # mTOR 通路
        "MTOR", "RPTOR", "RICTOR", "MLST8",
        "AKT1", "AKT2", "AKT3",
        "RPS6KB1", "RPS6KB2",
        "EIF4EBP1", "EIF4EBP2", "EIF4E",
        "TSC1", "TSC2", "RHEB",
        "DEPTOR", "PRAS40", "AKT1S1",
        "RRAGA", "RRAGB", "RRAGC", "RRAGD",
        "LAMTOR1", "LAMTOR2", "LAMTOR3", "LAMTOR4", "LAMTOR5",
        # PI3K-AKT
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
        "PIK3R1", "PIK3R2", "PIK3R3",
        "PTEN", "PDK1",
        # AMPK
        "PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2",
        "PRKAG1", "PRKAG2", "PRKAG3",
        "STK11", "CAB39", "STRADA", "STRADB",
        # IGF-1 / Insulin
        "IGF1", "IGF1R", "IGF2", "IGF2R",
        "INS", "INSR", "IRS1", "IRS2",
        "GH1", "GHR", "GHRH", "GHRHR",
        # Sirtuins (NAD+ sensors)
        "SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5",
        "SIRT6", "SIRT7",
        "NAMPT", "NMNAT1", "NMNAT2", "NMNAT3",
        "CD38", "BST1",  # NAD+ consumers
        "PARP1", "PARP2",
        # FOXO
        "FOXO1", "FOXO3", "FOXO4", "FOXO6",
        # 卡路里限制模拟
        "PPARGC1A", "NRF1", "NRF2",
        "TFEB", "CREB1", "CRTC1", "CRTC2",
        # FGF21 / GDF15
        "FGF21", "FGFR1", "KLB",
        "GDF15", "GFRAL",

        # ================================================================
        # 六、线粒体功能障碍 (Mitochondrial Dysfunction)
        # ================================================================
        # 线粒体生物合成
        "PPARGC1A", "PPARGC1B", "TFAM", "TFB1M", "TFB2M",
        "NRF1", "NFE2L2", "PPARA", "PPARD", "PPARG",
        "ESRRA", "ESRRG",
        # 线粒体融合/分裂
        "MFN1", "MFN2", "OPA1",
        "DNM1L", "FIS1", "MFF", "MIEF1", "MIEF2",
        # 线粒体自噬 (Mitophagy)
        "PINK1", "PRKN", "PARK2", "PARK7",
        "BNIP3", "BNIP3L", "FUNDC1",
        "OPTN", "SQSTM1", "CALCOCO2",
        # 电子传递链
        "MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND5", "MT-ND6",
        "NDUFA1", "NDUFA2", "NDUFA4", "NDUFA8", "NDUFA9",
        "NDUFB8", "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS7", "NDUFS8",
        "SDHA", "SDHB", "SDHC", "SDHD",
        "UQCRC1", "UQCRC2", "CYC1",
        "MT-CO1", "MT-CO2", "MT-CO3", "COX4I1", "COX5A", "COX5B",
        "COX6A1", "COX6B1", "COX7A2", "COX8A",
        "MT-ATP6", "MT-ATP8", "ATP5A1", "ATP5B", "ATP5C1", "ATP5O",
        "MT-CYB",
        # 线粒体 ROS 防御
        "SOD2", "GPX1", "GPX4", "PRDX3", "PRDX5",
        "TXN2", "TXNRD2",
        # 线粒体其他
        "TOMM20", "TOMM22", "TOMM40", "TOMM70",
        "TIMM23", "TIMM44", "TIMM50",
        "VDAC1", "VDAC2", "VDAC3",
        "ANT1", "SLC25A4", "SLC25A5", "SLC25A6",
        "CYCS", "ENDOG", "AIFM1", "HTRA2",
        "HSPD1", "HSPE1", "LONP1", "CLPP",
        "SURF1", "LRPPRC",
        "UCP1", "UCP2", "UCP3",

        # ================================================================
        # 七、细胞衰老 (Cellular Senescence) — GenAge + CellAge + SenMayo
        # ================================================================
        # 衰老核心效应器
        "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "CDKN2C", "CDKN2D",
        "RB1", "RBL1", "RBL2",
        "TP53", "TP53BP1", "MDM2", "MDM4",
        "E2F1", "E2F2", "E2F3", "E2F4",
        # 衰老标志基因 (SenMayo + CellAge)
        "GLB1", "SA-B-GAL",
        "LMNB1", "LMNB2", "LMNA",
        "HMGB1", "HMGB2",
        "H2AFX", "H2AJ",
        "CD44", "CDKN1A", "CDKN2A",
        "DEC1", "BHLHE40", "BHLHE41",
        "DEP1", "PTPRJ", "DPP4",
        "ICAM1", "VCAM1", "SELE",
        "SERPINE1", "PAI1",
        "IGFBP3", "IGFBP5", "IGFBP7",
        "VIM", "KRT8", "KRT18",
        "COL1A1", "COL1A2", "COL3A1", "FN1",
        "MMP1", "MMP2", "MMP3", "MMP7", "MMP9", "MMP10",
        "MMP12", "MMP13", "MMP14",
        "TIMP1", "TIMP2", "TIMP3",
        "PLAT", "PLAU", "PLAUR",
        "TGFB1", "TGFB2", "TGFB3",
        "TGFBR1", "TGFBR2",
        "SMAD2", "SMAD3", "SMAD4",
        # GenAge 人类衰老基因（全列表关键补充）
        "AGER", "RAGE", "AKT1", "APOE", "AR", "AREG",
        "ATG5", "ATG7", "ATF2", "ATM", "BUB1B", "BUB1",
        "CAT", "CAV1", "CCL2", "CCL3", "CCL4", "CCL5",
        "CCNA2", "CCNB1", "CCND1", "CCNE1", "CCNE2",
        "CDK1", "CDK2", "CDK4", "CDK6",
        "CEBPA", "CEBPB", "CEBPD", "CEBPG",
        "CLU", "CREBBP", "CRYAB",
        "CSF1", "CSF2", "CSF3",
        "CXCL1", "CXCL2", "CXCL8", "CXCL10", "CXCL12",
        "DDIT3", "E2F1", "EGF", "EGFR",
        "ERCC1", "ERCC2",
        "FGF1", "FGF2", "FGF7", "FGF21",
        "FOS", "FOSL1", "FOSL2",
        "GADD45A", "GADD45B", "GADD45G",
        "GCLC", "GCLM",
        "GPX1", "GPX4",
        "GSK3B",
        "HDAC1", "HDAC2", "HDAC3",
        "HGF", "HMGA1", "HMGA2",
        "HMOX1", "HO1",
        "HSPA1A", "HSPA1B",
        "IFNA1", "IFNB1", "IFNG",
        "IGFBP1", "IGFBP2", "IGFBP3", "IGFBP4", "IGFBP5", "IGFBP7",
        "IL1A", "IL1B", "IL4", "IL6", "IL8",
        "IL10", "IL15", "IL18",
        "JUN", "JUNB", "JUND",
        "KL", "KLOTHO",
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14",
        "MKI67",
        "MYC", "MAX", "MNT",
        "NFE2L2", "NFKB1", "NFKB2", "RELA", "RELB",
        "NGF", "NGFR",
        "NOS2", "NOS3",
        "NOX4",
        "PARP1",
        "PCNA",
        "PLK1",
        "PPARA", "PPARD", "PPARG",
        "PRDX1", "PRDX2", "PRDX3",
        "PTEN", "PTGS2", "COX2",
        "RAD50", "RAD51",
        "SERPINB2", "PAI2",
        "SIRT1", "SIRT6", "SIRT7",
        "SOD1", "SOD2",
        "SP1", "SP3",
        "STAT1", "STAT3", "STAT5A", "STAT5B",
        "TERT", "TFAM",
        "TFRC",
        "TNF", "TNFRSF1A", "TNFRSF1B",
        "TP53", "TP63", "TP73",
        "TXN", "TXN2",
        "UBB", "UBC",
        "VEGFA", "VEGFB", "VEGFC",
        "WRN", "XRCC5", "XRCC6",
        "ZEB1", "ZEB2", "SNAI1", "SNAI2",

        # ================================================================
        # 八、干细胞耗竭 (Stem Cell Exhaustion)
        # ================================================================
        "OCT4", "POU5F1", "SOX2", "NANOG", "KLF4", "MYC",
        "LIN28A", "LIN28B",
        "SALL4", "UTF1", "DPPA3", "REX1", "ZFP42",
        "TERT", "WRN", "BLM",
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4",
        "JAG1", "JAG2", "DLL1", "DLL3", "DLL4",
        "HES1", "HES5", "HEY1", "HEY2",
        "WNT1", "WNT3A", "WNT5A", "WNT10B",
        "CTNNB1", "AXIN1", "AXIN2", "APC",
        "TCF3", "TCF4", "TCF7", "TCF7L1", "TCF7L2", "LEF1",
        "SHH", "IHH", "DHH", "PTCH1", "SMO", "GLI1", "GLI2", "GLI3",
        "BMP2", "BMP4", "BMP7", "BMPR1A", "BMPR1B", "BMPR2",
        "SMAD1", "SMAD5", "SMAD9",
        "LIF", "LIFR", "IL6ST", "GP130",
        "HOXB4", "HOXA9", "HOXA10",
        "BMI1", "PCGF4", "RING1A", "RING1B",
        "MLL1", "KMT2A", "MLL5", "KMT2E",
        "CD34", "KIT", "FLT3", "THPO", "MPL",
        "GATA1", "GATA2", "GATA3",
        "RUNX1", "RUNX2", "RUNX3",
        "CEBPA", "PU1", "SPI1",

        # ================================================================
        # 九、细胞间通讯改变 (Altered Intercellular Communication)
        #     — 炎症衰老 (Inflammaging) + SASP
        # ================================================================
        # SASP 核心细胞因子
        "IL1A", "IL1B", "IL1RN",
        "IL6", "IL6R", "IL6ST",
        "IL8", "CXCL8",
        "TNF", "TNFRSF1A", "TNFRSF1B",
        "IFNG", "IFNGR1", "IFNGR2",
        "TGFB1", "TGFB2", "TGFB3",
        # 趋化因子 (SASP)
        "CCL1", "CCL2", "CCL3", "CCL4", "CCL5", "CCL7", "CCL8",
        "CCL11", "CCL13", "CCL17", "CCL20", "CCL22", "CCL26",
        "CXCL1", "CXCL2", "CXCL3", "CXCL5", "CXCL6",
        "CXCL9", "CXCL10", "CXCL11", "CXCL12", "CXCL13",
        "CX3CL1", "CX3CR1",
        "XCL1", "XCL2",
        # 炎症小体
        "NLRP1", "NLRP3", "NLRC4", "NLRP6", "NLRP12",
        "AIM2", "PYCARD", "CASP1", "CASP4", "CASP5",
        "IL18", "IL1B", "GSDMD", "GSDME",
        # TLR 信号
        "TLR1", "TLR2", "TLR3", "TLR4", "TLR5", "TLR6",
        "TLR7", "TLR8", "TLR9", "TLR10",
        "MYD88", "TIRAP", "TRIF", "TICAM1", "TRAM", "TICAM2",
        "IRAK1", "IRAK2", "IRAK4", "TRAF6",
        # NF-κB 通路
        "NFKB1", "NFKB2", "RELA", "RELB", "REL",
        "IKBKB", "IKBKG", "IKBKE", "TBK1",
        "NFKBIA", "NFKBIB", "NFKBIE", "NFKBIZ",
        # JAK-STAT
        "JAK1", "JAK2", "JAK3", "TYK2",
        "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
        "SOCS1", "SOCS2", "SOCS3", "CISH",
        # 生长因子 (SASP)
        "EGF", "FGF1", "FGF2", "FGF7", "FGF21",
        "HGF", "MSTN", "GDF11",
        "IGF1", "IGF2", "IGFBP1", "IGFBP2", "IGFBP3",
        "IGFBP4", "IGFBP5", "IGFBP7",
        "PDGFA", "PDGFB", "PDGFC", "PDGFD",
        "VEGFA", "VEGFB", "VEGFC",
        "CSF1", "CSF2", "CSF3",
        "AREG", "EREG", "HBEGF", "TGFA",
        # 基质金属蛋白酶 (SASP)
        "MMP1", "MMP2", "MMP3", "MMP7", "MMP8", "MMP9",
        "MMP10", "MMP12", "MMP13", "MMP14",
        "TIMP1", "TIMP2", "TIMP3",
        "SERPINE1", "SERPINB2", "PLAT", "PLAU",
        # 其他 SASP 因子
        "ICAM1", "VCAM1", "SELE", "SELP",
        "PTGS2", "COX2",
        "S100A8", "S100A9", "S100A12",
        "HMGB1", "HMGN1",
        "MIF", "MICB", "MICA",
        "FAS", "FASLG",
        "TRAIL", "TNFSF10", "TNFRSF10A", "TNFRSF10B",

        # ================================================================
        # 十、细胞凋亡抵抗与铁死亡 (Apoptosis Resistance & Ferroptosis)
        # ================================================================
        # BCL-2 家族
        "BCL2", "BCL2L1", "BCL2L2", "BCLW",
        "BCL2A1", "MCL1",
        "BAX", "BAK1", "BOK",
        "BAD", "BID", "BIM", "BCL2L11",
        "PUMA", "BBC3", "NOXA", "PMAIP1",
        "BIK", "BMF", "HRK",
        # 胱天蛋白酶
        "CASP1", "CASP2", "CASP3", "CASP4", "CASP5",
        "CASP6", "CASP7", "CASP8", "CASP9", "CASP10", "CASP14",
        "APAF1", "DIABLO", "SMAC", "XIAP", "BIRC5", "SURVIVIN",
        "CYCS", "AIFM1", "ENDOG",
        # 程序性坏死
        "RIPK1", "RIPK3", "MLKL", "FADD", "TRADD",
        # 铁死亡
        "GPX4", "ACSL4", "LPCAT3",
        "SLC7A11", "SLC3A2",
        "TFRC", "FTH1", "FTL",
        "STEAP3", "NCOA4", "IREB2",
        "ALOX5", "ALOX12", "ALOX15",
        "HMOX1", "NFE2L2", "KEAP1",
        "SAT1", "SAT2", "SMS",
        "GCLC", "GCLM", "GSR", "GSS",
        "CHAC1", "SLC40A1", "CP", "TF",
        "ATF3", "ATF4", "DDIT3", "TRIB3", "SESN2",

        # ================================================================
        # 十一、代谢失调 (Metabolic Dysregulation)
        # ================================================================
        # 糖代谢
        "HK1", "HK2", "GCK",
        "GPI", "PFKL", "PFKM", "PFKP",
        "ALDOA", "ALDOB", "ALDOC",
        "GAPDH", "PGK1", "PGAM1", "ENO1", "ENO2",
        "PKM", "PKLR",
        "LDHA", "LDHB",
        "PDHA1", "PDHB", "PDK1", "PDK2", "PDK3", "PDK4",
        "CS", "ACO1", "ACO2",
        "IDH1", "IDH2", "IDH3A", "IDH3B", "IDH3G",
        "OGDH", "SUCLA2", "SUCLG1", "SUCLG2",
        "SDHA", "SDHB", "SDHC", "SDHD",
        "FH", "MDH1", "MDH2",
        "PC", "PCK1", "PCK2",
        "G6PC", "G6PC2", "G6PC3",
        "FBP1", "FBP2",
        # 脂代谢
        "PPARA", "PPARD", "PPARG",
        "SREBF1", "SREBF2", "SCAP",
        "ACACA", "ACACB",
        "FASN",
        "CPT1A", "CPT1B", "CPT1C", "CPT2",
        "ACADM", "ACADL", "ACADVL",
        "HADHA", "HADHB",
        "ACSL1", "ACSL3", "ACSL4", "ACSL5", "ACSL6",
        "LPIN1", "LPIN2", "LPIN3",
        "PNPLA2", "PLIN2", "PLIN3", "PLIN5",
        "LIPE", "MGLL",
        "FABP1", "FABP3", "FABP4", "FABP5", "FABP7",
        # 胆固醇代谢
        "HMGCR", "HMGCS1", "HMGCS2",
        "LDLR", "VLDLR", "LRP1", "LRP2",
        "ABCA1", "ABCG1",
        "APOA1", "APOB", "APOE", "CLU",
        "CYP27A1", "CYP46A1", "CYP7A1",
        "LPL", "LIPC", "LCAT",
        "SCARB1", "SR-BI",
        # 氨基酸/谷氨酰胺
        "GLS", "GLS2", "GLUL", "GLUD1",
        "GOT1", "GOT2", "GPT", "GPT2",
        "CBS", "CTH", "CDO1",
        "MTHFR", "MTR", "MTRR",
        "BHMT", "BHMT2",
        "ASNS", "ASS1", "ASL",
        "ARG1", "ARG2",
        # 酮体
        "HMGCL", "BDH1", "BDH2", "OXCT1", "ACAT1", "ACAT2",
        # 一碳代谢
        "SHMT1", "SHMT2",
        "MTHFD1", "MTHFD2", "MTHFD1L",
        "ATIC", "GART",
        "DHFR", "TYMS",

        # ================================================================
        # 十二、氧化应激防御 (Oxidative Stress Defense)
        # ================================================================
        "SOD1", "SOD2", "SOD3",
        "CAT",
        "GPX1", "GPX2", "GPX3", "GPX4", "GPX5", "GPX6", "GPX7", "GPX8",
        "PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5", "PRDX6",
        "TXN", "TXN2", "TXNRD1", "TXNRD2", "TXNRD3",
        "GSR", "GSS", "GCLC", "GCLM",
        "NFE2L2", "KEAP1", "BACH1",
        "HMOX1", "NQO1", "NQO2",
        "GSTA1", "GSTA4", "GSTM1", "GSTP1", "GSTT1",
        "MSRA", "MSRB1", "MSRB2", "MSRB3",
        "SRXN1", "SRX1",
        "SESN1", "SESN2", "SESN3",
        "MT1A", "MT1E", "MT1F", "MT1G", "MT1H", "MT1X", "MT2A", "MT3",
        "NOX1", "NOX2", "CYBB", "NOX3", "NOX4", "NOX5",
        "DUOX1", "DUOX2",
        "XDH", "XO",

        # ================================================================
        # 十三、细胞外基质硬化 (ECM Stiffening & Fibrosis)
        # ================================================================
        "COL1A1", "COL1A2", "COL2A1", "COL3A1",
        "COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6",
        "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
        "FN1", "FBN1", "ELN", "EMD",
        "VIM", "DES", "ACTG2", "ACTA2",
        "TNC", "TNN", "TNR", "TNXB",
        "LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4",
        "MMP1", "MMP2", "MMP3", "MMP7", "MMP8", "MMP9",
        "MMP10", "MMP12", "MMP13", "MMP14", "MMP15", "MMP16",
        "TIMP1", "TIMP2", "TIMP3", "TIMP4",
        "SERPINE1", "PLAT", "PLAU", "PLAUR",
        "TGFB1", "TGFB2", "TGFB3",
        "CTGF", "CCN2", "CYR61", "CCN1",
        "YAP1", "TAZ", "WWTR1",
        "ITGA1", "ITGA2", "ITGA5", "ITGA6", "ITGAV",
        "ITGB1", "ITGB3", "ITGB5", "ITGB6",
        "ROCK1", "ROCK2", "RHOA",

        # ================================================================
        # 十四、细胞周期与增殖 (Cell Cycle & Proliferation)
        # ================================================================
        "CDK1", "CDK2", "CDK4", "CDK6", "CDK7",
        "CCNA1", "CCNA2", "CCNB1", "CCNB2", "CCNB3",
        "CCND1", "CCND2", "CCND3",
        "CCNE1", "CCNE2",
        "CDC20", "CDC25A", "CDC25B", "CDC25C",
        "CDC6", "CDT1",
        "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "PLK1", "PLK2", "PLK3", "PLK4", "PLK5",
        "AURKA", "AURKB", "AURKC",
        "BUB1", "BUB1B", "BUB3", "MAD2L1", "MAD1L1",
        "MKI67", "PCNA", "TOP2A",
        "WEE1", "MYT1", "PKMYT1",
        "CKS1B", "CKS2",
        "SKP2", "FBXW7", "FBXO5",
        "CUL1", "CUL2", "CUL3", "CUL4A", "CUL4B", "CUL5", "CUL7",
        "APC", "CDC27", "CDC16", "ANAPC1",

        # ================================================================
        # 十五、MAPK / ERK / p38 / JNK 应激信号
        # ================================================================
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10",
        "MAPK11", "MAPK12", "MAPK13", "MAPK14",
        "MAP2K1", "MAP2K2", "MAP2K3", "MAP2K4", "MAP2K5",
        "MAP2K6", "MAP2K7",
        "MAP3K1", "MAP3K5", "MAP3K7", "MAP3K8", "MAP3K11",
        "RAF1", "BRAF", "ARAF",
        "HRAS", "KRAS", "NRAS",
        "DUSP1", "DUSP2", "DUSP4", "DUSP5", "DUSP6",
        "DUSP7", "DUSP8", "DUSP9", "DUSP10", "DUSP16",
        "JUN", "JUNB", "JUND",
        "FOS", "FOSB", "FOSL1", "FOSL2",
        "ATF2", "ATF3", "ATF4", "ATF6",
        "CREB1", "CREB5",
        "ELK1", "ELK3", "ELK4",
        "MAX", "MNT", "MXD1", "MXD3", "MXD4",

        # ================================================================
        # 十六、免疫衰老 (Immunosenescence)
        # ================================================================
        "CD28", "CD27", "CD57", "B3GAT1",
        "KLRG1", "LAG3", "PDCD1", "CTLA4", "TIGIT", "HAVCR2",
        "CD244", "CD160", "BTLA",
        "IL2", "IL2RA", "IL2RB", "IL2RG",
        "IL7", "IL7R",
        "IL15", "IL15RA",
        "CD3E", "CD3D", "CD3G", "CD247",
        "CD4", "CD8A", "CD8B",
        "GZMA", "GZMB", "GZMK", "PRF1",
        "IFNG", "TNF", "IL6",
        "CCR7", "CD45RA", "CD45RO", "PTPRC",
        "CD69", "CD25", "HLA-DRA", "HLA-DRB1",
        "CD19", "MS4A1", "CD20",
        "CD14", "CD16", "FCGR3A", "FCGR3B",
        "CD33", "SIGLEC3", "CD11B", "ITGAM",
        "NKG2D", "KLRK1", "NKG2A", "KLRC1",
        "MICB", "MICA", "ULBP1", "ULBP2", "ULBP3",
        "KLRC2", "KLRC3",
        "FOXP3", "TBX21", "GATA3", "RORC",

        # ================================================================
        # 十七、昼夜节律 (Circadian Rhythm & Aging)
        # ================================================================
        "CLOCK", "BMAL1", "ARNTL", "BMAL2", "ARNTL2",
        "PER1", "PER2", "PER3",
        "CRY1", "CRY2",
        "NR1D1", "REV-ERBA", "NR1D2", "REV-ERBB",
        "RORA", "RORB", "RORC",
        "NPAS2",
        "DBP", "TEF", "HLF",
        "CSNK1D", "CSNK1E",
        "FBXL3", "FBXW11", "BTRC",
        "SIRT1",

        # ================================================================
        # 十八、铁代谢 / 铜代谢 (Metal Homeostasis)
        # ================================================================
        "TFRC", "TF", "TFR2", "FTH1", "FTL",
        "SLC40A1", "FPN1", "HAMP", "HFE", "HJV", "HFE2",
        "STEAP1", "STEAP2", "STEAP3", "STEAP4",
        "SLC11A2", "DMT1",
        "SLC39A8", "ZIP8", "SLC39A14", "ZIP14",
        "SLC30A1", "SLC30A3", "SLC30A10",
        "ATP7A", "ATP7B", "CP", "CERULOPLASMIN",
        "SLC31A1", "CTR1", "SLC31A2", "CTR2",
        "COMMD1", "CCS", "SOD1",
        "MT1A", "MT2A", "MT3",
        "IREB2", "IRP2", "ACO1", "IRP1",
    ]
    # 去重
    aging_core_genes = list(dict.fromkeys(aging_core_genes))
    log.info("  使用文献 fallback Aging 基因列表 (%d 个基因)", len(aging_core_genes))
    return pd.DataFrame({"gene": aging_core_genes})


# ===================================================================
# 主处理流程
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="疾病-基因关联数据自动获取与整合脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python disease_gene_scraper.py                     # 默认运行
  python disease_gene_scraper.py --cache             # 使用缓存（跳过已下载）
  python disease_gene_scraper.py --force-download    # 强制重新下载
  python disease_gene_scraper.py --disgenet-key YOUR_API_KEY  # 提供 DisGeNET API key
  python disease_gene_scraper.py --use-fallback      # 直接使用 fallback 基因列表
        """,
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="使用缓存文件（默认行为）",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载所有数据（忽略缓存）",
    )
    parser.add_argument(
        "--disgenet-key",
        type=str,
        default=None,
        help="DisGeNET API key（注册地址: https://www.disgenet.org/）",
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="直接使用文献 fallback 基因列表，跳过网络请求",
    )
    parser.add_argument(
        "--core-gene-file",
        type=str,
        default=str(CORE_GENE_FILE),
        help=f"核心基因集文件路径（默认: {CORE_GENE_FILE}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"输出文件路径（默认: {OUTPUT_FILE}）",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    force = args.force_download
    cache = args.cache

    # 如果强制下载，清除缓存
    if force:
        log.info("强制下载模式：清除所有缓存文件")
        for f in CACHE_DIR.glob("*"):
            f.unlink()

    # ------------------------------------------------------------------
    # 读取核心基因集
    # ------------------------------------------------------------------
    core_gene_path = Path(args.core_gene_file)
    if not core_gene_path.exists():
        log.error("核心基因集文件不存在: %s", core_gene_path)
        log.error(
            "请先运行 L1 分析生成 %s，或通过 --core-gene-file 指定路径",
            CORE_GENE_FILE.name,
        )
        sys.exit(1)

    core_df = pd.read_csv(core_gene_path)
    if "gene" not in core_df.columns:
        log.error("核心基因集文件缺少 'gene' 列，可用列: %s", core_df.columns.tolist())
        sys.exit(1)

    core_genes = set(core_df["gene"].dropna().astype(str).str.strip().unique())
    log.info("核心基因集: %d 个唯一基因", len(core_genes))

    session = create_session()

    # ------------------------------------------------------------------
    # Fallback 模式
    # ------------------------------------------------------------------
    if args.use_fallback:
        log.info("使用 fallback 基因列表模式")
        ad_genes_raw = set(build_ad_fallback_genes()["gene"].tolist())
        aging_genes_raw = set(build_aging_fallback_genes()["gene"].tolist())
        log.info("AD fallback 基因: %d", len(ad_genes_raw))
        log.info("Aging fallback 基因: %d", len(aging_genes_raw))
    else:
        # ------------------------------------------------------------------
        # 数据采集: AD 基因
        # ------------------------------------------------------------------
        log.info("=" * 60)
        log.info("第一步: 获取阿尔茨海默病（AD）关联基因")
        log.info("=" * 60)

        ad_genes_all = []

        # 1. AlzGene
        alzgene_cache = CACHE_DIR / "alzgene_manual.csv"
        if not cache or force or not alzgene_cache.exists():
            alzgene_df = fetch_alzgene(session, alzgene_cache)
        else:
            if alzgene_cache.exists():
                alzgene_df = pd.read_csv(alzgene_cache)
                log.info("  AlzGene 缓存命中: %s", alzgene_cache)
            else:
                alzgene_df = fetch_alzgene(session, alzgene_cache)
        if not alzgene_df.empty:
            ad_genes_all.extend(alzgene_df["gene"].tolist())

        # 2. DisGeNET (AD)
        disgenet_ad_cache = CACHE_DIR / "disgenet_ad.csv"
        if not cache or force or not disgenet_ad_cache.exists():
            disgenet_ad_df = fetch_disgenet(
                session, "Alzheimer's Disease", disgenet_ad_cache, args.disgenet_key
            )
        else:
            disgenet_ad_df = pd.read_csv(disgenet_ad_cache)
            log.info("  DisGeNET (AD) 缓存命中: %s", disgenet_ad_cache)
        if not disgenet_ad_df.empty and "gene" in disgenet_ad_df.columns:
            ad_genes_all.extend(disgenet_ad_df["gene"].tolist())

        # 3. 若 AlzGene 和 DisGeNET 均无数据，使用 fallback
        if len(ad_genes_all) == 0:
            log.warning("  AlzGene 和 DisGeNET 均无 AD 数据，启用 fallback AD 基因列表")
            ad_genes_all = build_ad_fallback_genes()["gene"].tolist()

        ad_genes_raw = set(g.strip().upper() for g in ad_genes_all if g and str(g).strip())
        log.info("AD 原始基因数（合并后去重）: %d", len(ad_genes_raw))

        # ------------------------------------------------------------------
        # 数据采集: Aging 基因
        # ------------------------------------------------------------------
        log.info("=" * 60)
        log.info("第二步: 获取衰老（Aging）关联基因")
        log.info("=" * 60)

        aging_genes_all = []

        # 1. GenAge
        genage_cache = CACHE_DIR / "genage_human.csv"
        if not cache or force or not genage_cache.exists():
            genage_df = fetch_genage(session, genage_cache)
        else:
            if genage_cache.exists():
                genage_df = pd.read_csv(genage_cache)
                log.info("  GenAge 缓存命中: %s", genage_cache)
            else:
                genage_df = fetch_genage(session, genage_cache)
        if not genage_df.empty and "gene" in genage_df.columns:
            aging_genes_all.extend(genage_df["gene"].tolist())

        # 2. CellAge
        cellage_cache = CACHE_DIR / "cellage.csv"
        if not cache or force or not cellage_cache.exists():
            cellage_df = fetch_cellage(session, cellage_cache)
        else:
            if cellage_cache.exists():
                cellage_df = pd.read_csv(cellage_cache)
                log.info("  CellAge 缓存命中: %s", cellage_cache)
            else:
                cellage_df = fetch_cellage(session, cellage_cache)
        if not cellage_df.empty and "gene" in cellage_df.columns:
            aging_genes_all.extend(cellage_df["gene"].tolist())

        # 3. DisGeNET (Aging)
        disgenet_aging_cache = CACHE_DIR / "disgenet_aging.csv"
        if not cache or force or not disgenet_aging_cache.exists():
            disgenet_aging_df = fetch_disgenet(
                session, "Aging", disgenet_aging_cache, args.disgenet_key
            )
        else:
            disgenet_aging_df = pd.read_csv(disgenet_aging_cache)
            log.info("  DisGeNET (Aging) 缓存命中: %s", disgenet_aging_cache)
        if not disgenet_aging_df.empty and "gene" in disgenet_aging_df.columns:
            aging_genes_all.extend(disgenet_aging_df["gene"].tolist())

        # 4. 若 GenAge, CellAge 和 DisGeNET 均无数据，使用 fallback
        if len(aging_genes_all) == 0:
            log.warning("  GenAge, CellAge 和 DisGeNET 均无 Aging 数据，启用 fallback Aging 基因列表")
            aging_genes_all = build_aging_fallback_genes()["gene"].tolist()

        aging_genes_raw = set(
            g.strip().upper() for g in aging_genes_all if g and str(g).strip()
        )
        log.info("Aging 原始基因数（合并后去重）: %d", len(aging_genes_raw))

    # ------------------------------------------------------------------
    # 交集与去重
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("第三步: 与核心基因集取交集并去重")
    log.info("=" * 60)

    # 核心基因也转大写进行匹配
    core_genes_upper = set(g.upper() for g in core_genes)

    ad_genes_in_network = ad_genes_raw & core_genes_upper
    aging_genes_in_network = aging_genes_raw & core_genes_upper

    log.info("AD:   原始 %d → 交集后 %d (%.1f%%)",
             len(ad_genes_raw), len(ad_genes_in_network),
             100 * len(ad_genes_in_network) / max(len(ad_genes_raw), 1))
    log.info("Aging: 原始 %d → 交集后 %d (%.1f%%)",
             len(aging_genes_raw), len(aging_genes_in_network),
             100 * len(aging_genes_in_network) / max(len(aging_genes_raw), 1))

    # ------------------------------------------------------------------
    # 构建输出表格
    # ------------------------------------------------------------------
    rows = []
    for gene in sorted(ad_genes_in_network):
        # 保持原始核心基因集的大小写
        rows.append({"disease": "AD", "gene": gene})
    for gene in sorted(aging_genes_in_network):
        rows.append({"disease": "Aging", "gene": gene})

    output_df = pd.DataFrame(rows)

    # 按 (disease, gene) 去重
    output_df = output_df.drop_duplicates(subset=["disease", "gene"])

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_df.to_csv(output_path, index=False)
    log.info("=" * 60)
    log.info("输出文件: %s", output_path)
    log.info("总关联数: %d (AD: %d, Aging: %d)",
             len(output_df),
             len(output_df[output_df["disease"] == "AD"]),
             len(output_df[output_df["disease"] == "Aging"]))
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 展示前10行预览
    # ------------------------------------------------------------------
    print("\n--- 输出预览 (前10行) ---")
    print(output_df.head(10).to_string(index=False))
    print(f"\n共 {len(output_df)} 条 disease-gene 关联记录。\n")


if __name__ == "__main__":
    main()
