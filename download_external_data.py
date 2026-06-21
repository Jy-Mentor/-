#!/usr/bin/env python3
"""
download_external_data.py — 从外部权威数据库和GitHub开源项目下载生物数据
===============================================================
替换 module3_hgt.py 中所有硬编码数据，全部数据来源可追溯。
遵循 FAIR (Findable, Accessible, Interoperable, Reusable) 数据原则。

数据源与GitHub仓库:
1. MSigDB (Broad Institute) — 基因通路映射 (KEGG/Reactome/Hallmark/WikiPathways/GO)
   文献: Subramanian et al., PNAS 2005; Liberzon et al., Cell Systems 2015
   GitHub: https://github.com/GSEA-MSigDB/msigdb
   网站: https://www.gsea-msigdb.org/gsea/msigdb/

2. PanglaoDB (Franzén et al., Database 2019) — 细胞类型标记基因
   文献: Franzén et al., Database 2019, PMID: 30929243
   GitHub: https://github.com/oscar-franzen/PanglaoDB
   网站: https://panglaodb.se/

3. CellChatDB (Jin et al., Nature Communications 2021) — 配体-受体对
   文献: Jin et al., Nature Communications 2021, PMID: 33597528
   GitHub: https://github.com/sqjin/CellChat
   数据: inst/extdata/interaction_input_CellChatDB.csv

4. PubChem PUG REST API (Kim et al., NAR 2021) — 化合物理化性质
   文献: Kim et al., NAR 2021, PMID: 33137181
   文档: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
   GitHub: https://github.com/ncbi/PubChem (NCBI官方)

5. STITCH (Szklarczyk et al., NAR 2021) — 化合物-蛋白互作
   文献: Szklarczyk et al., NAR 2021, PMID: 33125081
   网站: http://stitch.embl.de/
   物种: 10090 (Mus musculus)

6. DisGeNET (Piñero et al., NAR 2020) — 疾病-基因关联
   文献: Piñero et al., NAR 2020, PMID: 31680162
   网站: https://www.disgenet.org/
   GitHub: https://github.com/DisGeNET/DisGeNET-SQLite

7. STRING PPI (Szklarczyk et al., NAR 2021) — 蛋白-蛋白互作
   文献: Szklarczyk et al., NAR 2021, PMID: 33237311
   API: https://string-db.org/api/
   物种: 10090 (Mus musculus)

8. TRRUST v2 (Han et al., NAR 2018) — 转录因子-靶基因调控
   文献: Han et al., NAR 2018, PMID: 29087512
   网站: https://www.grnpedia.org/trrust/

9. FerrDb V2 (Zhou & Bao, NAR 2023) — 铁死亡调控因子数据库
   文献: Zhou & Bao, NAR 2023, PMID: 36305826
   网站: http://www.zhounan.org/ferrdb/

10. CellAge (Avelar et al., Genome Biology 2020) — 细胞衰老基因数据库
    文献: Avelar et al., Genome Biology 2020, PMID: 32264951
    网站: https://genomics.senescence.info/cells/

11. SenMayo (Saul et al., Nature Communications 2022) — 衰老基因集
    文献: Saul et al., Nature Communications 2022, PMID: 35999225
    GitHub: https://github.com/JuliaSaul/SenMayo

12. mygene.info (Wu et al., NAR 2013) — 基因注释聚合
    文献: Wu et al., NAR 2013, PMID: 23175614
    API: https://mygene.info/v3/
    GitHub: https://github.com/biothings/mygene.info

13. KEGG (Kanehisa et al., NAR 2021) — 京都基因与基因组百科全书
    文献: Kanehisa et al., NAR 2021, PMID: 33125081
    API: https://rest.kegg.jp/

14. Reactome (Jassal et al., NAR 2020) — 生物学通路数据库
    文献: Jassal et al., NAR 2020, PMID: 31691815
    GitHub: https://github.com/reactome

输出文件 (写入 network_files/ 目录):
- msigdb_gene_pathways.csv     — 基因→通路映射 (来自 MSigDB)
- panglaodb_celltype_markers.csv — 细胞类型→标记基因 (来自 PanglaoDB)
- cellchat_lr_pairs.csv        — 配体-受体配对 (来自 CellChatDB)
- pubchem_compound_props.csv   — 化合物理化性质 (来自 PubChem)
- stitch_compound_targets.csv  — 化合物-靶点互作 (来自 STITCH)
- disgenet_disease_genes.csv   — 疾病-基因关联 (来自 DisGeNET)
- string_ppi_edges.csv         — 蛋白-蛋白互作 (来自 STRING API)
- trrust_tf_target.csv         — TF-靶基因调控 (来自 TRRUST v2)
- gene_pathway_enrichment_external.csv — 基因通路映射 (MSigDB标准化)
"""

import csv
import logging
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "network_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 配置加载器: 从 external_db_config.yaml 加载所有查询参数 (替代硬编码)
# =============================================================================
def _load_db_config() -> dict:
    """从 external_db_config.yaml 加载外部数据库查询配置

    替代所有硬编码的化合物CID、疾病关键词、KEGG通路ID、细胞类型列表。
    配置文件: network_files/external_db_config.yaml
    """
    config_file = OUT_DIR / "external_db_config.yaml"
    config = {
        'compounds': {},
        'diseases': {},
        'kegg_pathways': {},
        'target_cell_types': [],
    }
    if not config_file.exists():
        logger.warning(f"外部数据库配置文件不存在: {config_file}, 使用内置默认值")
        return _get_default_db_config()
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"外部数据库配置加载: {config_file.name}")
    except ImportError:
        logger.warning("PyYAML 未安装, 使用内置默认值")
        return _get_default_db_config()
    except Exception as e:
        traceback.print_exc()
        logger.error("外部数据库配置加载失败: %s，使用内置默认值", e)
        return _get_default_db_config()
    return config


def _get_default_db_config() -> dict:
    """内置默认配置 (仅作为回退, 所有值均可从外部数据库公开查询验证)"""
    return {
        'compounds': {
            'BCP': 5281515, 'VC': 54670067, 'Fer-1': 4063092,
            'DFO': 2973, 'Lip-1': 42608917, 'Erastin': 11294402,
            'RSL3': 25195377, 'ML162': 609375,
        },
        'diseases': {
            'CIRI': {'keywords': ['cerebral ischemia', 'stroke', 'brain ischemia',
                                   'cerebral infarction', 'ischemic stroke']},
            'AD': {'keywords': ['alzheimer', 'alzheimer disease', "alzheimer's disease"]},
            'Aging': {'keywords': ['aging', 'ageing', 'senescence', 'cellular senescence']},
        },
        'kegg_pathways': {
            'hsa04216': 'Ferroptosis', 'hsa00480': 'Glutathione_metabolism',
            'hsa04140': 'Autophagy', 'hsa04210': 'Apoptosis',
            'hsa04150': 'mTOR_signaling', 'hsa04151': 'PI3K_Akt_pathway',
            'hsa04010': 'MAPK_signaling', 'hsa04630': 'JAK_STAT_pathway',
            'hsa04064': 'NF-kB_signaling', 'hsa04621': 'NLRP3_inflammasome',
            'hsa04115': 'p53_pathway', 'hsa04066': 'HIF1_signaling',
            'hsa00640': 'Propanoate_metabolism', 'hsa00010': 'Glycolysis',
            'hsa00020': 'TCA_cycle', 'hsa01212': 'Fatty_acid_metabolism',
            'hsa00590': 'Arachidonic_acid_metabolism', 'hsa00190': 'Oxidative_phosphorylation',
            'hsa04310': 'Wnt_signaling', 'hsa04330': 'Notch_signaling',
            'hsa04340': 'Hedgehog_signaling', 'hsa04350': 'TGF-beta_signaling',
            'hsa04370': 'VEGF_signaling', 'hsa04020': 'Calcium_signaling',
            'hsa04024': 'cAMP_signaling', 'hsa04022': 'cGMP_PKG_pathway',
            'hsa04015': 'Rap1_signaling', 'hsa04014': 'Ras_signaling',
            'hsa04910': 'Insulin_signaling', 'hsa04920': 'Adipocytokine',
            'hsa04710': 'Circadian_rhythm', 'hsa04720': 'Long_term_potentiation',
            'hsa04728': 'Dopaminergic_synapse', 'hsa04724': 'Glutamatergic_synapse',
            'hsa04727': 'GABAergic_synapse', 'hsa04726': 'Serotonergic_synapse',
            'hsa04725': 'Cholinergic_synapse', 'hsa04360': 'Axon_guidance',
            'hsa04510': 'Focal_adhesion', 'hsa04512': 'ECM_receptor',
            'hsa04514': 'Cell_adhesion', 'hsa04530': 'Tight_junction',
            'hsa04540': 'Gap_junction', 'hsa04610': 'Complement_cascade',
            'hsa04612': 'Antigen_processing', 'hsa04640': 'Hematopoietic_cell_lineage',
            'hsa04144': 'Endocytosis', 'hsa04145': 'Phagosome',
            'hsa04142': 'Lysosome', 'hsa04146': 'Peroxisome',
            'hsa03010': 'Ribosome', 'hsa03040': 'Spliceosome',
            'hsa03013': 'RNA_transport', 'hsa02010': 'ABC_transporters',
            'hsa00980': 'Drug_metabolism_cytochrome_P450', 'hsa00982': 'Xenobiotic_metabolism',
            'hsa00100': 'Steroid_biosynthesis', 'hsa00230': 'Nucleotide_metabolism',
            'hsa00250': 'Amino_acid_metabolism', 'hsa01200': 'Carbon_metabolism',
            'hsa03015': 'mRNA_surveillance', 'hsa03060': 'Protein_export',
            'hsa04141': 'Protein_processing_ER', 'hsa04120': 'Ubiquitin_proteasome',
            'hsa04650': 'Natural_killer_cytotoxicity', 'hsa04660': 'T_cell_receptor',
            'hsa04662': 'B_cell_receptor', 'hsa04666': 'Fc_gamma_phagocytosis',
            'hsa04670': 'Leukocyte_transendothelial_migration', 'hsa04611': 'Platelet_activation',
        },
        'target_cell_types': [
            'Neuron', 'Neurons', 'Neuronal stem cells', 'Neural stem cells',
            'Microglia', 'Astrocyte', 'Astrocytes', 'Oligodendrocyte', 'Oligodendrocytes',
            'Endothelial cells', 'Endothelial', 'Pericytes', 'Pericyte',
            'Radial glia', 'Bergmann glia', 'Ependymal cells',
            'Interneurons', 'Pyramidal neurons', 'GABAergic neurons',
            'Glutamatergic neurons', 'Dopaminergic neurons', 'Cholinergic neurons',
            'Granule cells', 'Purkinje cells',
        ],
    }

# =============================================================================
# 辅助函数
# =============================================================================

def _safe_request(url, method='GET', timeout=60, **kwargs):
    """带重试的HTTP请求"""
    import traceback

    import requests

    # 默认添加浏览器 User-Agent, 避免部分站点(如 PanglaoDB)返回 403
    headers = kwargs.pop('headers', {})
    headers.setdefault(
        'User-Agent',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )

    for attempt in range(3):
        try:
            if method == 'GET':
                resp = requests.get(url, headers=headers, timeout=timeout, **kwargs)
            else:
                resp = requests.post(url, headers=headers, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception:
            traceback.print_exc()
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def _download_github_file_bytes(owner: str, repo: str, path: str, ref: str | None = None) -> bytes:
    """通过 GitHub Contents API 下载文件并返回原始 bytes

    用于替代 raw.githubusercontent.com, 后者在中国大陆网络环境下
    经常出现连接重置/超时(错误 10054)。GitHub Contents API 通过
    api.github.com 返回 base64 编码内容, 稳定性更好。

    Args:
        owner: 仓库所有者
        repo: 仓库名
        path: 文件路径
        ref: 分支/tag/commit SHA, 可选

    Returns:
        文件原始二进制内容
    """
    import base64
    import traceback

    import requests

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        'User-Agent': 'download_external_data.py',
        'Accept': 'application/vnd.github.v3+json',
    }
    params = {'ref': ref} if ref else {}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            if 'content' not in payload:
                raise ValueError(f"GitHub API 响应缺少 content: {list(payload.keys())}")
            return base64.b64decode(payload['content'])
        except Exception:
            traceback.print_exc()
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


# =============================================================================
# 1. MSigDB 基因通路映射
# =============================================================================
# 文献: Subramanian et al., PNAS 2005; Liberzon et al., Cell Systems 2015
# 数据: https://www.gsea-msigdb.org/gsea/msigdb/
# 使用: Mouse gene sets (gene symbols), KEGG + Reactome + Hallmark

# MSigDB 2023.1 Mouse release 公开 GMT 下载链接
# 原 download_geneset.jsp 接口已要求登录, 现改用 Broad 公开静态文件
# 来源: https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.1.Mm/
MSIGDB_RELEASE = '2023.1.Mm'
MSIGDB_BASE_URL = (
    'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/'
    f'{MSIGDB_RELEASE}'
)
MSIGDB_MOUSE_GENESETS = {
    # Hallmark gene sets (50个, 概括生物学状态)
    'hallmark': f'{MSIGDB_BASE_URL}/mh.all.v{MSIGDB_RELEASE}.symbols.gmt',
    # Reactome pathways
    'reactome': f'{MSIGDB_BASE_URL}/m2.cp.reactome.v{MSIGDB_RELEASE}.symbols.gmt',
    # WikiPathways
    'wikipathways': f'{MSIGDB_BASE_URL}/m2.cp.wikipathways.v{MSIGDB_RELEASE}.symbols.gmt',
    # GO Biological Process
    'go_bp': f'{MSIGDB_BASE_URL}/m5.go.bp.v{MSIGDB_RELEASE}.symbols.gmt',
    # KEGG 单独文件在 2023.1.Mm 中不再提供, 由 download_kegg_pathway_genes() 通过 KEGG REST API 补充
}

def download_msigdb_gene_pathways():
    """从 MSigDB 下载小鼠基因通路映射"""
    out_file = OUT_DIR / "msigdb_gene_pathways.csv"
    if out_file.exists():
        logger.info(f"  MSigDB 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 MSigDB 基因通路映射 ===")
    rows = []
    for category, url in MSIGDB_MOUSE_GENESETS.items():
        try:
            logger.info(f"  下载 {category}...")
            resp = _safe_request(url)
            # GMT 格式: 每行 \t 分隔, 第一列通路名, 第二列URL, 后续列为基因
            for line in resp.text.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                pathway_name = parts[0].strip()
                genes = [g.strip() for g in parts[2:] if g.strip()]
                for gene in genes:
                    rows.append({
                        'pathway': pathway_name,
                        'gene': gene.upper(),
                        'source': f'MSigDB_{category}',
                    })
            logger.info(f"    {category}: {len(set(r['pathway'] for r in rows[-len(genes):])) if rows else 0} 通路")
        except Exception as e:
            logger.warning(f"  {category} 下载失败: {e}")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['pathway', 'gene', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  MSigDB 保存: {out_file} ({len(rows)} 行, {len(set(r['pathway'] for r in rows))} 通路)")
    else:
        logger.warning("  MSigDB 下载无数据!")
    return out_file


# =============================================================================
# 2. PanglaoDB 细胞类型标记基因
# =============================================================================
# 文献: Franzén et al., Database 2019; https://panglaodb.se/
# GitHub: https://github.com/oscar-franzen/PanglaoDB

PANGLAODB_MARKER_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"

def download_panglaodb_markers():
    """从 PanglaoDB 下载细胞类型标记基因

    细胞类型列表: 从 external_db_config.yaml 加载 (替代硬编码)
    文献: Franzén et al., Database 2019, PMID: 30929243
    """
    import gzip
    import io

    out_file = OUT_DIR / "panglaodb_celltype_markers.csv"
    if out_file.exists():
        logger.info(f"  PanglaoDB 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 PanglaoDB 细胞类型标记基因 ===")
    # 从配置文件加载目标细胞类型 (替代硬编码列表)
    db_config = _load_db_config()
    target_cell_types = db_config.get('target_cell_types', [])

    rows = []
    try:
        resp = _safe_request(PANGLAODB_MARKER_URL)
        with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
            content = gz.read().decode('utf-8')
            lines = content.strip().split('\n')
            headers = lines[0].split('\t')
            # 查找列索引
            species_col = None
            celltype_col = None
            gene_col = None
            for i, h in enumerate(headers):
                hl = h.lower().strip()
                if 'species' in hl:
                    species_col = i
                elif 'cell type' in hl or 'cell_type' in hl:
                    celltype_col = i
                elif 'gene' in hl or 'symbol' in hl:
                    gene_col = i

            if species_col is None or celltype_col is None or gene_col is None:
                logger.warning(f"  PanglaoDB 列名不匹配: {headers}")
                # 尝试默认列映射
                species_col = 0
                celltype_col = 5
                gene_col = 1

            for line in lines[1:]:
                parts = line.split('\t')
                if len(parts) <= max(species_col, celltype_col, gene_col):
                    continue
                species = parts[species_col].strip()
                cell_type = parts[celltype_col].strip()
                gene = parts[gene_col].strip()
                if 'Mus musculus' in species or 'Mm' in species:
                    # 只保留中枢神经系统相关细胞类型
                    for tc in target_cell_types:
                        if tc.lower() in cell_type.lower():
                            rows.append({
                                'celltype': cell_type,
                                'gene': gene.upper(),
                                'species': 'Mus musculus',
                                'source': 'PanglaoDB',
                            })
                            break
        logger.info(f"  PanglaoDB: {len(rows)} 条标记基因关联")
    except Exception as e:
        logger.warning(f"  PanglaoDB 下载失败: {e}")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['celltype', 'gene', 'species', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  PanglaoDB 保存: {out_file}")
    else:
        logger.warning("  PanglaoDB 无数据, 使用已有的 network_files/celltype_marker_genes.csv")
    return out_file


# =============================================================================
# 3. CellChatDB 配体-受体对
# =============================================================================
# 文献: Jin et al., Nature Communications 2021
# GitHub: https://github.com/sqjin/CellChat (v1 已归档, 数据以 .rda 形式存在)
# 数据位置: data/CellChatDB.mouse.rda (v1 归档 commit)
# 解析方式: 通过 GitHub Contents API 获取 base64 二进制, 再用 rdata 解析

# sqjin/CellChat v1 最后一个包含 CellChatDB.mouse.rda 的 commit
CELLCHAT_V1_COMMIT = "5d3ac89bb06c071802ba4e231e5ff0068adff2b2"


def download_cellchat_lr_pairs():
    """从 CellChatDB GitHub 归档仓库下载小鼠配体-受体对

    CellChat v1 仓库中 LR 数据以 R .rda 格式存储, 不再提供 CSV。
    本函数通过 GitHub Contents API 获取该文件 base64 内容,
    使用 rdata 解析后提取 interaction 数据框的 ligand/receptor/pathway_name。
    """
    out_file = OUT_DIR / "cellchat_lr_pairs.csv"
    if out_file.exists():
        logger.info(f"  CellChatDB 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 CellChatDB 配体-受体对 ===")
    logger.info(f"  来源: sqjin/CellChat @ {CELLCHAT_V1_COMMIT}")

    rows = []
    try:
        import io

        import rdata

        raw_bytes = _download_github_file_bytes(
            'sqjin', 'CellChat', 'data/CellChatDB.mouse.rda', ref=CELLCHAT_V1_COMMIT
        )
        logger.info(f"  下载 .rda 文件: {len(raw_bytes)} bytes")

        parsed = rdata.read_rda(io.BytesIO(raw_bytes))
        cellchat_db = parsed.get('CellChatDB.mouse')
        if cellchat_db is None:
            raise KeyError("CellChatDB.mouse 对象不存在于 .rda 文件")

        interaction = cellchat_db.get('interaction')
        if interaction is None:
            raise KeyError("CellChatDB.mouse$interaction 不存在")

        required_cols = {'ligand', 'receptor', 'pathway_name'}
        missing_cols = required_cols - set(interaction.columns)
        if missing_cols:
            raise KeyError(f"interaction 数据框缺少列: {missing_cols}")

        for _, row in interaction.iterrows():
            ligand = str(row['ligand']).strip()
            receptor = str(row['receptor']).strip()
            pathway = str(row['pathway_name']).strip()
            if ligand and receptor:
                rows.append({
                    'ligand': ligand.upper(),
                    'receptor': receptor.upper(),
                    'pathway': pathway,
                    'source': 'CellChatDB',
                })
        logger.info(f"  CellChatDB: {len(rows)} 对LR")
    except Exception:
        import traceback
        traceback.print_exc()
        logger.warning("  CellChatDB 下载失败, 详见上方异常")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ligand', 'receptor', 'pathway', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  CellChatDB 保存: {out_file}")
    return out_file


# =============================================================================
# 4. PubChem 化合物理化性质
# =============================================================================
# 文献: Kim et al., NAR 2021; https://pubchem.ncbi.nlm.nih.gov/
# API: PUG REST (https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
# 无需API密钥, 免费公开

def download_pubchem_compound_props():
    """从 PubChem PUG REST API 获取化合物理化性质

    化合物CID列表: 从 external_db_config.yaml 加载 (替代硬编码)
    文献: Kim et al., NAR 2021, PMID: 33137181
    API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
    """
    out_file = OUT_DIR / "pubchem_compound_props.csv"
    if out_file.exists():
        logger.info(f"  PubChem 化合物属性 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 PubChem 化合物理化性质 ===")

    # 从配置文件加载化合物CID (替代硬编码)
    db_config = _load_db_config()
    compound_cids = db_config.get('compounds', {})

    # 请求属性列表
    properties = [
        'MolecularWeight', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount',
        'TPSA', 'RotatableBondCount', 'CanonicalSMILES', 'IUPACName',
        'MolecularFormula', 'ExactMass',
    ]

    rows = []
    for name, cid in compound_cids.items():
        try:
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/"
                f"property/{','.join(properties)}/JSON"
            )
            resp = _safe_request(url)
            data = resp.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            rows.append({
                'compound': name,
                'cid': cid,
                'MW': props.get('MolecularWeight', ''),
                'LogP': props.get('XLogP', ''),
                'HBD': props.get('HBondDonorCount', ''),
                'HBA': props.get('HBondAcceptorCount', ''),
                'TPSA': props.get('TPSA', ''),
                'RotB': props.get('RotatableBondCount', ''),
                'SMILES': props.get('CanonicalSMILES', ''),
                'IUPACName': props.get('IUPACName', ''),
                'MolecularFormula': props.get('MolecularFormula', ''),
                'source': 'PubChem_PUG_REST',
            })
            logger.info(f"  {name} (CID:{cid}): MW={props.get('MolecularWeight', 'N/A')}")
        except Exception as e:
            logger.warning(f"  {name} (CID:{cid}) 获取失败: {e}")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'compound', 'cid', 'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotB',
                'SMILES', 'IUPACName', 'MolecularFormula', 'source',
            ])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  PubChem 保存: {out_file}")
    return out_file


# =============================================================================
# 5. STITCH 化合物-靶点互作
# =============================================================================
# 文献: Szklarczyk et al., NAR 2021; Kuhn et al., NAR 2014
# 数据: http://stitch.embl.de/download/ (小鼠, chemical-protein links)
# 物种: 10090 (Mus musculus)
# 注意: STITCH/STRING 下载服务器对自动化请求限制较严, 经常返回 403/404,
#       且官方 bulk 下载链接版本会变化。本函数保留尝试, 失败时仅记录警告,
#       不阻塞后续 DisGeNET/STRING API 等数据源。

STITCH_DOWNLOAD_URL = (
    "http://stitch.embl.de/download/"
    "chemical_protein_links.v5.0/10090.protein_chemical.links.detailed.v5.0.tsv.gz"
)

STITCH_CHEMICAL_MAP_URL = (
    "http://stitch.embl.de/download/"
    "chemicals.v5.0.tsv.gz"
)

def download_stitch_compound_targets():
    """从 STITCH 下载化合物-蛋白互作数据

    化合物CID列表: 从 external_db_config.yaml 加载 (替代硬编码)
    文献: Szklarczyk et al., NAR 2021, PMID: 33125081
    """
    import gzip
    import io
    out_file = OUT_DIR / "stitch_compound_targets.csv"
    if out_file.exists():
        logger.info(f"  STITCH 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 STITCH 化合物-靶点互作 ===")

    # 从配置文件加载化合物CID (替代硬编码)
    db_config = _load_db_config()
    compound_cids = db_config.get('compounds', {})

    rows = []
    try:
        # 1. 下载 chemical-protein links
        logger.info("  下载 STITCH chemical-protein links...")
        resp = _safe_request(STITCH_DOWNLOAD_URL)
        with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
            content = gz.read().decode('utf-8')
            lines = content.strip().split('\n')
            # 解析 header
            lines[0].strip().split('\t')
            # expected: chemical, protein, experimental, ...
            cid_to_stitch = {}
            for name, pubchem_cid in compound_cids.items():
                cid_to_stitch[f"CID{str(pubchem_cid).zfill(9)}"] = name

            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                chemical = parts[0].strip()
                protein = parts[1].strip()
                combined_score = int(parts[-1]) if parts[-1].isdigit() else 0
                if chemical in cid_to_stitch and combined_score >= 400:
                    # STITCH蛋白ID格式: 10090.ENSMUSPxxxxxxxx
                    # 需要映射到基因符号
                    gene_id = protein.split('.')[-1] if '.' in protein else protein
                    rows.append({
                        'compound': cid_to_stitch[chemical],
                        'gene': gene_id,
                        'score': combined_score,
                        'source': 'STITCH_v5',
                    })
        logger.info(f"  STITCH: {len(rows)} 条化合物-靶点关联 (score >= 400)")
    except Exception:
        import traceback
        traceback.print_exc()
        logger.warning("  STITCH 下载失败, 详见上方异常 (服务器可能限制自动化下载)")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['compound', 'gene', 'score', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  STITCH 保存: {out_file}")
    return out_file


# =============================================================================
# 6. DisGeNET 疾病-基因关联
# =============================================================================
# 文献: Piñero et al., NAR 2020; https://www.disgenet.org/
# 公开数据: 可使用 curated gene-disease associations 数据集

# DisGeNET curated gene-disease associations (GitHub 开源镜像, 无需注册)
# 原始仓库: https://github.com/dhimmel/disgenet
# 数据版本: DisGeNET v3.0 (May 2015), Open Database License
# 下载方式: 使用 GitHub Contents API (见 _download_github_file_bytes),
#          替代 raw.githubusercontent.com, 避免网络连接重置

def download_disgenet_disease_genes():
    """从 DisGeNET GitHub 开源镜像下载疾病-基因关联 (公开 curated 数据集)

    疾病关键词列表: 从 external_db_config.yaml 加载 (替代硬编码)
    数据来源: https://github.com/dhimmel/disgenet (DisGeNET v3.0, May 2015)
    许可证: Open Database License
    文献: Piñero et al., NAR 2020, PMID: 31680162
    """
    import gzip
    import io
    out_file = OUT_DIR / "disgenet_disease_genes.csv"
    if out_file.exists():
        logger.info(f"  DisGeNET 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 DisGeNET 疾病-基因关联 (GitHub 镜像) ===")

    # 从配置文件加载疾病关键词 (替代硬编码)
    db_config = _load_db_config()
    target_diseases_config = db_config.get('diseases', {})

    disease_hits = {d: set() for d in target_diseases_config}
    rows = []
    try:
        # 使用 GitHub Contents API 替代 raw.githubusercontent.com,
        # 避免在中国大陆网络下连接重置/超时
        raw_bytes = _download_github_file_bytes(
            'dhimmel', 'disgenet', 'download/curated_gene_disease_associations.txt.gz'
        )
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
            content = gz.read().decode('utf-8')
            lines = content.strip().split('\n')
            header = lines[0].strip().split('\t')
            # 查找列索引
            gene_col = disease_col = disease_name_col = score_col = None
            for i, h in enumerate(header):
                hl = h.lower().strip()
                if 'genesymbol' in hl or 'gene_symbol' in hl:
                    gene_col = i
                elif 'diseaseid' in hl or 'disease_id' in hl:
                    disease_col = i
                elif 'diseasename' in hl or 'disease_name' in hl or 'diseasetype' in hl:
                    disease_name_col = i
                elif 'score' in hl:
                    score_col = i

            if gene_col is None or (disease_col is None and disease_name_col is None):
                logger.warning(f"  DisGeNET 列名不匹配: {header}")
                # 默认映射
                gene_col = 1
                disease_col = 3
                disease_name_col = 4
                score_col = 5

            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) <= max(gene_col, disease_col or 0, disease_name_col or 0):
                    continue
                gene = parts[gene_col].strip().upper()
                disease_name = parts[disease_name_col].strip().lower() if disease_name_col is not None else ''
                disease_id = parts[disease_col].strip() if disease_col is not None else ''
                try:
                    score = float(parts[score_col]) if score_col is not None and score_col < len(parts) else 0.0
                except ValueError:
                    score = 0.0

                for target_disease, info in target_diseases_config.items():
                    for kw in info.get('keywords', []):
                        if kw.lower() in disease_name:
                            if gene not in disease_hits[target_disease]:
                                disease_hits[target_disease].add(gene)
                                rows.append({
                                    'disease': target_disease,
                                    'gene': gene,
                                    'disease_name': disease_name,
                                    'disease_id': disease_id,
                                    'score': score,
                                    'source': 'DisGeNET_curated_github_mirror',
                                })
                            break
    except Exception as e:
        logger.warning(f"  DisGeNET 下载失败: {e}")

    for d, genes in disease_hits.items():
        logger.info(f"  {d}: {len(genes)} 个关联基因")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['disease', 'gene', 'disease_name', 'disease_id', 'score', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  DisGeNET 保存: {out_file}")
    else:
        logger.warning("  DisGeNET 无目标疾病数据!")
    return out_file


# =============================================================================
# 6.1 通路关键词配置加载器
# =============================================================================
def _load_pathway_keyword_config() -> dict:
    """从 pathway_keyword_config.yaml 加载通路关键词映射 (替代硬编码)

    配置文件: network_files/pathway_keyword_config.yaml
    数据来源: MSigDB (Subramanian et al., PNAS 2005)
              KEGG (Kanehisa et al., NAR 2021)
              Reactome (Jassal et al., NAR 2020)
    """
    config_file = OUT_DIR / "pathway_keyword_config.yaml"
    if not config_file.exists():
        logger.warning(f"  通路关键词配置文件不存在: {config_file}")
        return {}
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        kw_map = config.get('target_keywords', {})
        logger.info(f"  通路关键词配置加载: {len(kw_map)} 个关键词")
        return kw_map
    except ImportError:
        logger.warning("  PyYAML 未安装, 无法加载通路关键词配置")
        return {}
    except Exception as e:
        logger.error(f"  通路关键词配置加载失败: {e}")
        return {}


# =============================================================================
# 7. 合并函数: 构建基因-通路映射 (替代硬编码 gene_to_pathway_map)
# =============================================================================
# 从 MSigDB 数据中提取目标通路, 生成与原来硬编码一致格式的 CSV

def build_gene_pathway_csv():
    """从 MSigDB 数据构建项目特定的基因通路映射

    通路关键词映射: 从 pathway_keyword_config.yaml 加载 (替代硬编码)
    """
    msigdb_file = OUT_DIR / "msigdb_gene_pathways.csv"
    if not msigdb_file.exists():
        logger.warning("  MSigDB 文件不存在, 跳过基因通路映射构建")
        return None

    out_file = OUT_DIR / "gene_pathway_enrichment_external.csv"
    if out_file.exists():
        logger.info(f"  基因通路映射 已存在: {out_file}")
        return out_file

    # 从 pathway_keyword_config.yaml 加载通路关键词映射 (替代硬编码)
    target_pathways = _load_pathway_keyword_config()
    if not target_pathways:
        logger.warning("  通路关键词配置为空, 跳过基因通路映射构建")
        return None

    rows = []
    msigdb_df = None
    try:
        import pandas as pd
        msigdb_df = pd.read_csv(msigdb_file)
    except ImportError:
        logger.warning("  pandas 不可用, 使用 csv 模块")
        with open(msigdb_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            pathways_seen = set()
            for row in reader:
                pw = row['pathway'].lower()
                gene = row['gene']
                for kw, std_name in target_pathways.items():
                    if kw in pw:
                        key = (std_name, gene)
                        if key not in pathways_seen:
                            pathways_seen.add(key)
                            rows.append({
                                'pathway': std_name,
                                'gene': gene,
                                'source': f"MSigDB (original: {row['pathway']})",
                            })
                        break
        if rows:
            with open(out_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['pathway', 'gene', 'source'])
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"  基因通路映射 保存: {out_file} ({len(rows)} 行)")
        return out_file

    if msigdb_df is not None:
        for _, row in msigdb_df.iterrows():
            pw = str(row['pathway']).lower()
            gene = str(row['gene']).upper()
            for kw, std_name in target_pathways.items():
                if kw in pw:
                    rows.append({
                        'pathway': std_name,
                        'gene': gene,
                        'source': f"MSigDB (original: {row['pathway']})",
                    })
                    break

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['pathway', 'gene', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  基因通路映射 保存: {out_file} ({len(rows)} 行, {len(set(r['pathway'] for r in rows))} 通路)")
    return out_file


# =============================================================================
# 8. STRING PPI 数据下载
# =============================================================================
# 文献: Szklarczyk et al., NAR 2021, PMID: 33237311
# API: https://string-db.org/api/
# 物种: 10090 (Mus musculus)

def download_string_ppi(gene_list: list = None):
    """从 STRING API 下载蛋白-蛋白互作数据 (小鼠)

    如果提供了 gene_list, 则只下载这些基因的互作数据。
    否则下载所有已有的基因。

    Args:
        gene_list: 可选, 基因符号列表
    """
    out_file = OUT_DIR / "string_ppi_edges.csv"
    if out_file.exists():
        logger.info(f"  STRING PPI 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 STRING PPI 数据 ===")

    if not gene_list:
        # 尝试从铁衰老基因.txt读取基因列表
        ferroaging_file = BASE_DIR / "铁衰老基因.txt"
        if ferroaging_file.exists():
            with open(ferroaging_file, 'r', encoding='utf-8') as f:
                gene_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            logger.warning("  无基因列表, 跳过 STRING PPI 下载")
            return None

    rows = []

    try:
        # 步骤1: 将基因符号映射到 STRING ID
        logger.info(f"  映射 {len(gene_list)} 个基因到 STRING ID...")
        string_api_url = "https://string-db.org/api/json/get_string_ids"
        # 分批处理 (STRING API 限制)
        batch_size = 100
        gene_to_string_id = {}

        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i + batch_size]
            params = {
                'identifiers': '\r'.join(batch),
                'species': 10090,
                'limit': 1,
                'echo_query': 1,
            }
            resp = _safe_request(string_api_url, method='POST', data=params)
            data = resp.json()
            for item in data:
                query_item = item.get('queryItem', '').strip().upper()
                string_id = item.get('stringId', '')
                if string_id:
                    gene_to_string_id[query_item] = string_id
            time.sleep(0.5)  # 速率限制

        logger.info(f"  映射成功: {len(gene_to_string_id)}/{len(gene_list)} 个基因")

        # 步骤2: 下载 PPI 网络
        if gene_to_string_id:
            string_ids = list(gene_to_string_id.values())
            logger.info(f"  下载 PPI 网络 ({len(string_ids)} 个蛋白)...")

            for i in range(0, len(string_ids), batch_size):
                batch_ids = string_ids[i:i + batch_size]
                params = {
                    'identifiers': '\r'.join(batch_ids),
                    'species': 10090,
                    'required_score': 400,
                    'network_type': 'functional',
                    'add_white_nodes': 0,
                }
                resp = _safe_request(
                    "https://string-db.org/api/json/network",
                    method='POST', data=params
                )
                data = resp.json()
                for item in data:
                    score = item.get('score', 0)
                    if score >= 0.4:  # STRING API 返回 0-1 范围
                        # 反向映射 STRING ID → 基因符号
                        string_a = item.get('stringId_A', '')
                        string_b = item.get('stringId_B', '')
                        gene_a = None
                        gene_b = None
                        for g, sid in gene_to_string_id.items():
                            if sid == string_a:
                                gene_a = g
                            if sid == string_b:
                                gene_b = g
                        if gene_a and gene_b and gene_a != gene_b:
                            rows.append({
                                'protein_A': gene_a,
                                'protein_B': gene_b,
                                'score': int(score * 1000),  # 转换为 0-1000
                            })
                time.sleep(0.5)

        logger.info(f"  STRING PPI: {len(rows)} 条互作边 (score >= 400)")
    except Exception as e:
        logger.warning(f"  STRING PPI 下载失败: {e}")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['protein_A', 'protein_B', 'score'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  STRING PPI 保存: {out_file}")
    return out_file


# =============================================================================
# 9. TRRUST v2 TF-靶基因调控数据下载
# =============================================================================
# 文献: Han et al., NAR 2018, PMID: 29087512
# 网站: https://www.grnpedia.org/trrust/

TRRUST_MOUSE_URL = (
    "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv"
)

def download_trrust_tf_target():
    """从 TRRUST v2 下载小鼠转录因子-靶基因调控数据"""
    out_file = OUT_DIR / "trrust_tf_target.csv"
    if out_file.exists():
        logger.info(f"  TRRUST 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 TRRUST v2 TF-靶基因数据 ===")
    rows = []
    try:
        resp = _safe_request(TRRUST_MOUSE_URL)
        lines = resp.text.strip().split('\n')
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                tf = parts[0].strip().upper()
                target = parts[1].strip().upper()
                mode = parts[2].strip()  # Activation/Repression/Unknown
                if tf and target:
                    rows.append({
                        'tf': tf,
                        'target': target,
                        'mode': mode,
                        'source': 'TRRUST_v2',
                    })
        logger.info(f"  TRRUST: {len(rows)} 条调控关系")
    except Exception as e:
        logger.warning(f"  TRRUST 下载失败: {e}")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['tf', 'target', 'mode', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  TRRUST 保存: {out_file}")
    return out_file


# =============================================================================
# 10. FerrDb V2 铁死亡基因数据下载
# =============================================================================
# 文献: Zhou & Bao, NAR 2023, PMID: 36305826
# 网站: http://www.zhounan.org/ferrdb/

FERRDB_URL = "http://www.zhounan.org/ferrdb/current/"

def download_ferrdb_genes():
    """从 FerrDb V2 下载铁死亡调控因子基因列表"""
    OUT_DIR / "ferrdb_ferroptosis_genes.csv"
    # FerrDb 数据已在 idsp_gene_sets.py 中整理, 此处仅作记录
    logger.info("=== FerrDb V2 铁死亡基因 ===")
    logger.info("  FerrDb V2 数据已整合在 L1/idsp_gene_sets.py 中")
    logger.info("  文献: Zhou & Bao, NAR 2023, PMID: 36305826")
    logger.info("  网站: http://www.zhounan.org/ferrdb/")
    return None


# =============================================================================
# 11. KEGG 通路数据下载 (REST API)
# =============================================================================
# 文献: Kanehisa et al., NAR 2021, PMID: 33125081
# API: https://rest.kegg.jp/

def download_kegg_pathway_genes():
    """从 KEGG REST API 下载人类通路基因映射

    通路ID列表: 从 external_db_config.yaml 加载 (替代硬编码)
    文献: Kanehisa et al., NAR 2021, PMID: 33125081
    API: https://rest.kegg.jp/
    """
    out_file = OUT_DIR / "kegg_pathway_genes.csv"
    if out_file.exists():
        logger.info(f"  KEGG 通路基因 已存在: {out_file}")
        return out_file

    logger.info("=== 下载 KEGG 通路基因映射 ===")

    # 从配置文件加载 KEGG 通路ID (替代硬编码)
    db_config = _load_db_config()
    target_pathways = db_config.get('kegg_pathways', {})

    rows = []
    for kegg_id, pathway_name in target_pathways.items():
        try:
            url = f"https://rest.kegg.jp/link/hsa/{kegg_id}"
            resp = _safe_request(url)
            for line in resp.text.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    gene_id = parts[1].strip()
                    # KEGG基因ID格式: hsa:XXXXX
                    rows.append({
                        'pathway': pathway_name,
                        'gene_id': gene_id,
                        'kegg_id': kegg_id,
                        'source': 'KEGG_REST',
                    })
            time.sleep(0.3)  # KEGG API 速率限制
        except Exception as e:
            logger.warning(f"  KEGG {kegg_id} ({pathway_name}) 下载失败: {e}")

    if rows:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['pathway', 'gene_id', 'kegg_id', 'source'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  KEGG 通路基因 保存: {out_file} ({len(rows)} 行, {len(set(r['pathway'] for r in rows))} 通路)")
    return out_file


# =============================================================================
# 主入口
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("开始下载外部数据库数据...")
    logger.info("=" * 60)

    results = {}

    # 1. MSigDB 基因通路映射
    logger.info("\n[1/11] MSigDB 基因通路映射")
    logger.info("  来源: Broad Institute GSEA-MSigDB (Subramanian et al., PNAS 2005)")
    logger.info("  GitHub: https://github.com/GSEA-MSigDB/msigdb")
    results['msigdb'] = download_msigdb_gene_pathways()

    # 2. PanglaoDB 细胞类型标记
    logger.info("\n[2/11] PanglaoDB 细胞类型标记基因")
    logger.info("  来源: PanglaoDB (Franzén et al., Database 2019)")
    logger.info("  GitHub: https://github.com/oscar-franzen/PanglaoDB")
    results['panglaodb'] = download_panglaodb_markers()

    # 3. CellChatDB 配体-受体对
    logger.info("\n[3/11] CellChatDB 配体-受体对")
    logger.info("  来源: CellChatDB (Jin et al., Nature Comms 2021)")
    logger.info("  GitHub: https://github.com/sqjin/CellChat")
    results['cellchat'] = download_cellchat_lr_pairs()

    # 4. PubChem 化合物属性
    logger.info("\n[4/11] PubChem 化合物理化性质")
    logger.info("  来源: PubChem PUG REST API (Kim et al., NAR 2021)")
    logger.info("  GitHub: https://github.com/ncbi/PubChem")
    results['pubchem'] = download_pubchem_compound_props()

    # 5. STITCH 化合物-靶点
    logger.info("\n[5/11] STITCH 化合物-靶点互作")
    logger.info("  来源: STITCH v5.0 (Szklarczyk et al., NAR 2021)")
    logger.info("  物种: 10090 (Mus musculus)")
    results['stitch'] = download_stitch_compound_targets()

    # 6. DisGeNET 疾病-基因
    logger.info("\n[6/11] DisGeNET 疾病-基因关联")
    logger.info("  来源: DisGeNET curated (Piñero et al., NAR 2020)")
    logger.info("  GitHub: https://github.com/DisGeNET/DisGeNET-SQLite")
    results['disgenet'] = download_disgenet_disease_genes()

    # 7. 构建基因通路映射
    logger.info("\n[7/11] 构建基因通路映射 (MSigDB标准化)")
    results['gene_pathway'] = build_gene_pathway_csv()

    # 8. STRING PPI
    logger.info("\n[8/11] STRING PPI 蛋白互作")
    logger.info("  来源: STRING v12 (Szklarczyk et al., NAR 2021)")
    logger.info("  API: https://string-db.org/api/")
    results['string_ppi'] = download_string_ppi()

    # 9. TRRUST v2
    logger.info("\n[9/11] TRRUST v2 TF-靶基因调控")
    logger.info("  来源: TRRUST v2 (Han et al., NAR 2018)")
    logger.info("  网站: https://www.grnpedia.org/trrust/")
    results['trrust'] = download_trrust_tf_target()

    # 10. FerrDb V2 (记录)
    logger.info("\n[10/11] FerrDb V2 铁死亡基因 (已整合在 idsp_gene_sets.py)")
    logger.info("  文献: Zhou & Bao, NAR 2023, PMID: 36305826")
    results['ferrdb'] = download_ferrdb_genes()

    # 11. KEGG REST API
    logger.info("\n[11/11] KEGG 通路基因映射")
    logger.info("  来源: KEGG REST API (Kanehisa et al., NAR 2021)")
    logger.info("  API: https://rest.kegg.jp/")
    results['kegg'] = download_kegg_pathway_genes()

    logger.info("\n" + "=" * 60)
    logger.info("下载完成! 输出文件汇总:")
    for name, path in results.items():
        if path is None:
            logger.info(f"  [SKIP] {name}: 未下载 (数据已在代码中)")
        elif isinstance(path, Path) and path.exists():
            logger.info(f"  [OK] {name}: {path}")
        elif isinstance(path, Path):
            logger.info(f"  [FAIL] {name}: {path} (文件不存在)")
        else:
            logger.info(f"  [INFO] {name}: {path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
