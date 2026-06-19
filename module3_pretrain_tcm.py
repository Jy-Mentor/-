"""
Large-scale pre-training for the AttentiveFP molecular encoder.

NOTE: The previous 12-molecule TCM-only GraphCL pre-training (Plan A) is deprecated
because it was statistically invalid and risked data leakage. This script implements
Plan B + C:
  - Real large-scale self-supervised pre-training on MoleculeNet (BBBP) + a ChEMBL
    subset (~2 M compounds) downloaded from public repositories.
  - The 63 project compounds (20 core + 43 expansion) are held out and never seen
    during pre-training; they are used only for downstream validation.

Data sources:
  - MoleculeNet BBBP: data/pretrain/BBBP.csv (2050 blood-brain-barrier labels)
    Reference: Wu et al. "MoleculeNet: a benchmark for molecular machine learning",
    Chem. Sci. 2018. https://moleculenet.org
  - ChEMBL 33 chemical representations:
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_chemreps.txt.gz
    Reference: Mendez et al. "ChEMBL: towards direct deposition of bioassay data",
    NAR 2019.

Method references:
  - AttentiveFP: Xiong et al., J Med Chem 2020.
    GitHub: https://github.com/OpenDrugAI/AttentiveFP
  - GraphCL / NT-Xent: Sun et al., ICLR 2020.
    PyG implementation pattern: simmzx/MolCLR_AttFP
  - ChEMBL downloader reference: cthoyt/chembl-downloader
"""

import argparse
import logging
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import AttentiveFP
from torch_geometric.utils import dropout_edge, subgraph

BASE_DIR = Path(__file__).parent
NETWORK_DIR = BASE_DIR / "network_files"
PRETRAIN_DIR = BASE_DIR / "data" / "pretrain"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ChEMBL 33 chemreps URL (EBI FTP)
CHEMBL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/"
    "chembl_33_chemreps.txt.gz"
)
CHEMBL_LOCAL = PRETRAIN_DIR / "chembl_33_chemreps.txt.gz"


def _smiles_to_data(smiles: str):
    """Convert SMILES to PyG Data with real RDKit atom/bond features.

    Feature encoding is kept identical to module3_hgt.py so that the pre-trained
    weights can be loaded directly into the downstream AttentiveFP encoder.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_types = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    hybrid_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.OTHER,
    ]
    chirality_types = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]

    xs = []
    for atom in mol.GetAtoms():
        x = [0] * (len(atom_types) + 1)
        if atom.GetAtomicNum() in atom_types:
            x[atom_types.index(atom.GetAtomicNum())] = 1
        else:
            x[-1] = 1

        h = [0] * 6
        h[min(atom.GetDegree(), 5)] = 1
        x.extend(h)

        f = [0] * 5
        f[min(max(atom.GetFormalCharge() + 2, 0), 4)] = 1
        x.extend(f)

        c = [0] * 4
        if atom.GetChiralTag() in chirality_types:
            c[chirality_types.index(atom.GetChiralTag())] = 1
        else:
            c[-1] = 1
        x.extend(c)

        hs = [0] * 5
        hs[min(atom.GetTotalNumHs(), 4)] = 1
        x.extend(hs)

        hyb = [0] * 4
        if atom.GetHybridization() in hybrid_types:
            hyb[hybrid_types.index(atom.GetHybridization())] = 1
        else:
            hyb[-1] = 1
        x.extend(hyb)

        x.append(int(atom.GetIsAromatic()))
        x.append(int(atom.IsInRing()))

        xs.append(x)

    x = torch.tensor(xs, dtype=torch.float)

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

        e = [0] * 4
        if bond.GetBondType() in bond_types:
            e[bond_types.index(bond.GetBondType())] = 1
        else:
            e[-1] = 1

        e.append(int(bond.GetIsConjugated()))
        e.append(int(bond.IsInRing()))

        s = [0] * 6
        if bond.GetStereo() in stereo_types:
            s[stereo_types.index(bond.GetStereo())] = 1
        else:
            s[-1] = 1
        e.extend(s)

        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Remove isolated atoms (e.g., salt counter-ions) so num_nodes matches edge_index.
    # This prevents index mismatches during GraphCL subgraph augmentation.
    if edge_index.numel() == 0:
        return None
    connected = torch.unique(edge_index)
    if connected.size(0) < 2:
        return None
    mapping = torch.full((x.size(0),), -1, dtype=torch.long)
    mapping[connected] = torch.arange(connected.size(0), dtype=torch.long)
    x = x[connected]
    edge_index = mapping[edge_index]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _canonical_smiles(smiles: str):
    """Return RDKit canonical SMILES or None if invalid."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _download_chembl(url: str = CHEMBL_URL, local: Path = CHEMBL_LOCAL):
    """Download ChEMBL 33 chemreps if not already present."""
    if local.exists():
        logger.info(f"ChEMBL file already exists: {local}")
        return local

    logger.info(f"Downloading ChEMBL chemreps from {url}")
    logger.info("This is ~233 MB and may take a few minutes ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(local, "wb") as out:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total and downloaded % (10 * chunk_size) == 0:
                logger.info(
                    f"Downloaded {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
                )
    logger.info(
        f"ChEMBL downloaded: {local} ({local.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    return local


def _load_core_smiles() -> set:
    """Load canonical SMILES of project compounds to exclude from pre-training."""
    core_file = NETWORK_DIR / "compound_smiles.csv"
    core = set()
    if not core_file.exists():
        logger.warning("compound_smiles.csv not found; cannot exclude core compounds")
        return core

    df = pd.read_csv(core_file)
    for smi in df["CanonicalSMILES"].dropna().astype(str):
        cano = _canonical_smiles(smi)
        if cano:
            core.add(cano)
    logger.info(f"Core compounds to exclude from pre-training: {len(core)}")
    return core


def _load_bbbp_smiles():
    """Load valid SMILES from MoleculeNet BBBP."""
    bbbp_file = PRETRAIN_DIR / "BBBP.csv"
    if not bbbp_file.exists():
        raise FileNotFoundError(f"BBBP.csv not found at {bbbp_file}")

    df = pd.read_csv(bbbp_file)
    smiles_list = df["smiles"].dropna().astype(str).tolist()
    valid = []
    for smi in smiles_list:
        data = _smiles_to_data(smi)
        if data is not None and data.num_nodes >= 2:
            valid.append(data)
    logger.info(f"BBBP valid molecules: {len(valid)}/{len(smiles_list)}")
    return valid


def _load_chembl_smiles(
    n_samples: int = 50_000,
    max_heavy_atoms: int = 100,
    seed: int = 42,
):
    """Sample valid drug-like molecules from ChEMBL 33 chemreps.

    Excludes any molecule whose canonical SMILES matches the project core set
    to avoid data leakage into downstream validation.
    """
    local = _download_chembl()
    core = _load_core_smiles()

    logger.info("Reading ChEMBL SMILES (random sample pool) ...")
    # Load a larger pool than needed; parsing all 2 M is slow and memory-heavy.
    min(500_000, n_samples * 20)
    df = pd.read_csv(
        local,
        sep="\t",
        compression="gzip",
        usecols=["canonical_smiles"],
    )
    smiles_pool = df["canonical_smiles"].dropna().astype(str).tolist()
    logger.info(f"ChEMBL raw SMILES loaded: {len(smiles_pool)}")

    rng = random.Random(seed)
    rng.shuffle(smiles_pool)

    selected = []
    skipped_core = 0
    skipped_large = 0
    skipped_invalid = 0

    for smi in smiles_pool:
        if len(selected) >= n_samples:
            break

        cano = _canonical_smiles(smi)
        if cano is None:
            skipped_invalid += 1
            continue
        if cano in core:
            skipped_core += 1
            continue

        data = _smiles_to_data(smi)
        if data is None:
            skipped_invalid += 1
            continue
        if data.x.size(0) > max_heavy_atoms:
            skipped_large += 1
            continue

        selected.append(data)

    logger.info(
        f"ChEMBL selected: {len(selected)} | skipped invalid: {skipped_invalid}, "
        f"core overlap: {skipped_core}, too large: {skipped_large}"
    )
    return selected


class GraphCLAugmentor:
    """Generate augmented views of molecular graphs for contrastive learning."""

    def __init__(self, node_drop_prob=0.1, edge_drop_prob=0.2, subgraph_ratio=0.8):
        self.node_drop_prob = node_drop_prob
        self.edge_drop_prob = edge_drop_prob
        self.subgraph_ratio = subgraph_ratio

    def __call__(self, data: Data):
        aug_data = data.clone()
        aug_data = self._node_drop(aug_data)
        aug_data = self._edge_perturb(aug_data)
        if random.random() < 0.5:
            aug_data = self._subgraph(aug_data)
        return aug_data

    def _node_drop(self, data: Data):
        if data.num_nodes <= 3 or self.node_drop_prob <= 0:
            return data
        keep_mask = torch.rand(data.num_nodes) > self.node_drop_prob
        if keep_mask.sum() < 2:
            keep_mask[:2] = True
        node_idx = torch.where(keep_mask)[0]
        edge_index, edge_attr = subgraph(
            node_idx, data.edge_index, data.edge_attr, relabel_nodes=True
        )
        if edge_attr.dim() == 1:
            edge_attr = torch.zeros(
                (edge_index.size(1), data.edge_attr.size(1)), dtype=torch.float
            )
        return Data(x=data.x[node_idx], edge_index=edge_index, edge_attr=edge_attr)

    def _edge_perturb(self, data: Data):
        if self.edge_drop_prob <= 0:
            return data
        edge_index, edge_attr = dropout_edge(
            data.edge_index, p=self.edge_drop_prob, force_undirected=True
        )
        if edge_attr.dim() == 1 or edge_attr.size(0) != edge_index.size(1):
            edge_attr = torch.zeros(
                (edge_index.size(1), data.edge_attr.size(1)), dtype=torch.float
            )
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)

    def _subgraph(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes <= 3 or data.edge_index.numel() == 0:
            return data
        num_sample = max(2, int(num_nodes * self.subgraph_ratio))
        node_idx = set()
        candidates = set(range(num_nodes))
        start = random.randint(0, num_nodes - 1)
        node_idx.add(start)
        candidates.remove(start)
        frontier = {start}

        edge_index = data.edge_index
        if edge_index.max().item() >= num_nodes:
            return data

        while len(node_idx) < num_sample and frontier:
            cur = frontier.pop()
            mask = (edge_index[0] == cur) | (edge_index[1] == cur)
            if mask.sum() == 0:
                continue
            nbrs = (
                torch.cat([edge_index[0][mask], edge_index[1][mask]]).unique().tolist()
            )
            for n in nbrs:
                if n in candidates and len(node_idx) < num_sample:
                    node_idx.add(n)
                    candidates.remove(n)
                    frontier.add(n)

        node_idx = torch.tensor(sorted(node_idx), dtype=torch.long)
        node_idx = node_idx[(node_idx >= 0) & (node_idx < num_nodes)]
        if node_idx.numel() < 2:
            return data

        edge_index, edge_attr = subgraph(
            node_idx, data.edge_index, data.edge_attr, relabel_nodes=True
        )
        if edge_attr.dim() == 1:
            edge_attr = torch.zeros(
                (edge_index.size(1), data.edge_attr.size(1)), dtype=torch.float
            )
        return Data(x=data.x[node_idx], edge_index=edge_index, edge_attr=edge_attr)


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized temperature-scaled cross entropy loss (GraphCL)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)
    n = z1.size(0)
    mask = torch.eye(2 * n, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)]).reshape(2 * n, 1)
    loss = -torch.log(pos.exp() / sim.exp().sum(dim=1, keepdim=True))
    return loss.mean()


def _collate_views(view_list, device):
    """Collate a list of augmented Data objects into tensors on target device."""
    xs = [v.x for v in view_list]
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    node_offset = 0
    for i, v in enumerate(view_list):
        edge_indices.append(v.edge_index + node_offset)
        edge_attrs.append(v.edge_attr)
        batch_vec.append(torch.full((v.x.size(0),), i, dtype=torch.long))
        node_offset += v.x.size(0)
    x = torch.cat(xs, dim=0).to(device)
    edge_index = torch.cat(edge_indices, dim=1).to(device)
    edge_attr = torch.cat(edge_attrs, dim=0).to(device)
    batch_vec = torch.cat(batch_vec, dim=0).to(device)
    return x, edge_index, edge_attr, batch_vec


def pretrain(
    chembl_samples: int = 50_000,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    out_channels: int = 64,
    temperature: float = 0.5,
    seed: int = 42,
):
    """GraphCL pre-training on BBBP + ChEMBL; save AttentiveFP weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    bbbp_data = _load_bbbp_smiles()
    chembl_data = _load_chembl_smiles(n_samples=chembl_samples, seed=seed)

    data_list = bbbp_data + chembl_data
    logger.info(
        f"Total pre-training molecules: {len(data_list)} (BBBP {len(bbbp_data)}, ChEMBL {len(chembl_data)})"
    )

    if len(data_list) < batch_size:
        raise RuntimeError("Not enough valid molecules for pre-training")

    in_channels = data_list[0].x.size(1)
    edge_dim = data_list[0].edge_attr.size(1)

    model = AttentiveFP(
        in_channels=in_channels,
        hidden_channels=out_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=2,
        num_timesteps=2,
        dropout=0.0,
    ).to(device)

    augmentor = GraphCLAugmentor(
        node_drop_prob=0.1, edge_drop_prob=0.2, subgraph_ratio=0.8
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_graphs = len(data_list)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_graphs)
        for start in range(0, n_graphs, batch_size):
            idx = perm[start : start + batch_size].tolist()
            batch_graphs = [data_list[i] for i in idx]

            optimizer.zero_grad()

            # Generate two augmented views per molecule (CPU augmentation)
            view1_list = [augmentor(g) for g in batch_graphs]
            view2_list = [augmentor(g) for g in batch_graphs]

            x1, ei1, ea1, b1 = _collate_views(view1_list, device)
            x2, ei2, ea2, b2 = _collate_views(view2_list, device)

            z1 = model(x1, ei1, ea1, batch=b1)
            z2 = model(x2, ei2, ea2, batch=b2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f}")

    save_path = CHECKPOINT_DIR / "pretrained_attentivefp_moleculenet.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Pre-trained AttentiveFP saved: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Large-scale AttentiveFP pre-training on MoleculeNet/ChEMBL (deprecated TCM-only Plan A)"
    )
    parser.add_argument(
        "--chembl-samples",
        type=int,
        default=50_000,
        help="Number of ChEMBL molecules to sample (default 50000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of pre-training epochs (default 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (default 128)"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pretrain(
        chembl_samples=args.chembl_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_channels=args.out_channels,
        temperature=args.temperature,
        seed=args.seed,
    )
