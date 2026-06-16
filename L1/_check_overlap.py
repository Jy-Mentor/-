import gzip, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from l1_dual_analysis import parse_series_matrix, parse_gpl6883_annotation, find_file, DATA_DIRS, GPL6883_ANNOT

# Load the annotation
probe_map = parse_gpl6883_annotation(GPL6883_ANNOT)
print(f"Annotation probes: {len(probe_map)}")
print(f"Sample annotation keys: {list(probe_map.keys())[:5]}")

# Load the expression matrix
sm_file = find_file(DATA_DIRS['GSE16561'], ['series_matrix'])
print(f"Series matrix file: {sm_file}")
expr_df = parse_series_matrix(sm_file)
print(f"Expression matrix shape: {expr_df.shape}")
print(f"Sample matrix index: {list(expr_df.index[:5])}")

# Check overlap
overlap = set(expr_df.index) & set(probe_map.keys())
print(f"Overlap: {len(overlap)}")
if overlap:
    print(f"Sample overlapped probes: {list(overlap)[:5]}")
else:
    print(f"Matrix probes start like: {list(expr_df.index[:5])}")
    print(f"Annot probes start like: {list(probe_map.keys())[:5]}")