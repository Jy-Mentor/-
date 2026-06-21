"""Print tail of SwissTargetPrediction predict.php response."""
import requests

SMILES = "CC(=O)Nc1ccc(O)c(C)c1"
PREDICT_URL = "https://www.swisstargetprediction.ch/predict.php"

sub = requests.post(
    PREDICT_URL,
    data={"organism": "Homo_sapiens", "smiles": SMILES},
    timeout=60,
)
text = sub.text
print("status:", sub.status_code, "len:", len(text))
print("--- tail ---")
print(text[-3000:])
