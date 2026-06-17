"""Test full Open Targets query for AD targets"""
import requests

q = """
{
  disease(efoId: "MONDO_0004975") {
    id
    name
    associatedTargets(page: 0, size: 500) {
      count
      rows {
        target {
          id
          approvedSymbol
        }
        score
        overallAssociationScore
      }
    }
  }
}
"""
r = requests.post('https://api.platform.opentargets.org/api/v4/graphql', 
                 json={'query': q}, timeout=30,
                 headers={'Content-Type': 'application/json'})
print(f'Status: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    dt = data.get('data', {}).get('disease', {})
    targets = dt.get('associatedTargets', {})
    print(f'Disease: {dt.get("name")} ({dt.get("id")})')
    print(f'Count: {targets.get("count")}')
    rows = targets.get('rows', [])
    print(f'Rows returned: {len(rows)}')
    for row in rows[:20]:
        t = row['target']
        print(f'  {t["approvedSymbol"]}  score={row["score"]:.4f}  overall={row["overallAssociationScore"]:.4f}')
else:
    print(r.text[:1000])