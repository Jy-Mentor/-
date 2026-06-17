"""Test correct Open Targets API query"""
import requests

pagination_formats = [
    'associatedTargets(pageSize: 500)',
    'associatedTargets(index: 0, limit: 500)',
    'associatedTargets(first: 500)',
]

for pf in pagination_formats:
    q = f"""{{disease(efoId: "MONDO_0004975") {{id name {pf} {{count rows {{target {{id approvedSymbol}} score}}}}}}}}"""
    r = requests.post('https://api.platform.opentargets.org/api/v4/graphql', 
                     json={'query': q}, timeout=30,
                     headers={'Content-Type': 'application/json'})
    print(f'{pf}: status={r.status_code}')
    if r.status_code == 200:
        data = r.json()
        at = data.get('data', {}).get('disease', {}).get('associatedTargets', {})
        print(f'  count={at.get("count")}, rows={len(at.get("rows",[]))}')
        if at.get('rows'):
            for row in at['rows'][:5]:
                print(f'  {row["target"]["approvedSymbol"]} score={row["score"]}')
    else:
        print(f'  {r.text[:200]}')
    print()