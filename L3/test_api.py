"""Test Open Targets API for AD"""
import requests

# Try simpler Open Targets query - AD as MONDO:0004975
queries = [
    ('MONDO_0004975 (Alzheimer disease)', '{disease(efoId:"MONDO_0004975"){id name}}'),
    ('EFO_0000249', '{disease(efoId:"EFO_0000249"){id name}}'),
    ('search AD', '{search(queryString:"Alzheimer",entityNames:["disease"],pageSize:5){total hits{id name}}}'),
]

for label, q in queries:
    try:
        r = requests.post('https://api.platform.opentargets.org/api/v4/graphql', 
                         json={'query': q}, timeout=30,
                         headers={'Content-Type': 'application/json'})
        print(f'{label}: status={r.status_code}')
        if r.status_code == 200:
            print(r.json())
        else:
            print(r.text[:300])
    except Exception as e:
        print(f'{label}: {e}')
    print()