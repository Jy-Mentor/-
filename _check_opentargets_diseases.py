"""查询 OpenTargets 中 CIRI/卒中相关疾病的正确 EFO ID。"""

import requests

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query SearchDisease($query: String!) {
  search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: 10}) {
    total
    hits {
      id
      name
      entity
      description
    }
  }
}
"""

for q in ["stroke", "ischemic stroke", "cerebral ischemia", "brain ischemia",
          "reperfusion injury", "hypoxic ischemic encephalopathy", "cerebral infarction"]:
    r = requests.post(OT_URL, json={"query": SEARCH_QUERY, "variables": {"query": q}})
    data = r.json()
    print(f"\nQuery: {q}")
    hits = data.get("data", {}).get("search", {}).get("hits", [])
    for h in hits[:5]:
        print(f"  {h['id']} | {h['name']}")
