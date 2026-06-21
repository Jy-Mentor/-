"""测试 DGIdb GraphQL API."""

import requests

QUERY = """
{
  genes(names: ["ACSL4", "GPX4", "PTGS2"]) {
    nodes {
      name
      interactions {
        drug { name }
        interactionTypes { type }
        interactionScore
      }
    }
  }
}
"""

resp = requests.post(
    "https://dgidb.org/api/graphql",
    json={"query": QUERY},
    headers={"Content-Type": "application/json"},
    timeout=30,
)
print("status:", resp.status_code)
print("content-type:", resp.headers.get("content-type"))
print(resp.text[:1000])
