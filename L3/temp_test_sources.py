"""Test real data source downloads"""
import requests, json, zipfile, io, csv

# Test Open Targets Genetics API for AD genes
query = """
{
  disease(efoId: "MONDO_0004975") {
    id
    name
    associatedTargets(pageSize: 500) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
      }
    }
  }
}
"""
try:
    r = requests.post(
        'https://api.platform.opentargets.org/api/v4/graphql',
        json={'query': query},
        timeout=30
    )
    print(f'Open Targets AD: status={r.status_code}')
    if r.status_code == 200:
        data = r.json()
        targets = data.get('data', {}).get('disease', {}).get('associatedTargets', {})
        print(f'Total AD associated targets: {targets.get("count", "N/A")}')
        rows = targets.get('rows', [])
        print(f'First 10:')
        for row in rows[:10]:
            t = row['target']
            print(f'  {t["approvedSymbol"]} - score={row["score"]}')
except Exception as e:
    print(f'Open Targets FAIL: {e}')

# Test GenAge zip extraction
print('\n=== GenAge ZIP test ===')
rz = requests.get('https://hagr.ageing-map.org/genes/human_genes.zip', timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
print(f'Downloaded {len(rz.content)} bytes')
with zipfile.ZipFile(io.BytesIO(rz.content)) as z:
    print(f'Files: {z.namelist()}')
    for fname in z.namelist():
        with z.open(fname) as f:
            content = f.read().decode('utf-8')
            lines = content.strip().split('\n')
            print(f'{fname}: {len(lines)} lines')
            print(f'Header: {lines[0]}')
            print(f'First 3 data rows:')
            for l in lines[1:4]:
                print(f'  {l}')

# Test CellAge zip extraction
print('\n=== CellAge ZIP test ===')
rz = requests.get('https://hagr.ageing-map.org/cells/cellAge.zip', timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
print(f'Downloaded {len(rz.content)} bytes')
with zipfile.ZipFile(io.BytesIO(rz.content)) as z:
    print(f'Files: {z.namelist()}')
    for fname in z.namelist():
        with z.open(fname) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8'))
            header = next(reader)
            data = list(reader)
            print(f'{fname}: {len(data)} data rows')
            print(f'Header: {header}')
            print(f'First 3 rows:')
            for row in data[:3]:
                print(f'  {row}')

# Test LongevityMap
print('\n=== LongevityMap ZIP test ===')
rz = requests.get('https://hagr.ageing-map.org/longevity/longevity_genes.zip', timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
print(f'Downloaded {len(rz.content)} bytes')
with zipfile.ZipFile(io.BytesIO(rz.content)) as z:
    print(f'Files: {z.namelist()}')
    for fname in z.namelist():
        with z.open(fname) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8'))
            header = next(reader)
            data = list(reader)
            print(f'{fname}: {len(data)} data rows')
            print(f'Header: {header}')