import gzip
with gzip.open(r'c:\Users\Jy-Mentor-7\Desktop\铁衰老\L1\GPL6883.annot.gz', 'rt', encoding='latin-1') as f:
    in_table = False
    for line in f:
        l = line.strip()
        if l == '!platform_table_begin':
            in_table = True
            header = f.readline().strip().split('\t')
            print('Header columns:')
            for j, h in enumerate(header):
                h_clean = h.strip('"')
                print(f'  [{j}] {h_clean}')
            # Print first 5 data rows
            for k in range(5):
                row = f.readline().strip().split('\t')
                print(f'Row {k}:', row[:6])
            break