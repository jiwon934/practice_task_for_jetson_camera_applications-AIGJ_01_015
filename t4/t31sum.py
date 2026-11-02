import csv, collections, sys
name = sys.argv[1]
rows = [r for r in csv.reader(open("/home/jiwon934/eg2-lab/applab/runs/t4_trace.csv")) if (r[8] if len(r) > 9 else r[-1]) == name]
by = collections.defaultdict(list)
for r in rows: by[r[1]].append(r)
for tid, rs in by.items():
    v = [int(r[4]) for r in rs]; L = [float(r[5]) for r in rs]; R = [float(r[6]) for r in rs]; f = [float(r[7]) for r in rs]
    print(f"id{tid} frames {len(rs)} joints mean {sum(v)/len(v):.1f} min {min(v)} max {max(v)} >=15: {100*sum(x>=15 for x in v)/len(v):.0f}%  Lconf {sum(L)/len(L):.2f} Rconf {sum(R)/len(R):.2f} facing {sum(f)/len(f):.2f}")
