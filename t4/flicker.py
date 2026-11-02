"""t4_trace.csv의 한 구간(mark)에서 사람별 라벨 흐름을 요약: 라벨 바뀐 횟수, 깜빡임(1초 안에 2번 이상 바뀜) 횟수,
목표 라벨이 켜진 '구간' 수(0.3초 이상 이어진 것), 라벨별 프레임 비율"""
import csv, collections, sys
name, target = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
rows = [r for r in csv.reader(open("/home/jiwon934/eg2-lab/applab/runs/t4_trace.csv")) if (r[8] if len(r) > 9 else r[-1]) == name]
by = collections.defaultdict(list)
for r in rows: by[r[1]].append((float(r[0]), r[2]))
for tid, seq in by.items():
    if len(seq) < 20: continue
    ch = [seq[i][0] for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1]]
    fl, i = 0, 0
    while i < len(ch):                      # 1초 창 안에 바뀜 2번 이상이면 깜빡임 1회, 그 창은 건너뜀
        j = i
        while j + 1 < len(ch) and ch[j + 1] - ch[i] <= 1.0: j += 1
        if j - i + 1 >= 2: fl += 1; i = j + 1
        else: i += 1
    segs, cur, start = [], None, None
    for t, l in seq + [(seq[-1][0] + 1, None)]:
        if l != cur:
            if cur is not None: segs.append((cur, start, t))
            cur, start = l, t
    on = [s for s in segs if s[0] == target and s[2] - s[1] >= 0.3] if target else []
    dist = collections.Counter(l for _, l in seq)
    print(f"id{tid} {seq[-1][0]-seq[0][0]:.0f}s frames {len(seq)} changes {len(ch)} flicker {fl}"
          + (f" | {target} episodes(>=0.3s) {len(on)}" if target else "") + f" | {dict(dist.most_common(4))}")
