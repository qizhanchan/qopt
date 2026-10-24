# 1. 准备只含两个目标实例的 problems dir
mkdir -p /tmp/qopt_prove /tmp/qopt_prove_results
cp bench/problems/miplib/gen-ip054.json bench/problems/miplib/gen-ip002.json /tmp/qopt_prove/

# 2. 三档 budget 跑默认配置（无 cuts、most-fractional、tableau backend）
#    parallel 2 因为只有两道题；30m 这档跑完约 30 分钟
for budget in 5m 10m 30m; do
    echo "=== default budget=$budget ==="
    /opt/homebrew/bin/go run ./bench/cmd/qopt_bench \
        -problems /tmp/qopt_prove \
        -out /tmp/qopt_prove_results/default_$budget.json \
        -time-limit $budget \
        -repeat 1 -warmup 0 -parallel 2 \
        -node-selection dfs
done

# 3. （可选）顺手再跑一档 pseudocost branching 对照 —— 对 general-integer + 浮点系数
#    的 gen-ip054 / gen-ip002 它理论上比 most-fractional 强不少
/opt/homebrew/bin/go run ./bench/cmd/qopt_bench \
    -problems /tmp/qopt_prove \
    -out /tmp/qopt_prove_results/pseudocost_30m.json \
    -time-limit 30m \
    -repeat 1 -warmup 0 -parallel 2 \
    -node-selection dfs -branching pseudocost

# 4. 读结果：关注 status 是否从 time_limit 变成 optimal，最后一笔 incumbent 时间
for f in /tmp/qopt_prove_results/*.json; do
    echo "=== $(basename $f) ==="
    python3 -c "
import json
with open('$f') as fp:
    r = json.load(fp)
for it in r['items']:
    print(f'  {it[\"name\"]:20s} status={it[\"status\"]:12s} obj={it[\"objective\"]:.4f} nodes={it.get(\"nodes\",0)} elapsed={it.get(\"elapsed_ms\",0)/1000:.1f}s')
    for ev in (it.get('incumbents') or [])[-3:]:
        print(f'      last incumbent @ t={ev[\"elapsed_ms\"]/1000:.1f}s node={ev[\"node\"]} obj={ev[\"objective\"]:.4f}')
"
done
