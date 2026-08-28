"""Search-DB exposure: layernorm/rmsnorm proposals whose primary width is
below 333 ran the checker's full battery, including the OOB variant. Lists
them and whether their recorded verdict depends on non_power_of_two."""
import sqlite3, json, re, glob
for db in ['../../adversarial_results/search_history.db',
           '../../adversarial_results/cfa_rerun_2026-08-20/search_history.db',
           '../../adversarial_results/cfa_rerun_postfix_2026-08-21/search_history.db']:
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    cur = con.cursor()
    cur2 = con.cursor()   # separate cursor: the nested verdict query must
                          # not clobber the outer iteration
    n_tot = n_small = 0
    for pid, op, pj in cur.execute(
            "select proposal_id, operator, proposal_json from proposals "
            "where operator in ('layernorm','rmsnorm')"):
        n_tot += 1
        dims = [[int(v) for v in sh.split(',') if v.strip()] for sh in
                re.findall(r'"shape"\s*:\s*\[([0-9,\s]+)\]', pj)]
        if any(len(d) == 2 and d[1] < 333 for d in dims):
            n_small += 1
            for row in cur2.execute(
                    "select is_hit, hit_mutants, verdict_json from verdicts "
                    "where proposal_id=?", (pid,)):
                v = row[2] or "{}"
                print(db.split('/')[-2], pid[:8], op, dims,
                      "is_hit", row[0], "hits", row[1],
                      "| verdict mentions non_power_of_two:",
                      "non_power_of_two" in v,
                      "| summary:", json.loads(v).get("failure_summary", "")[:120])
    print(db.split('/')[-2] or 'main', f"layernorm/rmsnorm proposals: {n_tot}, width<333: {n_small}")
    con.close()
