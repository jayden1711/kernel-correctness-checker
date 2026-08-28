"""Independent blast-radius recount (local, no GPU): affected-record counts
across every banked corpus arm, catch attribution, and margin-table
dependency. Duplicates nothing from theory_closure -- fresh fingerprints."""
import json, gzip, glob, collections
BANKS = (glob.glob('../n_samples_curve_2026-08-25/arms/*.json.gz')
         + glob.glob('../scope_detect_2026-08-26/arms/*.json.gz')
         + glob.glob('../gram_screen_2026-08-27/arms/*.json.gz')
         + glob.glob('../check_timing_2026-08-25/arms/*.json.gz'))
tot = collections.Counter(); catch_dep = []
sulp_min = {}
for p in BANKS:
    try: d = json.load(gzip.open(p, 'rt'))
    except Exception: continue
    for e in d.get('entries', []):
        if e['op'] in ('layernorm','rmsnorm'):
            det = e['mutant'].get('detail') or ''
            if 'non_power_of_two' in det:
                catch_dep.append((p, e['op'], e['mutant']['name']))
        packs=[e['mutant']['records']]+[r['records'] for r in e.get('refs',[])]
        for recs in packs:
            for r in recs:
                if (r['name']=='adversarial_non_power_of_two'
                        and e['op'] in ('layernorm','rmsnorm')):
                    tot[(p.split('/')[1], e['op'])] += 1
                for sc in (r.get('scope_flags') or []):
                    if (sc.get('sulp_median') is not None
                            and e['op'] in ('layernorm','rmsnorm')):
                        k=(p.split('/')[1], e['op'])
                        c=sulp_min.get(k)
                        if c is None or sc['sulp_median'] < c[0]:
                            sulp_min[k]=(sc['sulp_median'], r['name'])
print("affected records per (round, op) summed over that round's arms:")
for k,v in sorted(tot.items()): print('  ', k, v)
print("catches attributed to non_power_of_two:", catch_dep or "NONE")
print("per-op MIN s/ulp attained by:", {k:v[1] for k,v in sorted(sulp_min.items())})
