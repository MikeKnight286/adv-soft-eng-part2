"""Reproduces every number in the paper "What Breaks in Open-Source Games? Symptom Categories of Reported Bugs
in OSSGameBench and Testing Strategies to Detect Them", from the raw OSSGameBench tables to the last table.

Usage:  python -I reproduce.py [path/to/OSSGameBench]     (default: data/tables/OSSGameBench)
Needs:  pandas, numpy, pyarrow. Runtime: under a minute. Each block of output is headed with the paper section or table it supports.

The manual labels (model raters and the human check) cannot be recomputed, so they are embedded below as strings,
one character per item, in the order of the seeded samples that this script draws. The script checks that its draws
select exactly the issues and repositories that were labelled (SHA-256 of the keys) before it uses the labels.
"""
import hashlib, json, re, sys
from math import comb, sqrt
import numpy as np, pandas as pd

D = (sys.argv[1] if len(sys.argv) > 1 else 'data/tables/OSSGameBench').rstrip('/') + '/'
SEED, CAP, MINB, B = 42, 200, 20, 5000

# ---------------------------------------------------------------- embedded labels (Section 3.6 lists who made each)
# bug audit: 1 bug, 0 not a bug, u unclear; 200 issues in audit-draw order
AUDIT_R1 = '1uu101111111101111010111110101111010111u111111111u0111101011u111101101101100u0101011111111u10u111111111111111111u1u011111u11011111u1011110011111u110u011111110101011110111u11111011111101100011111111100'  # Claude Sonnet 5.5
AUDIT_R2 = '11110111111110111101011u110101111010111u111111111u01111010111111101101101100001010111111110101111111111111111111011011111111111111u1011110011111111u1011111110101011110111011111011111101110011111111110'  # Claude Opus 5.5, blind
HUMAN_AUDIT = '111110001u11011u11u0'  # the author, blind; 20 audit issues in the order of the seeded sub-sample
# symptoms: l logic_behaviour, u ui_rendering, c crash_hang, b build_install_env, d data_content, p persistence_io, o other, n nonbug
DEV_GOLD = 'bclupbooccluclublllcblobulobccupppdlclccouuulloudpcodluunbcluculldbpulnloodlcnlcldnolcolclbuullloduu'  # Sonnet
TEST_GOLD = 'uuclcuudlcuuouolbcnuldlcucnlluudlllluoullluclllbullouclldlllblucdlblllulnulcllonodblluoucnllulonuuod'  # Sonnet
TEST_R2 = 'uuclcuudlcuuououbcouldlcucncloudlllluoullluclllcullouclcdlllblucdlbnllulnllculonolblluoucnlcnconuuod'  # Opus, blind
# repository type: g game, t tool; 391 union-set repositories sorted case-insensitively
REPO_MANUAL = 'gggttgtggggtgggttgtgggggggtgggttgtttggtggggtgggggtggtgggggtggtttgggtgtgtggggggggggggggtggtgttgggggggtgggtgtggttgggggggtggttttttggtttgttgtgtgttgttgtggtgggggggggttggggtggggtgggtggggggtggggggttggtgtgggtgggttggggttggggtttgtgtggtgggtgggggtgtgtgtgttggtgggggggtgttttggtttttttttgtgttttgtgtttggggtggtgttgtggtttgtttgttgtttgtgtggggtgtggtgtgggggtgggtgggggttggggggtggggtgtttttggtgggtgtttggggggggttttgtgtg'  # Opus
REPO_R2 = 'gggttgtggggtgggttgtgggggggtgggttgtttggtggggtgggggtggtgggggtggtttgggtgtgtggggggggggggggtggtgtggggggggtgggtgtggttggtggggtttttttttggtttgttgtgtgttgttgtggtgggggggggttggggtggggtgggtggggggtggggtgttggtgtgggtgggttggggttggggtttgtgtggggggtgggggtgtgtgtgttggtgggggggtgttttggttgttttttgtgttttgtgtttggtgtggtgttgtggtttgtttgttgtttgtgtggggtgtggtgtgggggtgggtgggggttggggggtggggtgtttttggtgggtgtttggggggggttttgtgtg'  # Opus, blind
HUMAN_REPO = 'ggttgtgtggtggggggttggtgtgtttgg'  # the author, blind; 30 repositories in the order of the seeded sub-sample
KEYS = {'audit': '6eefcef4bc778590', 'dev': '4678ae6d2698da04', 'test': '2c01b2bdc843dfe6', 'repos': 'd1097435de91f96b'}
CAT = {'l': 'logic_behaviour', 'u': 'ui_rendering', 'c': 'crash_hang', 'b': 'build_install_env', 'd': 'data_content',
       'p': 'persistence_io', 'o': 'other', 'n': 'nonbug'}
TYPE = {'g': 'game', 't': 'tool'}

def check_keys(name, keys):  # the seeded draw must select exactly the labelled items
    assert hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:16] == KEYS[name], f'{name}: sample differs from the labelled one'
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{k}/{n} = {k / n:.2f} [{c - h:.2f}, {c + h:.2f}]'
def kappa(a, b):  # Cohen's kappa
    a, b = pd.Series(list(a)), pd.Series(list(b)); po = (a == b).mean()
    pe = sum(a.value_counts(normalize=True).get(c, 0) * b.value_counts(normalize=True).get(c, 0) for c in set(a) | set(b))
    return po, (po - pe) / (1 - pe)
assert kappa('aabb', 'aabb') == (1.0, 1.0) and abs(kappa('ab', 'ba')[1] + 1) < 1e-9
def head(t): print(f'\n{"=" * 100}\n{t}\n{"=" * 100}')
pct = lambda x: f'{100 * x:.1f}%'

# ---------------------------------------------------------------- 3.1 data
head('3.1 Design and data')
proj = pd.read_parquet(D + 'project.parquet', columns=['project', 'topics', 'languages'])
iss = pd.read_parquet(D + 'issue.parquet', columns=['project', 'number', 'title', 'body', 'labels', 'created_at', 'contributor'])
pr = pd.read_parquet(D + 'pr.parquet', columns=['project', 'number', 'merged_at'])
ip = pd.read_parquet(D + 'issue_pr.parquet')
print(f'repositories {len(proj):,}  issues {len(iss):,}  pull requests {len(pr):,}  issue-PR links {len(ip):,}')
# 20 repositories the dataset authors exclude (`duplicated_repos_v1` in their analysis-2.ipynb): old copies of renamed
# or transferred repositories that are also in the dataset.
DUPLICATE_REPOS = {r.lower() for r in [
    'Dragon-Fox-Collective/SBEPIS', 'lVlyke/starfield-mod-loader', 'kyh/vibed-games', 'lexi-the-cute/catgirl-engine',
    'benpollarduk/BP.AdventureFramework', 'MathiasGruber/TheNinjaRPG', 'alcalyn/hex', 'sefinek24/genshin-impact-reshade',
    'TeamFluXis/fluXis', 'alextangxiao/tengine', 'srmeier/knightonline', 'zerulight/erupe', 'dylan-power/dinosaur-exploder',
    'minetest/minetest_game', 'mihailris/voxelengine-cpp', 'LostArtefacts/TR1X', 'joshzcold/cold-family-feud',
    'fundament-software/feathergui', 'themeparkworld/opentpw', 'draftbot-a-discord-adventure/draftbot']}
print('duplicate-copy repositories excluded:', proj.project.str.lower().isin(DUPLICATE_REPOS).sum())
key = iss.title.fillna('') + '\0' + iss.body.fillna('').str.slice(0, 500) + '\0' + iss.created_at.fillna('').astype(str) + '\0' + iss.contributor.fillna('').astype(str)
nrep = iss.groupby(key).project.nunique()
print(f'issues (same title, body, creation time, author) that appear in more than one repository: {key.map(nrep).gt(1).sum():,}')

# ---------------------------------------------------------------- 3.2 bug rules + audit
head('3.2 Identifying bugs')
BUG_RE = re.compile(r'bug|defect|crash|regression|glitch', re.I)
iss['b_label'] = iss.labels.apply(lambda ls: any(BUG_RE.search(l) for l in (ls if ls is not None else [])))
merged = pr[pr.merged_at.notna()].rename(columns={'number': 'pr_number'})[['project', 'pr_number']]
fixed = ip.merge(merged, on=['project', 'pr_number'])[['project', 'issue_number']].drop_duplicates()
iss = iss.merge(fixed.rename(columns={'issue_number': 'number'}).assign(b_pr=True), on=['project', 'number'], how='left')
iss['b_pr'] = iss.b_pr.fillna(False).astype(bool)
iss['rule'] = np.select([iss.b_label & iss.b_pr, iss.b_label, iss.b_pr], ['both', 'label_only', 'pr_only'], 'none')
print({k: f'{v:,}' for k, v in iss.rule.value_counts().items()})
audit = pd.concat([iss[iss.rule == r].sample(n, random_state=SEED) for r, n in [('label_only', 70), ('pr_only', 70), ('both', 60)]])
audit = audit.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffled: raters never saw the rule
check_keys('audit', audit.project + '#' + audit.number.astype(str))
audit['r1'], audit['r2'] = list(AUDIT_R1), list(AUDIT_R2)

# analysis sets: repositories with >= 20 bug issues, at most 200 sampled issues each, duplicate copies excluded
def analysis_set(rules):
    b = iss[iss.rule.isin(rules)]
    n = b.groupby('project').size(); b = b[b.project.isin(n[n >= MINB].index)]
    return b.sample(frac=1, random_state=SEED).groupby('project').head(CAP)
prim_dup = analysis_set(['label_only', 'both'])  # the dev/held-out samples were drawn before the exclusion
prim = prim_dup[~prim_dup.project.str.lower().isin(DUPLICATE_REPOS)]
union = analysis_set(['label_only', 'both', 'pr_only']); union = union[~union.project.str.lower().isin(DUPLICATE_REPOS)]
print(f'label rule: {prim.project.nunique()} repositories, {len(prim):,} issues | union rule: {union.project.nunique()} repositories, {len(union):,} issues')

# ---------------------------------------------------------------- 3.3 classifier (rules frozen after dev tuning)
RULES = [
 ('nonbug', r'\bcve-\d{4}|whitesource|\bci\b|\btests?\b.*\b(hang|flaky)|assertTimeout|doctest'),
 ('build_install_env', r'\bcompil|\bbuild (fail|error)|\bcmake\b|\bninja\b|\bmeson\b|\bmake\b.*error|\binstall(ing|ation)?\b|\bdpkg\b|\bapt(-get)?\b|\.dll\b|missing: lib|importerror|no module named|dependenc|linker|\bgcc\b|\bglslc\b|failed to (start|launch)|fails? to start|won.?t (start|launch)|create session failed|not found'),
 ('persistence_io', r'\bsav(e|es|ed|ing)\b|\bload(ing)? (a )?(save|game)|\bsave game|doesn.?t save|not sav|\bconfig\b.*(revert|sav|reset)|\bexport|\bimport(ing)?\b|\bserializ|\bcorrupt|\.json\b|\bsettings? (are |is )?(not|reset|revert|lost)'),
 ('crash_hang', r'crash|segfault|sigsegv|sigfpe|sigabrt|core dumped|exception|\bnpe\b|nullpointer|null pointer|nullreference|traceback|stack ?trace|assertion|\bfreez|\bfroze|\bhang|unresponsive|not respond|panic|typeerror|\bfatal\b|\bboom\b|stuck|runtime error|répond plus|ne répond|provoque une erreur|エラー|ошибк'),
 ('network_input_audio_perf', r'\bsync(ed|hroni[sz])?\b|\bdesync|multiplayer|\bserver\b.*(connect|join)|\bnetwork|\blag(gy)?\b|\bfps\b|frame ?rate|slow(down| performance)?\b|performance|memory (leak|usage)|\bleak|\baudio\b|\bsound|\bmusic\b|\bvolume|\bkeyboard|\bkeybind|\bkeys?\b.*(press|hold|don.?t)|\bmouse\b|\bcontroller\b|\bgamepad|\bjoystick|\binput\b|\bhotkey|\bshortcut|\bclick(ing)?\b'),
 ('data_content', r'\btypo|опечатк|\btranslat|\blocali[sz]|\bwording|\bspelling|\bgrammar|\bmissing (text|string|texture|file|asset|dialog)|\btext\b.*(wrong|incorrect|missing)|\bdescription\b.*(wrong|incorrect)|\bdata\b.*(wrong|incorrect|missing)|read wrongly|incorrect (value|default)|default (value|limit)'),
 ('ui_rendering', r'\brender|\bdraw(n|ing)?\b|\bsprite|\btexture|\bshader|\bgraphic|\bvisual|\bglitch|\bflicker|\bmisalign|\boverlap|\bclip(ping|ped)?\b|\banimation|\bscreen\b|\bmenu\b|\bbutton|\btooltip|\bwindow\b|\bdialog|\bgui\b|\bui\b|\bicon|\bfont\b|\bcolou?r|\btransparen|\bblack (screen|bar)|\bpopup|\boverlay|\bdisplay|\bresolution|\bzoom|\bscroll|\bhud\b|\bcamera|\bimage\b|\bgraph\b|\bparticle|\boutfit|\bstyled?\b|\bstyling'),
]
FROZEN = '7593f23f3737f8aeb446a328c818f8673791f5736d49ae93d6012b5b30f4b409'  # SHA-256 recorded at the freeze
assert hashlib.sha256(json.dumps([[c, p] for c, p in RULES] + ['logic_behaviour']).encode()).hexdigest() == FROZEN
RX = [(c, re.compile(p, re.I)) for c, p in RULES]
def classify(title, body):  # title first, then the first 1,500 characters of the body; first match wins
    for text in (title or '', (body or '')[:1500]):
        for cat, rx in RX:
            if rx.search(text): return 'other' if cat == 'network_input_audio_perf' else cat
    return 'logic_behaviour'  # fallback
COARSE = {'data_content': 'other', 'persistence_io': 'other'}
CATS = ['logic_behaviour', 'ui_rendering', 'crash_hang', 'build_install_env', 'other']

dt = prim_dup.sample(200, random_state=SEED)
dev, test = dt.iloc[:100].copy(), dt.iloc[100:].copy()
check_keys('dev', dev.project + '#' + dev.number.astype(str)); check_keys('test', test.project + '#' + test.number.astype(str))
for d, g in [(dev, DEV_GOLD), (test, TEST_GOLD)]:
    d['gold'] = [CAT[x] for x in g]; d['pred'] = [classify(t, b) for t, b in zip(d.title, d.body)]
test['r2'] = [CAT[x] for x in TEST_R2]

# ---------------------------------------------------------------- 3.4 repository types
TOOL_TOPICS = {'game-engine', 'engine', 'framework', 'library', 'sdk', 'tool', 'tools', 'toolkit', 'editor', 'game-framework',
               'engine-3d', 'game-library', 'map-editor', 'level-editor', 'api', 'server', 'launcher', 'emulator',
               'game-development-tools', 'middleware'}
topics = proj.topics.fillna('').str.split(';').apply(lambda ts: {t.strip().lower() for t in ts if t.strip()})
proj['topic'] = ['tool' if t & TOOL_TOPICS else 'game' for t in topics]
repos = pd.DataFrame({'project': sorted(union.project.unique(), key=str.lower)})
check_keys('repos', repos.project)
repos['manual'], repos['r2'] = [TYPE[x] for x in REPO_MANUAL], [TYPE[x] for x in REPO_R2]
types = proj.set_index('project').topic.rename('topic').to_frame()
for col in ['manual', 'r2']:  # repositories outside the union set (only in the held-out sample) fall back to the topic rule
    types[col] = repos.set_index('project')[col].reindex(types.index).fillna(types.topic)

# ---------------------------------------------------------------- 3.6 / 4.1 validity of the labels, Table 1
head('4.1 Validity of the labels (and Table 1)')
print('rule precision, unclear counted as not a bug (Wilson 95% CI):')
for r in ['label_only', 'both', 'pr_only']:
    g = audit[audit.rule == r]; print(f'  {r:10s} rater 1 {wilson((g.r1 == "1").sum(), len(g))}   rater 2 {wilson((g.r2 == "1").sum(), len(g))}')
po, k = kappa(audit.r1, audit.r2); print(f'audit, model vs model (3 classes): agreement {po:.3f}, kappa {k:.2f}')
ha = audit.sample(20, random_state=SEED).copy(); ha['h'] = list(HUMAN_AUDIT)
for c, n in [('r1', 'rater 1'), ('r2', 'rater 2')]:
    po3, k3 = kappa(ha.h, ha[c]); po2, k2 = kappa(ha.h.replace('u', '0'), ha[c].replace('u', '0'))
    print(f'audit, human vs {n}: 3 classes agreement {po3:.2f} kappa {k3:.2f} | unclear = not bug: agreement {po2:.2f} kappa {k2:.2f}')
lr = ha[ha.rule.isin(['label_only', 'both'])]
print(f'human check, label-rule issues called bugs: human {(lr.h == "1").sum()}/{len(lr)}, rater 1 {(lr.r1 == "1").sum()}, rater 2 {(lr.r2 == "1").sum()}; human precision {wilson((lr.h == "1").sum(), len(lr))}')
po, k = kappa(test.gold, test.r2); print(f'held-out symptoms, model vs model (7 classes): agreement {po:.2f}, kappa {k:.2f}')
print(f'classifier accuracy: development set {(dev.gold == dev.pred).mean():.2f}, held-out set {(test.gold == test.pred).mean():.2f}')
print('held-out set, per category:')
for c in sorted(set(test.gold) | set(test.pred)):
    tp = ((test.gold == c) & (test.pred == c)).sum(); P = (test.pred == c).sum(); R = (test.gold == c).sum()
    print(f'  {c:18s} true {R:3d} predicted {P:3d} precision {tp / P if P else float("nan"):.2f} recall {tp / R if R else float("nan"):.2f}')
po, k = kappa(repos.manual, repos.r2); print(f'repository type, model vs model: agreement {po:.3f}, kappa {k:.2f} (n={len(repos)})')
hr = repos.sample(30, random_state=SEED).copy(); hr['h'] = [TYPE[x] for x in HUMAN_REPO]
for c, n in [('manual', 'manual labels'), ('r2', 'blind rater')]:
    po, k = kappa(hr.h, hr[c]); print(f'repository type, human vs {n}: agreed {(hr.h == hr[c]).sum()}/30, kappa {k:.2f}')
tp = types.reindex(repos.project)
hit = ((tp.topic == 'tool') & (tp.manual == 'tool')).sum()
print(f'topic rule against manual labels: recall {hit}/{(tp.manual == "tool").sum()} = {hit / (tp.manual == "tool").sum():.2f}, precision {hit}/{(tp.topic == "tool").sum()} = {hit / (tp.topic == "tool").sum():.2f}')

# ---------------------------------------------------------------- symptom shares and game-vs-tool comparison
def classified(s):
    a = s[['project', 'number', 'title', 'body', 'labels']].copy()
    a['cat'] = [classify(t, b) for t, b in zip(a.title, a.body)]
    a = a[a.cat != 'nonbug'].copy(); a['cat'] = a.cat.replace(COARSE); return a
def compare(a, type_col, label):  # difference in mean per-repository share (tool - game), repository bootstrap + Holm
    share = pd.crosstab(a.project, a.cat, normalize='index').reindex(columns=CATS, fill_value=0)
    rt = types[type_col].reindex(share.index)
    rng = np.random.default_rng(SEED); g, t = share[rt == 'game'].values, share[rt == 'tool'].values
    d = np.array([t[rng.integers(0, len(t), len(t))].mean(0) - g[rng.integers(0, len(g), len(g))].mean(0) for _ in range(B)])
    obs = t.mean(0) - g.mean(0)
    p = np.minimum(1, 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)))  # two-sided bootstrap p
    holm = np.empty(5); run = 0
    for r, i in enumerate(np.argsort(p)): run = max(run, (5 - r) * p[i]); holm[i] = min(1, run)
    print(f'\n{label}: {len(g)} game / {len(t)} tool repositories')
    print(f'  {"category":18s} {"game":>6s} {"tool":>6s}  difference [95% CI]    Holm p   Bonferroni 99% CI')
    for i, c in enumerate(CATS):
        hp = '<0.001' if holm[i] == 0 else f'{holm[i]:.3f}'
        print(f'  {c:18s} {pct(g.mean(0)[i]):>6s} {pct(t.mean(0)[i]):>6s}  {100 * obs[i]:+.1f} [{100 * np.percentile(d[:, i], 2.5):+.1f}, {100 * np.percentile(d[:, i], 97.5):+.1f}]   {hp:7s} [{100 * np.percentile(d[:, i], .5):+.1f}, {100 * np.percentile(d[:, i], 99.5):+.1f}]')
    return share, rt

A = classified(prim)
head('4.2 Symptom distribution (Table 2, pooled and median columns)')
pooled = A.cat.value_counts(normalize=True).reindex(CATS)
share0 = pd.crosstab(A.project, A.cat, normalize='index').reindex(columns=CATS, fill_value=0)
print(f'classified bug issues: {len(A):,} (predicted non-bugs excluded: {len(prim) - len(A)})')
for c in CATS: print(f'  {c:18s} pooled {pct(pooled[c])}  median per repository {pct(share0[c].median())}')
print(f'logic + UI pooled: {pct(pooled.logic_behaviour + pooled.ui_rendering)}')

head('4.3 Games versus tools (Table 3, primary columns, and the sensitivity analyses)')
share, rt = compare(A, 'manual', 'PRIMARY: label rule, manual repository types')
compare(classified(union), 'manual', 'union rule, manual repository types')
compare(A, 'topic', 'label rule, topic-rule repository types')
compare(A, 'r2', "label rule, blind second rater's repository types")

# ---------------------------------------------------------------- correction for classifier error (Algorithm 1)
head('4.2 / 4.3 Correction for classifier error (Table 2 corrected column, Table 3 corrected column)')
rng = np.random.default_rng(SEED)
G, T = share[rt == 'game'].values, share[rt == 'tool'].values
q = pooled.values
GOLD = CATS + ['nonbug']
t = test[['project', 'gold', 'pred']].replace(COARSE)
t = t[t.pred != 'nonbug'].merge(types.manual.rename('repo_type'), left_on='project', right_index=True).reset_index(drop=True)
def cmat(d):  # rows: predicted category; columns: P(true category | predicted)
    return pd.crosstab(d.pred, d.gold, normalize='index').reindex(index=CATS, columns=GOLD, fill_value=0).to_numpy(copy=True)
def correct(x, C):  # predicted shares -> corrected shares among true bugs (non-bug dropped, renormalised)
    y = (x @ C)[..., :len(CATS)]; return y / y.sum(-1, keepdims=True)
C0 = cmat(t); pool0 = correct(q, C0); diff0 = correct(T, C0).mean(0) - correct(G, C0).mean(0)
bp, bd = [], []
for _ in range(B):  # resample the held-out pairs and the repositories together
    C = cmat(t.iloc[rng.integers(0, len(t), len(t))].reset_index(drop=True))
    C[C.sum(1) == 0] = C0[C.sum(1) == 0]
    bp.append(correct(q, C))
    bd.append(correct(T[rng.integers(0, len(T), len(T))], C).mean(0) - correct(G[rng.integers(0, len(G), len(G))], C).mean(0))
bp, bd = np.array(bp), np.array(bd)
ci = lambda m, i: f'[{100 * np.percentile(m[:, i], 2.5):+.1f}, {100 * np.percentile(m[:, i], 97.5):+.1f}]'
print(f'held-out pairs used: {len(t)}')
for i, c in enumerate(CATS): print(f'  {c:18s} pooled corrected {pct(pool0[i])} {ci(bp, i)}   tool - game corrected {100 * diff0[i]:+.1f} {ci(bd, i)}')
print(f'logic + UI corrected: {pct(pool0[0] + pool0[1])}')

# ---------------------------------------------------------------- fallback checks
head('4.3 Fallback checks: keyword-matched issues, fallback precision, text length')
m = share[CATS[1:]].div(share[CATS[1:]].sum(axis=1), axis=0).dropna()  # shares among keyword-matched issues only
mrt = rt.reindex(m.index); Gm, Tm = m[mrt == 'game'].values, m[mrt == 'tool'].values
d = np.array([Tm[rng.integers(0, len(Tm), len(Tm))].mean(0) - Gm[rng.integers(0, len(Gm), len(Gm))].mean(0) for _ in range(B)])
print('tool - game among keyword-matched issues:')
for i, c in enumerate(CATS[1:]): print(f'  {c:18s} {100 * (Tm.mean(0)[i] - Gm.mean(0)[i]):+.1f} {ci(d, i)}')
f = t[t.pred == 'logic_behaviour']; k = f.groupby('repo_type').apply(lambda g: (g.gold == 'logic_behaviour').sum()); n = f.repo_type.value_counts()
print(f'fallback precision on held-out set: games {k.game}/{n.game}, tools {k.tool}/{n.tool}')
N1, N2, K = n.game, n.tool, k.game + k.tool  # two-sided Fisher exact test
ph = lambda x: comb(N1, x) * comb(N2, K - x) / comb(N1 + N2, K)
print(f'  Fisher exact two-sided p = {sum(ph(x) for x in range(max(0, K - N2), min(N1, K) + 1) if ph(x) <= ph(k.game) + 1e-12):.3f}')
x = A.assign(repo_type=types.manual.reindex(A.project).values)
x['len'] = x.title.fillna('').str.len() + x.body.fillna('').str.slice(0, 1500).str.len()
x['f'] = x.cat == 'logic_behaviour'; x['q'] = pd.qcut(x.len, 10, labels=False)
print('median text length (title + first 1,500 characters of body):', x.groupby('repo_type').len.median().to_dict())
raw = x.groupby('repo_type').f.mean(); r = x.pivot_table(index='q', columns='repo_type', values='f', aggfunc='mean')
w = x[x.repo_type == 'game'].q.value_counts(normalize=True).sort_index()
print(f'issue-level fallback (logic) share, tool - game: {100 * (raw.tool - raw.game):+.1f} points; tools reweighted to the game length deciles: {100 * ((r.tool * w).sum() - raw.game):+.1f}')

# ---------------------------------------------------------------- language
head('4.3 Programming language (Table 3 adjusted column)')
prim_lang = proj.set_index('project').languages.map(lambda s: max(json.loads(s), key=json.loads(s).get) if s and json.loads(s) else 'none')
GROUP = {'C': 'C/C++', 'C++': 'C/C++', 'C#': 'C#', 'Java': 'Java/Kotlin', 'Kotlin': 'Java/Kotlin',
         'JavaScript': 'JS/TS', 'TypeScript': 'JS/TS', 'Python': 'Python'}
lang = prim_lang.reindex(share.index).map(lambda l: GROUP.get(l, 'other'))
cc = lang == 'C/C++'
print(f'C/C++ share: tools {(cc & (rt == "tool")).sum()}/{(rt == "tool").sum()} = {pct((cc & (rt == "tool")).sum() / (rt == "tool").sum())}, games {(cc & (rt == "game")).sum()}/{(rt == "game").sum()} = {pct((cc & (rt == "game")).sum() / (rt == "game").sum())}')
print('mean build/install/environment share by language group:', {g: pct(v) for g, v in share.groupby(lang).build_install_env.mean().items()})
X = np.column_stack([np.ones(len(share)), (rt == 'tool').values] + [(lang == g).values for g in ['C/C++', 'C#', 'Java/Kotlin', 'JS/TS', 'Python']]).astype(float)
Y = share.values
coef = lambda idx: np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0][1]  # tool coefficient, OLS with language dummies
b0 = coef(np.arange(len(Y))); bb = np.array([coef(rng.integers(0, len(Y), len(Y))) for _ in range(B)])
for i, c in enumerate(CATS): print(f'  {c:18s} adjusted for language {100 * b0[i]:+.1f} {ci(bb, i)}')
for name, msk in [('C/C++', cc), ('not C/C++', ~cc)]:
    s = share[msk].build_install_env; r = rt[msk]
    print(f'build/install/environment within {name}: games {pct(s[r == "game"].mean())}, tools {pct(s[r == "tool"].mean())}, difference {100 * (s[r == "tool"].mean() - s[r == "game"].mean()):+.1f}')

# ---------------------------------------------------------------- 4.4 labels
head('4.4 Symptoms and issue labels')
LABEL_RX = {'crash_hang': r'crash|freeze|hang', 'ui_rendering': r'\bui\b|graphic|render|visual|display|gui|cosmetic',
            'build_install_env': r'build|compil|install|packag'}
lab = A.labels.apply(lambda ls: ' '.join(ls) if ls is not None else '')
for c, rx in LABEL_RX.items():
    g = lab[A.cat == c]; hit = g.str.contains(rx, case=False)
    print(f'  {c:18s} {len(g):,} issues, {hit.sum()} ({pct(hit.mean())}) carry a label naming the symptom')
pc = ((test.gold == 'crash_hang') & (test.pred == 'crash_hang')).sum() / (test.pred == 'crash_hang').sum()
print(f'likely true crash reports: {pc:.2f} x {(A.cat == "crash_hang").sum():,} = {pc * (A.cat == "crash_hang").sum():,.0f}')
