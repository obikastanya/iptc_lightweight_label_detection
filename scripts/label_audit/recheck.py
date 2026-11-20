"""Post-hoc recheck (sensitivity analysis) of the article-level label audit.

Unblinded reread of every disagree item whose final status is "wrong" or "both_acceptable". The
judgments below were made by reading the article text (original language) together with the section
name, mapped label, Annotator A, adjudicator and human decision. Official decisions are unchanged;
this script only writes
    results/label_audit/recheck_disagreements.csv
    results/label_audit/recheck_summary.md
and reuses the estimator in analyze.py (not edited).

    python scripts/label_audit/recheck.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import analyze
from analyze import bootstrap, build_items, estimate
from common import AUDIT_ORIGINS, OUT, ROOT, TEST_PATH, LABELS
from statickd.testset.iptc_map import map_section, normalise

MAP_TABLE = ROOT / "src" / "statickd" / "testset" / "iptc_reference" / "mapping_table.csv"

ART, CON, CRI, DIS, ECO, EDU, ENV, HEA, HUM, LAB, LIF, POL, REL, SCI, SOC, SPO, WEA = LABELS

# item_id: (category, recheck_status, recheck_labels, content_summary, rationale, needs_author)
# recheck_status = what the editorial label would be under this reread (correct / both_acceptable / wrong).
J = {
    "A004": ("true_mismatch", "wrong", [SOC, POL],
             "Punjab Scheduled Castes Commission summons MP Ravneet Bittu over a complaint about an objectionable viral video.",
             "Caste-rights commission hearing against an MP; nothing religious in the text. 'Religious' maps correctly; this article was filed there.", False),
    "A012": ("true_mismatch", "wrong", [DIS],
             "Eleven-year-old boy killed when a dilapidated compound wall collapses in Davanagere; case filed against owner.",
             "Accidental death; the police case against the house owner is a one-line follow-up.", False),
    "A017": ("true_mismatch", "wrong", [POL],
             "Electoral court approves Campo Grande mayor's 2024 campaign accounts after irregularities were corrected.",
             "Campaign finance is IPTC politics > election > campaign finance; the court approval is routine election administration, no offence or trial.", False),
    "A021": ("section_acceptable", "both_acceptable", [ART, HEA],
             "Two students hold self-made paintings carrying a stay-home COVID-19 prevention message.",
             "Concur with both: the paintings are the story, the COVID message is the theme.", False),
    "A027": ("unclear", "wrong", [LIF, ECO],
             "Tips: air conditioner on heat mode uses less electricity than an electric heater, lowering winter bills.",
             "Household appliance advice framed as saving on the electricity bill; no IPTC node fits well; economy as co-topic is arguable either way.", True),
    "A028": ("true_mismatch", "wrong", [DIS],
             "Car ploughs into fans celebrating Barcelona's La Liga title, injuring 15; driver arrested.",
             "The crash is the event; the title win is the occasion plus one context paragraph.", False),
    "A032": ("annotation_error", "correct", [HUM, ART],
             "Survivor contestant and a rival contestant's husband trade insults on Instagram after her elimination.",
             "Personal social-media feud between reality-TV personalities: IPTC human interest > celebrity ('life and behaviour of famous people'). The show itself is only backstory.", True),
    "A034": ("true_mismatch", "wrong", [HEA, ART],
             "Punjab closes museums, including Virasat-e-Khalsa, until April because of rising COVID-19 cases.",
             "Pandemic closure of museums; the Sikh heritage museum and Hola Mohalla appeal are incidental, not religion.", False),
    "A036": ("true_mismatch", "wrong", [LIF],
             "Le Rove restricts vehicle access to two calanques on summer weekends; residents need passes.",
             "Access rule to manage summer crowds and traffic; no environmental protection content.", False),
    "A037": ("true_mismatch", "wrong", [ART],
             "Cast of Punjabi film Kali Jotta promotes the movie at CT University.",
             "Film promotion; the university is only the venue.", False),
    "A046": ("true_mismatch", "wrong", [LIF],
             "Bois des Naix park recommended as a cool place for walks, play and sport during a heatwave.",
             "Leisure outing guide (IPTC park and playground); nature is description, not environmental news.", False),
    "A050": ("true_mismatch", "wrong", [ART, HUM],
             "LE SSERAFIM's Sakura responds to criticism of the group's Coachella vocals.",
             "The subject is a music performance and its reviews; celebrity framing is secondary.", False),
    "A052": ("section_acceptable", "both_acceptable", [ENV, ECO],
             "Cabinet approves CITIIS 2.0 urban programme funded by AFD/KfW loans and EU grant for waste management.",
             "Concur with both: loan-financed investment programme whose content is waste management and climate.", False),
    "A055": ("true_mismatch", "wrong", [CRI],
             "Interior minister reports traffic penalties issued after helicopter and drone inspections.",
             "Traffic law enforcement statistics; the minister is only the source.", False),
    "A061": ("true_mismatch", "wrong", [POL, ECO],
             "Profile of oil magnate Harold Hamm's influence on Trump's Israel, Iran and energy policy.",
             "Lobbying and policy influence; the Iran war is one outcome mentioned, not reported as conflict news.", False),
    "A071": ("section_acceptable", "both_acceptable", [POL, EDU],
             "New female deputy minister of education appointed; minister outlines her portfolio and sector reform tasks.",
             "Government appointment, but half the text is education-sector reform and her education portfolio; education is a fair co-topic.", False),
    "A073": ("section_acceptable", "both_acceptable", [HUM, ART],
             "Viral clip: woman confronts talk-show host Rocío Sánchez Azuara on air for snubbing neighbours.",
             "About the host's personal behaviour (celebrity) on her own TV show; both labels fit equally.", False),
    "A074": ("true_mismatch", "wrong", [ENV],
             "Automated Krakow air-quality report with PM2.5 and PM10 readings and norms.",
             "Air pollution is IPTC environment > environmental pollution; not meteorology.", False),
    "A081": ("section_acceptable", "both_acceptable", [HEA, LIF],
             "Three vegetable juices recommended for losing belly fat within a week.",
             "Weight-loss diet advice fits both health > diet and lifestyle > wellness.", False),
    "A092": ("true_mismatch", "wrong", [HEA],
             "WHO warns of camel flu (MERS) risk during the Qatar World Cup; symptoms and precautions.",
             "Disease warning; the World Cup is only the setting.", False),
    "A098": ("true_mismatch", "wrong", [LAB, POL],
             "BKN opens national portal for PPPK government contract-employee registration; document requirements.",
             "Public-sector recruitment notice; no business or economic content.", False),
    "A100": ("true_mismatch", "wrong", [CON, POL],
             "Quota-reform protests spread to public universities; class boycotts, road blockades, clashes with police.",
             "Protest movement over government job quotas (IPTC civil unrest); campuses are the setting.", False),
    "A101": ("section_acceptable", "both_acceptable", [CRI, HUM],
             "Actor William Levy arrested after refusing to pay a restaurant bill; bail set by judge.",
             "Concur with both: arrest and charges of a celebrity.", False),
    "A119": ("true_mismatch", "wrong", [LAB],
             "Sucofindo laboratory job vacancies: seven positions, requirements and how to apply.",
             "Job advertisement (labour > employment).", False),
    "A121": ("true_mismatch", "wrong", [LIF],
             "Making lampshades at home instead of buying expensive ones for hall decoration.",
             "Home decor (IPTC lifestyle > house and home > interior decoration), not fashion.", False),
    "A124": ("true_mismatch", "wrong", [SPO],
             "Hardik Pandya turns 30: career facts and five memorable cricket performances.",
             "Pure cricket; no politics.", False),
    "A125": ("true_mismatch", "wrong", [SOC],
             "Boy scolded over mobile gaming dies by suicide; father kills himself minutes later.",
             "Suicide is IPTC society > death and dying; no offence, the police case is formal.", False),
    "A126": ("true_mismatch", "wrong", [ECO, HEA],
             "Pharmacy cooperative PROSYFAPE's chairman presents its model, services and CSR at a health conference.",
             "Corporate presentation of a pharma distributor; health remarks support the business story.", False),
    "A127": ("true_mismatch", "wrong", [CRI],
             "Ministry of Justice explains Ukrainian inheritance law and children's compulsory share.",
             "Civil law explainer (IPTC crime, law and justice > law); no economic content.", False),
    "A129": ("true_mismatch", "wrong", [LAB, POL],
             "Government plans fixed pension of 40-45% of last salary for employees by reforming NPS.",
             "Pension is IPTC labour > retirement > pension or politics > pension policy; the investment-return line is minor.", False),
    "A134": ("true_mismatch", "wrong", [CRI],
             "Former Derbyshire cricketer jailed for killing a cyclist while drunk-driving.",
             "Court sentence for a driving offence; cricket only identifies the person.", False),
    "A137": ("true_mismatch", "wrong", [CON, CRI],
             "Police raid Chittagong University halls after rival Chhatra League factions clash; weapons seized.",
             "Politically motivated campus violence and police action; nothing about education.", False),
    "A139": ("section_acceptable", "both_acceptable", [CRI, POL],
             "Pensioner injured by police at pension protest files criminal complaint against minister Bullrich.",
             "Criminal complaint against a sitting security minister after a pension protest; politics is a substantial co-topic.", False),
    "A142": ("true_mismatch", "wrong", [ECO],
             "Binance leaves Russia at end of January and stops ruble support, selling to CommEX.",
             "Cryptocurrency exchange market exit (IPTC economy > cryptocurrency).", False),
    "A145": ("annotation_error", "correct", [ART],
             "Vogue-based guide to preppy-style shoes in fashion for autumn 2025.",
             "Footwear trends are IPTC arts > fashion ('styles and trends in clothing, footwear'); lifestyle has no fashion node.", False),
    "A149": ("true_mismatch", "wrong", [CRI],
             "Kaspersky finds evidence linking North Korean hackers to the Bangladesh Bank reserve heist.",
             "Cybercrime attribution; the central bank is the victim, not the topic.", False),
    "A160": ("section_acceptable", "both_acceptable", [LAB, POL],
             "Deputy minister answers MPs' questions in parliament on bus drivers' working hours after a crash.",
             "Parliamentary question time on drivers' working hours; parliament and minister make politics a co-topic.", False),
    "A162": ("section_acceptable", "both_acceptable", [HEA, ECO],
             "Jokowi opens red palm cooking oil plant; product described as cheaper and nutritious, anti-stunting.",
             "Concur with both: nutrition content and downstreaming/price.", False),
    "A166": ("section_acceptable", "both_acceptable", [CRI, CON],
             "Parliament backs bill dropping criminal liability for soldiers' first desertion if they return.",
             "Criminal-law change limited to wartime military desertion; the war/military dimension is a fair co-topic.", False),
    "A176": ("true_mismatch", "wrong", [POL, CON],
             "Iran's foreign minister makes rare visit to Egypt to discuss regional escalation.",
             "Diplomacy wire story filed under 'Aktien' by a finance portal; no economic content.", False),
    "A177": ("true_mismatch", "wrong", [HEA, SOC],
             "Church-run San Camilo centre provides palliative care; award from Red Cross; calls for palliative law.",
             "Palliative care is health; church affiliation and pro-life framing are brief.", False),
    "A181": ("annotation_error", "correct", [ART],
             "Tips for choosing and wearing high heels without foot pain or shoe bite.",
             "Wearing footwear styles is IPTC arts > fashion; Annotator A's lifestyle has no fitting node.", False),
    "A185": ("true_mismatch", "wrong", [ENV],
             "Young mako shark sighted off Porto Cesareo; framed as sign of marine biodiversity.",
             "Wildlife sighting and biodiversity (environment > nature); no scientific research reported.", False),
    "A189": ("section_acceptable", "both_acceptable", [HUM, ART],
             "Johnny Depp recalls being frightened by sudden fame after Edward Scissorhands, in a Tim Burton documentary.",
             "His experience of fame is celebrity (human interest); the film and documentary are arts. Both fit.", False),
    "A191": ("true_mismatch", "wrong", [LAB],
             "Pelindo II (IPC) opens outsourced administrator recruitment at head office and seven branches.",
             "Job advertisement.", False),
    "A196": ("true_mismatch", "wrong", [LAB],
             "UKSSSC releases admit cards for forest constable physical efficiency test.",
             "Government job recruitment test, not education.", False),
    "A198": ("section_acceptable", "both_acceptable", [ART, POL],
             "Oscars ceremony: Jimmy Kimmel mocks Donald and Melania Trump; political speeches.",
             "Concur with both.", False),
    "A203": ("true_mismatch", "wrong", [WEA],
             "Siberian cold grips Ardahan; Göle records minus 19 degrees.",
             "Temperature report is weather.", False),
    "A205": ("true_mismatch", "wrong", [DIS],
             "Goods vehicle carrying pilgrims hits a tree near Ramdurg, killing six.",
             "Road accident.", False),
    "A219": ("true_mismatch", "wrong", [LAB],
             "Russian-crewed fishing vessel arrested in Namibia over unpaid wages; embassy assists.",
             "Wage dispute is the cause and subject; embassy support is consular, not politics.", False),
    "A220": ("true_mismatch", "wrong", [POL],
             "Modi opens G20 parliamentary speakers' summit, praising Indian democracy.",
             "Politics; nothing about cinema.", False),
    "A222": ("section_acceptable", "both_acceptable", [ART, POL],
             "Dozens of reporters leave the Pentagon after refusing Hegseth's new press-access rules.",
             "Government restriction of press access: media and government policy weigh equally (adjudicator also said both).", True),
    "A231": ("section_acceptable", "both_acceptable", [POL, ECO],
             "Parliamentary fisheries research commission completes work; sector problems and demands listed.",
             "Concur with both.", False),
    "A237": ("section_acceptable", "both_acceptable", [CRI, SOC],
             "Child bride elopes with husband's nephew; arrest on husband's complaint; child welfare committee involved.",
             "Concur with both.", False),
    "A249": ("true_mismatch", "wrong", [HEA],
             "Seven health benefits of mango, citing Cleveland Clinic dietitians.",
             "Nutrition advice; filed under infobae /salud/ciencia.", False),
    "A260": ("true_mismatch", "wrong", [CRI],
             "Madras High Court denies anticipatory bail in POCSO case, stresses child's welfare.",
             "Court ruling in sexual-abuse case; nothing about cinema.", False),
    "A264": ("true_mismatch", "wrong", [EDU],
             "Education minister presents Princess Maha Chakri awards to teachers and praises their role.",
             "Teacher awards ceremony; the minister presides, no political content.", False),
    "A265": ("true_mismatch", "wrong", [CON],
             "Iranian army drones strike US bases in Kuwait and Bahrain.",
             "Military attack.", False),
    "A269": ("true_mismatch", "wrong", [ART],
             "K-pop group Apink announces seventh solo concert at Christmas.",
             "Concert announcement is music (arts); nothing about personal life.", False),
    "A283": ("true_mismatch", "wrong", [ECO],
             "Krishnagiri okra farmers lose income as oversupply drops price to Rs 10 per kg.",
             "Agricultural prices (economy > agriculture); no politics.", False),
    "A288": ("section_acceptable", "both_acceptable", [SOC, LIF],
             "Athens residents protest overtourism with anti-tourist graffiti amid housing crisis.",
             "Overtourism is about tourism itself; social protest is primary but travel and tourism is a real co-topic.", False),
    "A291": ("true_mismatch", "wrong", [ENV],
             "Migratory birds arrive at Jahangirnagar University lakes; species counts, bird awareness fair.",
             "Wildlife (environment > nature); the campus is only the location.", False),
}

CATS = ["true_mismatch", "mapping_error", "section_acceptable", "annotation_error", "unclear"]
VALID = ("correct", "wrong", "both_acceptable")


def sections_of(r) -> list[str]:
    s = r["source_label"]
    return json.loads(s) if isinstance(s, str) and s.startswith("[") else [s]


def trigger_names(r) -> list[str]:
    """Section names of this article that the mapping table maps to its label (normalised)."""
    if r["source"] != "ccnews_jsonld":
        return [normalise(r["source_label"])]
    out = []
    for s in sections_of(r):
        if map_section(s, r["lang"]) == r["label"] and normalise(s) not in out:
            out.append(normalise(s))
    return out


def main():
    full = pd.read_json(TEST_PATH, lines=True, dtype={"domain": str, "url": str})
    pop = full[full["label_origin"].isin(AUDIT_ORIGINS)].copy()
    analyze.POP = pop["source"].value_counts().to_dict()

    df = build_items()
    sel = df[df["status"].isin(["wrong", "both_acceptable"])].copy()
    assert set(sel["item_id"]) == set(J), set(sel["item_id"]) ^ set(J)
    assert all(v[0] in CATS and v[1] in VALID for v in J.values())

    # test-set index: (normalised section name, label) for CC-NEWS; (source, lang, category) for the others
    cc = pop[pop["source"] == "ccnews_jsonld"]
    cc_names = [{normalise(s) for s in (json.loads(x) if x.startswith("[") else [x])} for x in cc["source_label"]]

    def n_same(r):
        if r["source"] != "ccnews_jsonld":
            m = (pop["source"] == r["source"]) & (pop["lang"] == r["lang"]) & (pop["source_label"] == r["source_label"])
            return int(m.sum())
        names = set(trigger_names(r))
        return int(sum(bool(names & ns) and lab == r["label"] for ns, lab in zip(cc_names, cc["label"])))

    rows = []
    for _, r in sel.sort_values("item_id").iterrows():
        cat, rs, labs, summ, rat, need = J[r["item_id"]]
        assert len(summ.split()) <= 20, (r["item_id"], len(summ.split()))
        assert len(rat.split()) <= 30, (r["item_id"], len(rat.split()))
        assert all(l in LABELS for l in labs)
        # consistency of recheck_status with recheck_label
        if rs == "correct":
            assert labs[0] == r["label"], r["item_id"]
        elif rs == "both_acceptable":
            assert r["label"] in labs, r["item_id"]
        if cat == "true_mismatch":
            assert rs == "wrong" and labs[0] != r["label"], r["item_id"]
        secs = [r["source_label"]]
        if r["source"] == "ccnews_jsonld":
            secs = list(dict.fromkeys(s for s in sections_of(r) if len(normalise(s)) <= 40))
        d = r["adj_decision"]
        if d in ("option_1", "option_2"):
            origin = r[f"{d}_origin"]
            won = r["label"] if origin == "section" else r["A_primary"]
            adj = f"{'section' if origin == 'section' else 'llm_A'} wins ({won}), {r['adj_confidence']}"
        else:
            adj = f"{d}, {r['adj_confidence']}"
        if r["routed"]:
            h = str(r["h_decision"]).strip()
            if h.upper() in ("A", "B"):
                hd = f"{r[f'option_{h.upper()}']} ({r[f'option_{h.upper()}_origin']})"
            else:
                hd = h.lower()
        else:
            hd = "not routed (adjudicator final)"
        rows.append({
            "item_id": r["item_id"], "doc_id": r["doc_id"], "source": r["source"], "lang": r["lang"],
            "section_name": " ; ".join(secs), "mapped_label": r["label"],
            "llm_primary": r["A_primary"], "llm_secondary": r["A_secondary"] if isinstance(r["A_secondary"], str) else "",
            "adjudication": adj, "human_decision": hd, "final_status": r["status"],
            "content_summary": summ, "recheck_category": cat, "recheck_label": " | ".join(labs),
            "rationale": rat, "needs_author": need,
            "recheck_status": rs, "trigger_section_name": " ; ".join(trigger_names(r)),
            "n_test_same_section_and_label": n_same(r),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "recheck_disagreements.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ sensitivity
    st = df.set_index("item_id")
    st["recheck"] = st["status"]
    st.loc[out["item_id"], "recheck"] = out.set_index("item_id")["recheck_status"]
    # literal rule: every section_acceptable / annotation_error item counted correct
    st["literal"] = st["status"]
    lit = out.loc[out["recheck_category"].isin(["section_acceptable", "annotation_error"]), "item_id"]
    st.loc[lit, "literal"] = "correct"
    st = st.reset_index()

    res = {}
    for col in ("status", "recheck", "literal"):
        e = estimate(st, col)
        e.update(bootstrap(st, col))
        res[col] = e
    assert abs(res["status"]["acc_strict"] - 0.793) < 0.0006, res["status"]["acc_strict"]

    write_summary(out, st, res, pop)
    print("wrote recheck_disagreements.csv, recheck_summary.md")
    for k, v in res.items():
        print(k, f"strict {v['acc_strict']:.4f} {v['acc_strict_ci']}  lenient {v['acc_lenient']:.4f} {v['acc_lenient_ci']}")


def pct(x):
    return f"{100 * x:.1f}%"


def ci(c):
    return f"{100 * c[0]:.1f}-{100 * c[1]:.1f}%"


def write_summary(out, st, res, pop):
    L = ["# Post-hoc recheck of the label-audit disagreements (sensitivity analysis)", "",
         "Generated by `scripts/label_audit/recheck.py`. Official audit decisions are final and unchanged; this "
         "is an unblinded reread of the 62 disagree items whose final status is `wrong` (55) or "
         "`both_acceptable` (7). Each article was reread in its original language with the section name, the "
         "mapped label, Annotator A, the adjudicator and the human decision visible. Item-level judgments: "
         "`recheck_disagreements.csv`.", "",
         f"Mapping table: `src/statickd/testset/iptc_reference/mapping_table.csv` (exported by "
         "`src/statickd/testset/iptc_map.py`; byte-identical to `_archive/code_v1/testset_jsonld/iptc_reference/"
         "mapping_table.csv`). Re-running `map_section` / `iptc_label` on the stored section names reproduces the "
         "label of all 8,650 audit-population articles (6,050 CC-NEWS, 2,600 external), so this is the table the "
         "unified test set was built with.", "",
         "`recheck_status` is the status the editorial label would get under this reread (correct / "
         "both_acceptable / wrong). Concurring with an original `both_acceptable` verdict is recorded as "
         "`section_acceptable` with `recheck_status = both_acceptable` (no change).", ""]

    L += ["## 1. Counts per category", "", "| category | all | " + " | ".join(sorted(out["source"].unique())) + " |",
          "|---|---|" + "---|" * out["source"].nunique()]
    srcs = sorted(out["source"].unique())
    for c in CATS:
        g = out[out["recheck_category"] == c]
        L.append(f"| {c} | {len(g)} | " + " | ".join(str(int((g['source'] == s).sum())) for s in srcs) + " |")
    L.append(f"| total | {len(out)} | " + " | ".join(str(int((out['source'] == s).sum())) for s in srcs) + " |")
    L.append("")
    L.append("Status change implied by the reread (original final status -> recheck_status):")
    L.append("")
    L.append("| original | recheck correct | recheck both_acceptable | recheck wrong |")
    L.append("|---|---|---|---|")
    for o in ("wrong", "both_acceptable"):
        g = out[out["final_status"] == o]
        L.append(f"| {o} ({len(g)}) | " + " | ".join(str(int((g['recheck_status'] == s).sum()))
                                                   for s in ("correct", "both_acceptable", "wrong")) + " |")
    L.append("")
    for c in ("annotation_error", "section_acceptable", "unclear"):
        g = out[out["recheck_category"] == c]
        if c == "section_acceptable":
            g = g[g["final_status"] == "wrong"]
            head = "section_acceptable items that were `wrong` (now both_acceptable)"
        else:
            head = c
        L.append(f"- {head}: " + ", ".join(f"{r.item_id} ({r.mapped_label} vs {r.llm_primary})" for r in g.itertuples()))
    L.append("")

    L += ["## 2. Mapping errors", "",
          "None. In every one of the 62 items the section name itself is mapped to the IPTC topic it names "
          "(e.g. Religious -> religion, Crime -> crime, Fashion -> arts > fashion, Celebrities/明星/Famosos -> "
          "human interest > celebrity, Pogoda -> weather, Наука и технологии -> science and technology). The "
          "mismatches come from where the newsroom or dataset filed the article, not from the mapping table, "
          "so no mapping rule has to be propagated to other test articles.", "",
          "Patterns that recur in the sample and are worth knowing (not mapping errors). Counts are articles in "
          "the 8,650-article audit population carrying the same section/category and mapped label:", ""]
    pats = []
    for r in out.itertuples():
        pats.append((r.source, r.lang if r.source != "ccnews_jsonld" else "", r.trigger_section_name, r.mapped_label,
                     r.n_test_same_section_and_label, r.item_id, r.recheck_status))
    pt = pd.DataFrame(pats, columns=["source", "lang", "section", "label", "n_test", "item", "rs"])
    agg = (pt.groupby(["source", "lang", "section", "label", "n_test"])
           .agg(items=("item", lambda x: ", ".join(x)), n=("item", "size"),
                still_wrong=("rs", lambda x: int((x == "wrong").sum())))
           .reset_index().sort_values(["n", "n_test"], ascending=False))
    n_single = int((agg["n"] == 1).sum())
    agg = agg[agg["n"] >= 2]
    L += ["| source | lang | section / category (normalised) | mapped label | test articles with it | rechecked items | still wrong |",
          "|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        L.append(f"| {r.source} | {r.lang} | {r.section} | {r.label} | {r.n_test} | {r.items} | {r.still_wrong} |")
    L.append("")
    L.append(f"The other {n_single} sections/categories occur in one rechecked item each; their test-set counts are in "
             "the CSV column `n_test_same_section_and_label`.")
    L.append("")
    hi = pop[pop["label"] == HUM]
    hs = st[st["label"] == HUM]
    l3ta = st[(st["source"] == "l3cube_indic") & (st["lang"] == "ta")]
    L += ["Observations:", "",
          f"- Celebrity sections (human interest > celebrity; {len(hi):,} audit-population articles, all CC-NEWS): "
          f"{int((~hs['agree']).sum())} of the {len(hs)} sampled human-interest items were disagreements. Annotator A labelled most celebrity stories "
          "`arts`. On reread, A032 is human interest, A073/A189 are both, A101 both (unchanged); A050 and "
          "A269 are music news filed under a celebrity section. The mapping follows IPTC; the gap is an annotator "
          "tendency plus K-pop/entertainment news in star sections.",
          "- Fashion (arts > fashion): A145 and A181 are fashion/footwear articles that the LLM chain put under "
          "`lifestyle`, which has no fashion node in IPTC; the editorial label was right. A121 (lampshades, Marathi "
          "'Fashion') is a real misfiling.",
          f"- L3Cube Tamil categories are noisy: {int((l3ta['status'] == 'wrong').sum())} of {len(l3ta)} sampled Tamil items "
          "are wrong (Politics: cricket birthday, okra prices; Cinema: G20 summit, POCSO bail). This is dataset "
          "category noise (50 test articles per category), not mapping.",
          "- Section names that are fine but carry off-topic articles: IndoNews/Liputan6 `bisnis` with job "
          "vacancies and civil-service recruitment (A098, A119, A191; regex count of recruitment terms in the 50 "
          "`bisnis` test articles: 6), finanzen.net `Aktien` with general dpa-AFX wire news (A176), L3Cube "
          "`education`/`Crime`/`sports` with exam notices, accidents and off-field stories.", ""]

    L += ["## 3. Sensitivity of the estimated editorial-label accuracy", "",
          "Same population-weighted estimator and stratified bootstrap (2,000 resamples, same seed) as "
          "`scripts/label_audit/analyze.py` (functions imported, script not edited). Only the status of the 62 "
          "rechecked items changes; spot-check error e stays 0.", "",
          "| scenario | strict | 95% CI | lenient | 95% CI |", "|---|---|---|---|---|"]
    lab = {"status": "official (final decisions)",
           "recheck": "recheck (item-level recheck_status)",
           "literal": "upper bound: every section_acceptable and annotation_error item counted correct"}
    for k in ("status", "recheck", "literal"):
        v = res[k]
        L.append(f"| {lab[k]} | {pct(v['acc_strict'])} | {ci(v['acc_strict_ci'])} | {pct(v['acc_lenient'])} | "
                 f"{ci(v['acc_lenient_ci'])} |")
    L.append("")
    L.append("Per stratum (recheck):")
    L.append("")
    L.append("| stratum | official strict / lenient | recheck strict / lenient |")
    L.append("|---|---|---|")
    for s, v in res["recheck"]["strata"].items():
        o = res["status"]["strata"][s]
        L.append(f"| {s} | {pct(o['acc_strict'])} / {pct(o['acc_lenient'])} | {pct(v['acc_strict'])} / {pct(v['acc_lenient'])} |")
    L.append("")
    L.append("The recheck row is the realistic sensitivity: strict moves only through the 3 items that become "
             "correct; lenient additionally through the `wrong` items that become both_acceptable. The upper-bound "
             "row also turns all both_acceptable-type items (including the 7 original ones) into strict correct, "
             "which overstates accuracy because those articles have a second, equally valid topic. The unclear "
             "item (A027) keeps its official status in every row.")
    L.append("")

    na = out[out["needs_author"]]
    L += ["## 4. Items for the author", ""]
    for r in na.itertuples():
        L.append(f"- **{r.item_id}** ({r.source}, {r.lang}; section {r.section_name!s}; official: {r.human_decision}; "
                 f"recheck: {r.recheck_label}, {r.recheck_category}). {r.rationale}")
    L.append("")
    L.append("Not flagged but worth noting: A145 and A181 flip LLM-only verdicts (adjudicator high confidence, not "
             "routed) to correct; the remaining section_acceptable items are medium-confidence co-topic calls.")
    (OUT / "recheck_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
