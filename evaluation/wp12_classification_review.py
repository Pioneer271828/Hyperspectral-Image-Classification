"""Release a protocol-scoped classification and cost review from approved aggregates."""

import csv
import json
from itertools import product
from math import isclose
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from evaluation import wp12_p4_short_review as p4, wp12_release_results as native
from evaluation import wp12_support_head as support, wp12_support_head_cost as support_cost
from evaluation import wp12_support_head_provenance as support_provenance
from evaluation.classification_metrics import metrics_from_confusion
from model.research_config import digest
from research_runtime import ROOT, hash_file, require_frozen_code, write_json

SCHEMA = "wp12_classification_cost_review_v1"
SCENES = {"paviaU":9,"xuzhou":9,"whu-honghu":22}
SEEDS = (64,512,2048)
ARMS = tuple(product(("none","x_residual"),("legacy_three_heads","primary_only")))
ARM_NAMES = dict(zip(ARMS,("无残差＋三头监督","无残差＋仅融合监督","残差＋三头监督","残差＋仅融合监督"),strict=True))
RESULT_DIR = "m4s2-issue135-analysis02"
HEAD_DIR = "wp05-wp06-support-head-deployment-20261120-qa"
RAW_PINS = {
    "launch": ("m4s2-issue135-full01/launch.json","c3527a7d0422a5c605b8b7271fa41644784c5a3e91145fde198983fb7df4e805"),
    "result": (RESULT_DIR+"/results.json","83e3a270efcd3293060a58063d10a0ea7b0b0dda36071a920d10bfd25b2cc61a"),
    "audit": (RESULT_DIR+"/independent-receipt.json","1c17cd49bf848518e77fc3e60be9f0016435cdc3ffa20fc32f365a1eb0f9b5c7"),
    "summary": (RESULT_DIR+"/summary.json","a2b34ace698c0fe1d257349ea13ba0dfe728e748c24db3f97183a3dcf4edbcb6"),
    "heads": (HEAD_DIR+"/head-deployment-evidence.json","82594d01a21264f67c3012ff34fdeb11402b69d56905c680e6d967dd65054f00"),
    "migration": ("wp05-resource-migration-serial-control01/migration.json","62abd45b3cbdf3437d03a3f590094434b89941afba6e0dd45015dc6447a5ba03"),
    "approval": ("wp05-resource-migration-serial-control01/release.json","44ae429be6255f3da5e3d9b3f42b16bbb7543c755925f6da1c1fcd3d72464310"),
}
require = native.require


def pin(root, key):
    path,sha = RAW_PINS[key]
    return dict(path=str(root/path),sha256=sha)


def measure(values):
    values = list(values)
    return dict(values=values,**native.describe(values))


def agree(first, second):
    require(isclose(first,second,rel_tol=0,abs_tol=1e-10),"Historical summary differs from sealed audit")


def complete_factorial(root):
    launch,result,audit,summary = (native.document(root,pin(root,k)) for k in ("launch","result","audit","summary"))
    policy = "exact_state_trainval_decisions_two_precision_sensitivity_v1"
    require(launch["study"]["protocol"]=="m4_supervision_factorial_full_v1"
            and launch["code"]["commit"]==result["source"]["commit"]==audit["source"]["commit"]==
                "b36f1c976120eb8abd7d74e7f44295227550e9ea"
            and result["consumer"]["commit"]==audit["consumer"]["commit"]==
                "9d5318803d926eacebf8984bc4e6a7bed388bbc4"
            and result["policy"]==audit["policy"]==policy and audit["status"]=="passed"
            and result["status"]=="completed" and result["optimizer_updates"]==audit["optimizer_updates"]==0
            and audit["precision_gate_agreement"] is True and audit["probability_calibration_validated"] is False
            and audit["original_probability_policy"]=="not_stably_passed"
            and summary==json.loads((ROOT/"docs/architecture/m4-s2-results.json").read_bytes())
            and all(audit[k]==value for k,value in summary.items()),
            "Wrong full-budget protocol, scorer, independent review or repository summary")
    schedule = {}
    for item in launch["cases"]:
        mode,supervision,seed = (item[k] for k in ("mode","supervision","seed"))
        key = (mode,supervision,seed)
        settings = item["settings"]
        require(key not in schedule and item["name"]==f"{mode}-{supervision}-{seed}"
                and settings["split_root_seed"]==64 and settings["model_seed"]==settings["schedule_seed"]==seed
                and settings["optimizer_steps"]==audit["training_reviews"][item["name"]]["updates"]==1503
                and settings["test_scoring"] is launch["test_scoring"] is False,
                "Full-budget training case changed")
        schedule[key] = dict(case=item["name"],optimizer_steps=1503,
                             config_sha256=digest(settings["config"]),
                             selected_step=audit["training_reviews"][item["name"]]["selected_step"])
    require(set(schedule)=={(*arm,seed) for arm,seed in product(ARMS,SEEDS)}
            and launch["expected_updates"]==18036,"Incomplete registered factorial")

    rows = {}
    for item in result["rows"]:
        mode,supervision,seed,scene,precision = (item[k] for k in ("mode","supervision","seed","dataset","precision"))
        key = (mode,supervision,seed,scene,precision)
        expected = {"fused","graph","cnn"} if supervision=="legacy_three_heads" else {"fused"}
        require(key not in rows and set(item["scores"])==expected and item["case"]==schedule[key[:3]]["case"],
                "Duplicate aggregate, absent trained head or foreign case")
        heads = {}
        for h,saved in item["scores"].items():
            measured = metrics_from_confusion(saved["confusion"],range(1,SCENES[scene]+1))
            for field in ("oa","aa"):
                agree(measured[field],saved[field])
            for entry,recall in zip(measured["per_class"],saved["recall"],strict=True):
                agree(entry["recall"],recall)
            heads[h] = measured
        rows[key] = heads
    require(set(rows)=={(*a,s,c,p) for a,s,c,p in product(ARMS,SEEDS,SCENES,("fp32","fp64"))}
            and sum(map(len,rows.values()))==audit["metrics_recomputed"]==144
            and sum(len(h["per_class"]) for group in rows.values() for h in group.values())==1920,
            "Incomplete three-scene two-precision metrics")
    for scene in SCENES:
        population = {tuple(row["support"] for row in heads["fused"]["per_class"])
                      for key,heads in rows.items() if key[3]==scene}
        require(len(population)==1,"Same-protocol factorial changed an evaluation denominator")

    tables, paired, protections = {},{},{}
    for precision in ("fp32","fp64"):
        view = tables[precision] = {}
        for mode,supervision in ARMS:
            arm = (mode,supervision)
            scenes = {}
            for scene in SCENES:
                scenes[scene] = {k:measure(100*rows[(*arm,seed,scene,precision)]["fused"][k] for seed in SEEDS)
                                 for k in ("oa","aa")}
            macro = measure(sum(scenes[scene]["oa"]["values"][i] for scene in SCENES)/len(SCENES)
                            for i,_ in enumerate(SEEDS))
            tag = "-".join(arm)
            expected = audit["tables"][precision][tag]
            for metric,old in [(macro,expected["macro_oa"]),*((scenes[s][k],expected["datasets"][s][k])
                            for s,k in product(SCENES,("oa","aa")))]:
                for a,b in zip(metric["values"],old["values"],strict=True):
                    agree(a,b)
                agree(metric["mean"],old["mean"])
                agree(metric["sample_sd"],old["std"])
            view[tag] = dict(name=ARM_NAMES[arm],scenes=scenes,macro_oa=macro,
                             arms=[schedule[(*arm,seed)] for seed in SEEDS])
        values = [view["-".join(arm)]["macro_oa"]["values"] for arm in ARMS]
        a,b,c,d = values
        contrasts = paired[precision] = {
            "candidate_vs_baseline":measure(y-x for x,y in zip(a,d,strict=True)),
            "residual_legacy":measure(y-x for x,y in zip(a,c,strict=True)),
            "residual_primary":measure(y-x for x,y in zip(b,d,strict=True)),
            "supervision_none":measure(y-x for x,y in zip(a,b,strict=True)),
            "supervision_residual":measure(y-x for x,y in zip(c,d,strict=True)),
            "interaction":measure(z-y-x+w for w,x,y,z in zip(a,b,c,d,strict=True)),
        }
        for key,value in audit["paired"][precision].items():
            for a,b in zip(contrasts[key]["values"],value["values"],strict=True):
                agree(a,b)
        subset = protections[precision] = audit["gates"][precision]
        for gate,control in zip(subset,ARMS[1:3],strict=True):
            require(gate["control"]==list(control) and gate["passed"] is True,"Registered comparison did not pass")
            tag = "-".join(control)
            best = view["-".join(ARMS[-1])]
            for change in gate["aa"]:
                scene = change["dataset"]
                agree(best["scenes"][scene]["aa"]["mean"]-view[tag]["scenes"][scene]["aa"]["mean"],change["delta_pp"])
        classes = audit["class_changes"][precision]
        require(len(classes)==2*sum(SCENES.values()) and len({(tuple(c["control"]),c["dataset"],c["class"]) for c in classes})
                ==len(classes),"Incomplete audited class protection")
        for case in classes:
            scene,cls = case["dataset"],case["class"]
            original = tuple(case["control"])
            count = launch["train_class_counts"][scene][cls-1]
            delta = sum(100*(rows[(*ARMS[-1],s,scene,precision)]["fused"]["per_class"][cls-1]["recall"]-
                rows[(*original,s,scene,precision)]["fused"]["per_class"][cls-1]["recall"]) for s in SEEDS)/len(SEEDS)
            protected = count<=launch["study"]["gate"]["small_class_max_train_labels"]
            require(case["train_count"]==count and case["protected"]==protected
                    and case["passed"]==(not protected or delta>=-2),"Class protection does not match actual denominator")
            agree(delta,case["delta_pp"])
    return dict(policy=policy,seeds=SEEDS,launch_plan=launch["study"],tables=tables,paired=paired,
                registered_protections=protections,classes=audit["class_changes"],
                manifest={k:RAW_PINS[k] for k in ("launch","result","audit","summary")},
                headings=dict(fused="all four combinations",graph_cnn="two three-head supervision arms"),
                saved_heads=[dict(mode=key[0],supervision=key[1],seed=key[2],scene=key[3],precision=key[4],
                                  heads=heads) for key,heads in rows.items()],
                fits_unchanged=12,new_query_reads=0,probability_calibration_validated=False)


def stage_summary(heads):
    require(heads["schema"]=="wp12_support_head_aggregate_v1" and heads["develop_only"] is True,
            "Expected the archived frozen-head development review")
    summary = {}
    for key, scene in (("selected","xuzhou"),("whu-honghu","whu-honghu")):
        selected = heads["results"][key]
        require(selected["source"]["scene"]==scene and selected["registered_candidate"] is False,
                "Frozen-head review belongs to another scene or publication")
        summary[scene] = selected
    return summary


def support_review(root, link, reviewer):
    reviewed = native.document(root,link)
    require(reviewed["schema"]==support.SCHEMA and reviewed["case"] in support.PINS
            and reviewed["development_only"] is True and reviewed["backbone_optimizer_updates"]==0
            and reviewed["raw_gt_loaded"] is reviewed["query_accessed"] is False,
            "Expected a current support-head development review")
    support_provenance.ancestor(reviewed["producer"],reviewer)
    case = reviewed["case"]
    _,source = support.frozen_case(root,case)
    result = reviewed["results"]
    identity = reviewed["measurement_identity"]
    require(result["source"]==source and result["registered_candidate"] is False,
            "Support-head result differs from its frozen case")
    plan,_,verified = support_cost.evidence(root,reviewed["cost"],identity["plan_producer"],source,
                                          result["artifact"],reviewer=reviewer)
    require(verified==identity and plan["selection"]==result["deployed_selection"],
            "Support-head review modified the measured identity or selection")
    raw = native.document(root,reviewed["cost"])
    return dict(scene=source["scene"],reviewer=reviewed["producer"],review_sha256=link["sha256"],
                cost_sha256=reviewed["cost"]["sha256"],identity=verified,selection=plan["selection"],
                model_and_head_load_seconds=raw["model_and_head_load_seconds"],
                input_load_seconds=raw["input_load_seconds"],resident=raw["resident"],
                rss_kib=raw["rss_kib"],cuda_peak_bytes=raw["cuda_peak_bytes"])


def p4_grid(result):
    cells = {(entry["k"],entry["method"]):len(entry["accepted_arms"]) for entry in result["roster_cells"]}
    accepted = len(result["cases"])
    require(set(cells)==set(product((1,5),("scratch","shared_transfer"))) and len(result["roster_cells"])==4
            and sum(cells.values())==accepted and accepted+len(result["pending"])==20
            and ((accepted==20)==(result["combined_acceptance"] is not None))
            and all(n in (0,5) for n in cells.values()),
            "P4 report requires a disjoint partial or complete registered roster")
    return cells


def historical_pairs(cohorts):
    matched = {}
    for row in cohorts:
        if row["source"]!="frozen54" or row["head"]!="fused":
            continue
        group = matched.setdefault((row["method"],row["arm"]),{})
        scene = row["scene"]
        require(scene in SCENES and scene not in group and tuple(row["model_seeds"])==SEEDS
                and row["role"]=="historical_frozen_authorized_aggregate",
                "Frozen54 paired three-scene quality roster differs")
        group[scene] = {key:row[key] for key in ("oa","aa","protocol","measurement_scope","cost")}
    require(len(matched)==6 and all(set(group)==set(SCENES) for group in matched.values())
            and all(len({group[scene]["protocol"] for group in matched.values()})==1 for scene in SCENES),
            "Frozen54 has no complete common-settings group")
    return [dict(method=m,arm=a,scenes=group) for (m,a),group in sorted(matched.items())]


def text_report(result):
    m4 = result["full_factorial"]
    lines = ["# WP12 分类精度与成本审阅",
        "",
        "## 完整同协议联合训练",
        "",
        "三景固定划分 root64，每类 1% 训练、0.5% 验证。固定全带超图与 DomainBN 校准选步，"
        "每组合用 64、512、2048 三种子，各 1,503 更新。表内为已封存 FP32 计分，样本标准差按三种子计算。"
        "本次从保存混淆矩阵复算，新增 query 评分为零。",
        "",
        "| 残差与监督 | PaviaU OA (%) | 徐州 OA (%) | 洪湖 OA (%) | 三景等权 OA (%) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    def fmt(x):
        return f"{x['mean']:.4f} ± {x['sample_sd']:.4f}"
    for row in m4["tables"]["fp32"].values():
        scene_scores = [fmt(row["scenes"][scene]["oa"]) for scene in SCENES]
        lines.append("| "+" | ".join([row["name"],*scene_scores,fmt(row["macro_oa"])])+" |")
    paired = m4["paired"]["fp32"]
    lines+=["",
        f"残差＋仅融合监督对同批无残差＋三头基线为 {fmt(paired['candidate_vs_baseline'])} 个百分点。"
        f"对无残差＋仅融合监督和残差＋三头两项登记比较分别为 {fmt(paired['residual_primary'])} 和 "
        f"{fmt(paired['supervision_residual'])} 个百分点。两项逐类保护在两种精度下均通过。"
        "保护允许在冻结界线内退化。仅融合监督组的图/CNN 辅助头未训练，逐类表仅公布融合头。",
        "",
        "## 已发布冻结 54 例的同设置三景比较",
        "",
        "下表仍为三个种子。每景在相同冻结指标、原发布点及整带观察下配对，场景自身划分与 M4-S2 不同。"
        "同一训练方式在组内比较无残差、容量对照和区域残差，各头和逐类见统一表。",
        "",
        "| 训练方式 / 结构 | PaviaU 融合 OA | 徐州融合 OA | 洪湖融合 OA |",
        "| --- | ---: | ---: | ---: |"]
    for row in result["historical_pairs"]:
        values = [row["scenes"][scene]["oa"] for scene in SCENES]
        arm = f"`{row['method']}/{row['arm']}`"
        lines.append("| "+" | ".join([arm,*(f"{x['mean']:.2%} ± {x['sample_sd']:.2%}" for x in values)])+" |")
    lines += ["", "## 真实短训与固定开发读出",
        "",
        "| 证据 | 成果和角色 |",
        "| --- | --- |",
        ]
    p4_result = result["p4"]
    accepted = len(p4_result["cases"])
    pavia,xuzhou = (p4_result["other_real_scopes"][scene] for scene in ("paviaU","xuzhou"))
    lines += [f"| PaviaU 原生释放点 | {pavia['cases']} 例、{pavia['steps']:,} 步，开发角色及三头分母见原统一表 |",
              f"| 徐州真实短训 | 旧本地固定多尺度对照 {xuzhou['cases']} 臂、{xuzhou['steps']:,} 步，原空间开发角色 |",f"| 洪湖 P4 | {accepted}/20 例，支持种子 2048、模型 512、采样 64，22 类每类 5 个开发点 |",
              "", "| 洪湖 P4 支持量 | 从头 | 共享迁移 |",
              "| --- | ---: | ---: |"]
    cells = p4_grid(p4_result)
    for k in (1,5):
        lines.append(f"| K{k} | {cells[k,'scratch']}/5 | {cells[k,'shared_transfer']}/5 |")
    lines += ["", "| 洪湖 P4 同视图 none 基线 | 融合正确 | 图正确 | CNN 正确 |",
              "| --- | ---: | ---: | ---: |"]
    for cell in p4_result["roster_cells"]:
        prefix = f"s{cell['support_seed']}-k{cell['k']}-{cell['method']}-m{cell['model_seed']}-"
        baseline = p4_result["cases"].get(prefix+"none")
        if baseline is not None:
            values = [f"{round(baseline['heads'][h]['oa']*110)}/110" for h in ("fused","graph","cnn")]
            lines.append("| "+" | ".join([f"K{cell['k']} {cell['method']}",*values])+" |")
    lines+=["","PaviaU、徐州和洪湖的短训划分、支持量和原生设置不同。现有材料没有共同的三景短训准确率。"
            "同协议三景结果见上表，旧 54 例的同设置记录可在统一主表按 method、arm、scene 和 seed 精确选取。",
            "", "| 冻结开发读出，中央缺带 | 原融合 OA | 固定全带头 OA | 同视图头 OA |",
            "| --- | ---: | ---: | ---: |"]
    for scene,row in result["support_heads"].items():
        view = row["views"]["central_gap"]
        values = [view["native"]["fused"],*(view[name]["all"]["metrics"] for name in ("fixed_full","same_view"))]
        lines.append("| "+" | ".join([scene,*(f"{v['micro_oa']:.2%}" for v in values)])+" |")
    lines+=["","开发读出在徐州类 1/8 与洪湖类 6 产生新增错误，两景逐类保护均未过。"
            "原生模型继续为默认。M4-S2 仅支持原历史协议内的比较，独立空间、跨仪器和概率校准各需自己的证据。",
            "", "## 成本范围", "",
            "| 成本来源 | 工作量与边界 |", "| --- | --- |",
            "| 冻结 54 例 | 已验 54/54、648 次、162 次预热，RTX 4090 常驻前向。采集不含载入、传输和构图准备。本轮只读重组 |",
            "| 两景支持头 | 授权开发块的 CPU 单线程，单次支持拟合、独立加载和部署每块分开计量 |",
            "",
            "机器回执保留完整联合混淆、逐类保护、短训来源和 P4 两个物理槽、支持头 CPU 成本与既有统一表。"
            "尚未建立可跨场景、跨协议的排行榜，也没有新的生产默认放行。",
            ""]
    lines += ["", "| 成本组 | 原测工作量 | 汇总毫秒 |", "| --- | --- | ---: |"]
    for scene,values in result["historical_cost_ms"].items():
        lines.append(f"| 冻结 54 · {scene} | 18 例各自 p50，再取中位数 | {values:.2f} |")
    for scene,readout in result["support_heads"].items():
        cost = readout["deployment"]["all"]["cost"]
        per_block = cost["resident_per_block"]
        samples = " / ".join(f"{per_block[key]['median_ms']:.2f}" for key in ("native","fixed_full","same_view"))
        lines.append(f"| 冻结头 · {scene} | 两块三视图，原生 / 固定全带 / 同视图 | {samples} |")
    lines += ["","原冻结头成本来自旧格式，原文设备为 CPU，实际执行 SHA 以及实体设备身份未在旧格式单列。"]
    if result["support_measurements"]:
        lines += ["","| 分类头测量复核 | 计划制作 SHA | 实际执行 SHA | 记录的设备 |",
                  "| --- | --- | --- | --- |"]
    for row in result["support_measurements"]:
        identity = row["identity"]
        actual = identity["execution"]
        worker = "旧格式未记录" if actual is None else actual["commit"]
        host = identity["reported_device"] if actual is None else (
            actual["hardware"]["physical_uuid"] or actual["hardware"]["name"] or actual["hardware"]["device"])
        lines.append(f"| {row['scene']} · {row['selection']} | {identity['plan_producer']} | {worker} | {host} |")
    lines += ["",
              "去身份论文表 `paired-scene-seed.csv` 给出同景同种子 OA/AA 和成对百分点差。",
              "`deployment-cost-evidence.csv` 分别列常驻、加载、缺测范围和设备来源。",
              "冻结头与冻结 54 的既有负结果各自保留，真实 P4 的缺格及未测精度不参与排名。"]
    return "\n".join([*lines,""])


def paper_tables(folder, result, historical_cases):
    paired_fields = ("protocol","precision","scene","seed","head","candidate","baseline","denominator",
                     "candidate_oa","baseline_oa","delta_oa_pp","candidate_aa","baseline_aa","delta_aa_pp",
                     "protection_status")
    cost_fields = ("source","scene","seed","configuration","scope","status","reported_device",
                   "physical_uuid","plan_producer","execution_commit","p50_ms","p95_ms","load_ms",
                   "peak_allocated_bytes","source_sha256")
    flags = result["full_factorial"]["registered_protections"]
    short = result["full_factorial"]["saved_heads"]
    rows = {(r["precision"],r["scene"],r["seed"],r["mode"],r["supervision"]):r["heads"] for r in short}
    matched = [r for r in historical_cases if r["source"]=="frozen54"]
    named = {(r["scene"],r["seed"],r["method"],r["arm"]):r for r in matched}
    paired_name = "paired-scene-seed.csv"
    cost_name = "deployment-cost-evidence.csv"
    with (folder/paired_name).open("x",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(paired_fields)

        def pairing(protocol,precision,scene,seed,candidate,baseline,heads,controls,protection):
            for head,a in heads.items():
                if head not in controls:
                    continue
                b = controls[head]
                require(a["count"]==b["count"],"Paper pair changed the evaluation population")
                writer.writerow((protocol,precision,scene,seed,head,candidate,baseline,a["count"],
                                 a["oa"],b["oa"],100*(a["oa"]-b["oa"]),a["aa"],b["aa"],100*(a["aa"]-b["aa"]),
                                 protection))

        for (precision,scene,seed,mode,supervision),heads in sorted(rows.items()):
            arm = (mode,supervision)
            if arm==ARMS[0]:
                continue
            for control in (ARMS[:3] if arm==ARMS[-1] else ARMS[:1]):
                gate = next((r for r in flags[precision] if arm==ARMS[-1] and r["control"]==list(control)),None)
                pairing("m4_supervision_factorial_full_v1",precision,scene,seed,"-".join(arm),"-".join(control),
                        heads,rows[(precision,scene,seed,*control)],
                        "registered_passed" if gate and gate["passed"] else "descriptive_same_protocol")
        for (scene,seed,method,arm),candidate in sorted(named.items()):
            if arm=="none":
                continue
            baseline = named[(scene,seed,method,"none")]
            require(candidate["protocol"]==baseline["protocol"],"Frozen54 seed pair changed protocol")
            pairing(candidate["protocol"],"recorded",scene,seed,f"{method}/{arm}",f"{method}/none",
                    candidate["heads"],baseline["heads"],"archived_primary_benefit_not_supported")

    with (folder/cost_name).open("x",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=cost_fields)
        writer.writeheader()
        for row in matched:
            cost = row["cost"]
            writer.writerow(dict(source="frozen54",scene=row["scene"],seed=row["seed"],
                configuration=f"{row['method']}/{row['arm']}",scope=cost["scope"],status="historical_measured",
                reported_device="cuda:0",physical_uuid=row["device"],
                p50_ms=cost["p50_ms"],p95_ms=cost["p95_ms"],peak_allocated_bytes=cost["peak_allocated_bytes"],
                source_sha256=cost["receipt"]))
        for scene,row in result["support_heads"].items():
            old = row["deployment"]["all"]
            for config,values in old["cost"]["resident_per_block"].items():
                writer.writerow(dict(source="support_head_archive",scene=scene,configuration=config,
                    scope=support_cost.TIMING_SCOPE,status="legacy_execution_not_recorded",
                    reported_device="cpu",p50_ms=values["median_ms"],source_sha256=old["receipt_sha256"]["cost"]))
            writer.writerow(dict(source="support_head_archive",scene=scene,configuration="all",
                scope="model_and_head_load",status="legacy_execution_not_recorded",reported_device="cpu",
                load_ms=1000*old["cost"]["loading_seconds"],source_sha256=old["receipt_sha256"]["cost"]))
        for row in result["support_measurements"]:
            identity = row["identity"]
            actual = identity["execution"]
            scope = dict(source="support_head_replay",scene=row["scene"],plan_producer=identity["plan_producer"],
                         execution_commit=None if actual is None else actual["commit"],
                         physical_uuid=None if actual is None else actual["hardware"]["physical_uuid"],
                         reported_device=identity["reported_device"],status=identity["identity_status"],
                         peak_allocated_bytes=row["cuda_peak_bytes"] if actual is not None else None,
                         source_sha256=row["cost_sha256"])
            for block,view in row["resident"].items():
                for config,values in view["items"].items():
                    time = values["timing"]
                    writer.writerow(dict(**scope,configuration=f"{row['selection']}/{block}/{config}",
                        scope=identity["timing_scope"],p50_ms=time["p50_ms"],p95_ms=time["p95_ms"]))
            writer.writerow(dict(**scope,configuration=row["selection"],scope="model_and_head_load",
                                 load_ms=1000*row["model_and_head_load_seconds"]))
        for scene in SCENES:
            writer.writerow(dict(source="m4_factorial",scene=scene,configuration="all",
                                 scope="matched_deployment_device_benchmark",status="unmeasured"))
    return {name:hash_file(folder/name) for name in (paired_name,cost_name)}


def publish(root, out, code, acceptance=None, support_receipts=()):
    factorial = complete_factorial(root)
    heads = stage_summary(native.document(root,pin(root,"heads")))
    measurements = [support_review(root,link,code["commit"]) for link in support_receipts]
    require(len({r["cost_sha256"] for r in measurements})==len(measurements),
            "The same support deployment may not be counted twice")
    p4_result = p4.run(root,pin(root,"migration"),pin(root,"approval"),acceptance)
    require(len(p4_result["cases"]) in (10,20) and p4_result["raw_gt_loaded"] is p4_result["query_accessed"] is False,
            "P4 scope must supply the registered accepted archive")
    with TemporaryDirectory(dir=out.parent,prefix=".wp12-classification-") as tmp:
        folder = Path(tmp)/"publication"
        folder.mkdir()
        replay = native.publish(root,folder/"native-real",code)
        receipt_path = Path(replay["path"])
        old = native.document(root,replay)
        aggregate = json.loads((receipt_path.parent/"cohorts.json").read_bytes())
        cases = json.loads((receipt_path.parent/"cases.json").read_bytes())
        durations = {scene:median(row["cost"]["p50_ms"] for row in cases
                                  if row["source"]=="frozen54" and row["scene"]==scene) for scene in SCENES}
        result = dict(schema=SCHEMA,producer=code["commit"],full_factorial=factorial,p4=p4_result,
                      historical_pairs=historical_pairs(aggregate),historical_cost_ms=durations,
                      support_heads=heads,support_measurements=measurements,
                      native_real=dict(receipt=receipt_path.relative_to(folder).as_posix(),
                                        sha256=replay["sha256"],files=old["files"],frozen_cost=old["cost_lifecycle"]),
                      new_query_reads=0,new_training_updates=0,new_cost_measurements=0)
        result["paper_files"] = paper_tables(folder,result,cases)
        write_json(folder/"classification-review.json",result)
        (folder/"classification-review.md").write_text(text_report(result),encoding="utf-8")
        require_frozen_code(code)
        folder.rename(out)
    artifacts = ("classification-review.json","classification-review.md","native-real/receipt.json",*result["paper_files"])
    return {name:hash_file(out/name) for name in artifacts}
