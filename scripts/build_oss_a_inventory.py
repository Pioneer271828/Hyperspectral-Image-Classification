"""Build the OSS-A crosswalk from the pinned module and delivery records."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


COMMIT = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
SHORT_TESTED = re.compile(r"(?:tested|生产|源码)\s*[` ]*([0-9a-f]{7,12})(?![0-9a-f])")
SOURCE_PATH = re.compile(r"(?<![/\w])(?:[\w.-]+/)*[\w.-]+\.(?:py|json)(?![\w.])")
PLAN = "docs/references/completed-module-oss-review-20261113.json"
STATUS = "docs/architecture/current-status.json"
INDEX = "docs/architecture/module-map.json"
ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "C0": ("登记、预算、恢复和独立证据", "外部 trainer 不能保持原更新、选模和证据身份"),
    "C1": ("数据空间、标签角色和传感器准入", "只审空间算子和协议，数据权限仍由项目合同控制"),
    "M1": ("真实波段子集、条件和逐类曝光", "上游动态输入不等价于真实删带及图观测隔离"),
    "M2": ("可变波段集合映射、波长与锚点", "DOFA 只作量纲参照，动态卷积和锚点混合职责不同"),
    "M3": ("区域归属、固定或学习图和稀疏传播", "必须先匹配归属、归一化、空行及重复边"),
    "M4": ("区域均值回投与像素残差", "普通残差和解混重建与该算子职责不同"),
    "M5": ("两阶段 CNN/GCN 骨干和分类读出", "保留双支路、区域回投、DomainBN 和三头语义"),
    "M6": ("区域像素对齐与方向消息", "上游整网或视觉 token 消息不能直接接入"),
    "M7": ("候选支持内边权、方向与归一化", "保持自环、边方向和原候选支持"),
    "M8": ("融合、主头/辅助头和监督", "保持类别语义、分支预算和监督开关"),
    "E0": ("冻结表征、机制干预和质量指标", "独立 oracle 可复算，指标库不替代访问和审计合同"),
    "E1": ("观测覆盖调制", "普通 FiLM 不等价于覆盖描述，原许可有限制"),
    "E2": ("每景条件多尺度与图半径控制", "timm 逐通道路由、BN 和 split 需通过职责门槛"),
    "E3": ("区域远程支持和固定图受控优化", "MiTA 的 KV 路由不能替代显式固定候选图"),
    "E4a": ("源/目标隔离、初始化与迁移", "外部域协议不覆盖本项目的家族和标签边界"),
    "E4b": ("无标签顺序适配与恢复", "须保留先预测后更新、场景重置和顺序隔离"),
    "E4c": ("未知隔离、开放集评分与独立确认", "只审评分组件，阈值和未知类证据权限仍需独立验证"),
    "MON": ("逐层持续监控与小类保护", "库的默认重要度不能替代持续窗口或逐类保护"),
    "MASK": ("依赖组、可逆屏蔽和物理缩宽", "DepGraph 仅核通道结构，保留逐类准入"),
    "LR": ("物理低秩分解与部署", "CP/Tucker、LoRA 与原 SVD 分解分别核验"),
    "PTQ": ("固定校准与物理整数推理", "算子和硬件覆盖必须实测"),
    "QAT": ("量化训练、恢复和落地", "fake quant 不等价于物理低比特"),
    "TQ": ("三值训练、编码和解码", "原权重/比例/位编码需逐项对齐"),
    "DY": ("冻结出口、轻重回退、阈值和成本", "早退出、双模型回退及独立阈值分别保留"),
    "CUR": ("固定曝光样本/观测课程", "外部 loader 不能替代类保护及实际更新台账"),
    "WP12": ("三头、逐类和全链路成本证据", "外部指标仅可交叉复算，保持现有收据和计时协议"),
}
P4_REVISIONS = {
    65:"89253cbb5fab42762b0129e8522b13b9ff3dbebf",
    66:"d656caddf345f2cd2a87adf7393b8b27691d54ce",
    67:"d656caddf345f2cd2a87adf7393b8b27691d54ce",
    68:"1218070d302d7c23b9134f42275b987488c5ff9d",
    69:"737cd3a5087ab2e0a5f0ed5e7bab53419d6c5105",
    70:"00bfb597543e289d2c24bafafaad30e7b99e6cea",
    71:"c8fa0e45d6e791d539aeb8b4454083344861adb0",
    72:"a3abde4892f8f00767e9e87140b78b6e081ad7a5",
    73:"635213d7dd1c442b4b0fc886acc1d388db2378bb",
    74:"c826803bf02862e704d1ae32cd24dd62fd45778c",
}

# Mixed ledger rows name distinct delivered responsibilities.
EXPLICIT = {
    1:"M4 M8",2:"M3 M5",3:"M6 M7 M8",4:"M2",5:"M1 E1",6:"E4a",
    7:"MON MASK",8:"MASK LR PTQ TQ DY E3",9:"C0",10:"M4 E4a",11:"M4 E4a",
    12:"M4 E4a",15:"C0 E4a",16:"E4a E0",17:"E4a E0",18:"C1 E4a",19:"C1 E4a",
    20:"M2 E4a",21:"C1 E4a",22:"M1 E4a",23:"E1 E4a",24:"E1 E4a",25:"C0 E0",
    32:"E0",33:"E0 C0",38:"M8",39:"C0 E4a",42:"E0 C0",43:"E0 E4a",44:"E0",
    45:"C0 E4a",46:"C0 E4a",47:"C0 E4a",48:"E0 E4a",49:"E0 E4a",50:"MON",
    51:"MON MASK",52:"MASK",53:"MASK M5",54:"MASK C0",55:"MASK C0",56:"MASK WP12",
    57:"MON MASK",58:"MON MASK",59:"MON MASK",60:"MON M5",61:"MON M5",62:"C0",
    63:"C1 E4a",64:"C1",65:"C1 C0",66:"C1 E4a",67:"C1",68:"C1 C0",
    69:"C1 E4a M8",70:"C1 E4a",71:"E4a M8",72:"E4a C0",73:"E4a E0",
    74:"E4b",75:"E3 M7",76:"LR",77:"PTQ QAT",78:"PTQ QAT C0",79:"TQ",80:"TQ C0",
    81:"DY E0",82:"DY M5",83:"CUR M1",84:"CUR",85:"CUR C0",86:"DY M8 C0",
    87:"DY C0",88:"DY C1",89:"DY WP12",90:"DY WP12",91:"DY E0",
    92:"CUR C0",93:"CUR M1",94:"E3 M3",95:"E3 M7 C0",96:"E3 C0",97:"E3 E0",
    98:"E3 M7 E0",99:"E3 WP12",100:"E0 M3",101:"E0 C1",102:"E0 C1",
    103:"E0",104:"E4c",105:"E4c",106:"E4c",107:"C1 E3 M8",108:"DY M8 E0",
    109:"CUR C0",110:"C1 E4a",111:"C1",112:"C1 C0",113:"M1 C1 E0",114:"M1 CUR C0",
    115:"E1 M1 CUR",116:"E1 M1 CUR",117:"E1 M1 E0",118:"E1 C0",119:"C0 M3",
    120:"M3 C0",121:"C0 DY",122:"M6 M7 M8",123:"M4 M8",124:"E0",
    125:"E0 M7",126:"E0 M7",127:"M7 E0 C0",128:"E0 C1",129:"E0 M3",
    130:"E0",131:"E0",132:"E0 E4a",133:"E0",134:"E0",135:"E3 E0 M7",
    136:"C1 E3 E0",137:"E3 M7 E0",138:"E0 M5",139:"E0 M2",
    140:"E2 C0",141:"M6 E3 C0",142:"E4b C0",143:"E4a C1",144:"E4c C1",
    145:"QAT TQ WP12",146:"DY C0",147:"DY WP12",148:"QAT TQ DY WP12",
    149:"LR C0",150:"LR PTQ WP12",151:"LR WP12",152:"CUR C0",153:"CUR E0 M1",
    154:"CUR WP12",155:"CUR M1",156:"CUR M1 C0",157:"CUR WP12",158:"CUR M1 E0 WP12",
    159:"E0 E3 M7 WP12",160:"E3 E2 WP12",161:"E0 E3 M7 WP12",
    162:"C1 E2 E3 WP12",163:"C1 E2 E3 C0",164:"M1 E2 E0 WP12",
    165:"C1 M1 E0 E2 WP12",166:"C0 M3 E2 WP12",
}
DOCUMENTATION_ONLY = {9, 62}
MIXED_BOUNDARIES = {
    3:"仅交付交互、边权和融合已有实现；完整组合仍需独立验收",
    8:"仅已交付的工程部件；压缩、动态研究闭环待完成",
    9:"计划登记，当前行无独立实现",
    62:"规划核对，当前行无独立实现",
}
NO_EXACT_CANDIDATE = {"C0", "M4"}
FAMILY_ALIASES = {"baseline":"C0", "E4a-2":"E4a", "MASK-1":"MASK", "MASK-2":"MASK", "MON-01":"MON"}


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, encoding="utf-8", check=False)


def sha(value):
    answer = git("rev-parse", "--verify", value + "^{commit}")
    return answer.stdout.strip() if answer.returncode == 0 else None


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def report_commits(text):
    lines = [line for line in text.splitlines() if re.search(r"SHA|tested|提交|生产|冻结|commit", line, re.I)]
    return list(dict.fromkeys(match for line in lines for match in COMMIT.findall(line)))


def source_paths(text, revisions):
    paths = dict.fromkeys(SOURCE_PATH.findall(text))
    matches = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        for revision in revisions:
            if git("cat-file", "-e", f"{revision}:{path}").returncode == 0:
                matches[path] = revision
                break
    return [{"path":path,"commit":revision} for path,revision in matches.items()]


def families(ordinal, identifier):
    if ordinal in EXPLICIT:
        return EXPLICIT[ordinal].split()
    found = re.findall(r"\bM[678]", identifier)
    return (found or ["E4a"]) + (["E4a"] if found and identifier.startswith("G4 ") else [])


def build(base):
    state, index, plan = map(read, (STATUS, INDEX, PLAN))
    modules = state["modules"]
    if len(modules) != 166 or len({row["id"] for row in modules}) != 166:
        raise ValueError("OSS-A frozen ledger roster differs")
    candidates = {
        key:[row["repository"] for row in plan["entries"] if key in row["modules"]]
        for key in FAMILIES
    }
    mapped = []
    for i, row in enumerate(modules, 1):
        report = row["report"]
        raw = subprocess.check_output(["git","show",f"{base}:{report}"],cwd=ROOT)
        report_text = raw.decode("utf-8")
        keys = families(i, row["id"])
        stated = list(dict.fromkeys(COMMIT.findall(row["evidence"]) + report_commits(report_text)))
        confirmed = [value for value in stated if sha(value) == value]
        confirmed += [resolved for token in SHORT_TESTED.findall(row["evidence"])
                      if (resolved := sha(token)) and resolved not in confirmed]
        source = source_paths(report_text, confirmed[:8]) if i not in DOCUMENTATION_ONLY else []
        recorded = ([P4_REVISIONS[i]] if i in P4_REVISIONS else [])
        if any(value not in stated for value in recorded):
            raise ValueError(f"Changed recorded revision for {row['id']}")
        report_commit = git("log","-1","--format=%H",base,"--",report).stdout.strip()
        mapped.append(dict(
            ordinal=i,id=row["id"],state=row["state"],report=report,report_sha256=hashlib.sha256(raw).hexdigest(),
            scope="planning_only" if i in DOCUMENTATION_ONLY else "delivered_evidence_and_listed_remaining_work",
            boundary=MIXED_BOUNDARIES.get(i,row["remaining"]),
            implementations=[dict(family=key,duty=FAMILIES[key][0],candidates=candidates[key],
                                  disposition=("no_same_duty_replacement_found" if key in NO_EXACT_CANDIDATE
                                               or not candidates[key] else "family_candidates_require_scope_review"),
                                  reason=FAMILIES[key][1]) for key in keys],
            report_version_commit=report_commit,recorded_tested_revisions=recorded,
            resolved_report_revisions=confirmed,unresolved_report_hashes=[value for value in stated if value not in confirmed],
            source_mentions_verified=source,
            revision_boundary=("report_revisions_keep_distinct_run_roles" if confirmed else
                               "no_run_sha_claimed; report and indexed baseline are separately pinned"),
        ))
    navigation = []
    for row in index["modules"] + index["experiment_gates"]:
        role_shas = {key:value for key,value in row.items()
                     if key.endswith("_commit") and isinstance(value,str) and COMMIT.fullmatch(value)}
        path_refs = [{"path":path,"baseline_present":git("cat-file","-e",f"{base}:{path.rstrip('/')}").returncode==0}
                     for path in row.get("paths", [])]
        navigation.append(dict(id=row["id"],guide=row.get("guide"),paths=path_refs,
                               tests=row.get("tests",[]),baseline=base,
                               indexed_revisions=role_shas,
                               family=FAMILY_ALIASES.get(row["id"],row["id"] if row["id"] in FAMILIES else None)))
    counts = Counter(key["family"] for row in mapped for key in row["implementations"])
    return dict(schema=1,issue=461,baseline=base,status=STATUS,index=INDEX,initial_inventory=PLAN,
                ledger_count=len(mapped),navigation_count=len(navigation),family_counts=dict(counts),
                ledger=mapped,navigation=navigation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.baseline) != args.baseline or git("diff","--quiet",args.baseline,"--",STATUS,INDEX,PLAN).returncode:
        parser.error("Frozen inventory inputs required")
    result = build(args.baseline)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    print(json.dumps({key:result[key] for key in ("ledger_count","navigation_count","family_counts")}))


if __name__ == "__main__":
    main()
