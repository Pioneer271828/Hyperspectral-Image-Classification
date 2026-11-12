"""Single-use disk allocations for the bounded Xuzhou route campaign."""

from transfer import matched_route_admission as admission, registered_development as dev


POLICY = "wp06_xuzhou_route_allocation_v1"
ARM_NAMES = ("none", "important", "low_weight", "random:64", "random:512", "random:2048")


def segments(budget):
    if type(budget) is not int or not 7 < budget <= 256:
        raise ValueError("Expected a registered budget between 8 and 256")
    return {"full":(0,budget,None), "split-first":(0,7,None), "split-last":(7,budget,"split-first"),
            "completed":(budget,budget,"split-last")}


def routes(budget, arm_names):
    return {arm:{name:{"start":start,"stop":stop,"previous":previous,
                        "directory":f"{arm.replace(':','-')}-{name}"}
                 for name,(start,stop,previous) in segments(budget).items()} for arm in arm_names}


def location(root, claim, arm, name):
    row = claim["routes"][arm][name]
    folder = root/"attempts"/row["directory"]
    token = root/"claims"/(row["directory"]+".json")
    for path in (folder,token):
        if path.resolve() != path:
            raise ValueError("Attempt or claim path escapes the registered root")
    return folder, token


def read_attempt(root, allocation_pin, allocation, arm, name):
    folder, token = location(root, allocation, arm, name)
    claim = dev._pin(token)
    value = admission.read(claim)
    result = admission.read(dev._pin(token.with_suffix(".result.json")))
    expected = allocation["routes"][arm][name]
    admission.require(value == {"allocation":allocation_pin,"arm":arm,"segment":name,
                                 "registration":allocation["registrations"][arm],
                                 "output":str(folder),"interval":expected,"prior":value["prior"]},
                      "Claim has the wrong arm, registration or budget")
    status = admission.read(result["attempt"])
    first, last = expected["start"], expected["stop"]
    expected_status = "completed" if last==allocation["budget"] else "partial"
    admission.require(result["claim"]==claim and result["status"]==expected_status
        and result["attempt"]["path"]==str(folder/"status.json") and status["status"]==expected_status
        and status["registration"]==value["registration"] and status["code"]["commit"]==allocation["commit"]
        and status["code"]["source_hash"]==allocation["source_hash"]
        and status["total_budget"]==allocation["budget"] and status["query"]==0
        and status["initial_updates"]==first and status["target_updates"]==status["completed_updates"]==last
        and status["new_updates"]==last-first and status["checkpoint"]["path"]==str(folder/f"step-{last}.pt"),
        "Stored attempt differs from its frozen execution interval")
    previous = expected["previous"]
    if previous is None:
        admission.require(value["prior"] is None,"Fresh execution cannot have a prefix")
    else:
        earlier, prior = read_attempt(root, allocation_pin, allocation, arm, previous)
        admission.require(value["prior"]==prior["attempt"] and earlier["checkpoint"] in status["inputs"],
                          "Attempt did not consume its own successful prefix")
    return status,result


def execute(root, allocation_pin, allocation, arm, name, code, *, resume=None):
    admission.require(allocation["policy"]==POLICY and allocation["commit"]==code["commit"]
        and allocation["source_hash"]==code["source_hash"] and arm in allocation["routes"]
        and name in allocation["routes"][arm], "Unknown arm, stage or consumer")
    folder, token = location(root, allocation, arm, name)
    interval = allocation["routes"][arm][name]
    previous = interval["previous"]
    checkpoint = None
    if previous is None:
        admission.require(resume is None,"Fresh arm starts from its registered source")
    else:
        earlier, prior = read_attempt(root, allocation_pin, allocation, arm, previous)
        admission.require(resume==prior["attempt"],"Cross-arm or cross-registration prefix rejected")
        checkpoint = earlier["checkpoint"]
    if folder.exists() or token.exists():
        raise FileExistsError("Registered execution interval was already consumed")
    token.parent.mkdir(parents=True,exist_ok=True)
    registration = allocation["registrations"][arm]
    claim = dev._json_new(token,{"allocation":allocation_pin,"arm":arm,"segment":name,
                                "registration":registration,"output":str(folder),
                                "interval":interval,"prior":resume})
    result_path = token.with_suffix(".result.json")
    try:
        status = dev.execute_route(registration,out=folder,to_step=interval["stop"],expected_commit=code["commit"],
            resume=checkpoint,measure_at=2 if arm=="donor" and name=="full" else None,real_claim=claim)
        result = dev._json_new(result_path,{"claim":claim,"status":status["status"],"attempt":dev._pin(folder/"status.json")})
        read_attempt(root,allocation_pin,allocation,arm,name)
        return result
    except BaseException as error:
        status_path = folder/"status.json"
        if not result_path.exists():
            dev._json_new(result_path,{"claim":claim,"status":"failed","error":repr(error),
                                      "attempt":dev._pin(status_path) if status_path.is_file() else None})
        raise
