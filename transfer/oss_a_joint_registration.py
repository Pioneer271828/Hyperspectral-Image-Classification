"""Register the orthogonal joint development matrix without enabling real inputs."""

from hashlib import sha256
from itertools import product

from joint_supervision import MODES, SUPERVISION, SupervisionRuntime, supervision_settings
from joint_training import schedule_for
from model.research_config import digest
from research_runtime import tensor_hash


ADAPTERS = ("wavelength","index")
SEEDS = (64,512,2048)
SCENES = ("paviaU","xuzhou","whu-honghu")


class OrthogonalRuntime(SupervisionRuntime):
    def __init__(self,residual,supervision,adapter_mode):
        super().__init__(residual,supervision)
        if adapter_mode not in ADAPTERS:
            raise ValueError("Unregistered VCA comparison mode")
        self.adapter_mode = adapter_mode

    def validate(self,settings):
        if (settings["config"]["model"]["adapter_attention_mode"]!=self.adapter_mode
                or settings.get("oss_a_adapter")!=self.adapter_mode
                or settings["optimizer_steps"]!=12):
            raise ValueError("Orthogonal training settings differ")
        return super().validate(settings)


def options(base,residual,supervision,adapter_mode):
    if adapter_mode not in ADAPTERS or base["config"]["model"]["adapter_attention_mode"]!="wavelength":
        raise ValueError("Fixed reference adapter required")
    opts = supervision_settings(base,residual,supervision)
    opts["config"]["model"]["adapter_attention_mode"] = adapter_mode
    opts["oss_a_adapter"] = adapter_mode
    OrthogonalRuntime(residual,supervision,adapter_mode).validate(opts)
    return opts


def bind_inputs(items):
    if tuple(items)!=SCENES or any(any(key.startswith("test_") for key in item) for item in items.values()):
        raise ValueError("Train/validation-only three-scene input required")
    hashes = {}
    fields = ("net_input","Q","A","wavelengths","train_samples_gt","val_samples_gt")
    for scene,item in items.items():
        if item["fwhm"] is not None:
            raise ValueError("Frozen reference has no FWHM")
        hashes[scene] = {field:tensor_hash(item[field]) for field in fields}
    return hashes


def register(base_options,items,*,data_identity,source_commit):
    fixed = bind_inputs(items)
    cells = []
    for residual,supervision,adapter,seed in product(MODES,SUPERVISION,ADAPTERS,SEEDS):
        opt = options(base_options(seed),residual,supervision,adapter)
        schedule = schedule_for(opt)
        if len(schedule)!=12:
            raise ValueError("Frozen actual optimizer budget differs")
        cells.append(dict(residual=residual,supervision=supervision,adapter=adapter,seed=seed,
                          settings_sha256=digest(opt),schedule_sha256=digest(schedule),
                          updates=len(schedule),test_scoring=False))
    return dict(schema="oss-a-joint-registration-v1",source_commit=source_commit,scope="synthetic_only",
                data_identity=data_identity,inputs=fixed,cells=cells,registered_updates=288,
                validation="existing_calibrated_first_maximum_steps_3_6_9_12",
                real_training_authorized=False,query_reads=0)
