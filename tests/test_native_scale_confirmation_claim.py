"""Exercise the one-use confirmation gate with isolated, untrained synthetic endpoints."""

from contextlib import contextmanager
from copy import deepcopy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest import TestCase
from unittest.mock import patch

import torch

from model.research_config import digest
from transfer import native_scale_campaign as campaign, native_scale_training as training
from transfer import registered_development as transport
from transfer.native_scale_fixture import Fixture


CASE = "9-64-off"
CLAIM = "confirmation-claim.json"
COMPLETE = "confirmation-complete.json"


@contextmanager
def confirmation_fixture(root, producer):
    """Make an isolated frozen-format fixture without optimizing any model."""
    matrix_dir = root/"matrix"
    matrix_dir.mkdir()
    spec = training.contract(9,64,"off",producer)
    model = training.baseline(9,64)
    state = {name:value.detach().clone() for name,value in model.state_dict().items()}
    registration = transport._json_new(matrix_dir/"registration.json",spec)
    matrix = transport._json_new(matrix_dir/"matrix.json",dict(producer=producer,cases={CASE:registration}))
    # The cursor and the accepted audit receipt below exercise admission.
    # The tensors are newly initialized, with no optimizer or formal run.
    checkpoint = matrix_dir/"untrained-endpoints.pt"
    with io.BytesIO() as stream:
        torch.save(dict(registration=digest(spec),cursor=spec["steps"],model=state,
                        selected=dict(step=0,state=state)),stream)
        checkpoint.write_bytes(stream.getvalue())
    receipt = dict(exact=True,endpoints={"selected":{},"last":{}},stages=[
        dict(updates=spec["steps"],checkpoint=transport._pin(checkpoint)),dict(updates=0)])
    audit = transport._json_new(root/"audit.json",dict(producer=producer,matrix=matrix,cases={CASE:receipt},
                                                     main_updates=9216,hard_updates=96,confirmation_reads=0))
    # Reduce only the case roster, leaving authentication and the actual
    # synthetic confirmation observations and forwards on their normal path.
    with patch.object(campaign,"case_names",return_value=(CASE,)),patch.object(campaign,"CLASSES",(9,)):
        frozen = campaign.freeze(root,matrix,audit,root/"freeze.json",producer)
        yield matrix_dir,matrix,frozen


@contextmanager
def observe_payloads(after_materialize=None):
    """Count generated payloads and successful calls through the real fixture methods."""
    counts = dict(generated=0,materialized=0)
    payload, materialize = Fixture._payload, Fixture.materialize

    def traced_payload(self,tile,profile):
        result = payload(self,tile,profile)
        if self._role=="confirmation" and (tile,profile.key) not in seen:
            seen.add((tile,profile.key))
            counts["generated"] += 1
        return result

    def traced_materialize(self,*args,**kwargs):
        result = materialize(self,*args,**kwargs)
        if self._role=="confirmation":
            counts["materialized"] += 1
            if after_materialize is not None:
                after_materialize()
        return result

    seen = set()
    with patch.object(Fixture,"_payload",traced_payload),patch.object(Fixture,"materialize",traced_materialize):
        yield counts


class NativeConfirmationClaimTests(TestCase):
    def test_invalid_preconditions_do_not_claim_or_read(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with confirmation_fixture(root,"code") as (folder,_,frozen),observe_payloads() as counts:
                valid = campaign.read(frozen,root)
                bad_model = deepcopy(valid)
                bad_model["cases"][CASE]["selected_state"] = "wrong"
                bad_catalog = dict(valid,interventions=["wrong"])
                bad_audit = dict(valid,audit=dict(valid["audit"],sha256="0"*64))
                for name,body in (("model",bad_model),("catalog",bad_catalog),("audit",bad_audit)):
                    bad = transport._json_new(root/f"wrong-{name}.json",body)
                    with self.subTest(name=name),self.assertRaises(ValueError):
                        campaign.confirm(root,bad,root/"out","code")
                with self.assertRaises(ValueError):
                    campaign.confirm(root,frozen,root/"out","wrong-producer")
                self.assertEqual(counts,dict(generated=0,materialized=0))
                self.assertFalse((folder/CLAIM).exists())
                self.assertFalse((root/"out").exists())

    def test_first_payload_failure_and_concurrent_attempt_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entered,release = Event(),Event()
            errors = []

            def stop():
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("probe release timed out")
                raise RuntimeError("injected after actual held-out materialization")

            with confirmation_fixture(root,"code") as (folder,matrix,frozen):
                first = root/"first"

                def launch():
                    try:
                        campaign.confirm(root,frozen,first,"code")
                    except Exception as exc:
                        errors.append(exc)

                with observe_payloads(stop) as counts:
                    thread = Thread(target=launch)
                    thread.start()
                    try:
                        self.assertTrue(entered.wait(10))
                        claim = campaign.read(transport._pin(folder/CLAIM),root)
                        self.assertEqual(claim["matrix"],matrix)
                        self.assertEqual(claim["freeze"],frozen)
                        self.assertEqual(claim["producer"],"code")
                        self.assertEqual(claim["output"],str(first))
                        with self.assertRaises(FileExistsError):
                            campaign.confirm(root,frozen,root/"concurrent","code")
                        self.assertFalse((root/"concurrent").exists())
                        self.assertEqual(counts,dict(generated=1,materialized=1))
                    finally:
                        release.set()
                        thread.join(10)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual([str(exc) for exc in errors],["injected after actual held-out materialization"])
                    self.assertFalse((folder/".lock").exists())
                    for out in (first,root/"different"):
                        with self.assertRaises(ValueError):
                            campaign.confirm(root,frozen,out,"code")
                    self.assertEqual(counts,dict(generated=1,materialized=1))
                    self.assertFalse((folder/COMPLETE).exists())
                    self.assertFalse((root/"different").exists())

    def test_result_published_before_completion_failure_keeps_claim(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with confirmation_fixture(root,"code") as (folder,_,frozen),observe_payloads() as counts:
                first = root/"first"
                original = campaign._confirmation_json

                def interrupt(path,body):
                    if path.name==COMPLETE:
                        self.assertTrue((first/"confirmation.json").is_file())
                        raise RuntimeError("injected before completion marker")
                    return original(path,body)

                with patch.object(campaign,"_confirmation_json",side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,"before completion"):
                        campaign.confirm(root,frozen,first,"code")
                self.assertEqual(counts,dict(generated=8,materialized=16))
                result = campaign.read(transport._pin(first/"confirmation.json"),root)
                self.assertEqual(result["claim"],transport._pin(folder/CLAIM))
                self.assertFalse((folder/COMPLETE).exists())
                self.assertFalse((folder/".lock").exists())
                for out in (first,root/"different"):
                    with self.assertRaises(ValueError):
                        campaign.confirm(root,frozen,out,"code")
                self.assertEqual(counts,dict(generated=8,materialized=16))
                self.assertFalse((root/"different").exists())

    def test_claim_sync_failure_rejects_before_payload_and_keeps_link(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with confirmation_fixture(root,"code") as (folder,_,frozen),observe_payloads() as counts:
                claim = folder/CLAIM
                fsync = campaign.os.fsync

                def fail_after_link(fd):
                    if claim.exists():
                        raise OSError("injected claim directory sync failure")
                    return fsync(fd)

                with patch.object(campaign.os,"fsync",side_effect=fail_after_link):
                    with self.assertRaisesRegex(OSError,"claim directory sync"):
                        campaign.confirm(root,frozen,root/"first","code")
                self.assertTrue(claim.is_file())
                self.assertFalse((folder/".lock").exists())
                with self.assertRaisesRegex(ValueError,"already claimed"):
                    campaign.confirm(root,frozen,root/"different","code")
                self.assertEqual(counts,dict(generated=0,materialized=0))

    def test_success_and_legacy_completion_block_repeat(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with confirmation_fixture(root,"code") as (folder,matrix,frozen),observe_payloads() as counts:
                first = root/"first"
                result = campaign.confirm(root,frozen,first,"code")
                claim = transport._pin(folder/CLAIM)
                completed = campaign.read(transport._pin(folder/COMPLETE),root)
                self.assertEqual(completed["claim"],claim)
                self.assertEqual(completed["matrix"],matrix)
                self.assertEqual(completed["confirmation"],result)
                self.assertEqual(campaign.read(result,root)["claim"],claim)
                self.assertEqual(counts,dict(generated=8,materialized=16))
                for out in (first,root/"different"):
                    with self.assertRaises(ValueError):
                        campaign.confirm(root,frozen,out,"code")
                self.assertEqual(counts,dict(generated=8,materialized=16))
            legacy = root/"legacy"
            legacy.mkdir()
            with confirmation_fixture(legacy,"code") as (folder,_,frozen),observe_payloads() as counts:
                transport._json_new(folder/COMPLETE,dict(producer="code",legacy=True))
                with self.assertRaises(ValueError):
                    campaign.confirm(legacy,frozen,legacy/"out","code")
                self.assertEqual(counts,dict(generated=0,materialized=0))
                self.assertFalse((folder/CLAIM).exists())
