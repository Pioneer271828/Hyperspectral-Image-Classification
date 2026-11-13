"""Write-once dispatch journal for a released comparison controller."""
from dataclasses import dataclass
from pathlib import Path

from model.research_config import canonical_json
from transfer import registered_development as transport
from transfer.development_bank import _atomic_new
from transfer.native_scale_campaign import read


def seal(path, row):
    _atomic_new(path, (canonical_json(row) + "\n").encode())
    return transport._pin(path)


class JournalRejected(ValueError):
    pass


@dataclass(frozen=True)
class Operation:
    journal: Path
    attempt: int
    order: int

    @property
    def base(self):
        return self.journal / f"attempt{self.attempt:04d}-operation{self.order:04d}"

    def path(self, suffix):
        return self.base.with_suffix(suffix)


class Journal:
    """The parent seals a dispatch before forking and acknowledges verified exits."""
    def __init__(self, root, folder, identity, expected):
        self.root, self.folder, self.identity, self.expected = root, folder, identity, expected
        self.first = folder / "failure.json"

    def load(self, pin):
        return read(pin, self.root)

    def value(self, path):
        return self.load(transport._pin(path))

    def failure(self, attempt, *, dispatch=None, reason, returncode=None):
        if not self.first.exists():
            seal(self.first, {"schema": "wp07_dual_first_failure_v2", "identity": self.identity,
                              "attempt": attempt, "dispatch": dispatch, "reason": reason,
                              "returncode": returncode})
        return transport._pin(self.first)

    def _reconcile(self):
        attempts = sorted(self.folder.glob("attempt-*.json"))
        recognized = set()
        for i, path in enumerate(attempts):
            if path.name != f"attempt-{i:04d}.json":
                raise JournalRejected("Missing or unexpected dispatch attempt")
            attempt_pin = transport._pin(path)
            if self.load(attempt_pin) != {"attempt": i, "identity": self.identity}:
                raise JournalRejected("Attempt owner differs")
            recognized.add(path)
            matching = sorted(self.folder.glob(f"attempt{i:04d}-operation*.dispatch.json"))
            operations = []
            for order, dispatch in enumerate(matching):
                op = Operation(self.folder, i, order)
                if dispatch != op.path(".dispatch.json"):
                    raise JournalRejected("Discontinuous dispatch ledger")
                base = {**self.value(dispatch)}
                expected = self.expected.get((base.get("case"), base.get("phase")))
                if expected is None or base != {**expected, "attempt": attempt_pin, "order": order}:
                    raise JournalRejected("Dispatch does not belong to its frozen route")
                launch_path, ack_path = op.path(".launch.json"), op.path(".ack.json")
                if not launch_path.is_file() or not ack_path.is_file():
                    raise JournalRejected(f"Unacknowledged operation {op.base.name}")
                dp, lp, ap = map(transport._pin, (dispatch, launch_path, ack_path))
                launch, ack = map(self.load, (lp, ap))
                if set(launch) != {"dispatch", "pid"} or launch["dispatch"] != dp or type(launch["pid"]) is not int or launch["pid"] <= 0:
                    raise JournalRejected("Invalid launch identity")
                if set(ack) != {"launch", "returncode", "receipt"} or ack["launch"] != lp or ack["returncode"] not in (0, 75):
                    raise JournalRejected("Unexpected operation exit acknowledgement")
                receipt = op.path(".receipt.json")
                if ack["returncode"] == 0:
                    if not receipt.is_file() or ack["receipt"] != transport._pin(receipt):
                        raise JournalRejected("Acknowledged science receipt is missing or changed")
                    row = self.load(ack["receipt"])
                    self.check_receipt(row, expected, launch["pid"])
                    result = self.load(row["result"])
                    if base["phase"] == "audit":
                        if result["case"] != base["case"] or result["completed_resume_updates"] != 0:
                            raise JournalRejected("Committed audit changed")
                    else:
                        from research_runtime import hash_file
                        checkpoint = Path(result["checkpoint"]["path"])
                        if (result["phase"] != base["phase"] or checkpoint.resolve() != checkpoint
                                or self.root not in checkpoint.parents
                                or hash_file(checkpoint) != result["checkpoint"]["sha256"]):
                            raise JournalRejected("Committed scientific checkpoint changed")
                        self.load(result["registration"])
                    recognized.add(receipt)
                elif ack["receipt"] is not None or receipt.exists():
                    raise JournalRejected("Resource wait produced an inconsistent receipt")
                recognized.update((dispatch, launch_path, ack_path))
                operations.append(ap)
            summary = self.folder / f"summary-{i:04d}.json"
            if not summary.is_file() or not (value := self.value(summary)):
                raise JournalRejected("Attempt has no complete verified summary")
            if value.get("attempt") != attempt_pin or value.get("operation_acks") != operations or value.get("status") not in ("completed", "waiting"):
                raise JournalRejected("Attempt summary conflicts with the committed journal")
            recognized.add(summary)
        evidence = {path for path in self.folder.iterdir()
                    if path.name.startswith(("attempt", "summary")) and path.suffix == ".json"}
        if evidence != recognized:
            raise JournalRejected("Unclaimed or altered dispatch evidence")
        return len(attempts)

    @staticmethod
    def check_receipt(row, expected, pid):
        if (any(row.get(k) != v for k, v in expected.items()) or row["pid"] != pid
                or type(row["new_updates"]) is not int or row["new_updates"] < 0):
            raise JournalRejected("Subprocess receipt differs from its dispatch")

    def start(self):
        if self.first.exists():
            row = self.value(self.first)
            if row.get("schema") != "wp07_dual_first_failure_v2" or row.get("identity") != self.identity:
                raise JournalRejected("Unknown first-failure gate")
            raise JournalRejected("Previously failed dispatch requires independent review")
        try:
            number = self._reconcile()
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.failure(None, reason=str(error))
            raise JournalRejected(f"Dispatch reconciliation refused: {error}") from error
        return number, seal(self.folder / f"attempt-{number:04d}.json", {"attempt": number, "identity": self.identity})
