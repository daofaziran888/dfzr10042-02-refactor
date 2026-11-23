"""样本统计证据：分群比较与实验分析共用的样本读取层。

分群比较需要样本数、均值和方差来做显著性检验，实验分析需要两组均值和
提升幅度；这两处原本各自遍历原始样本列表。本模块把样本读取收敛为一份
逻辑，产出 :class:`SampleEvidence`，两处入口再各自基于同一份证据计算原有
的统计结果，既有输入输出保持不变。

证据除统计量外，还登记每个观测的独立身份（``observation_id``）：

* 重复送达同一编号只计一次；
* 同号异值进入待核队列，不进入当前样本，直到显式确认；
* 异号同额是不同观测，各自进入样本。

实验分析发布时冻结一份证据快照，保证显著性与提升来自同一批次；每次发布
保留当时的证据，历史结果可以原地重算。
"""
from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ACCEPTED",
    "CONFIRMED",
    "DUPLICATE",
    "PENDING_VERIFICATION",
    "REJECTED",
    "UNKNOWN",
    "AnalysisPublisher",
    "ExperimentAnalysis",
    "ExperimentAnalysisPublisher",
    "IngestResult",
    "Observation",
    "ObservationIdentity",
    "ObservationLedger",
    "PublishedAnalysis",
    "PublicationRegistry",
    "SampleEvidence",
    "SampleLedger",
    "SampleObservation",
    "SampleStatistics",
    "SampleStatisticsEvidence",
    "analyze_evidence",
    "analyze_experiment",
    "analyze_sample_evidence",
    "build_evidence",
    "build_sample_evidence",
    "build_sample_statistics",
    "collect_evidence",
    "collect_sample_evidence",
    "collect_sample_statistics",
    "compute_sample_statistics",
    "compute_welch_ttest",
    "create_sample_evidence",
    "get_sample_evidence",
    "load_sample_evidence",
    "load_sample_statistics",
    "make_sample_evidence",
    "normalize_group",
    "read_evidence",
    "read_sample_evidence",
    "read_sample_statistics",
    "read_samples",
    "run_experiment_analysis",
    "sample_statistics_evidence",
    "settle_experiment_analysis",
    "shared_sample_evidence",
    "start_analysis",
    "summarize_samples",
    "welch_ttest",
]

# 观测入样状态
ACCEPTED = "accepted"
DUPLICATE = "duplicate"
PENDING_VERIFICATION = "pending_verification"
CONFIRMED = "confirmed"
REJECTED = "rejected"
UNKNOWN = "unknown"

# 分群比较使用 a/b，实验分析使用 control/treatment；统一到后者。
_GROUP_ALIASES = {
    "a": "control",
    "group_a": "control",
    "groupa": "control",
    "control": "control",
    "ctrl": "control",
    "对照组": "control",
    "b": "treatment",
    "group_b": "treatment",
    "groupb": "treatment",
    "treatment": "treatment",
    "treat": "treatment",
    "实验组": "treatment",
}


def normalize_group(group: str) -> str:
    """把常见的分组别名归一到 control / treatment。"""
    text = str(group).strip()
    return _GROUP_ALIASES.get(text.casefold(), text)


# ============================================================
# 内部工具
# ============================================================

def _coerce_value(value: Any) -> float:
    if isinstance(value, Mapping):
        for key in ("value", "amount", "数值", "值"):
            if key in value:
                return float(value[key])
        raise ValueError("观测记录缺少 value")
    return float(value)


def _coerce_values(values: Iterable[Any] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("样本应为数值序列，而不是单个字符串")
    return tuple(_coerce_value(value) for value in values)


def _digest(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _default_batch_id(
    group_a_name: str,
    group_a: "SampleStatistics",
    group_b_name: str,
    group_b: "SampleStatistics",
) -> str:
    parts: list[str] = [group_a_name]
    parts.extend(
        f"{obs_id}:{value!r}"
        for obs_id, value in zip(group_a.observation_ids, group_a.values)
    )
    parts.append(group_b_name)
    parts.extend(
        f"{obs_id}:{value!r}"
        for obs_id, value in zip(group_b.observation_ids, group_b.values)
    )
    return f"batch:{_digest(parts)}"


def _coerce_observation(item: Any, default_group: str = "control") -> "Observation":
    if isinstance(item, Observation):
        return item
    if isinstance(item, ObservationIdentity):
        return Observation(item.observation_id, item.group, item.value, item.source)
    if isinstance(item, Mapping):
        obs_id = item.get("observation_id", item.get("obs_id", item.get("id")))
        if obs_id is None:
            raise ValueError("观测记录缺少 observation_id")
        group = item.get("group", default_group)
        return Observation(
            str(obs_id),
            str(group),
            _coerce_value(item),
            str(item.get("source", "")),
        )
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        # (id, value[, group[, source]])
        obs_id, value = item[0], item[1]
        group = item[2] if len(item) > 2 else default_group
        source = item[3] if len(item) > 3 else ""
        return Observation(str(obs_id), str(group), float(value), str(source))
    raise TypeError(f"无法识别的观测：{item!r}")


def _coerce_observations(
    items: Iterable[Any] | None,
    default_group: str,
) -> tuple["Observation", ...] | None:
    if items is None:
        return None
    return tuple(_coerce_observation(item, default_group) for item in items)


# ============================================================
# 观测身份
# ============================================================

@dataclass(frozen=True, init=False)
class ObservationIdentity:
    """一个观测的独立身份：编号 + 分组 + 数值。

    去重看编号，不看数值；因此同号异值不能被当成两个样本，
    异号同额则天然是两个独立观测。
    """

    observation_id: str
    group: str
    value: float
    source: str = ""

    def __init__(
        self,
        observation_id: str | None = None,
        group: str = "control",
        value: Any = 0.0,
        source: str = "",
        *,
        obs_id: str | None = None,
        id: str | None = None,
        amount: Any | None = None,
    ) -> None:
        object.__setattr__(
            self, "observation_id",
            str(observation_id if observation_id is not None else (obs_id or id or "")).strip(),
        )
        object.__setattr__(self, "group", normalize_group(group))
        object.__setattr__(
            self, "value",
            float(amount if amount is not None else _coerce_value(value)),
        )
        object.__setattr__(self, "source", str(source or ""))

    @property
    def obs_id(self) -> str:
        return self.observation_id

    @property
    def id(self) -> str:
        return self.observation_id

    @property
    def amount(self) -> float:
        return self.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "group": self.group,
            "value": self.value,
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, init=False)
class Observation:
    """待导入样本的观测。

    编号是去重和冲突判断的唯一依据；``source`` 记录报送来源，
    便于在证据里追溯这次导入的身份。
    """

    observation_id: str
    group: str
    value: float
    source: str = ""

    def __init__(
        self,
        observation_id: str | None = None,
        group: str | float | int = "control",
        value: Any | None = None,
        source: str = "",
        *,
        obs_id: str | None = None,
        id: str | None = None,
        amount: Any | None = None,
    ) -> None:
        # 兼容 Observation("T3", 30, "treatment") 的写法。
        if isinstance(group, (int, float)) and not isinstance(group, bool):
            numeric = float(group)
            if isinstance(value, str) and value.strip():
                group, value = value, numeric
            else:
                group = "control" if value is None else group
                value = numeric if value is None else value

        obs_id_value = observation_id if observation_id is not None else (obs_id or id or "")
        resolved_value = value if value is not None else (amount if amount is not None else 0.0)
        object.__setattr__(self, "observation_id", str(obs_id_value).strip())
        object.__setattr__(self, "group", normalize_group(str(group)))
        object.__setattr__(self, "value", float(_coerce_value(resolved_value)))
        object.__setattr__(self, "source", str(source or ""))
        if not str(obs_id_value).strip():
            raise ValueError("observation_id 不能为空")

    @property
    def identity(self) -> ObservationIdentity:
        return ObservationIdentity(
            self.observation_id, self.group, self.value, self.source,
        )

    @property
    def obs_id(self) -> str:
        return self.observation_id

    @property
    def id(self) -> str:
        return self.observation_id

    @property
    def amount(self) -> float:
        return self.value

    def as_dict(self) -> dict[str, Any]:
        return self.identity.as_dict()

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


# 语义别名：观测 / 导入观测
SampleObservation = Observation


@dataclass(frozen=True)
class IngestResult:
    """一次观测导入的结果。"""

    status: str
    observation: Observation
    changed: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (ACCEPTED, DUPLICATE, CONFIRMED)

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def duplicate(self) -> bool:
        return self.status == DUPLICATE

    @property
    def pending(self) -> bool:
        return self.status == PENDING_VERIFICATION

    @property
    def conflict(self) -> bool:
        return self.status == PENDING_VERIFICATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observation": self.observation.as_dict(),
            "changed": self.changed,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


# ============================================================
# 样本统计量与证据
# ============================================================

@dataclass(frozen=True)
class SampleStatistics:
    """一组样本的统计量。

    方差沿用原有口径：n >= 2 时使用无偏样本方差（分母 ``n - 1``）；
    样本不足时没有方差（``None``），由调用方保留原有的样本不足行为。
    """

    values: tuple[float, ...] = ()
    observation_ids: tuple[str, ...] = ()
    identities: tuple[ObservationIdentity, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _coerce_values(self.values))
        object.__setattr__(
            self, "observation_ids", tuple(str(item) for item in self.observation_ids),
        )
        object.__setattr__(
            self,
            "identities",
            tuple(
                item if isinstance(item, ObservationIdentity)
                else _coerce_observation(item).identity
                for item in self.identities
            ),
        )
        if self.observation_ids and len(self.observation_ids) != len(self.values):
            raise ValueError("observation_ids 与 values 数量不一致")
        if self.identities and len(self.identities) != len(self.values):
            raise ValueError("identities 与 values 数量不一致")

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def sample_size(self) -> int:
        return len(self.values)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def average(self) -> float:
        return self.mean

    @property
    def variance(self) -> float | None:
        if len(self.values) < 2:
            return None
        mean = self.mean
        return sum((value - mean) ** 2 for value in self.values) / (len(self.values) - 1)

    @property
    def var(self) -> float | None:
        return self.variance

    @property
    def std(self) -> float | None:
        variance = self.variance
        return None if variance is None else math.sqrt(variance)

    @property
    def is_empty(self) -> bool:
        return not self.values

    @property
    def is_sufficient(self) -> bool:
        return len(self.values) >= 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "values": list(self.values),
            "observation_ids": list(self.observation_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class SampleEvidence:
    """一次样本读取的完整证据：两组统计量 + 观测身份 + 批次号。"""

    group_a: SampleStatistics
    group_b: SampleStatistics
    group_a_name: str = "a"
    group_b_name: str = "b"
    batch_id: str = ""
    observations: tuple[ObservationIdentity, ...] = ()
    revision: int | None = None
    duplicates: tuple[ObservationIdentity, ...] = ()
    conflicts: tuple[ObservationIdentity, ...] = ()

    # ---------- 分群比较口径 ----------

    @property
    def n_a(self) -> int:
        return self.group_a.n

    @property
    def n_b(self) -> int:
        return self.group_b.n

    @property
    def mean_a(self) -> float:
        return self.group_a.mean

    @property
    def mean_b(self) -> float:
        return self.group_b.mean

    @property
    def var_a(self) -> float | None:
        return self.group_a.variance

    @property
    def var_b(self) -> float | None:
        return self.group_b.variance

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return self.group_a.observation_ids + self.group_b.observation_ids

    # ---------- 实验分析口径 ----------

    @property
    def control(self) -> SampleStatistics:
        return self.group_a

    @property
    def treatment(self) -> SampleStatistics:
        return self.group_b

    @property
    def control_statistics(self) -> SampleStatistics:
        return self.group_a

    @property
    def treatment_statistics(self) -> SampleStatistics:
        return self.group_b

    @property
    def n_control(self) -> int:
        return self.group_a.n

    @property
    def n_treatment(self) -> int:
        return self.group_b.n

    @property
    def control_mean(self) -> float:
        return self.group_a.mean

    @property
    def treatment_mean(self) -> float:
        return self.group_b.mean

    @property
    def sample_size(self) -> dict[str, int]:
        return {"control": self.n_control, "treatment": self.n_treatment}

    @property
    def means(self) -> dict[str, float]:
        return {"control": self.control_mean, "treatment": self.treatment_mean}

    @property
    def variances(self) -> dict[str, float | None]:
        return {"control": self.var_a, "treatment": self.var_b}

    @property
    def empty(self) -> bool:
        return self.group_a.is_empty and self.group_b.is_empty

    @property
    def is_empty(self) -> bool:
        return self.empty

    @property
    def sufficient(self) -> bool:
        return self.group_a.is_sufficient and self.group_b.is_sufficient

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)

    def statistics_for(self, group: str) -> SampleStatistics | None:
        name = normalize_group(group)
        if name == normalize_group(self.group_a_name):
            return self.group_a
        if name == normalize_group(self.group_b_name):
            return self.group_b
        return None

    def has_observation(self, observation_id: str) -> bool:
        return str(observation_id) in self.observation_ids

    def identity_of(self, observation_id: str) -> ObservationIdentity | None:
        wanted = str(observation_id)
        for observation in self.observations:
            if observation.observation_id == wanted:
                return observation
        return None

    def lift(self) -> float:
        if self.control_mean > 0:
            return round(
                (self.treatment_mean - self.control_mean) / self.control_mean, 4,
            )
        return 0.0

    def ttest(self) -> dict[str, Any]:
        return welch_ttest(self)

    def significance(self) -> dict[str, Any]:
        return welch_ttest(self)

    def with_batch_id(self, batch_id: str) -> "SampleEvidence":
        return replace(self, batch_id=batch_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_a": self.group_a.as_dict(),
            "group_b": self.group_b.as_dict(),
            "group_a_name": self.group_a_name,
            "group_b_name": self.group_b_name,
            "batch_id": self.batch_id,
            "revision": self.revision,
            "observations": [obs.as_dict() for obs in self.observations],
            "duplicates": [obs.as_dict() for obs in self.duplicates],
            "conflicts": [obs.as_dict() for obs in self.conflicts],
            "n_a": self.n_a,
            "n_b": self.n_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "var_a": self.var_a,
            "var_b": self.var_b,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "lift": self.lift(),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    @classmethod
    def from_samples(
        cls,
        group_a_values: Sequence[Any] | None = None,
        group_b_values: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> "SampleEvidence":
        return read_sample_evidence(group_a_values, group_b_values, **kwargs)


# 语义别名：样本统计证据
SampleStatisticsEvidence = SampleEvidence


def _build_statistics(
    values: Sequence[Any] | None,
    observations: Sequence[Observation] | None,
    group_name: str,
    fallback_base: bool = True,
) -> tuple[SampleStatistics, tuple[ObservationIdentity, ...], tuple[ObservationIdentity, ...]]:
    """把一组原始样本读成统计量。

    返回 (统计量, 重复观测, 冲突观测)。同一个编号只计一次：
    同号同值记为重复，同号异值不入样、进入冲突待核。
    """
    if values is None:
        degraded = _build_statistics_from_observations(observations, group_name)
        return degraded

    parsed_values = _coerce_values(values)
    duplicates: list[ObservationIdentity] = []
    conflicts: list[ObservationIdentity] = []

    if observations:
        parsed_observations = tuple(observations)
        if len(parsed_observations) != len(parsed_values):
            # 观测列表里带了重复/冲突条目时，以观测列表为准；但去重后的
            # 数值必须与原始样本一致，避免静默丢样本。
            statistics, dups, conflicts = _build_statistics_from_observations(
                parsed_observations, group_name,
            )
            if statistics.values != parsed_values:
                raise ValueError("观测身份数量与样本数量不一致")
            return statistics, dups, conflicts
        identities: list[ObservationIdentity] = []
        kept_values: list[float] = []
        seen: dict[str, ObservationIdentity] = {}
        for obs, value in zip(parsed_observations, parsed_values):
            identity = ObservationIdentity(obs.observation_id, group_name, value, obs.source)
            previous = seen.get(identity.observation_id)
            if previous is not None:
                if previous.value == identity.value:
                    duplicates.append(identity)
                else:
                    conflicts.append(identity)
                continue
            seen[identity.observation_id] = identity
            identities.append(identity)
            kept_values.append(value)
    elif fallback_base:
        identities = [
            ObservationIdentity(f"BASE:{group_name}:{index}", group_name, value, "base")
            for index, value in enumerate(parsed_values)
        ]
        kept_values = list(parsed_values)
    else:
        identities = [
            ObservationIdentity("", group_name, value, "base")
            for value in parsed_values
        ]
        kept_values = list(parsed_values)

    statistics = SampleStatistics(
        values=tuple(kept_values),
        observation_ids=tuple(obs.observation_id for obs in identities),
        identities=tuple(identities),
    )
    return statistics, tuple(duplicates), tuple(conflicts)


def _build_statistics_from_observations(
    observations: Sequence[Observation] | None,
    group_name: str,
) -> tuple[SampleStatistics, tuple[ObservationIdentity, ...], tuple[ObservationIdentity, ...]]:
    if observations is None:
        return SampleStatistics(), (), ()
    parsed = tuple(observations)
    identities: list[ObservationIdentity] = []
    duplicates: list[ObservationIdentity] = []
    conflicts: list[ObservationIdentity] = []
    seen: dict[str, ObservationIdentity] = {}
    for obs in parsed:
        identity = ObservationIdentity(obs.observation_id, group_name, obs.value, obs.source)
        previous = seen.get(identity.observation_id)
        if previous is not None:
            if previous.value == identity.value:
                duplicates.append(identity)
            else:
                conflicts.append(identity)
            continue
        seen[identity.observation_id] = identity
        identities.append(identity)
    statistics = SampleStatistics(
        values=tuple(obs.value for obs in identities),
        observation_ids=tuple(obs.observation_id for obs in identities),
        identities=tuple(identities),
    )
    return statistics, tuple(duplicates), tuple(conflicts)



def read_sample_evidence(
    group_a_values: Sequence[Any] | None = None,
    group_b_values: Sequence[Any] | None = None,
    *,
    observations_a: Sequence[Any] | None = None,
    observations_b: Sequence[Any] | None = None,
    group_a_name: str = "a",
    group_b_name: str = "b",
    batch_id: str = "",
    revision: int | None = None,
    observations: Sequence[Any] | None = None,
) -> SampleEvidence:
    """单次读取两组样本，生成共用的统计证据。

    这是分群比较与实验分析唯一的样本读取入口。既接受原始数值列表，
    也接受带身份编号的观测；原始列表中的每个值会获得确定性的基础编号，
    保证证据里的每个观测都有独立身份。

    也可以只传一个 ``observations`` 序列，由每项自带的 ``group`` 分到两组。
    """
    if observations is not None:
        parsed_all = _coerce_observations(observations, "control") or ()
        obs_a = tuple(
            obs for obs in parsed_all
            if obs.group == normalize_group(group_a_name)
        )
        obs_b = tuple(
            obs for obs in parsed_all
            if obs.group == normalize_group(group_b_name)
        )
        observations_a = observations_a if observations_a is not None else obs_a
        observations_b = observations_b if observations_b is not None else obs_b
        if group_a_values is None:
            group_a_values = tuple(obs.value for obs in obs_a)
        if group_b_values is None:
            group_b_values = tuple(obs.value for obs in obs_b)

    parsed_obs_a = _coerce_observations(observations_a, group_a_name)
    parsed_obs_b = _coerce_observations(observations_b, group_b_name)

    parsed_a, duplicates_a, conflicts_a = _build_statistics(
        group_a_values, parsed_obs_a, group_a_name,
    )
    parsed_b, duplicates_b, conflicts_b = _build_statistics(
        group_b_values, parsed_obs_b, group_b_name,
    )
    identities = parsed_a.identities + parsed_b.identities
    duplicate_identities = tuple(duplicates_a + duplicates_b)
    conflict_identities = tuple(conflicts_a + conflicts_b)
    if not batch_id:
        batch_id = _default_batch_id(
            group_a_name, parsed_a, group_b_name, parsed_b,
        )
    return SampleEvidence(
        group_a=parsed_a,
        group_b=parsed_b,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        batch_id=batch_id,
        observations=identities,
        revision=revision,
        duplicates=duplicate_identities,
        conflicts=conflict_identities,
    )


# 语义别名：一份样本读取逻辑
collect_sample_evidence = read_sample_evidence
build_sample_evidence = read_sample_evidence
read_samples = read_sample_evidence
load_sample_evidence = read_sample_evidence
collect_evidence = read_sample_evidence
build_evidence = read_sample_evidence


def compute_sample_statistics(
    values: Sequence[Any] | None,
    *,
    group: str = "a",
    observation_ids: Sequence[str] | None = None,
) -> SampleStatistics:
    """读取单组样本统计量；供需要单组均值和方差的调用方复用。"""
    observations = None
    if observation_ids is not None:
        observations = tuple(
            Observation(obs_id, group, value)
            for obs_id, value in zip(observation_ids, _coerce_values(values))
        )
    return _build_statistics(values, observations, normalize_group(group))


summarize_samples = compute_sample_statistics
build_sample_statistics = compute_sample_statistics
read_sample_statistics = compute_sample_statistics
collect_sample_statistics = compute_sample_statistics
load_sample_statistics = compute_sample_statistics

# 证据构建入口的其余命名，统一指向同一份读取逻辑。
create_sample_evidence = read_sample_evidence
make_sample_evidence = read_sample_evidence
get_sample_evidence = read_sample_evidence
shared_sample_evidence = read_sample_evidence
read_evidence = read_sample_evidence
sample_statistics_evidence = read_sample_evidence


# ============================================================
# Welch 检验：供分群比较直接调用
# ============================================================

def welch_ttest(evidence: SampleEvidence) -> dict[str, Any]:
    """基于样本证据执行 Welch's t-test，返回原有入口的统计结果。"""
    n_a, n_b = evidence.n_a, evidence.n_b
    if n_a < 2 or n_b < 2:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "reason": "样本不足",
        }

    mean_a = evidence.mean_a
    mean_b = evidence.mean_b
    var_a = evidence.var_a
    var_b = evidence.var_b
    if var_a is None or var_b is None:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "reason": "样本不足",
        }

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite 自由度
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df = df_num / df_den

    # 延迟导入，避免 sample_evidence 依赖 models 的其余内容。
    from app.models import _t_distribution_pvalue

    p_value = _t_distribution_pvalue(abs(t_stat), df)

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "n_a": n_a,
        "n_b": n_b,
    }


compute_welch_ttest = welch_ttest


# ============================================================
# 已发布分析：冻结证据、同批次统计、可重算
# ============================================================

@dataclass(frozen=True)
class PublishedAnalysis:
    """一次实验分析发布结果，绑定发布时实际采用的样本证据。"""

    report_id: str
    version: int
    evidence: SampleEvidence
    sample_source: str
    control_mean: float
    treatment_mean: float
    lift: float
    t_stat: float
    p_value: float
    significant: bool
    ledger_revision: int | None = None

    @property
    def batch_id(self) -> str:
        return self.evidence.batch_id

    @property
    def n_control(self) -> int:
        return self.evidence.n_a

    @property
    def n_treatment(self) -> int:
        return self.evidence.n_b

    # 分群比较口径的别名
    @property
    def n_a(self) -> int:
        return self.n_control

    @property
    def n_b(self) -> int:
        return self.n_treatment

    @property
    def mean_a(self) -> float:
        return self.control_mean

    @property
    def mean_b(self) -> float:
        return self.treatment_mean

    @property
    def sample_size(self) -> dict[str, int]:
        return {"control": self.n_control, "treatment": self.n_treatment}

    @property
    def means(self) -> dict[str, float]:
        return {"control": self.control_mean, "treatment": self.treatment_mean}

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return self.evidence.observation_ids

    @property
    def has_conflicts(self) -> bool:
        """结论是否有同号异值待核（待核观测本身不在样本里）。"""
        return self.evidence.has_conflicts

    @property
    def conflicts(self) -> tuple[ObservationIdentity, ...]:
        return self.evidence.conflicts

    @property
    def duplicate_count(self) -> int:
        return self.evidence.duplicate_count

    @property
    def data_source(self) -> str:
        return self.sample_source

    @property
    def significance_source(self) -> dict[str, Any]:
        """显著性实际使用的数据来源：与提升同一份证据批次。"""
        return {
            "evidence": "sample_statistics",
            "statistic": "welch_ttest",
            "sample_source": self.sample_source,
            "batch_id": self.batch_id,
            "ledger_revision": self.ledger_revision,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "observation_ids": list(self.observation_ids),
        }

    def recompute(self) -> dict[str, Any]:
        """按当时实际采用的样本证据重算，结果应与发布值一致。"""
        replay = analyze_evidence(
            self.evidence,
            report_id=self.report_id,
            version=self.version,
            sample_source=self.sample_source,
            ledger_revision=self.ledger_revision,
        )
        return replay.as_dict()

    def recompute_matches(self) -> bool:
        return self.recompute() == self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "batch_id": self.batch_id,
            "sample_source": self.sample_source,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "lift": self.lift,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "significant": self.significant,
            "observation_ids": list(self.observation_ids),
            "significance_source": self.significance_source,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


def analyze_evidence(
    evidence: SampleEvidence,
    *,
    report_id: str = "",
    version: int = 1,
    sample_source: str = "original",
    ledger_revision: int | None = None,
) -> PublishedAnalysis:
    """从同一份证据同时计算显著性和提升，避免跨批次拼接。"""
    result = welch_ttest(evidence)
    control_mean = evidence.mean_a if evidence.n_a else 0.0
    treatment_mean = evidence.mean_b if evidence.n_b else 0.0
    lift = 0.0
    if control_mean > 0:
        lift = round((treatment_mean - control_mean) / control_mean, 4)
    return PublishedAnalysis(
        report_id=report_id,
        version=version,
        evidence=evidence,
        sample_source=sample_source,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        lift=lift,
        t_stat=float(result.get("t_stat", 0.0)),
        p_value=float(result.get("p_value", 1.0)),
        significant=bool(result.get("significant", False)),
        ledger_revision=ledger_revision,
    )


analyze_sample_evidence = analyze_evidence


# ============================================================
# 观测台账：独立身份、去重、冲突待核
# ============================================================

class SampleLedger:
    """实验样本的观测台账。

    - 同号同值重复送达：幂等，不改变样本，计入 duplicate；
    - 同号异值：进入待核队列，不进入样本，直到显式确认；
    - 异号同额：编号不同，各自进入样本。
    """

    def __init__(
        self,
        control: Sequence[Any] | None = None,
        treatment: Sequence[Any] | None = None,
        *,
        observations: Sequence[Any] | None = None,
        group_names: tuple[str, str] = ("control", "treatment"),
        id_prefix: str = "BASE",
    ) -> None:
        self._lock = threading.RLock()
        self._accepted: dict[str, Observation] = {}
        self._order: list[str] = []
        self._conflicts: dict[str, list[Observation]] = {}
        self._duplicates: list[str] = []
        self._revision = 0
        self.group_names = (
            normalize_group(group_names[0]), normalize_group(group_names[1]),
        )
        parsed_observations = _coerce_observations(observations, self.group_names[0]) or ()
        # 某组一旦提供了观测身份，就以观测为准，避免原始值与观测重复登记。
        observed_groups = {obs.group for obs in parsed_observations}
        if control is not None and self.group_names[0] not in observed_groups:
            self.seed_group(control, self.group_names[0], id_prefix=id_prefix)
        if treatment is not None and self.group_names[1] not in observed_groups:
            self.seed_group(treatment, self.group_names[1], id_prefix=id_prefix)
        for item in parsed_observations:
            self.ingest(item)

    # ---------- 只读视图 ----------

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def duplicate_count(self) -> int:
        with self._lock:
            return len(self._duplicates)

    @property
    def duplicates(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._duplicates)

    def __len__(self) -> int:
        with self._lock:
            return len(self._accepted)

    def __contains__(self, observation_id: object) -> bool:
        with self._lock:
            return str(observation_id) in self._accepted

    def accepted(self, group: str | None = None) -> tuple[Observation, ...]:
        with self._lock:
            if group is None:
                return tuple(self._accepted[obs_id] for obs_id in self._order)
            wanted = normalize_group(group)
            return tuple(
                self._accepted[obs_id]
                for obs_id in self._order
                if self._accepted[obs_id].group == wanted
            )

    def accepted_ids(self, group: str | None = None) -> tuple[str, ...]:
        return tuple(obs.observation_id for obs in self.accepted(group))

    def pending(self, group: str | None = None) -> tuple[Observation, ...]:
        with self._lock:
            rows = [obs for items in self._conflicts.values() for obs in items]
            if group is None:
                return tuple(rows)
            wanted = normalize_group(group)
            return tuple(obs for obs in rows if obs.group == wanted)

    def conflicts(self) -> dict[str, tuple[Observation, ...]]:
        with self._lock:
            return {key: tuple(value) for key, value in self._conflicts.items()}

    # 别名
    pending_observations = pending
    accepted_observations = accepted

    # ---------- 写入 ----------

    def seed_group(
        self,
        values: Sequence[Any],
        group: str,
        *,
        observation_ids: Sequence[str] | None = None,
        source: str = "base",
        id_prefix: str = "BASE",
    ) -> tuple[Observation, ...]:
        """登记初始样本。基础样本不计入变更 revision。"""
        normalized = normalize_group(group)
        parsed_values = _coerce_values(values)
        if observation_ids is not None:
            ids = tuple(str(obs_id) for obs_id in observation_ids)
            if len(ids) != len(parsed_values):
                raise ValueError("observation_ids 与 values 数量不一致")
        else:
            ids = tuple(
                f"{id_prefix}:{normalized}:{index}"
                for index in range(len(parsed_values))
            )

        seeded: list[Observation] = []
        with self._lock:
            for obs_id, value in zip(ids, parsed_values):
                observation = Observation(obs_id, normalized, value, source)
                existing = self._accepted.get(obs_id)
                if existing is not None and existing.value != observation.value:
                    self._record_conflict(observation)
                    continue
                if obs_id not in self._accepted:
                    self._order.append(obs_id)
                self._accepted[obs_id] = observation
                seeded.append(observation)
        return tuple(seeded)

    # 别名
    seed = seed_group
    initialize = seed_group
    load = seed_group

    def ingest(self, observation: Any) -> IngestResult:
        """导入一个观测；同号同值幂等，同号异值待核。"""
        item = _coerce_observation(observation, self.group_names[0] if hasattr(self, "group_names") else "control")
        with self._lock:
            existing = self._accepted.get(item.observation_id)
            if existing is not None:
                if (
                    existing.group == item.group
                    and existing.value == item.value
                ):
                    self._duplicates.append(item.observation_id)
                    return IngestResult(
                        DUPLICATE,
                        item,
                        changed=False,
                        reason="同号同值重复送达，只计一次",
                    )
                self._record_conflict(item)
                return IngestResult(
                    PENDING_VERIFICATION,
                    item,
                    changed=False,
                    reason="同号异值，待核对后再入样",
                )

            if item.observation_id in self._conflicts:
                self._record_conflict(item)
                return IngestResult(
                    PENDING_VERIFICATION,
                    item,
                    changed=False,
                    reason="已有同号冲突待核，本次同样待核",
                )

            self._accepted[item.observation_id] = item
            self._order.append(item.observation_id)
            self._revision += 1
            return IngestResult(
                ACCEPTED,
                item,
                changed=True,
                reason="观测已入样",
            )

    # 别名
    add_observation = ingest
    record_observation = ingest
    add = ingest
    supplement = ingest
    report_observation = ingest

    def confirm(
        self,
        observation_id: Any,
        value: float | None = None,
    ) -> IngestResult:
        """核对冲突观测：确认指定数值后用它替换同号观测并入样。"""
        if isinstance(observation_id, Observation):
            value = observation_id.value if value is None else value
            observation_id = observation_id.observation_id
        obs_id = str(observation_id).strip()
        with self._lock:
            existing = self._accepted.get(obs_id)
            pending = list(self._conflicts.get(obs_id, []))
            if existing is None and not pending:
                return IngestResult(
                    UNKNOWN,
                    Observation(obs_id, self.group_names[0], 0.0),
                    changed=False,
                    reason="没有待核或已入样的观测",
                )

            candidates: list[Observation] = []
            if existing is not None:
                candidates.append(existing)
            candidates.extend(pending)

            target: Observation | None = None
            if value is not None:
                wanted = float(value)
                for candidate in candidates:
                    if candidate.value == wanted:
                        target = candidate
                        break
                if target is None:
                    base = existing or pending[0]
                    target = Observation(
                        obs_id, base.group, wanted, base.source or "confirmed",
                    )
            else:
                target = pending[-1] if pending else existing
            assert target is not None

            changed = (
                existing is None
                or existing.value != target.value
                or existing.group != target.group
            )
            if existing is None:
                self._order.append(obs_id)
            self._accepted[obs_id] = target
            self._conflicts.pop(obs_id, None)
            if changed:
                self._revision += 1
            return IngestResult(
                CONFIRMED,
                target,
                changed=changed,
                reason="冲突已核对，确认值已入样",
            )

    # 别名
    approve = confirm
    confirm_observation = confirm

    def reject(self, observation_id: Any, value: float | None = None) -> IngestResult:
        """驳回待核观测；已入样的原值保持不变。"""
        obs_id = str(observation_id).strip()
        with self._lock:
            pending = list(self._conflicts.get(obs_id, []))
            existing = self._accepted.get(obs_id)
            if not pending:
                return IngestResult(
                    UNKNOWN,
                    existing or Observation(obs_id, self.group_names[0], 0.0),
                    changed=False,
                    reason="没有待核观测",
                )
            rejected = pending[-1]
            if value is not None:
                for candidate in pending:
                    if candidate.value == float(value):
                        rejected = candidate
                        break
            self._conflicts.pop(obs_id, None)
            return IngestResult(
                REJECTED,
                rejected,
                changed=False,
                reason="待客观测已驳回",
            )

    reject_observation = reject

    def _record_conflict(self, observation: Observation) -> None:
        pending = self._conflicts.setdefault(observation.observation_id, [])
        if not any(
            item.value == observation.value and item.group == observation.group
            for item in pending
        ):
            pending.append(observation)

    # ---------- 证据快照 ----------

    def snapshot(
        self,
        *,
        batch_id: str = "",
        group_names: tuple[str, str] | None = None,
    ) -> SampleEvidence:
        evidence, _ = self.snapshot_with_revision(
            batch_id=batch_id, group_names=group_names,
        )
        return evidence

    read = snapshot
    evidence = snapshot
    build_evidence = snapshot
    sample_evidence = snapshot

    def snapshot_with_revision(
        self,
        *,
        batch_id: str = "",
        group_names: tuple[str, str] | None = None,
    ) -> tuple[SampleEvidence, int]:
        """在同一把锁内冻结样本与 revision，保证快照原子。"""
        with self._lock:
            group_a_name, group_b_name = group_names or self.group_names
            obs_a = tuple(
                self._accepted[obs_id]
                for obs_id in self._order
                if self._accepted[obs_id].group == normalize_group(group_a_name)
            )
            obs_b = tuple(
                self._accepted[obs_id]
                for obs_id in self._order
                if self._accepted[obs_id].group == normalize_group(group_b_name)
            )
            evidence = read_sample_evidence(
                tuple(obs.value for obs in obs_a),
                tuple(obs.value for obs in obs_b),
                observations_a=obs_a,
                observations_b=obs_b,
                group_a_name=group_a_name,
                group_b_name=group_b_name,
                batch_id=batch_id,
                revision=self._revision,
            )
            # 待核冲突不进入样本，但要在证据里留痕，便于结论标注来源。
            pending = tuple(
                obs for obs_id in sorted(self._conflicts) for obs in self._conflicts[obs_id]
            )
            evidence = replace(
                evidence,
                duplicates=tuple(obs.identity for obs in self._accepted.values()
                                 if obs.observation_id in self._duplicates)
                or evidence.duplicates,
                conflicts=evidence.conflicts + pending,
            )
            return evidence, self._revision


# 语义别名：观测台账
ObservationLedger = SampleLedger


# ============================================================
# 实验分析发布：同一次分析只发布一次
# ============================================================

class PublicationRegistry:
    """按分析标识登记发布结果，保证并发下同一次分析只发布一次。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._published: dict[str, PublishedAnalysis] = {}
        self._history: dict[str, list[PublishedAnalysis]] = {}

    def get(self, key: str) -> PublishedAnalysis | None:
        with self._lock:
            return self._published.get(key)

    def history(self, key: str) -> tuple[PublishedAnalysis, ...]:
        with self._lock:
            return tuple(self._history.get(key, ()))

    def publish(
        self,
        key: str,
        factory: Any,
    ) -> tuple[PublishedAnalysis, bool]:
        """并发调用时只有一个 factory 真正执行；其余拿到同一份发布结果。"""
        with self._lock:
            existing = self._published.get(key)
            if existing is not None:
                return existing, False
            result = factory() if callable(factory) else factory
            self._published[key] = result
            self._history.setdefault(key, []).append(result)
            return result, True

    def publish_once(
        self,
        key: str,
        factory: Any,
    ) -> PublishedAnalysis:
        result, _ = self.publish(key, factory)
        return result

    def append(self, key: str, analysis: PublishedAnalysis) -> PublishedAnalysis:
        """追加一版显式重算的历史结果，不覆盖已发布版本。"""
        with self._lock:
            self._history.setdefault(key, []).append(analysis)
            return analysis

    # 别名
    record = publish_once

    def reset(self) -> None:
        with self._lock:
            self._published.clear()
            self._history.clear()


# 全局发布登记处：同一 report_id 的并发发布共享一次结果。
PUBLICATION_REGISTRY = PublicationRegistry()


class AnalysisPublisher:
    """一次实验分析的发布器，绑定台账并冻结证据批次。"""

    def __init__(
        self,
        analysis_id: str = "",
        *,
        ledger: SampleLedger | None = None,
        registry: PublicationRegistry | None = None,
        group_names: tuple[str, str] | None = None,
    ) -> None:
        self.analysis_id = str(analysis_id or "")
        self.ledger = ledger
        # 默认每次发布器自带登记处，避免不同分析之间串结果；
        # 需要跨实例去重时显式传入同一个 registry。
        self.registry = registry or PublicationRegistry()
        self.group_names = group_names
        self._lock = threading.RLock()

    def __enter__(self) -> "AnalysisPublisher":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    @property
    def _key(self) -> str:
        return self.analysis_id or f"analysis:{id(self)}"

    @property
    def published(self) -> PublishedAnalysis | None:
        return self.registry.get(self._key)

    def publish(
        self,
        evidence: SampleEvidence | None = None,
        *,
        report_id: str = "",
        version: int = 1,
    ) -> PublishedAnalysis:
        """发布一次分析；同一 key 并发调用只产生一次发布。"""
        def factory() -> PublishedAnalysis:
            snapshot = evidence
            revision: int | None = None
            source = "explicit"
            if snapshot is None:
                if self.ledger is None:
                    raise ValueError("缺少样本证据或观测台账")
                snapshot, revision = self.ledger.snapshot_with_revision(
                    group_names=self.group_names,
                )
                source = f"ledger:{snapshot.batch_id}"
            if snapshot.revision is None and revision is not None:
                snapshot = replace(snapshot, revision=revision)
            return analyze_evidence(
                snapshot,
                report_id=report_id or self.analysis_id,
                version=version,
                sample_source=source,
                ledger_revision=revision,
            )

        result, _ = self.registry.publish(self._key, factory)
        return result

    # 别名
    publish_once = publish
    publish_analysis = publish

    def history(self) -> tuple[PublishedAnalysis, ...]:
        return self.registry.history(self._key)


# 语义别名：实验分析发布器
ExperimentAnalysisPublisher = AnalysisPublisher


class ExperimentAnalysis:
    """一次实验分析的完整生命周期。

    ``begin`` 冻结当时的样本证据；在检验之后、结论组装之前补报观测时，
    可以按 ``original`` 用原完整输入结清，或用 ``latest`` 换入已确认的
    补报并重新检验。两种方式都只使用同一份证据，显著性和提升不会跨批次。
    """

    def __init__(
        self,
        report_id: str = "",
        *,
        control_data: Sequence[Any] | None = None,
        treatment_data: Sequence[Any] | None = None,
        ledger: SampleLedger | None = None,
        observations_control: Sequence[Any] | None = None,
        observations_treatment: Sequence[Any] | None = None,
        experiment: Any = None,
        metric: Any = None,
        registry: PublicationRegistry | None = None,
    ) -> None:
        self.report_id = str(report_id or "")
        self.experiment = experiment
        self.metric = metric
        self.registry = registry or PublicationRegistry()
        if ledger is not None:
            self.ledger = ledger
        elif control_data is not None or treatment_data is not None or observations_control or observations_treatment:
            self.ledger = SampleLedger(
                control_data or (),
                treatment_data or (),
                observations=tuple(observations_control or ()) + tuple(observations_treatment or ()),
            )
        else:
            self.ledger = SampleLedger()
        self._lock = threading.RLock()
        self._snapshot: SampleEvidence | None = None
        self._revision: int | None = None
        self._test_result: dict[str, Any] | None = None
        self._published: PublishedAnalysis | None = None

    # ---------- 冻结与检验 ----------

    def begin(self) -> SampleEvidence:
        """冻结当前样本，后续检验都基于这份证据。"""
        with self._lock:
            self._snapshot, self._revision = self.ledger.snapshot_with_revision()
            return self._snapshot

    # 别名
    freeze = begin
    snapshot = begin

    @property
    def evidence(self) -> SampleEvidence:
        if self._snapshot is None:
            return self.begin()
        return self._snapshot

    @property
    def revision(self) -> int | None:
        return self._revision

    def test(self) -> dict[str, Any]:
        """基于冻结证据做显著性检验，保留原分群比较的返回结构。"""
        with self._lock:
            evidence = self.evidence
            self._test_result = welch_ttest(evidence)
            return dict(self._test_result)

    compute_ttest = test
    significance = test

    def current_evidence(self) -> SampleEvidence:
        return self.ledger.snapshot()

    # ---------- 补报观测 ----------

    def add_observation(self, observation: Any) -> IngestResult:
        return self.ledger.ingest(observation)

    ingest = add_observation
    report_observation = add_observation
    supplement = add_observation

    def confirm_observation(
        self, observation_id: Any, value: float | None = None,
    ) -> IngestResult:
        return self.ledger.confirm(observation_id, value)

    confirm = confirm_observation

    # ---------- 结清与发布 ----------

    def settle(self, policy: str = "original") -> PublishedAnalysis:
        """组装结论。

        ``original``：按检验时的完整输入结清，补报的观测留待下一次分析；
        ``latest``：换入已确认的补报，基于更新后的完整样本重新检验。
        两种方式都只用一份证据，不会出现均值来自新样本、显著性来自旧样本。
        """
        with self._lock:
            if policy not in ("original", "latest"):
                raise ValueError("policy 只能是 original 或 latest")
            if policy == "original":
                evidence = self.evidence
                revision = self._revision
                source = "original"
            else:
                evidence, revision = self.ledger.snapshot_with_revision()
                source = "confirmed_supplement"
            return analyze_evidence(
                evidence,
                report_id=self.report_id,
                version=1,
                sample_source=source,
                ledger_revision=revision,
            )

    def publish(self, policy: str = "original") -> PublishedAnalysis:
        """发布结清结果；同一次分析并发调用只发布一次。

        结清动作放在登记处的临界区里执行：并发到达时只有一个线程
        真正冻结并计算，其余线程拿到同一份发布结果，因此显著性
        和提升一定来自同一批次。
        """
        key = f"experiment_analysis:{self.report_id or id(self)}"

        def factory() -> PublishedAnalysis:
            return self.settle(policy)

        published, _ = self.registry.publish(key, factory)
        self._published = published
        return published

    publish_once = publish

    @property
    def published(self) -> PublishedAnalysis | None:
        return self._published or self.registry.get(
            f"experiment_analysis:{self.report_id or id(self)}"
        )

    def history(self) -> tuple[PublishedAnalysis, ...]:
        return self.registry.history(
            f"experiment_analysis:{self.report_id or id(self)}"
        )

    def recompute(self) -> dict[str, Any]:
        if self._published is None:
            raise ValueError("尚未发布")
        return self._published.recompute()

    def reset(self) -> None:
        """重新开始一次分析会话（显式重算场景）。"""
        with self._lock:
            self._snapshot = None
            self._revision = None
            self._test_result = None
            self._published = None
            self.registry.reset()


# 语义别名
ExperimentAnalysisSession = ExperimentAnalysis
ExperimentResultAnalysis = ExperimentAnalysis


# ============================================================
# 函数式入口：一次跑完「检验 → 补报 → 结清」
# ============================================================

def start_analysis(
    report_id: str,
    control_data: Sequence[Any] | None = None,
    treatment_data: Sequence[Any] | None = None,
    *,
    observations_control: Sequence[Any] | None = None,
    observations_treatment: Sequence[Any] | None = None,
    experiment: Any = None,
    metric: Any = None,
) -> ExperimentAnalysis:
    """创建并冻结一次实验分析会话。"""
    session = ExperimentAnalysis(
        report_id,
        control_data=control_data or (),
        treatment_data=treatment_data or (),
        observations_control=observations_control,
        observations_treatment=observations_treatment,
        experiment=experiment,
        metric=metric,
    )
    session.begin()
    return session


def analyze_experiment(
    control_data: Sequence[Any] | None = None,
    treatment_data: Sequence[Any] | None = None,
    *,
    supplement: Sequence[Any] | IngestResult | None = None,
    policy: str = "original",
    report_id: str = "",
    observations_control: Sequence[Any] | None = None,
    observations_treatment: Sequence[Any] | None = None,
) -> PublishedAnalysis:
    """检验后补报观测，再按 ``original`` / ``latest`` 结清。

    ``original`` 只使用检验时冻结的原完整输入，``latest`` 换入补报后
    的完整样本重新检验；两种口径的显著性和提升都来自同一份证据。
    """
    session = start_analysis(
        report_id,
        control_data,
        treatment_data,
        observations_control=observations_control,
        observations_treatment=observations_treatment,
    )
    session.test()
    if supplement is not None:
        items = (supplement,) if isinstance(supplement, (Observation, IngestResult)) else supplement
        for item in items:
            session.add_observation(item.observation if isinstance(item, IngestResult) else item)
    return session.settle(policy)


run_experiment_analysis = analyze_experiment
settle_experiment_analysis = analyze_experiment
