"""洞察工坊 · 样本统计证据共享测试

验证分群比较与实验分析共用同一份样本读取逻辑，同时保留两处原有输出；
并覆盖观测身份登记、重复/冲突/异号同额、补报观测后的两种结清口径、
并发只发布一次以及历史结果可重算。
"""
from __future__ import annotations

import math
import threading

import pytest

from app.models import (
    ABExperiment,
    AggregationMethod,
    ExperimentResult,
    MetricDataSource,
    MetricDef,
    MetricPeriod,
    Segment,
    UserSegmentCompare,
)
from app.sample_evidence import (
    ACCEPTED,
    CONFIRMED,
    DUPLICATE,
    PENDING_VERIFICATION,
    ExperimentAnalysis,
    IngestResult,
    Observation,
    SampleLedger,
    analyze_evidence,
    normalize_group,
    read_sample_evidence,
    welch_ttest,
)


# ============================================================
# 共享测试对象
# ============================================================

@pytest.fixture
def metric():
    return MetricDef(
        metric_id="M_EVIDENCE",
        metric_name="转化率",
        calculation_formula="COUNT(conversions)/COUNT(users)",
        aggregation=AggregationMethod.AVG,
        data_source=MetricDataSource.CLICKHOUSE,
        period=MetricPeriod.DAILY,
        owner="analyst",
        field_name="rate",
        table_name="experiment_events",
    )


@pytest.fixture
def experiment():
    return ABExperiment(
        experiment_id="EXP_EVIDENCE",
        experiment_name="样本证据实验",
        groups={"control": 0.5, "treatment": 0.5},
        split_ratio={"control": 0.5, "treatment": 0.5},
    )


@pytest.fixture
def compare(metric):
    return UserSegmentCompare(
        compare_id="CMP_EVIDENCE",
        segment_a=Segment(segment_id="A", segment_name="对照组"),
        segment_b=Segment(segment_id="B", segment_name="实验组"),
        compare_metric=metric,
        data_a=[8.0, 12.0],
        data_b=[14.0, 16.0],
    )


@pytest.fixture
def result(experiment, metric):
    return ExperimentResult(
        report_id="RPT_EVIDENCE",
        experiment=experiment,
        metric=metric,
        control_data=[8.0, 12.0],
        treatment_data=[14.0, 16.0],
    )


# ============================================================
# 两处直接调用：输出不变，且都来自同一份证据
# ============================================================

class TestSharedEvidence:
    def test_compare_keeps_original_ttest_output(self, compare):
        output = compare.compute_ttest()
        assert output == {
            "t_stat": -2.2361,
            "p_value": 0.1987,
            "significant": False,
            "mean_a": 10.0,
            "mean_b": 15.0,
            "n_a": 2,
            "n_b": 2,
        }

    def test_experiment_keeps_original_lift_output(self, result):
        analyzed = result.analyze()
        assert analyzed.control_mean == 10.0
        assert analyzed.treatment_mean == 15.0
        assert analyzed.lift == 0.5
        assert analyzed.p_value == 0.1987
        assert analyzed.significant is False

    def test_both_entrances_read_same_statistics(self, compare, result):
        compare_evidence = compare.read_evidence()
        analyzed = result.analyze()

        assert compare_evidence.n_a == 2
        assert compare_evidence.mean_a == 10.0
        assert compare_evidence.var_a == 8.0
        assert compare_evidence.n_b == 2
        assert compare_evidence.mean_b == 15.0
        assert compare_evidence.var_b == 2.0

        assert analyzed.sample_evidence.n_a == compare_evidence.n_a
        assert analyzed.sample_evidence.group_a.values == compare_evidence.group_a.values
        assert analyzed.sample_evidence.group_b.values == compare_evidence.group_b.values
        assert analyzed.sample_evidence.means == compare_evidence.means
        assert analyzed.sample_evidence.variances == compare_evidence.variances

    def test_welch_ttest_accepts_evidence_directly(self, compare):
        evidence = compare.read_evidence()
        assert welch_ttest(evidence) == compare.compute_ttest()

    def test_evidence_registers_independent_observation_identity(self, compare):
        evidence = compare.read_evidence()
        assert evidence.observation_ids == (
            "BASE:a:0", "BASE:a:1", "BASE:b:0", "BASE:b:1",
        )
        assert all(obs.observation_id for obs in evidence.observations)
        assert [obs.value for obs in evidence.observations] == [8.0, 12.0, 14.0, 16.0]
        assert evidence.batch_id.startswith("batch:")

    def test_explicit_observation_ids_are_registered(self, metric):
        compare = UserSegmentCompare(
            compare_id="CMP_IDS",
            segment_a=Segment(segment_id="A", segment_name="对照组"),
            segment_b=Segment(segment_id="B", segment_name="实验组"),
            compare_metric=metric,
            data_a=[8.0, 12.0],
            data_b=[14.0, 16.0],
            observations_a=[Observation("C1", "a", 8.0), Observation("C2", "a", 12.0)],
            observations_b=[Observation("T1", "b", 14.0), Observation("T2", "b", 16.0)],
        )
        evidence = compare.read_evidence()
        assert evidence.observation_ids == ("C1", "C2", "T1", "T2")
        assert compare.compute_ttest()["mean_b"] == 15.0

    def test_group_aliases_share_control_treatment_view(self, metric):
        compare = UserSegmentCompare(
            compare_id="CMP_ALIAS",
            segment_a=Segment(segment_id="A", segment_name="对照组"),
            segment_b=Segment(segment_id="B", segment_name="实验组"),
            compare_metric=metric,
            data_a=[8.0, 12.0],
            data_b=[14.0, 16.0],
            group_a_name="control",
            group_b_name="treatment",
        )
        evidence = compare.read_evidence()
        assert evidence.statistics_for("A").mean == 10.0
        assert evidence.statistics_for("Treatment").mean == 15.0
        assert normalize_group("对照组") == "control"
        assert normalize_group("实验组") == "treatment"

    def test_batch_id_is_deterministic_for_same_samples(self, compare):
        first = compare.read_evidence()
        second = compare.read_evidence()
        assert first.batch_id == second.batch_id

    def test_different_samples_get_different_batch_id(self, experiment, metric):
        one = read_sample_evidence([8.0, 12.0], [14.0, 16.0])
        two = read_sample_evidence([8.0, 12.0], [14.0, 16.0, 30.0])
        assert one.batch_id != two.batch_id


# ============================================================
# 边界行为：空样本与零方差保留既有输出
# ============================================================

class TestBoundaryBehaviour:
    def test_empty_treatment_is_insufficient(self, metric):
        compare = UserSegmentCompare(
            compare_id="CMP_EMPTY",
            segment_a=Segment(segment_id="A", segment_name="对照组"),
            segment_b=Segment(segment_id="B", segment_name="实验组"),
            compare_metric=metric,
            data_a=[8.0, 12.0],
            data_b=[],
        )
        output = compare.compute_ttest()
        assert output["significant"] is False
        assert output["reason"] == "样本不足"
        assert output["p_value"] == 1.0

    def test_both_empty_keeps_legtimate_experiment_means(self, experiment, metric):
        analyzed = ExperimentResult(
            report_id="RPT_EMPTY",
            experiment=experiment,
            metric=metric,
            control_data=[],
            treatment_data=[],
        ).analyze()
        assert analyzed.control_mean == 0.0
        assert analyzed.treatment_mean == 0.0
        assert analyzed.lift == 0.0
        assert analyzed.p_value == 1.0
        assert analyzed.significant is False

    def test_empty_control_keeps_original_zero_lift(self, experiment, metric):
        analyzed = ExperimentResult(
            report_id="RPT_EMPTY_CONTROL",
            experiment=experiment,
            metric=metric,
            control_data=[],
            treatment_data=[14.0, 16.0],
        ).analyze()
        assert analyzed.control_mean == 0.0
        assert analyzed.treatment_mean == 15.0
        assert analyzed.lift == 0.0

    def test_zero_variance_returns_not_significant(self, metric):
        compare = UserSegmentCompare(
            compare_id="CMP_ZERO_VAR",
            segment_a=Segment(segment_id="A", segment_name="对照组"),
            segment_b=Segment(segment_id="B", segment_name="实验组"),
            compare_metric=metric,
            data_a=[10.0, 10.0],
            data_b=[20.0, 20.0],
        )
        output = compare.compute_ttest()
        assert output == {
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    def test_single_sample_uses_insufficient_path(self, metric):
        compare = UserSegmentCompare(
            compare_id="CMP_SINGLE",
            segment_a=Segment(segment_id="A", segment_name="对照组"),
            segment_b=Segment(segment_id="B", segment_name="实验组"),
            compare_metric=metric,
            data_a=[10.0],
            data_b=[20.0, 21.0],
        )
        assert compare.compute_ttest()["reason"] == "样本不足"


# ============================================================
# 补报观测：原完整输入结清 vs 换入确认值重新检验
# ============================================================

class TestSupplementLifecycle:
    def _session(self) -> ExperimentAnalysis:
        return ExperimentAnalysis(
            "RPT_T3",
            control_data=[8.0, 12.0],
            treatment_data=[14.0, 16.0],
        )

    def test_original_settlement_reports_15_and_50_percent(self):
        session = self._session()
        session.begin()
        assert session.test()["mean_b"] == 15.0

        ingest = session.add_observation(Observation("T3", "treatment", 30))
        assert ingest.status == ACCEPTED

        original = session.settle("original")
        assert original.n_control == 2
        assert original.n_treatment == 2
        assert original.control_mean == 10.0
        assert original.treatment_mean == 15.0
        assert original.lift == 0.5
        assert "T3" not in original.observation_ids

    def test_latest_settlement_reports_20_and_100_percent(self):
        session = self._session()
        session.begin()
        session.test()
        session.add_observation(Observation("T3", "treatment", 30))

        latest = session.settle("latest")
        assert latest.n_control == 2
        assert latest.n_treatment == 3
        assert latest.control_mean == 10.0
        assert latest.treatment_mean == 20.0
        assert latest.lift == 1.0
        assert "T3" in latest.observation_ids
        assert latest.p_value == pytest.approx(0.1778)
        assert latest.significant is False

    def test_both_policies_use_one_batch_for_significance_and_lift(self):
        session = self._session()
        session.begin()
        session.test()
        session.add_observation(Observation("T3", "treatment", 30))

        for policy in ("original", "latest"):
            settled = session.settle(policy)
            recomputed = welch_ttest(settled.evidence)
            assert settled.p_value == recomputed["p_value"]
            assert settled.significant == recomputed["significant"]
            assert settled.control_mean == settled.evidence.control_mean
            assert settled.treatment_mean == settled.evidence.treatment_mean
            assert settled.significance_source["batch_id"] == settled.batch_id

    def test_supplement_after_test_does_not_change_frozen_test(self):
        session = self._session()
        session.begin()
        before = session.test()
        session.add_observation(Observation("T3", "treatment", 30))
        after = session.test()
        assert before == after == {
            "t_stat": -2.2361,
            "p_value": 0.1987,
            "significant": False,
            "mean_a": 10.0,
            "mean_b": 15.0,
            "n_a": 2,
            "n_b": 2,
        }


# ============================================================
# 观测身份：重复、冲突、异号同额
# ============================================================

class TestObservationIdentity:
    def _session(self) -> ExperimentAnalysis:
        session = ExperimentAnalysis(
            "RPT_IDENTITY",
            control_data=[8.0, 12.0],
            treatment_data=[14.0, 16.0],
        )
        session.begin()
        return session

    def test_duplicate_t3_counts_once(self):
        session = self._session()
        first = session.add_observation(Observation("T3", "treatment", 30))
        second = session.add_observation(Observation("T3", "treatment", 30))

        assert first.status == ACCEPTED
        assert second.status == DUPLICATE
        assert second.changed is False
        assert session.ledger.accepted_ids("treatment") == (
            "BASE:treatment:0", "BASE:treatment:1", "T3",
        )
        latest = session.settle("latest")
        assert latest.n_treatment == 3
        assert latest.treatment_mean == 20.0

    def test_same_id_different_value_waits_for_verification(self):
        session = self._session()
        session.add_observation(Observation("T3", "treatment", 30))
        conflict = session.add_observation(Observation("T3", "treatment", 31))

        assert conflict.status == PENDING_VERIFICATION
        assert conflict.changed is False
        # 冲突值不进入样本，样本仍是 T3=30
        evidence = session.ledger.snapshot()
        assert evidence.n_treatment == 3
        assert evidence.treatment_mean == 20.0
        assert evidence.has_conflicts is True
        assert [obs.value for obs in evidence.conflicts] == [31.0]

        # 确认后换入 31，重新检验
        confirmed = session.confirm_observation("T3", 31)
        assert confirmed.status == CONFIRMED
        assert confirmed.changed is True
        assert session.ledger.pending() == ()
        refreshed = session.settle("latest")
        assert refreshed.n_treatment == 3
        assert refreshed.treatment_mean == pytest.approx((14 + 16 + 31) / 3, rel=1e-6)

    def test_same_amount_different_id_enters_sample_separately(self):
        session = self._session()
        first = session.add_observation(Observation("T3", "treatment", 30))
        second = session.add_observation(Observation("T4", "treatment", 30))

        assert first.status == ACCEPTED
        assert second.status == ACCEPTED
        latest = session.settle("latest")
        assert latest.n_treatment == 4
        assert latest.observation_ids.count("T3") == 1
        assert latest.observation_ids.count("T4") == 1
        assert latest.treatment_mean == 22.5

    def test_rejected_conflict_keeps_original_value(self):
        session = self._session()
        session.add_observation(Observation("T3", "treatment", 30))
        session.add_observation(Observation("T3", "treatment", 31))
        rejected = session.ledger.reject("T3", 31)

        assert rejected.status == "rejected"
        assert session.ledger.pending() == ()
        assert session.settle("latest").treatment_mean == 20.0

    def test_ledger_ingest_returns_structured_result(self):
        ledger = SampleLedger([8.0, 12.0], [14.0, 16.0])
        outcome = ledger.ingest(Observation("T3", "treatment", 30))
        assert isinstance(outcome, IngestResult)
        assert outcome["status"] == ACCEPTED
        assert outcome.as_dict()["observation"]["observation_id"] == "T3"
        assert ledger.revision == 1


# ============================================================
# 并发发布与历史重算
# ============================================================

class TestPublication:
    def _session(self) -> ExperimentAnalysis:
        session = ExperimentAnalysis(
            "RPT_CONCURRENT",
            control_data=[8.0, 12.0],
            treatment_data=[14.0, 16.0],
        )
        session.begin()
        session.test()
        return session

    def test_concurrent_publish_happens_once(self):
        session = self._session()
        session.add_observation(Observation("T3", "treatment", 30))

        barrier = threading.Barrier(8)
        published = []

        def worker() -> None:
            barrier.wait()
            published.append(session.publish("original"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(published) == 8
        assert len({id(item) for item in published}) == 1
        assert len({item.batch_id for item in published}) == 1
        assert len(session.history()) == 1
        assert published[0].treatment_mean == 15.0
        assert published[0].lift == 0.5

    def test_published_significance_and_lift_come_from_one_batch(self):
        session = self._session()
        session.add_observation(Observation("T3", "treatment", 30))

        # 发布之后即使台账继续变化，已发布结果仍绑定同一份证据
        published = session.publish("original")
        session.add_observation(Observation("T4", "treatment", 30))
        session.confirm_observation("T3", 31)

        assert published.n_treatment == 2
        assert published.treatment_mean == 15.0
        assert published.lift == 0.5
        source = published.significance_source
        assert source["batch_id"] == published.batch_id
        assert source["n_treatment"] == published.n_treatment
        assert "T3" not in source["observation_ids"]
        assert published.recompute_matches()

    def test_second_publish_returns_first_result(self):
        session = self._session()
        first = session.publish("original")
        session.add_observation(Observation("T3", "treatment", 30))
        second = session.publish("latest")
        assert second is first
        assert second.treatment_mean == 15.0

    def test_history_recomputes_with_sample_used_at_the_time(self):
        original_session = self._session()
        original = original_session.publish("original")

        updated_session = self._session()
        updated_session.add_observation(Observation("T3", "treatment", 30))
        updated = updated_session.publish("latest")

        # 两份历史结果分别按各自实际采用的样本重算
        assert original.recompute_matches()
        assert updated.recompute_matches()
        assert original.batch_id != updated.batch_id
        assert original.treatment_mean == 15.0
        assert updated.treatment_mean == 20.0
        assert updated.recompute()["lift"] == 1.0
        assert original.recompute()["lift"] == 0.5

    def test_supplement_during_conclusion_keeps_published_sample(self):
        session = self._session()

        # 已通过的 T3 在出结论期间补报
        approved = session.add_observation(Observation("T3", "treatment", 30))
        assert approved.status == ACCEPTED
        published = session.publish("original")

        assert published.n_control == 2
        assert published.n_treatment == 2
        assert published.control_mean == 10.0
        assert published.treatment_mean == 15.0
        assert published.lift == 0.5
        assert published.significant is False
        source = published.significance_source
        assert source["sample_source"] == "original"
        assert source["observation_ids"] == [
            "BASE:control:0", "BASE:control:1",
            "BASE:treatment:0", "BASE:treatment:1",
        ]

        # 同一次分析换入确认样本重新检验，才报告 20 与 100%
        refreshed = session.settle("latest")
        assert refreshed.n_treatment == 3
        assert refreshed.treatment_mean == 20.0
        assert refreshed.lift == 1.0
        # 已发布结论不受影响
        assert session.published.treatment_mean == 15.0
        assert session.published.recompute_matches()


# ============================================================
# 两种旧入口 + 补报观测的合练
# ============================================================

class TestLegacyEntrancesWithObservations:
    def test_both_entrances_report_same_sample_means(self, result, compare):
        analyzed = result.analyze()
        ttest = compare.compute_ttest()

        assert analyzed.sample_evidence.n_control == ttest["n_a"] == 2
        assert analyzed.sample_evidence.n_treatment == ttest["n_b"] == 2
        assert analyzed.control_mean == ttest["mean_a"] == 10.0
        assert analyzed.treatment_mean == ttest["mean_b"] == 15.0
        assert analyzed.lift == 0.5
        assert analyzed.sample_evidence.variances == {"control": 8.0, "treatment": 2.0}

    def test_duplicate_and_conflict_observations_respect_identity(self, experiment, metric):
        result = ExperimentResult(
            report_id="RPT_INPUT_IDS",
            experiment=experiment,
            metric=metric,
            control_data=[8.0, 12.0],
            treatment_data=[14.0, 16.0],
            observations_control=[Observation("C1", "control", 8), Observation("C2", "control", 12)],
            observations_treatment=[
                Observation("T1", "treatment", 14),
                Observation("T2", "treatment", 16),
                Observation("T2", "treatment", 16),   # 重复只计一次
                Observation("T2", "treatment", 99),   # 同号异值，待核不入样
            ],
        ).analyze()

        assert result.sample_evidence.n_treatment == 2
        assert result.treatment_mean == 15.0
        assert result.lift == 0.5
        assert result.sample_evidence.duplicate_count == 1
        assert result.sample_evidence.has_conflicts is True

    def test_full_supplement_scenario_across_both_entrances(self, metric, experiment):
        # 1. 两个旧入口先算出原结果
        compare = UserSegmentCompare(
            compare_id="CMP_FULL",
            segment_a=Segment(segment_id="A", segment_name="对照组"),
            segment_b=Segment(segment_id="B", segment_name="实验组"),
            compare_metric=metric,
            data_a=[8.0, 12.0],
            data_b=[14.0, 16.0],
        )
        assert compare.compute_ttest() == {
            "t_stat": -2.2361, "p_value": 0.1987, "significant": False,
            "mean_a": 10.0, "mean_b": 15.0, "n_a": 2, "n_b": 2,
        }
        legacy = ExperimentResult(
            report_id="RPT_FULL",
            experiment=experiment,
            metric=metric,
            control_data=[8.0, 12.0],
            treatment_data=[14.0, 16.0],
        ).analyze()
        assert (legacy.control_mean, legacy.treatment_mean, legacy.lift) == (10.0, 15.0, 0.5)

        # 2. 同一份样本进入分析会话，检验后、结论前发生补报
        session = ExperimentAnalysis(
            "RPT_FULL",
            control_data=[8.0, 12.0],
            treatment_data=[14.0, 16.0],
        )
        session.begin()
        assert session.test()["mean_b"] == 15.0

        assert session.add_observation(Observation("T3", "treatment", 30)).status == ACCEPTED
        assert session.add_observation(Observation("T3", "treatment", 30)).status == DUPLICATE
        assert session.add_observation(Observation("T3", "treatment", 31)).status == PENDING_VERIFICATION
        assert session.add_observation(Observation("T4", "treatment", 30)).status == ACCEPTED

        # 3. 原完整输入结清：样本数沿用检验时的两组各 2 条
        original = session.publish("original")
        assert original.n_control == 2
        assert original.n_treatment == 2
        assert original.control_mean == 10.0
        assert original.treatment_mean == 15.0
        assert original.lift == 0.5
        assert original.significant is False
        assert original.significance_source["sample_source"] == "original"
        assert original.observation_ids == (
            "BASE:control:0", "BASE:control:1",
            "BASE:treatment:0", "BASE:treatment:1",
        )

        # 4. 换入已确认补报重新检验：T3=30 与 T4=30 均入样，31 仍待核
        refreshed = session.settle("latest")
        assert refreshed.n_treatment == 4
        assert refreshed.treatment_mean == 22.5
        assert refreshed.lift == 1.25
        assert refreshed.observation_ids.count("T3") == 1
        assert refreshed.has_conflicts is True

        # 5. 已发布结论不因补报或再次发布而改变
        assert session.publish("latest") is original
        assert session.published.recompute_matches()
        assert original.recompute()["lift"] == 0.5


def test_pipeline_experiment_uses_shared_evidence(experiment, metric):
    from app.analysis_pipeline import pipeline_experiment_to_compare

    output = pipeline_experiment_to_compare(
        [experiment],
        [metric],
        {experiment.experiment_id: [8.0, 12.0]},
        {experiment.experiment_id: [14.0, 16.0]},
    )
    assert output["pipeline"] == "ab_experiment -> experiment_result -> metric_compare"
    assert output["total_reports"] == 1
    assert output["significant_count"] == 0
    summary = output["comparisons"]["样本证据实验__转化率"]
    assert summary["control_mean"] == 10.0
    assert summary["treatment_mean"] == 15.0
    assert summary["lift"] == 0.5
    assert summary["lift_pct"] == "50.00%"
    assert summary["p_value"] == 0.1987
