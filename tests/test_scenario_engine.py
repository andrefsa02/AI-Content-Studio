import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.core.database import Base, Scenario
from server.core.scenario_engine import extract_json, write_json_artifact
from server.core.scenario_models import (
    CausalLevel,
    AnalysisBranch,
    AnalysisDocument,
    CausalNode,
    EvidenceKind,
    ResearchDocument,
    ResearchItem,
    SourceReference,
)
from server.routers.scenario import queue_stage


class DeterministicScenarioAI:
    calls = []

    def __init__(self, config):
        self.config = config

    def _generate_text(self, prompt, as_json=False):
        self.__class__.calls.append({"prompt": prompt, "as_json": as_json})
        if "Research this hypothetical scenario" in prompt:
            return json.dumps({
                "summary": "A sudden stop would produce immediate physical hazards followed by cascading social effects.",
                "items": [
                    {"kind": "FACT", "claim": "Earth currently rotates once approximately every 24 hours.", "confidence": 0.99, "assumptions": [], "uncertainty": [], "source_ids": ["rotation-source"]},
                    {"kind": "ASSUMPTION", "claim": "The scenario changes rotational motion instantly without changing gravity.", "confidence": 0.95, "assumptions": ["The premise is physically idealized."], "uncertainty": ["An instantaneous stop is not naturally achievable."], "source_ids": []},
                    {"kind": "INFERENCE", "claim": "The atmosphere and oceans would continue moving relative to the surface.", "confidence": 0.88, "assumptions": ["The stop occurs across the whole solid Earth."], "uncertainty": ["Exact regional effects depend on geography."], "source_ids": ["rotation-source"]},
                    {"kind": "SPECULATION", "claim": "Long-term civilization could reorganize around permanent day and night regions.", "confidence": 0.42, "assumptions": ["Some population survives the initial event."], "uncertainty": ["Human adaptation is highly uncertain."], "source_ids": []},
                    {"kind": "SOURCE", "claim": "Reference for Earth rotation facts.", "confidence": 1.0, "assumptions": [], "uncertainty": [], "source_ids": ["rotation-source"]},
                    {"kind": "UNCERTAINTY", "claim": "The exact severity depends on how abruptly and uniformly rotation stops.", "confidence": 0.9, "assumptions": [], "uncertainty": ["The premise has no direct real-world analogue."], "source_ids": []},
                ],
                "sources": {
                    "rotation-source": {
                        "title": "Earth's rotation",
                        "url": "https://example.com/earth-rotation",
                        "publisher": "Deterministic Test Reference",
                        "reference": "test-source-rotation",
                    }
                },
            })
        if "Analyze this hypothetical scenario" in prompt:
            levels = [
                "HYPOTHESIS",
                "FIRST-ORDER EFFECT",
                "SECOND-ORDER EFFECT",
                "THIRD-ORDER EFFECT",
                "LONG-TERM CONSEQUENCE",
            ]
            return json.dumps({
                "hypothesis": "Earth stops rotating instantly while the scenario assumptions hold.",
                "branches": [{
                    "id": "main",
                    "label": "Physical-stop branch",
                    "assumptions": ["The stop is instantaneous and global."],
                    "uncertainty": ["The exact damage pattern depends on geography."],
                    "nodes": [
                        {"id": f"node-{index}", "level": level, "claim": f"Causal consequence at level {index}.", "confidence": 0.8 - index * 0.1, "assumptions": [], "uncertainty": ["Outcome varies by region."], "source_ids": [], "alternative_outcomes": [{"outcome": "A less severe regional result.", "confidence": 0.2, "assumptions": ["Local conditions reduce exposure."]}], "parent_ids": []}
                        for index, level in enumerate(levels)
                    ],
                }],
            })
        if "Create a timeline for" in prompt:
            phases = ["immediate", "hours_days", "weeks_months", "years", "decades_centuries"]
            return json.dumps({
                "entries": [
                    {"id": f"time-{index}", "phase": phase, "title": f"Timeline phase {index}", "consequence": f"Consequence during {phase}.", "confidence": 0.8 - index * 0.1, "assumptions": ["The scenario remains in effect."], "uncertainty": ["Long-range effects are uncertain."], "alternative_outcomes": [{"outcome": "An alternate adaptation path.", "confidence": 0.25, "assumptions": ["Survivors adapt differently."]}], "causal_node_ids": ["node-0"]}
                    for index, phase in enumerate(phases)
                ]
            })
        if "Write a clear narrated script" in prompt:
            return json.dumps({
                "title": "Earth Stops Rotating",
                "sections": [{"heading": "Immediate effects", "text": "The first moments are dominated by motion and uncertainty."}],
                "narration": "This is a hypothetical scenario. The immediate effects would be severe, while long-term outcomes remain uncertain and depend on the assumptions described.",
            })
        if "Create a visual plan" in prompt:
            return json.dumps({
                "scenes": [{"id": "scene-1", "narration": "The first moments unfold.", "visual_prompt": "A scientifically grounded cinematic view of a rotating Earth stopping, with clear atmospheric motion.", "timeline_entry_id": "time-0"}],
            })
        raise AssertionError("Unexpected Scenario AI prompt")


class ScenarioEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        engine = create_engine(f"sqlite:///{self.root / 'scenario.db'}")
        Base.metadata.create_all(bind=engine)
        self.engine = engine
        self.session = sessionmaker(bind=engine)()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_scenario_persists_required_fields(self):
        scenario = Scenario(
            id="scenario-test",
            premise="What if the Internet disappeared for 30 days?",
            title="Thirty Days Offline",
            status="created",
            artifact_dir=str(self.root / "scenario-test"),
        )
        self.session.add(scenario)
        self.session.commit()

        loaded = self.session.get(Scenario, "scenario-test")
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.premise.startswith("What if"))
        self.assertEqual(loaded.status, "created")
        self.assertIsNotNone(loaded.created_at)
        self.assertIsNotNone(loaded.updated_at)

    def test_research_requires_real_source_reference(self):
        source = SourceReference(title="Example source", url="https://example.com")
        document = ResearchDocument(
            items=[ResearchItem(kind=EvidenceKind.SOURCE, claim="A referenced source", confidence=1, source_ids=["source-1"])],
            sources={"source-1": source},
        )
        self.assertEqual(document.items[0].kind, EvidenceKind.SOURCE)

        with self.assertRaises(ValueError):
            ResearchDocument(
                items=[ResearchItem(kind=EvidenceKind.SOURCE, claim="Unsupported source", confidence=0.5)],
                sources={},
            )

    def test_causal_node_requires_ordered_level(self):
        node = CausalNode(id="h1", level=CausalLevel.HYPOTHESIS, claim="The premise occurs", confidence=0.8)
        self.assertEqual(node.level, CausalLevel.HYPOTHESIS)
        with self.assertRaises(ValueError):
            AnalysisDocument(
                hypothesis="The premise occurs",
                branches=[AnalysisBranch(id="b1", label="Main", nodes=[
                    node,
                    CausalNode(id="f1", level=CausalLevel.FIRST_ORDER, claim="A first effect", confidence=0.7),
                    CausalNode(id="h2", level=CausalLevel.HYPOTHESIS, claim="An invalid reset", confidence=0.2),
                ])],
            )

    def test_artifact_json_is_written_atomically(self):
        path = write_json_artifact(str(self.root), "research.json", {"items": [{"kind": "FACT"}]})
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["items"][0]["kind"], "FACT")

    def test_malformed_ai_json_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            extract_json("not json")

    def add_scenario(self, status="created"):
        scenario = Scenario(
            id=f"scenario-{status}",
            premise="What if Earth stopped rotating instantly?",
            title="Earth Stops Rotating",
            status=status,
            artifact_dir=str(self.root / f"scenario-{status}"),
        )
        self.session.add(scenario)
        self.session.commit()
        return scenario

    def assert_stage_rejected(self, status, stage):
        scenario = self.add_scenario(status)
        with patch("server.routers.scenario.create_job") as create_job, patch("server.routers.scenario.submit_to_queue") as submit:
            with self.assertRaises(Exception) as context:
                queue_stage(scenario.id, stage, self.session)
        self.assertEqual(context.exception.status_code, 409)
        self.assertIn(stage, context.exception.detail)
        create_job.assert_not_called()
        submit.assert_not_called()

    def test_failed_research_rejects_analyze(self):
        self.assert_stage_rejected("failed", "analyze")

    def test_missing_analysis_rejects_timeline(self):
        self.assert_stage_rejected("research_complete", "timeline")

    def test_missing_timeline_rejects_script(self):
        self.assert_stage_rejected("analysis_complete", "script")

    def test_missing_script_rejects_visual_plan(self):
        self.assert_stage_rejected("timeline_complete", "visual_plan")

    def test_valid_sequential_transitions_are_accepted(self):
        statuses_and_stages = [
            ("created", "research"),
            ("research_complete", "analyze"),
            ("analysis_complete", "timeline"),
            ("timeline_complete", "script"),
            ("script_complete", "visual_plan"),
        ]
        with patch("server.routers.scenario.create_job", side_effect=[f"job-{index}" for index in range(len(statuses_and_stages))]), patch("server.routers.scenario.submit_to_queue") as submit:
            for index, (status, stage) in enumerate(statuses_and_stages):
                scenario = self.add_scenario(status)
                result = queue_stage(scenario.id, stage, self.session)
                self.assertEqual(result["status"], f"{stage}_queued")
                self.assertEqual(result["job_id"], f"job-{index}")
        self.assertEqual(submit.call_count, len(statuses_and_stages))

    def test_complete_lifecycle_with_deterministic_mocked_ai(self):
        scenario = self.add_scenario()
        scenario_dir = self.root / "scenario-lifecycle"
        scenario.artifact_dir = str(scenario_dir)
        self.session.commit()
        scenario_dir.mkdir()
        DeterministicScenarioAI.calls = []

        stages = [
            ("research", "research_complete", "research_data", "research.json"),
            ("analyze", "analysis_complete", "analysis_data", "analysis.json"),
            ("timeline", "timeline_complete", "timeline_data", "timeline.json"),
            ("script", "script_complete", "script_data", "script.json"),
            ("visual_plan", "visual_plan_complete", "visual_plan_data", "visual_plan.json"),
        ]
        statuses = ["created", "research_complete", "analysis_complete", "timeline_complete", "script_complete"]
        job_index = 0

        def run_synchronously(function, *args, **kwargs):
            function(*args, **kwargs)

        with patch("server.core.scenario_engine.GoogleClient", DeterministicScenarioAI), \
                patch("server.core.scenario_engine.SessionLocal", side_effect=lambda: sessionmaker(bind=self.engine)()), \
                patch("server.core.scenario_engine.artifact_dir", return_value=str(scenario_dir)), \
                patch("server.core.scenario_engine.update_job"), \
                patch("server.core.scenario_engine.manager.broadcast"), \
                patch("server.routers.scenario.create_job", side_effect=lambda topic: f"mock-job-{topic}"), \
                patch("server.routers.scenario.submit_to_queue", side_effect=run_synchronously):
            for (stage, expected_status, field_name, artifact_name), expected_previous_status in zip(stages, statuses):
                self.session.expire_all()
                current = self.session.get(Scenario, scenario.id)
                self.assertEqual(current.status, expected_previous_status)
                result = queue_stage(scenario.id, stage, self.session)
                self.assertTrue(result["job_id"].startswith("mock-job-"))
                self.session.expire_all()
                current = self.session.get(Scenario, scenario.id)
                self.assertEqual(current.status, expected_status)
                self.assertEqual(current.current_stage, stage)
                self.assertIsNotNone(getattr(current, field_name))
                self.assertTrue((scenario_dir / artifact_name).exists())

            final = self.session.get(Scenario, scenario.id)
            self.assertEqual(final.status, "visual_plan_complete")
            self.assertEqual(final.current_stage, "visual_plan")
            self.assertEqual(len(DeterministicScenarioAI.calls), 5)
            self.assertTrue(all(call["as_json"] for call in DeterministicScenarioAI.calls))

        self.session.close()
        persisted_session = sessionmaker(bind=self.engine)()
        try:
            persisted = persisted_session.get(Scenario, scenario.id)
            self.assertEqual(persisted.status, "visual_plan_complete")
            self.assertIsNotNone(persisted.research_data)
            self.assertIsNotNone(persisted.analysis_data)
            self.assertIsNotNone(persisted.timeline_data)
            self.assertIsNotNone(persisted.script_data)
            self.assertIsNotNone(persisted.visual_plan_data)
            for artifact_name in ["scenario.json", "research.json", "analysis.json", "timeline.json", "script.json", "script.txt", "visual_plan.json"]:
                self.assertTrue((scenario_dir / artifact_name).exists(), artifact_name)
        finally:
            self.session = persisted_session


if __name__ == "__main__":
    unittest.main()
