import json
import logging
import os
import tempfile
from typing import Any, Dict, Type, TypeVar

from api_clients import GoogleClient, NewsApiClient
from config import load_config
from server.core.database import Scenario, SessionLocal
from server.core.queue import manager, update_job
from server.core.scenario_models import (
    AnalysisDocument,
    ResearchDocument,
    ScriptDocument,
    TimelineDocument,
    VisualPlanDocument,
)

T = TypeVar("T")
SCENARIO_OUTPUT_ROOT = "outputs"


def artifact_dir(scenario_id: str) -> str:
    path = os.path.join(SCENARIO_OUTPUT_ROOT, f"scenario_{scenario_id}")
    os.makedirs(path, exist_ok=True)
    return path


def write_json_artifact(directory: str, filename: str, payload: Dict[str, Any]) -> str:
    target = os.path.join(directory, filename)
    fd, temp_path = tempfile.mkstemp(prefix=f".{filename}.", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temp_path, target)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return target


def read_payload(scenario: Scenario, field_name: str) -> Dict[str, Any] | None:
    value = getattr(scenario, field_name)
    return json.loads(value) if value else None


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("scenario AI response must be a JSON object")
    return value


def emit(job_id: str, message: str, progress: float) -> None:
    update_job(job_id, step=message, progress=progress)
    manager.broadcast(job_id, {"type": "log", "data": message, "progress": progress})


def run_stage(scenario_id: str, job_id: str, stage: str) -> None:
    db = SessionLocal()
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        update_job(job_id, status="failed", error="Scenario not found")
        db.close()
        return

    try:
        scenario.status = f"{stage}_running"
        scenario.current_stage = stage
        scenario.job_id = job_id
        scenario.error = None
        db.commit()
        emit(job_id, f"Starting scenario {stage}...", 0.1)

        handlers = {
            "research": generate_research,
            "analyze": generate_analysis,
            "timeline": generate_timeline,
            "script": generate_script,
            "visual_plan": generate_visual_plan,
        }
        handlers[stage](db, scenario, job_id)
        completed_status = "analysis_complete" if stage == "analyze" else f"{stage}_complete"
        scenario.status = completed_status
        scenario.current_stage = stage
        db.commit()
        emit(job_id, f"Scenario {stage} complete.", 1.0)
        update_job(job_id, status="completed", result=scenario_id, progress=1.0)
    except Exception as exc:
        logging.exception("Scenario %s failed during %s", scenario_id, stage)
        db.rollback()
        scenario = db.get(Scenario, scenario_id)
        if scenario:
            scenario.status = "failed"
            scenario.current_stage = stage
            scenario.error = str(exc)
            db.commit()
        update_job(job_id, status="failed", error=str(exc), progress=1.0)
        manager.broadcast(job_id, {"type": "log", "data": f"Error: {exc}", "progress": 1.0})
    finally:
        db.close()


def generate_research(db, scenario: Scenario, job_id: str) -> None:
    emit(job_id, "Gathering structured research...", 0.25)
    config = load_config()
    client = GoogleClient(config)
    news = NewsApiClient(config.get("NEWS_API_KEY"))
    prompt = f"""Research this hypothetical scenario: {scenario.premise}

Return ONLY valid JSON with this shape:
{{"summary":"...","items":[{{"kind":"FACT|ASSUMPTION|INFERENCE|SPECULATION|SOURCE|UNCERTAINTY","claim":"...","confidence":0.0,"assumptions":[],"uncertainty":[],"source_ids":[]}}],"sources":{{"source-1":{{"title":"...","url":"https://...","publisher":"...","published_at":"...","reference":"..."}}}}}}

Use SOURCE items only for references you can actually provide. Never invent URLs, publications, quotations, or source metadata. Mark hypothetical reasoning as ASSUMPTION, INFERENCE, or SPECULATION. Include uncertainty explicitly."""
    raw = client._generate_text(prompt, as_json=True)
    document = ResearchDocument.model_validate(extract_json(raw))
    source_ids = set(document.sources)
    for item in document.items:
        if any(source_id not in source_ids for source_id in item.source_ids):
            raise ValueError("research item references an unknown source")
    payload = document.model_dump(mode="json")
    scenario.research_data = json.dumps(payload)
    write_json_artifact(artifact_dir(scenario.id), "research.json", payload)
    write_json_artifact(artifact_dir(scenario.id), "scenario.json", scenario_snapshot(scenario))


def generate_analysis(db, scenario: Scenario, job_id: str) -> None:
    research = read_payload(scenario, "research_data")
    if not research:
        raise ValueError("structured research is required before analysis")
    emit(job_id, "Building causal branches...", 0.35)
    client = GoogleClient(load_config())
    prompt = f"""Analyze this hypothetical scenario: {scenario.premise}
Research JSON:
{json.dumps(research)}

Return ONLY valid JSON with {{"hypothesis":"...","branches":[{{"id":"...","label":"...","assumptions":[],"uncertainty":[],"nodes":[{{"id":"...","level":"HYPOTHESIS|FIRST-ORDER EFFECT|SECOND-ORDER EFFECT|THIRD-ORDER EFFECT|LONG-TERM CONSEQUENCE","claim":"...","confidence":0.0,"assumptions":[],"uncertainty":[],"source_ids":[],"alternative_outcomes":[],"parent_ids":[]}}]}}]}}.
Model multiple branches when uncertainty changes the outcome. Do not present speculation as fact."""
    document = AnalysisDocument.model_validate(extract_json(client._generate_text(prompt, as_json=True)))
    payload = document.model_dump(mode="json")
    scenario.analysis_data = json.dumps(payload)
    write_json_artifact(artifact_dir(scenario.id), "analysis.json", payload)


def generate_timeline(db, scenario: Scenario, job_id: str) -> None:
    analysis = read_payload(scenario, "analysis_data")
    if not analysis:
        raise ValueError("causal analysis is required before timeline generation")
    emit(job_id, "Mapping consequences across time...", 0.45)
    client = GoogleClient(load_config())
    prompt = f"""Create a timeline for: {scenario.premise}
Analysis JSON:
{json.dumps(analysis)}

Return ONLY valid JSON: {{"entries":[{{"id":"...","phase":"immediate|hours_days|weeks_months|years|decades_centuries","title":"...","consequence":"...","confidence":0.0,"assumptions":[],"uncertainty":[],"alternative_outcomes":[],"causal_node_ids":[]}}]}}.
Use only relevant time horizons and preserve uncertainty."""
    document = TimelineDocument.model_validate(extract_json(client._generate_text(prompt, as_json=True)))
    payload = document.model_dump(mode="json")
    scenario.timeline_data = json.dumps(payload)
    write_json_artifact(artifact_dir(scenario.id), "timeline.json", payload)


def generate_script(db, scenario: Scenario, job_id: str) -> None:
    analysis = read_payload(scenario, "analysis_data")
    timeline = read_payload(scenario, "timeline_data")
    if not analysis or not timeline:
        raise ValueError("analysis and timeline are required before script generation")
    emit(job_id, "Writing scenario script...", 0.55)
    client = GoogleClient(load_config())
    prompt = f"""Write a clear narrated script for this hypothetical scenario: {scenario.premise}
Analysis: {json.dumps(analysis)}
Timeline: {json.dumps(timeline)}

Return ONLY valid JSON with {{"title":"...","sections":[],"narration":"..."}}. Clearly label assumptions and uncertainty. Never state speculation as established fact."""
    document = ScriptDocument.model_validate(extract_json(client._generate_text(prompt, as_json=True)))
    payload = document.model_dump(mode="json")
    scenario.script_data = json.dumps(payload)
    write_json_artifact(artifact_dir(scenario.id), "script.json", payload)
    with open(os.path.join(artifact_dir(scenario.id), "script.txt"), "w", encoding="utf-8") as handle:
        handle.write(document.narration)


def generate_visual_plan(db, scenario: Scenario, job_id: str) -> None:
    script = read_payload(scenario, "script_data")
    timeline = read_payload(scenario, "timeline_data")
    if not script or not timeline:
        raise ValueError("script and timeline are required before visual planning")
    emit(job_id, "Planning visuals...", 0.65)
    client = GoogleClient(load_config())
    prompt = f"""Create a visual plan for this scenario script. Return ONLY valid JSON with {{"scenes":[{{"id":"...","narration":"...","visual_prompt":"...","timeline_entry_id":"..."}}]}}.
Script: {json.dumps(script)}
Timeline: {json.dumps(timeline)}"""
    document = VisualPlanDocument.model_validate(extract_json(client._generate_text(prompt, as_json=True)))
    payload = document.model_dump(mode="json")
    scenario.visual_plan_data = json.dumps(payload)
    write_json_artifact(artifact_dir(scenario.id), "visual_plan.json", payload)


def scenario_snapshot(scenario: Scenario) -> Dict[str, Any]:
    return {
        "id": scenario.id,
        "premise": scenario.premise,
        "title": scenario.title,
        "status": scenario.status,
        "created_at": scenario.created_at.isoformat() if scenario.created_at else None,
        "updated_at": scenario.updated_at.isoformat() if scenario.updated_at else None,
        "artifact_dir": scenario.artifact_dir,
    }
