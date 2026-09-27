import json
import os
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from server.core.database import Scenario, get_db
from server.core.queue import create_job, submit_to_queue
from server.core.scenario_engine import artifact_dir, run_stage, scenario_snapshot
from server.core.scenario_models import ScenarioCreateRequest

router = APIRouter()

STAGE_REQUIREMENTS = {
    "research": ("created",),
    "analyze": ("research_complete",),
    "timeline": ("analysis_complete",),
    "script": ("timeline_complete",),
    "visual_plan": ("script_complete",),
}


def response_payload(scenario: Scenario) -> Dict[str, Any]:
    payload = scenario_snapshot(scenario)
    payload.update({
        "current_stage": scenario.current_stage,
        "job_id": scenario.job_id,
        "error": scenario.error,
        "research_data": json.loads(scenario.research_data) if scenario.research_data else None,
        "analysis_data": json.loads(scenario.analysis_data) if scenario.analysis_data else None,
        "timeline_data": json.loads(scenario.timeline_data) if scenario.timeline_data else None,
        "script_data": json.loads(scenario.script_data) if scenario.script_data else None,
        "visual_plan_data": json.loads(scenario.visual_plan_data) if scenario.visual_plan_data else None,
    })
    return payload


@router.post("", status_code=201)
def create_scenario(req: ScenarioCreateRequest, db: Session = Depends(get_db)):
    scenario_id = str(uuid.uuid4())
    directory = artifact_dir(scenario_id)
    scenario = Scenario(
        id=scenario_id,
        premise=req.premise.strip(),
        title=(req.title or req.premise.strip())[:200],
        status="created",
        artifact_dir=directory,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    from server.core.scenario_engine import write_json_artifact
    write_json_artifact(directory, "scenario.json", scenario_snapshot(scenario))
    return response_payload(scenario)


@router.get("")
def list_scenarios(db: Session = Depends(get_db)):
    scenarios = db.query(Scenario).order_by(Scenario.created_at.desc()).all()
    return [response_payload(scenario) for scenario in scenarios]


@router.get("/{scenario_id}")
def get_scenario(scenario_id: str, db: Session = Depends(get_db)):
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return response_payload(scenario)


@router.get("/{scenario_id}/status")
def get_scenario_status(scenario_id: str, db: Session = Depends(get_db)):
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {
        "id": scenario.id,
        "status": scenario.status,
        "current_stage": scenario.current_stage,
        "job_id": scenario.job_id,
        "error": scenario.error,
    }


def queue_stage(scenario_id: str, stage: str, db: Session):
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if scenario.status not in STAGE_REQUIREMENTS[stage]:
        raise HTTPException(status_code=409, detail=f"Scenario is not ready for {stage} from status '{scenario.status}'")
    job_id = create_job(f"scenario_{scenario_id}_{stage}")
    scenario.job_id = job_id
    scenario.status = f"{stage}_queued"
    scenario.current_stage = stage
    scenario.error = None
    db.commit()
    submit_to_queue(run_stage, scenario_id, job_id, stage)
    return {"scenario_id": scenario_id, "job_id": job_id, "status": scenario.status}


@router.post("/{scenario_id}/research")
def research_scenario(scenario_id: str, db: Session = Depends(get_db)):
    return queue_stage(scenario_id, "research", db)


@router.post("/{scenario_id}/analyze")
def analyze_scenario(scenario_id: str, db: Session = Depends(get_db)):
    return queue_stage(scenario_id, "analyze", db)


@router.post("/{scenario_id}/timeline")
def timeline_scenario(scenario_id: str, db: Session = Depends(get_db)):
    return queue_stage(scenario_id, "timeline", db)


@router.post("/{scenario_id}/script")
def script_scenario(scenario_id: str, db: Session = Depends(get_db)):
    return queue_stage(scenario_id, "script", db)


@router.post("/{scenario_id}/visual-plan")
def visual_plan_scenario(scenario_id: str, db: Session = Depends(get_db)):
    return queue_stage(scenario_id, "visual_plan", db)
