from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import pickle
import traceback
import json

router = APIRouter()


class PublishRequest(BaseModel):
    video_path: str
    title: str
    description: str
    tags: str
    privacy_status: str = "private"


def _read_seo(folder_path, default_title):
    seo_data = {
        "title": default_title,
        "description": "",
        "tags": ""
    }

    seo_json_path = os.path.join(folder_path, "seo_metadata.json")

    if os.path.exists(seo_json_path):
        try:
            with open(seo_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            seo_data["title"] = data.get("title", default_title)
            seo_data["description"] = data.get("description", "")

            tags = data.get("tags", [])
            if isinstance(tags, list):
                seo_data["tags"] = ", ".join(tags)
            else:
                seo_data["tags"] = str(tags)

        except Exception as e:
            print(f"Error reading {seo_json_path}: {e}")

    timestamps_path = os.path.join(folder_path, "timestamps.txt")

    if os.path.exists(timestamps_path):
        try:
            with open(timestamps_path, "r", encoding="utf-8") as f:
                ts = f.read()

            if seo_data["description"]:
                seo_data["description"] += f"\n\n{ts}"
            else:
                seo_data["description"] = ts

        except Exception as e:
            print(f"Error reading {timestamps_path}: {e}")

    return seo_data


@router.get("/library")
def get_publishing_library():
    projects = []

    # ---------------------------------------------------------
    # 1. Traditional workspace projects
    # ---------------------------------------------------------
    workspace_dir = "workspace"

    if os.path.exists(workspace_dir):
        try:
            folders = sorted(
                os.listdir(workspace_dir),
                key=lambda x: os.path.getctime(
                    os.path.join(workspace_dir, x)
                ),
                reverse=True
            )
        except Exception:
            folders = os.listdir(workspace_dir)

        for folder in folders:
            folder_path = os.path.join(workspace_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            video_file = os.path.join(folder_path, "final_podcast.mp4")

            if not os.path.exists(video_file):
                video_file = os.path.join(
                    folder_path,
                    "final_podcast_video.mp4"
                )

            if not os.path.exists(video_file):
                continue

            seo_data = _read_seo(folder_path, folder)

            projects.append({
                "id": folder,
                "name": folder.replace("_", " ").title(),
                "type": "workspace",
                "path": folder_path,
                "video_path": video_file,
                "seo": seo_data
            })

    # ---------------------------------------------------------
    # 2. Movie Explainer projects
    # ---------------------------------------------------------
    outputs_dir = "outputs"

    if os.path.exists(outputs_dir):
        try:
            folders = sorted(
                os.listdir(outputs_dir),
                key=lambda x: os.path.getctime(
                    os.path.join(outputs_dir, x)
                ),
                reverse=True
            )
        except Exception:
            folders = os.listdir(outputs_dir)

        for folder in folders:
            if not folder.startswith("explainer_"):
                continue

            folder_path = os.path.join(outputs_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            video_file = os.path.join(
                folder_path,
                "final_explainer.mp4"
            )

            if not os.path.exists(video_file):
                continue

            seo_data = _read_seo(
                folder_path,
                "Movie Explainer"
            )

            projects.append({
                "id": folder,
                "name": seo_data["title"] or "Movie Explainer",
                "type": "movie explainer",
                "path": folder_path,
                "video_path": video_file,
                "seo": seo_data
            })

    # Newest projects first
    projects.sort(
        key=lambda x: os.path.getctime(x["path"]),
        reverse=True
    )

    return {"projects": projects}


@router.post("/youtube")
def publish_to_youtube(req: PublishRequest):
    if not req.video_path or not os.path.exists(req.video_path):
        raise HTTPException(
            status_code=400,
            detail="Invalid video path."
        )

    if not os.path.exists("client_secrets.json"):
        raise HTTPException(
            status_code=400,
            detail="client_secrets.json not found! Please download your OAuth 2.0 Client IDs from Google Cloud Console."
        )

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        credentials = None

        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                credentials = pickle.load(token)

        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "client_secrets.json",
                    scopes=[
                        "https://www.googleapis.com/auth/youtube.upload"
                    ],
                    redirect_uri="http://localhost:8080/"
                )

                credentials = flow.run_local_server(port=8080)

            with open("token.pickle", "wb") as f:
                pickle.dump(credentials, f)

        youtube = build(
            "youtube",
            "v3",
            credentials=credentials
        )

        body = {
            "snippet": {
                "title": req.title,
                "description": req.description,
                "tags": [
                    t.strip()
                    for t in req.tags.split(",")
                    if t.strip()
                ],
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": req.privacy_status
            }
        }

        media = MediaFileUpload(
            req.video_path,
            chunksize=-1,
            resumable=True
        )

        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media
        )

        response = None

        while response is None:
            status, response = request.next_chunk()

        return {
            "message": "Video uploaded successfully!",
            "video_id": response.get("id")
        }

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
