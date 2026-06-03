"""Project log: records all operations and persists project state."""

import json
from datetime import datetime
from pathlib import Path

STEP_NAMES = [
    "切割视频",
    "拆分音轨",
    "转换文字",
    "文本翻译",
    "人声分离",
    "人声生成",
    "重新混音",
    "合成音轨",
    "人物锚定",
    "口型匹配",
    "合成视频",
]

# Steps whose completion is NOT required to unlock the next step.
OPTIONAL_STEPS = {"切割视频", "合成音轨"}

# Each step's prerequisites (must all be "done" to unlock).
STEP_DEPS: dict[str, list[str]] = {
    "切割视频": [],
    "拆分音轨": [],
    "转换文字": ["拆分音轨"],
    "文本翻译": ["转换文字"],
    "人声分离": ["拆分音轨"],
    "人声生成": ["人声分离", "文本翻译"],
    "重新混音": ["人声生成"],
    "合成音轨": ["拆分音轨", "文本翻译"],  # one-click shortcut
    "人物锚定": ["文本翻译"],
    "口型匹配": ["人物锚定"],
    "合成视频": ["口型匹配", "重新混音"],
}


class ProjectLog:
    """Tracks pipeline progress and operation history.

    Serialisable to a ``.aimovie.json`` file so the user can save / resume.
    """

    def __init__(self, name: str = ""):
        now = datetime.now().isoformat()
        self.name = name
        self.created_at = now
        self.updated_at = now
        self.video_path: str | None = None
        self.workspace_dir: str | None = None
        self.steps: dict[str, str] = {}     # step_name → status
        self.step_data: dict[str, dict] = {}  # step_name → arbitrary payload
        self.history: list[dict] = []

    # ── step helpers ──────────────────────────────────────────

    def step_status(self, step_name: str) -> str:
        """Return ``locked`` | ``ready`` | ``running`` | ``done`` | ``failed``."""
        if step_name not in STEP_NAMES:
            return "locked"
        deps = STEP_DEPS.get(step_name, [])
        if not deps:
            return self.steps.get(step_name, "ready")
        for dep in deps:
            if self.steps.get(dep) != "done":
                return "locked"
        return self.steps.get(step_name, "ready")

    def mark_step(self, step_name: str, status: str):
        assert step_name in STEP_NAMES
        self.steps[step_name] = status
        self.updated_at = datetime.now().isoformat()

    def set_step_data(self, step_name: str, data: dict):
        self.step_data[step_name] = data
        self.updated_at = datetime.now().isoformat()

    def first_ready_step(self) -> str | None:
        for name in STEP_NAMES:
            if self.step_status(name) == "ready":
                return name
        return None

    # ── history ───────────────────────────────────────────────

    def add_entry(self, step: str, action: str, detail: str | None = None):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "action": action,
            "detail": detail,
        })
        self.updated_at = datetime.now().isoformat()

    # ── persistence ───────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "video_path": self.video_path,
            "workspace_dir": self.workspace_dir,
            "steps": self.steps,
            "step_data": self.step_data,
            "history": self.history,
        }

    def save(self, path: Path):
        path = Path(path)
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectLog":
        data = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(name=data.get("name", ""))
        obj.created_at = data.get("created_at", obj.created_at)
        obj.updated_at = data.get("updated_at", obj.updated_at)
        obj.video_path = data.get("video_path")
        obj.workspace_dir = data.get("workspace_dir")
        obj.steps = data.get("steps", {})
        obj.step_data = data.get("step_data", {})
        obj.history = data.get("history", [])
        return obj
