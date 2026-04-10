from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable, Optional

from aoede.schemas import VoiceProfile


class VoiceProfileStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, voice_id: str):
        return self.root_dir / f"{voice_id}.json"

    def save(self, profile: VoiceProfile):
        path = self._path_for(profile.voice_id)
        path.write_text(profile.model_dump_json(indent=2))
        return profile

    def create(self, profile: VoiceProfile):
        if not profile.voice_id:
            profile = profile.model_copy(update={"voice_id": uuid.uuid4().hex})
        return self.save(profile)

    def load(self, voice_id: str):
        path = self._path_for(voice_id)
        if not path.exists():
            return None
        return VoiceProfile.model_validate_json(path.read_text())

    def delete(self, voice_id: str):
        path = self._path_for(voice_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list(self):
        profiles = []
        for path in sorted(self.root_dir.glob("*.json")):
            profiles.append(VoiceProfile.model_validate_json(path.read_text()))
        return profiles

    def resolve(
        self, voice_id: Optional[str], inline_profile: Optional[VoiceProfile]
    ):
        if inline_profile is not None:
            return inline_profile
        if voice_id is None:
            return None
        return self.load(voice_id)

    def export_index(self, path: Path):
        payload = [profile.model_dump() for profile in self.list()]
        path.write_text(json.dumps(payload, indent=2))

    def import_profiles(self, profiles: Iterable[VoiceProfile]):
        for profile in profiles:
            self.save(profile)
