"""Plain data shared by discovery, storage and experiments."""

from __future__ import annotations

import json
import ntpath
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

Kind = Literal["app", "folder", "file"]
LaunchType = Literal["shortcut", "executable", "app_id", "folder", "file"]


def is_maintenance(name: str, target: str, arguments: tuple[str, ...] = ()) -> bool:
    """Reject installer actions, without treating ordinary names like Remove.bg as commands."""
    if re.search(r"^(?:uninstall|деинсталл\w*|удаление|удалить)\b", name.strip(), re.I):
        return True
    executable = ntpath.basename(target).casefold()
    if re.fullmatch(r"(?:unins\d*|uninstall(?:er)?|maintenancetool)\.exe", executable):
        return True
    switches = {arg.casefold() for arg in arguments}
    if switches & {"/uninstall", "--uninstall", "-uninstall", "/repair", "--repair"}:
        return True
    return executable == "msiexec.exe" and bool(switches & {"/x", "/i", "/f", "/modify"})


def launch_identity(
    launch_type: str, target: str, arguments: tuple[str, ...] = (), working_directory: str = ""
) -> str:
    """Deduplicate launch targets, preserving distinct arguments and profiles."""
    target_key = target if launch_type == "app_id" else ntpath.normcase(ntpath.normpath(target))
    cwd_key = ntpath.normcase(ntpath.normpath(working_directory)) if working_directory else ""
    return json.dumps([launch_type, target_key, list(arguments), cwd_key], ensure_ascii=False)


@dataclass
class DiscoveredObject:
    name: str
    launch_target: str
    launch_type: LaunchType
    source: str
    source_key: str
    kind: Kind = "app"
    description: str = ""
    description_source: str = ""
    aliases: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    working_directory: str = ""
    identity: str = ""
    icon_target: str = ""
    resolved_target: str = ""
    available: bool = True

    def __post_init__(self) -> None:
        if not self.identity:
            self.identity = launch_identity(
                self.launch_type, self.launch_target, self.arguments, self.working_directory
            )


@dataclass
class CatalogObject:
    object_id: str
    name: str
    kind: Kind
    launch_type: LaunchType
    launch_target: str
    source: str
    identity: str
    description: str = ""
    description_source: str = ""
    aliases: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    working_directory: str = ""
    icon_target: str = ""
    resolved_target: str = ""
    enabled: bool = True
    available: bool = True
    favorite: bool = False
    last_seen: str = ""
    overrides: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def for_model(self) -> dict:
        """A complete candidate without private launch paths or executable instructions."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, value: dict) -> CatalogObject:
        value = dict(value)
        for key in ("aliases", "arguments", "overrides"):
            value[key] = tuple(value.get(key, ()))
        return cls(**value)


@dataclass
class ScanResult:
    objects: list[DiscoveredObject] = field(default_factory=list)
    successful_sources: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
