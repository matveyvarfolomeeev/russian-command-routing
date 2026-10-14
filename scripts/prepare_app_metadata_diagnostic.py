"""Author diagnostics for the twelve enriched apps, separately from the frozen STT."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/benchmarks/app-metadata-v2"
PURPOSES = {
    "Anki": "открой программу для интервального повторения учебных карточек",
    "Docker Desktop": "открой Docker для управления контейнерами",
    "Git Bash": "открой Git в оболочке Bash",
    "Git CMD": "открой Git через обычную командную строку Windows",
    "Git GUI": "открой графический интерфейс Git",
    "PyCharm Community Edition 2023.3.5": "открой бесплатную среду JetBrains для разработки на Python",
    "Roblox Studio": "открой среду для создания игр Roblox",
    "SharpKeys": "открой утилиту для переназначения клавиш клавиатуры через реестр",
    "SumatraPDF": "открой лёгкую читалку PDF и DjVu",
    "Windows Media Player Legacy": "открой старый проигрыватель Windows",
    "Графический планшет": "открой настройки графического планшета",
    "СОЛО на клавиатуре": "открой клавиатурный тренажёр СОЛО",
}


def run():
    updates = json.loads((FOLDER / "description-updates.json").read_text(encoding="utf-8"))
    assert {row["name"] for row in updates.values()} == set(PURPOSES)
    data = []
    for index, (object_id, item) in enumerate(updates.items()):
        for kind, query in (("name", "открой " + item["name"]), ("purpose", PURPOSES[item["name"]])):
            data.append({"id": f"app-metadata-{kind}-{index:02d}", "query": query,
                         "target": object_id, "profile": f"metadata-diagnostic/{kind}",
                         "group_id": object_id, "catalog_key": "real"})
    assert len(data) == len({row["id"] for row in data}) == 24
    path = FOLDER / "diagnostic-queries.jsonl"
    if path.exists():
        raise ValueError("Diagnostic already exists")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    (FOLDER / "diagnostic-role.json").write_text(json.dumps({
        "rows": 24, "apps": 12, "name_queries": 12, "purpose_queries": 12,
        "role": "Codex-authored descriptive diagnostics AFTER metadata collection; not blind, not STT",
        "purpose_queries_include_qualifiers": True, "used_for_training_or_selection": False,
        "collector_received_queries": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostic_rows": len(data), "role": "author_diagnostic_not_blind_not_STT"}))


if __name__ == "__main__":
    run()
