# Build on Windows: python -m PyInstaller --noconfirm JevLauncher.spec
from pathlib import Path

root = Path(SPECPATH)
package = root / "src" / "jev_launcher"

analysis = Analysis(
    [str(root / "scripts" / "gui_entry.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=[
        (str(package / "ui"), "jev_launcher/ui"),
        (str(package / "resources"), "jev_launcher/resources"),
        *[(str(root / "scripts" / name), "jev_launcher/runtime") for name in (
            "run_launcher_model.py", "benchmark_choice_quantization.py",
            "train_laya_kaggle.py", "train_nanojev_kaggle.py",
            "continue_nanojev_kaggle.py", "train_modernbert_kaggle.py",
        )],
    ] + (
        [(str(root / "models" / "launcher-runtime.json"), "jev_launcher/resources")]
        if (root / "models" / "launcher-runtime.json").is_file() else []
    ),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6.Qt3DCore", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"],
    noarchive=False,
)
# Windows Qt uses the OS-provided unversioned ICU API. A foreign ICU from PATH
# (for example Poppler's versioned exports) must not shadow the Windows library.
analysis.binaries = [
    entry for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in {"icuuc.dll", "icuin.dll", "icudt.dll"}
]
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [],
    exclude_binaries=True, name="JevLauncher",
    debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=str(package / "resources" / "launcher.ico"),
)
collection = COLLECT(
    exe, analysis.binaries, analysis.datas,
    strip=False, upx=False, name="JevLauncher",
)
