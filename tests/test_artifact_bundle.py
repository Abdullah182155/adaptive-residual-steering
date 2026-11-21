import hashlib
import json
import zipfile

from src.utils.artifact_bundle import create_session3_bundle


def test_session3_bundle_includes_files_manifest_hashes_and_checks_zip(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "session3_status.json").write_text('{"ok": true}', encoding="utf-8")
    (run_dir / "rollouts").mkdir()
    rollout = run_dir / "rollouts" / "shard0.jsonl"
    rollout.write_text('{"question_index": 1}\n', encoding="utf-8")
    archive_path = tmp_path / "run-1_artifacts.zip"

    result = create_session3_bundle(
        run_dir,
        archive_path,
        required_files=["session3_status.json", "rollouts/shard0.jsonl"],
    )

    assert result["complete"] is True
    assert result["archive_verified"] is True
    assert result["archive_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json"))
        assert manifest["complete"] is True
        assert "rollouts/shard0.jsonl" in archive.namelist()
        assert archive.namelist().count("artifact_manifest.json") == 1
        assert manifest["files"][0]["sha256"]

    # Rebuilding a bundle must not treat its previous manifest as a run artifact.
    second = create_session3_bundle(
        run_dir,
        archive_path,
        required_files=["session3_status.json", "rollouts/shard0.jsonl"],
    )
    assert second["complete"] is True
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist().count("artifact_manifest.json") == 1


def test_session3_bundle_keeps_partial_artifacts_and_reports_missing_requirements(tmp_path):
    run_dir = tmp_path / "partial-run"
    run_dir.mkdir()
    (run_dir / "session3_status.json").write_text("{}", encoding="utf-8")

    result = create_session3_bundle(
        run_dir,
        tmp_path / "partial.zip",
        required_files=["session3_status.json", "session3_router.pt"],
    )

    assert result["complete"] is False
    assert result["missing_required_files"] == ["session3_router.pt"]
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert "session3_status.json" in archive.namelist()
