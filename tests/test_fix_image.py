"""Tests for FixImageAction — GitOps-aware image-drift remediation (REM-FIXIMAGE).

Covers:
- should_execute gating (enabled, risk, allowlisted target, log signature)
- deployment resolution from pod ownerReferences (ReplicaSet -> Deployment)
- execute: patch mode (kubectl set image), gitops mode (manifest edit + git),
  dry-run, failure/timeout handling
- revert (restore previous image in patch mode)

The action never invents images: only operator-declared FIX_IMAGE_TARGETS
mappings are applied, and (by default) only when the pod's logs carry a
migration/image-drift signature.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from predictive_agent.actions.fix_image import (
    FixImageAction,
    find_target,
    log_has_drift_signature,
    resolve_deployment_name,
)
from predictive_agent.remediator import ActionContext

TARGETS = [
    {
        "namespace": "llm",
        "deployment": "open-webui",
        "container": "open-webui",
        "image": "172.25.24.36:5001/openedusuite-nix/open-webui:v0.13.3-nix",
        "mode": "patch",
    }
]


def _pod(namespace="llm", name="open-webui-x", node_name="clrz14-07"):
    return MagicMock(namespace=namespace, name=name, node_name=node_name,
                     state="FAILED", restart_count=52)


class TestFindTarget:
    def test_exact_match(self):
        assert find_target(TARGETS, "llm", "open-webui") == TARGETS[0]

    def test_namespace_mismatch(self):
        assert find_target(TARGETS, "default", "open-webui") is None

    def test_deployment_mismatch(self):
        assert find_target(TARGETS, "llm", "ollama") is None

    def test_empty_targets(self):
        assert find_target([], "llm", "open-webui") is None


class TestLogSignature:
    PATTERNS = ["alembic", "can't locate revision", "django.db.migrations", "digest mismatch"]

    def test_migration_signature_positive(self):
        log = "INFO alembic.runtime.migration: Can't locate revision identified by '9f6c2a8d1b43'"
        assert log_has_drift_signature(log, self.PATTERNS) is True

    def test_digest_mismatch_positive(self):
        assert log_has_drift_signature("ERROR image digest mismatch sha256:c4cd", self.PATTERNS) is True

    def test_no_signature(self):
        assert log_has_drift_signature("INFO serving http on :8080", self.PATTERNS) is False

    def test_case_insensitive(self):
        assert log_has_drift_signature("ALEMBIC upgrade failed", self.PATTERNS) is True


class TestDeploymentResolution:
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_rs_owner_ref(self, mock_run):
        calls = {"n": 0}

        def run(cmd, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                # pod owner ref -> ReplicaSet
                return MagicMock(returncode=0, stdout='[{"kind":"ReplicaSet","name":"open-webui-76844c76f6"}]', stderr="")
            # replicaset owner ref -> Deployment
            return MagicMock(returncode=0, stdout='[{"kind":"Deployment","name":"open-webui"}]', stderr="")

        mock_run.side_effect = run
        assert resolve_deployment_name("llm", "open-webui-x") == "open-webui"

    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_direct_deployment_owner(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout='[{"kind":"Deployment","name":"open-webui"}]', stderr="")
        assert resolve_deployment_name("llm", "open-webui-x") == "open-webui"

    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_no_owner_ref(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        assert resolve_deployment_name("llm", "open-webui-x") is None

    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_kubectl_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
        assert resolve_deployment_name("llm", "open-webui-x") is None


class TestShouldExecute:
    def test_disabled(self):
        action = FixImageAction(targets=TARGETS, enabled=False)
        assert action.should_execute(_pod(), None, 0.9) is False

    def test_low_risk(self):
        action = FixImageAction(targets=TARGETS, enabled=True)
        assert action.should_execute(_pod(), None, 0.5) is False

    def test_no_matching_target(self):
        action = FixImageAction(targets=TARGETS, enabled=True)
        pod = _pod(namespace="default", name="other-x")
        assert action.should_execute(pod, None, 0.9) is False

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.FixImageAction._log_matches")
    def test_log_signature_required_no_match(self, mock_log, mock_resolve):
        mock_resolve.return_value = "open-webui"
        mock_log.return_value = False
        action = FixImageAction(targets=TARGETS, enabled=True, require_log_signature=True)
        assert action.should_execute(_pod(), None, 0.9) is False
        mock_log.assert_called_once()

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.FixImageAction._log_matches")
    def test_log_signature_match(self, mock_log, mock_resolve):
        mock_resolve.return_value = "open-webui"
        mock_log.return_value = True
        action = FixImageAction(targets=TARGETS, enabled=True, require_log_signature=True)
        assert action.should_execute(_pod(), None, 0.9) is True

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    def test_no_log_signature_required(self, mock_resolve):
        mock_resolve.return_value = "open-webui"
        action = FixImageAction(targets=TARGETS, enabled=True, require_log_signature=False)
        assert action.should_execute(_pod(), None, 0.9) is True


class TestExecutePatchMode:
    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_patch_executes_set_image(self, mock_run, mock_resolve):
        mock_resolve.return_value = "open-webui"
        calls = {"n": 0}

        def run(cmd, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                # current-image probe
                return MagicMock(returncode=0, stdout="old-image:tag", stderr="")
            return MagicMock(returncode=0, stdout="deployment.apps/open-webui image updated", stderr="")

        mock_run.side_effect = run
        action = FixImageAction(targets=TARGETS, enabled=True, require_log_signature=False)
        ctx = ActionContext(namespace="llm", dry_run=False)
        res = action.execute("open-webui-x", ctx)
        assert res.success is True
        assert "open-webui" in res.command
        assert "v0.13.3-nix" in res.command
        assert "llm" in res.command

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_patch_dry_run(self, mock_run, mock_resolve):
        mock_resolve.return_value = "open-webui"
        mock_run.return_value = MagicMock(returncode=0, stdout="old-image:tag", stderr="")
        action = FixImageAction(targets=TARGETS, enabled=True, require_log_signature=False)
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=True))
        assert res.dry_run is True
        assert "--dry-run" in res.command

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    def test_execute_rechecks_enabled(self, mock_resolve):
        mock_resolve.return_value = "open-webui"
        action = FixImageAction(targets=TARGETS, enabled=False)
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=True))
        assert res.success is False
        assert "not enabled" in res.message.lower()

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_patch_failure_propagates(self, mock_run, mock_resolve):
        mock_resolve.return_value = "open-webui"
        calls = {"n": 0}

        def run(cmd, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return MagicMock(returncode=0, stdout="old-image:tag", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="error: unable to update")

        mock_run.side_effect = run
        action = FixImageAction(targets=TARGETS, enabled=True, require_log_signature=False)
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=False))
        assert res.success is False
        assert "error: unable to update" in res.message


class TestExecuteGitOpsMode:
    """gitops mode: manifest edit + git commit (+optional push) in a repo dir.

    Uses a real temp checkout (tmp_path) so the file edit is exercised for real;
    only subprocess (git) is mocked.
    """

    TARGET_GIT = [
        {
            "namespace": "llm",
            "deployment": "open-webui",
            "container": "open-webui",
            "image": "172.25.24.36:5001/openedusuite-nix/open-webui:v0.13.3-nix",
            "mode": "gitops",
        }
    ]

    MANIFEST = (
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        "  name: open-webui\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "      - name: open-webui\n"
        "        image: 172.25.24.36:5001/open-webui/open-webui@sha256:old\n"
    )

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_gitops_edits_manifest_and_commits(self, mock_run, mock_resolve, tmp_path):
        manifest = tmp_path / "open-webui.yaml"
        manifest.write_text(self.MANIFEST)
        mock_resolve.return_value = "open-webui"
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        action = FixImageAction(
            targets=self.TARGET_GIT, enabled=True,
            require_log_signature=False,
            gitops_repo_dir=str(tmp_path),
            gitops_manifest="open-webui.yaml",
            gitops_branch="main", gitops_push=True,
        )
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=False))
        assert res.success is True
        # manifest was really edited
        new_text = manifest.read_text()
        assert "v0.13.3-nix" in new_text
        assert "sha256:old" not in new_text
        # git workflow invoked
        cmd_str = ";".join(" ".join(c.args[0]) for c in mock_run.call_args_list)
        assert " add " in cmd_str
        assert " commit " in cmd_str
        assert " push " in cmd_str
        assert "open-webui.yaml" in cmd_str

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_gitops_no_push_when_disabled(self, mock_run, mock_resolve, tmp_path):
        (tmp_path / "open-webui.yaml").write_text(self.MANIFEST)
        mock_resolve.return_value = "open-webui"
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        action = FixImageAction(
            targets=self.TARGET_GIT, enabled=True,
            require_log_signature=False,
            gitops_repo_dir=str(tmp_path),
            gitops_manifest="open-webui.yaml",
            gitops_branch="main", gitops_push=False,
        )
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=False))
        assert res.success is True
        cmd_str = ";".join(" ".join(c.args[0]) for c in mock_run.call_args_list)
        assert " commit " in cmd_str
        assert " push " not in cmd_str

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_gitops_dry_run(self, mock_run, mock_resolve, tmp_path):
        manifest = tmp_path / "open-webui.yaml"
        manifest.write_text(self.MANIFEST)
        mock_resolve.return_value = "open-webui"
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        action = FixImageAction(
            targets=self.TARGET_GIT, enabled=True,
            require_log_signature=False,
            gitops_repo_dir=str(tmp_path),
            gitops_manifest="open-webui.yaml",
            gitops_branch="main", gitops_push=True,
        )
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=True))
        assert res.dry_run is True
        assert res.success is True
        assert "v0.13.3-nix" not in manifest.read_text()  # dry-run leaves repo untouched

    def test_replace_image_line_pure(self):
        from predictive_agent.actions.fix_image import replace_image_line
        out, old = replace_image_line(self.MANIFEST, "new-img:1")
        assert old == "172.25.24.36:5001/open-webui/open-webui@sha256:old"
        assert "image: new-img:1" in out
        assert "sha256:old" not in out

    def test_replace_image_line_no_image(self):
        from predictive_agent.actions.fix_image import replace_image_line
        with pytest.raises(ValueError):
            replace_image_line("apiVersion: v1\nkind: ConfigMap\n", "new-img:1")


class TestRevert:
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_revert_restores_old_image(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        action = FixImageAction(targets=TARGETS, enabled=True)
        prev = {
            "desired_state": {
                "prev_image": "old-image:tag",
                "mode": "patch",
                "container": "open-webui",
            }
        }
        res = action.revert("open-webui", prev, ActionContext(namespace="llm", dry_run=False))
        assert res is not None
        assert res.success is True
        assert "old-image:tag" in res.command

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_revert_resolves_deployment_from_pod_name(self, mock_run, mock_resolve):
        """Verification loop passes the POD name; revert must resolve the
        owning deployment (was a bug: kubectl set image deployment/<pod>)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        mock_resolve.return_value = "open-webui"
        action = FixImageAction(targets=TARGETS, enabled=True)
        prev = {
            "desired_state": {
                "prev_image": "old-image:v1", "mode": "patch", "container": "open-webui"
            }
        }
        res = action.revert("open-webui-6f9b4c8d5e-abc12", prev,
                            ActionContext(namespace="llm", dry_run=False))
        assert res is not None and res.success is True
        assert "deployment/open-webui" in res.command
        assert "open-webui=old-image:v1" in res.command
        mock_resolve.assert_called_once()

    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_revert_containerless_prev_image(self, mock_run, mock_resolve):
        """Legacy records without a container still revert (deployment-scoped)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        mock_resolve.return_value = "open-webui"
        action = FixImageAction(targets=TARGETS, enabled=True)
        prev = {"desired_state": {"prev_image": "old:v2", "mode": "patch"}}
        res = action.revert("open-webui-abc", prev,
                            ActionContext(namespace="llm", dry_run=False))
        assert res is not None and res.success is True
        assert "deployment/open-webui" in res.command
        assert "old:v2" in res.command


class TestExecuteNoImage:
    @patch("predictive_agent.actions.fix_image.resolve_deployment_name")
    def test_execute_fails_when_target_has_no_image(self, mock_resolve):
        """A malformed/allowlist entry without an image must fail gracefully,
        not raise KeyError (operational config could be edited by hand)."""
        mock_resolve.return_value = "open-webui"
        action = FixImageAction(
            targets=[{"namespace": "llm", "deployment": "open-webui", "mode": "patch"}],
            enabled=True, require_log_signature=False,
        )
        res = action.execute("open-webui-x", ActionContext(namespace="llm", dry_run=False))
        assert res is not None
        assert res.success is False
        assert "has no image" in res.message


class TestCurrentImage:
    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_current_image_all_containers(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="img-a:v1 img-b:v2", stderr="")
        action = FixImageAction(targets=TARGETS, enabled=True)
        img = action._current_image("llm", "multy")
        assert img == "img-a:v1"
        cmd = mock_run.call_args[0][0]
        assert "containers[*].image" in cmd[-1]

    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_current_image_target_container_filter(self, mock_run):
        """When the fix targets a specific container, its image — not
        container[0] — is the prev_image for revert (multi-container bug)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="img-b:v2", stderr="")
        action = FixImageAction(targets=TARGETS, enabled=True)
        img = action._current_image("llm", "multy", container="web")
        assert img == "img-b:v2"
        cmd = mock_run.call_args[0][0]
        assert "containers[?(@.name=='web')].image" in cmd[-1]

    @patch("predictive_agent.actions.fix_image.subprocess.run")
    def test_patch_apply_records_container_scoped_prev_image(self, mock_run):
        """_patch_apply captures prev_image from the TARGET container, and the
        recorded desired_state carries that container for revert."""
        def _fake_run(cmd, **kw):
            if "jsonpath" in cmd[-1] and "containers[?" in cmd[-1]:
                out = "img-web:old"
            else:
                out = "deployment.apps/open-webui image updated"
            return MagicMock(returncode=0, stdout=out, stderr="")
        mock_run.side_effect = _fake_run
        action = FixImageAction(targets=TARGETS, enabled=True)
        fix = {
            "namespace": "llm", "deployment": "multy", "container": "web",
            "image": "img-web:new", "mode": "patch",
        }
        res = action._patch_apply("llm", "multy", fix,
                                  ActionContext(namespace="llm", dry_run=False))
        assert res.success is True
        assert res.desired_state["prev_image"] == "img-web:old"
        assert res.desired_state["container"] == "web"
