"""FixImageAction — bounded GitOps-aware image-drift remediation.

Addresses the 2026-11-18 incident class: a workload CrashLoopBackOff caused by
version/image drift where the running image can no longer boot against shared
state (e.g. alembic migrations advanced the DB beyond the pinned image's
migration chain), and where a plain restart cannot help.

Design constraints (bounded autonomy):
  * The agent NEVER invents an image. It only applies operator-declared
    mappings from ``FIX_IMAGE_TARGETS`` (namespace + deployment + container ->
    target image).
  * By default it additionally requires the pod's recent logs to carry a
    migration/image-drift signature (alembic / "can't locate revision" /
    django migrations / prisma / knex / digest mismatch …) before it acts.
  * ``mode="patch"`` runs ``kubectl set image`` (works when no GitOps
    controller reverts direct patches).
  * ``mode="gitops"`` edits the manifest in a local checkout of the ArgoCD
    source repo and commits (+ optionally pushes). This survives controllers
    with selfHeal+prune that revert direct kubectl patches.
  * The action is reversible: the previous image is recorded and restored by
    ``revert`` after a failed verification (or on request).

Each execution is gated by the RemediationManager's usual safety policy,
cooldown, dedup, review tier and verification loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess

from predictive_agent import config
from predictive_agent.remediator import (
    ActionContext,
    RemediationAction,
    RemediationResult,
)

logger = logging.getLogger(__name__)

RISK_THRESHOLD = 0.7

DEFAULT_LOG_PATTERNS = [
    "alembic",
    "can't locate revision",
    "migration failed",
    "running upgrade",
    "django.db.migrations",
    "django_migrations",
    "prisma migrate",
    "knex",
    "sequelize",
    "digest mismatch",
    "manifest unknown",
    "no matching manifest",
    "image pull",
]


def _load_json_env(name: str, default):
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def find_target(targets, namespace: str, deployment: str):
    """Return the operator-declared fix target for (namespace, deployment)."""
    if not targets:
        return None
    for t in targets:
        if t.get("namespace") == namespace and t.get("deployment") == deployment:
            return t
    return None


def log_has_drift_signature(log_text: str, patterns) -> bool:
    """True when log text contains any migration/image-drift signature."""
    if not log_text:
        return False
    low = log_text.lower()
    for pat in patterns or []:
        if pat.lower() in low:
            return True
    return False


def _run(cmd, timeout: float):
    """subprocess helper returning (returncode, stdout, stderr); never raises."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except subprocess.TimeoutExpired:
        return -2, "", f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)


def resolve_deployment_name(namespace: str, pod_name: str, timeout: float = 15.0):
    """Resolve a pod to its owning Deployment via kubectl ownerReferences.

    Handles pod -> ReplicaSet -> Deployment chains. Returns the deployment
    name or None when the pod has no deployment owner (DaemonSet/StatefulSet/
    bare pod).
    """
    def _owner_refs(kind: str, name: str):
        cmd = [
            "kubectl", "get", kind, "-n", namespace, name,
            "-o", "jsonpath={.metadata.ownerReferences}",
        ]
        rc, out, _ = _run(cmd, timeout)
        if rc != 0 or not out:
            return None
        try:
            parsed = json.loads(out)
        except ValueError:
            return None
        if isinstance(parsed, list) and parsed:
            return parsed[0]
        return None

    pod_owner = _owner_refs("pod", pod_name)
    if not pod_owner:
        return None
    kind = pod_owner.get("kind", "")
    name = pod_owner.get("name", "")
    if kind == "Deployment":
        return name
    if kind == "ReplicaSet" and name:
        rs_owner = _owner_refs("rs", name)
        if rs_owner and rs_owner.get("kind") == "Deployment":
            return rs_owner.get("name")
    return None


def replace_image_line(manifest_text: str, new_image: str):
    """Replace the first ``image: <ref>`` line in a manifest.

    Returns ``(new_text, old_image_ref)``. Raises ValueError when the manifest
    has no image line.
    """
    lines = manifest_text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        m = re.match(r"^(\s*)image:\s*(\S+)\s*$", line)
        if m:
            old = m.group(2)
            lines[idx] = f"{m.group(1)}image: {new_image}\n"
            return "".join(lines), old
    raise ValueError("no image: line found in manifest")


class FixImageAction(RemediationAction):
    """Fix a deployment stuck in a migration/image-drift crash loop."""

    name = "fix_image"
    # Not idempotent in the P3 sense: a gitops re-run would produce a second
    # commit (the dedup window still prevents double execution).
    idempotent: bool = False
    # P6 bounded autonomy: changes a shared workload's image -> destructive tier
    # (at/above REMEDIATION_REVIEW_TIER it waits for human approval).
    risk_tier: str = "destructive"
    reversible: bool = True
    resolution_signals: tuple = ("state", "restart_count")

    def __init__(
        self,
        targets=None,
        enabled=None,
        require_log_signature=None,
        log_patterns=None,
        log_lines=None,
        threshold=None,
        gitops_repo_dir=None,
        gitops_manifest=None,
        gitops_branch=None,
        gitops_push=None,
        kubectl_timeout=None,
    ):
        self.targets = config.FIX_IMAGE_TARGETS if targets is None else targets
        self.enabled = config.FIX_IMAGE_ENABLED if enabled is None else bool(enabled)
        self.require_log_signature = (
            config.FIX_IMAGE_REQUIRE_LOG_SIGNATURE if require_log_signature is None
            else bool(require_log_signature)
        )
        self.log_patterns = (
            config.FIX_IMAGE_LOG_PATTERNS if log_patterns is None else list(log_patterns)
        )
        self.log_lines = (
            config.FIX_IMAGE_LOG_LINES if log_lines is None else int(log_lines)
        )
        self.threshold = (
            config.FIX_IMAGE_RISK_THRESHOLD if threshold is None else float(threshold)
        )
        self.gitops_repo_dir = (
            config.FIX_IMAGE_GITOPS_REPO_DIR if gitops_repo_dir is None
            else (gitops_repo_dir or "")
        )
        self.gitops_manifest = (
            config.FIX_IMAGE_GITOPS_MANIFEST if gitops_manifest is None
            else (gitops_manifest or "")
        )
        self.gitops_branch = (
            config.FIX_IMAGE_GITOPS_BRANCH if gitops_branch is None
            else (gitops_branch or "main")
        )
        self.gitops_push = (
            config.FIX_IMAGE_GITOPS_PUSH if gitops_push is None else bool(gitops_push)
        )
        self.kubectl_timeout = (
            config.FIX_IMAGE_KUBECTL_TIMEOUT_S if kubectl_timeout is None
            else float(kubectl_timeout)
        )
        self._last_prev: dict = {}

    # ── helpers ─────────────────────────────────────────────────────────────
    def _log_matches(self, namespace: str, pod_name: str, container=None) -> bool:
        """Fetch recent pod logs and look for a migration/image-drift signature."""
        for previous in (True, False):
            cmd = [
                "kubectl", "logs", "-n", namespace, pod_name,
                "--tail", str(self.log_lines),
            ]
            if previous:
                cmd.append("--previous")
            if container:
                cmd += ["-c", container]
            rc, out, _ = _run(cmd, self.kubectl_timeout)
            if rc == 0 and out and log_has_drift_signature(out, self.log_patterns):
                return True
        return False

    def _current_image(self, namespace: str, deployment: str):
        rc, out, _ = _run(
            [
                "kubectl", "get", "deployment", "-n", namespace, deployment,
                "-o", "jsonpath={.spec.template.spec.containers[*].image}",
            ],
            self.kubectl_timeout,
        )
        if rc != 0:
            return None
        parts = [p for p in out.split() if p]
        return parts[0] if parts else None

    # ── RemediationAction interface ─────────────────────────────────────────
    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        if not self.enabled:
            return False
        if risk_score < self.threshold:
            return False
        ns = getattr(pod_state, "namespace", "")
        pod = getattr(pod_state, "name", "")
        if not ns or not pod:
            return False
        deployment = resolve_deployment_name(ns, pod, self.kubectl_timeout)
        if not deployment:
            return False
        target = find_target(self.targets, ns, deployment)
        if not target:
            return False
        if self.require_log_signature:
            return self._log_matches(ns, pod, target.get("container"))
        return True

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Apply the operator-declared image fix to the pod's deployment.

        ``target`` is the pod name (the evaluate loop passes the pod); the
        owning deployment is resolved here. Preconditions are re-checked
        (defense in depth — the POMDP planner path does not call
        ``should_execute``).
        """
        ns = getattr(context, "namespace", "") or ""
        pod = target
        if not self.enabled:
            return RemediationResult(
                action=self.name, target=pod, success=False,
                dry_run=context.dry_run,
                message="fix_image not enabled",
            )
        deployment = resolve_deployment_name(ns, pod, self.kubectl_timeout)
        if not deployment:
            return RemediationResult(
                action=self.name, target=pod, success=False,
                dry_run=context.dry_run,
                message="fix_image: no owning deployment for pod",
            )
        fix = find_target(self.targets, ns, deployment)
        if not fix:
            return RemediationResult(
                action=self.name, target=pod, success=False,
                dry_run=context.dry_run,
                message=f"fix_image: no allowlisted target for {ns}/{deployment}",
            )
        if self.require_log_signature and not self._log_matches(
            ns, pod, fix.get("container")
        ):
            return RemediationResult(
                action=self.name, target=pod, success=False,
                dry_run=context.dry_run,
                message="fix_image: no migration/image-drift log signature",
            )

        mode = fix.get("mode", "patch")
        new_image = fix["image"]
        if mode == "gitops":
            return self._gitops_apply(ns, deployment, fix, context)
        return self._patch_apply(ns, deployment, fix, context)

    def _patch_apply(self, ns, deployment, fix, context) -> RemediationResult:
        new_image = fix["image"]
        container = fix.get("container")
        old_image = self._current_image(ns, deployment) or ""
        spec = (f"{container}=" if container else "") + new_image
        cmd = ["kubectl", "set", "image", f"deployment/{deployment}", spec, "-n", ns]
        if context.dry_run:
            cmd.append("--dry-run=client")
        rc, out, err = _run(cmd, self.kubectl_timeout)
        success = rc == 0
        message = out or err or ("image updated" if success else "kubectl set image failed")
        if success:
            message += f" ({old_image} -> {new_image})"
        result = RemediationResult(
            action=self.name,
            target=deployment,
            success=success,
            dry_run=context.dry_run,
            message=message,
            command=" ".join(cmd),
            desired_state={
                "prev_image": old_image or new_image,
                "mode": "patch",
                "container": container or "",
                "new_image": new_image,
            },
        )
        return result

    def _gitops_apply(self, ns, deployment, fix, context) -> RemediationResult:
        repo_dir = self.gitops_repo_dir
        manifest_rel = self.gitops_manifest
        if not repo_dir or not manifest_rel:
            return RemediationResult(
                action=self.name, target=deployment, success=False,
                dry_run=context.dry_run,
                message=(
                    "fix_image: gitops mode requires FIX_IMAGE_GITOPS_REPO_DIR "
                    "and FIX_IMAGE_GITOPS_MANIFEST"
                ),
            )
        manifest_path = os.path.join(repo_dir, manifest_rel)
        new_image = fix["image"]
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                old_text = fh.read()
            new_text, old_image = replace_image_line(old_text, new_image)
        except FileNotFoundError:
            return RemediationResult(
                action=self.name, target=deployment, success=False,
                dry_run=context.dry_run,
                message=f"fix_image: gitops manifest not found: {manifest_path}",
            )
        except ValueError as e:
            return RemediationResult(
                action=self.name, target=deployment, success=False,
                dry_run=context.dry_run,
                message=f"fix_image: gitops manifest edit failed: {e}",
            )

        commands = []
        if not context.dry_run:
            # write the edited manifest
            with open(manifest_path, "w", encoding="utf-8") as fh:
                fh.write(new_text)
            commands.append(["git", "-C", repo_dir, "add", manifest_rel])
            commands.append([
                "git", "-C", repo_dir, "commit", "-m",
                f"fix({deployment}): apply image fix -> {new_image} (predictive-agent fix_image)",
            ])
            if self.gitops_push:
                commands.append(
                    ["git", "-C", repo_dir, "push", "origin", self.gitops_branch]
                )
        else:
            # dry-run: simulate the workflow without touching the repo
            commands.append(["git", "-C", repo_dir, "add", manifest_rel])
            commands.append(["git", "-C", repo_dir, "commit", "-m", "(dry-run)"])
            if self.gitops_push:
                commands.append(["git", "-C", repo_dir, "push", "origin", self.gitops_branch])

        failures = []
        for cmd in commands:
            rc, out, err = _run(cmd, self.kubectl_timeout)
            if rc != 0:
                failures.append(f"{' '.join(cmd)}: {err or out}")
        success = not failures
        message = (
            "; ".join(failures) if failures
            else f"gitops image fix staged: {old_image} -> {new_image}"
        )
        return RemediationResult(
            action=self.name,
            target=deployment,
            success=success,
            dry_run=context.dry_run,
            message=message,
            command=" ; ".join(" ".join(c) for c in commands),
            desired_state={
                "prev_image": old_image or new_image,
                "mode": "gitops",
                "new_image": new_image,
            },
        )

    def revert(self, target: str, prev_state, context) -> RemediationResult:
        """Restore the previously-recorded image on failed verification."""
        ns = getattr(context, "namespace", "") or ""
        desired = (prev_state or {}).get("desired_state") or {}
        prev_image = desired.get("prev_image") or ""
        mode = desired.get("mode") or "patch"
        container = desired.get("container") or ""
        if not prev_image:
            return RemediationResult(
                action=self.name, target=target, success=False,
                dry_run=context.dry_run,
                message="fix_image: revert has no prev_image recorded",
            )
        if mode == "gitops":
            return self._gitops_revert(target, ns, context)
        spec = (f"{container}=" if container else "") + prev_image
        cmd = ["kubectl", "set", "image", f"deployment/{target}", spec, "-n", ns]
        if context.dry_run:
            cmd.append("--dry-run=client")
        rc, out, err = _run(cmd, self.kubectl_timeout)
        success = rc == 0
        return RemediationResult(
            action=self.name,
            target=target,
            success=success,
            dry_run=context.dry_run,
            message=(out or err) if not success else f"reverted image to {prev_image}",
            command=" ".join(cmd),
        )

    def _gitops_revert(self, target: str, ns: str, context) -> RemediationResult:
        repo_dir = self.gitops_repo_dir
        manifest_rel = self.gitops_manifest
        if not repo_dir or not manifest_rel:
            return RemediationResult(
                action=self.name, target=target, success=False,
                dry_run=context.dry_run,
                message="fix_image: gitops revert missing repo/manifest config",
            )
        # restore HEAD version of the manifest (un-committed local edits)
        cmd = ["git", "-C", repo_dir, "checkout", "--", manifest_rel]
        if context.dry_run:
            cmd.append("--dry-run" if False else "--force")
        rc, out, err = _run(cmd, self.kubectl_timeout)
        success = rc == 0
        return RemediationResult(
            action=self.name, target=target, success=success,
            dry_run=context.dry_run,
            message=(out or err) if not success else "gitops manifest restored (git checkout HEAD)",
            command=" ".join(cmd),
        )
