#!/usr/bin/env python3
"""Jenkins Deploy MCP Server

Provides MCP tools for discovering Jenkins jobs linked to this git repo,
listing branches, triggering builds, and checking build status.

Required environment variables:
  JENKINS_URL   - e.g. http://jenkins.51zhulie.com
  JENKINS_USER  - Jenkins username
  JENKINS_TOKEN - Jenkins API token
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("jenkins-deploy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jenkins_client() -> tuple[str, httpx.Auth]:
    url = os.environ.get("JENKINS_URL", "").rstrip("/")
    user = os.environ.get("JENKINS_USER", "")
    token = os.environ.get("JENKINS_TOKEN", "")
    if not url or not user or not token:
        raise RuntimeError(
            "JENKINS_URL, JENKINS_USER, and JENKINS_TOKEN must be set in the environment."
        )
    return url, httpx.BasicAuth(user, token)


def _get_git_remote_url(cwd: str | None = None) -> str:
    """Read remote.origin.url from .git/config in cwd (or current dir)."""
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=cwd or os.getcwd(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not get git remote URL: {result.stderr.strip()}")
    return result.stdout.strip()


def _normalize_git_url(url: str) -> str:
    """Normalize git URL for comparison (strip .git suffix, trailing slash)."""
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/").lower()


def _collect_jobs(base_url: str, auth: httpx.Auth, folder_path: str = "") -> list[dict]:
    """Recursively collect all Jenkins jobs (including those in folders)."""
    api_url = f"{base_url}{folder_path}/api/json?tree=jobs[name,url,_class]"
    resp = httpx.get(api_url, auth=auth, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()

    jobs: list[dict] = []
    for job in data.get("jobs", []):
        cls = job.get("_class", "")
        # Folder types contain nested jobs
        if "Folder" in cls or "WorkflowMultiBranchProject" in cls or "OrganizationFolder" in cls:
            sub_path = urlparse(job["url"]).path.rstrip("/")
            # Use the path relative to base_url
            rel_path = sub_path[len(urlparse(base_url).path.rstrip("/")):]
            jobs.extend(_collect_jobs(base_url, auth, rel_path))
        else:
            jobs.append({"name": job["name"], "url": job["url"], "class": cls})
    return jobs


def _get_job_scm_urls(job_url: str, auth: httpx.Auth) -> list[str]:
    """Fetch config.xml for a job and extract all SCM remote URLs."""
    config_url = job_url.rstrip("/") + "/config.xml"
    try:
        resp = httpx.get(config_url, auth=auth, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    # <hudson.plugins.git.UserRemoteConfig><url>…</url></hudson.plugins.git.UserRemoteConfig>
    urls = []
    for elem in root.iter("url"):
        parent = elem.tag
        text = (elem.text or "").strip()
        if text:
            urls.append(text)
    return urls


def _get_job_parameters(job_url: str, auth: httpx.Auth) -> list[str]:
    """Return the names of build parameters defined on the job."""
    api_url = job_url.rstrip("/") + "/api/json?tree=property[parameterDefinitions[name]]"
    try:
        resp = httpx.get(api_url, auth=auth, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    params: list[str] = []
    for prop in data.get("property", []):
        for param_def in prop.get("parameterDefinitions", []):
            name = param_def.get("name", "")
            if name:
                params.append(name)
    return params


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def find_jenkins_jobs(cwd: str = "") -> dict[str, Any]:
    """Discover Jenkins jobs whose SCM URL matches this git repository's remote origin.

    Args:
        cwd: Working directory to read the git remote URL from (defaults to current dir).

    Returns a list of matching jobs with name and URL.
    """
    base_url, auth = _jenkins_client()

    try:
        remote_url = _get_git_remote_url(cwd or None)
    except RuntimeError as e:
        return {"error": str(e)}

    normalized_remote = _normalize_git_url(remote_url)

    try:
        all_jobs = _collect_jobs(base_url, auth)
    except httpx.HTTPStatusError as e:
        return {"error": f"Jenkins API error: {e.response.status_code} {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}

    matched: list[dict] = []
    for job in all_jobs:
        scm_urls = _get_job_scm_urls(job["url"], auth)
        for scm_url in scm_urls:
            if _normalize_git_url(scm_url) == normalized_remote:
                matched.append({"name": job["name"], "url": job["url"]})
                break

    if not matched:
        return {
            "remote_url": remote_url,
            "total_jobs_checked": len(all_jobs),
            "matched_jobs": [],
            "message": "No Jenkins jobs found matching this git remote URL.",
        }

    return {
        "remote_url": remote_url,
        "total_jobs_checked": len(all_jobs),
        "matched_jobs": matched,
    }


@mcp.tool()
def list_branches(job_name: str = "", cwd: str = "") -> dict[str, Any]:
    """List remote branches available for deployment.

    Uses `git ls-remote --heads origin` to fetch branches from the remote,
    reusing existing git credentials (no extra token needed).

    Args:
        job_name: (Informational) Jenkins job name — not used for the git query.
        cwd: Working directory (defaults to current dir).

    Returns a sorted list of branch names.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "origin"],
            capture_output=True,
            text=True,
            cwd=cwd or os.getcwd(),
            timeout=30,
        )
        if result.returncode != 0:
            return {"error": f"git ls-remote failed: {result.stderr.strip()}"}
    except Exception as e:
        return {"error": str(e)}

    branches: list[str] = []
    for line in result.stdout.splitlines():
        # format: <sha>\trefs/heads/<branch>
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            branches.append(parts[1][len("refs/heads/"):])

    return {"job_name": job_name, "branches": sorted(branches)}


@mcp.tool()
def trigger_build(job_name: str, branch: str) -> dict[str, Any]:
    """Trigger a Jenkins build for the given job and branch.

    Automatically detects whether the job accepts parameters and chooses
    the correct endpoint (buildWithParameters vs build).  The branch
    parameter name is guessed from the job's defined parameters
    (first match among: BRANCH, branch, GIT_BRANCH, BRANCH_NAME).

    Args:
        job_name: Jenkins job name (as returned by find_jenkins_jobs).
        branch: Branch name to build.

    Returns build queue URL and queue item ID.
    """
    base_url, auth = _jenkins_client()
    job_url = f"{base_url}/job/{job_name}"

    # Fetch crumb for CSRF protection
    crumb_header: dict[str, str] = {}
    try:
        crumb_resp = httpx.get(
            f"{base_url}/crumbIssuer/api/json",
            auth=auth,
            timeout=10,
            follow_redirects=True,
        )
        if crumb_resp.status_code == 200:
            crumb_data = crumb_resp.json()
            crumb_header = {crumb_data["crumbRequestField"]: crumb_data["crumb"]}
    except Exception:
        pass  # CSRF disabled on this Jenkins instance

    params = _get_job_parameters(job_url, auth)

    # Determine branch parameter name
    branch_param_candidates = ["BRANCH", "branch", "GIT_BRANCH", "BRANCH_NAME"]
    branch_param = next((p for p in params if p in branch_param_candidates), None)

    if params and branch_param:
        endpoint = f"{job_url}/buildWithParameters"
        body = {branch_param: branch}
    elif params and not branch_param:
        # Job is parameterized but branch param name is non-standard — try common names
        endpoint = f"{job_url}/buildWithParameters"
        body = {params[0]: branch}
    else:
        # Not parameterized
        endpoint = f"{job_url}/build"
        body = {}

    try:
        resp = httpx.post(
            endpoint,
            auth=auth,
            headers=crumb_header,
            data=body,
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"Build trigger failed: HTTP {e.response.status_code}",
            "details": e.response.text[:500],
        }
    except Exception as e:
        return {"error": str(e)}

    queue_url = resp.headers.get("Location", "")
    # Extract queue item ID from URL like .../queue/item/123/
    queue_id: str | None = None
    parts = [p for p in queue_url.rstrip("/").split("/") if p]
    for i, part in enumerate(parts):
        if part == "item" and i + 1 < len(parts):
            queue_id = parts[i + 1]
            break

    return {
        "job_name": job_name,
        "branch": branch,
        "branch_param": branch_param or (params[0] if params else None),
        "queue_url": queue_url,
        "queue_id": queue_id,
        "message": f"Build triggered successfully. Queue URL: {queue_url}",
    }


@mcp.tool()
def get_build_status(job_name: str, build_number: str | int) -> dict[str, Any]:
    """Get the status of a specific Jenkins build.

    Args:
        job_name: Jenkins job name.
        build_number: Build number (integer) or "lastBuild".

    Returns build result, duration, URL, and whether it is still running.
    """
    base_url, auth = _jenkins_client()
    api_url = f"{base_url}/job/{job_name}/{build_number}/api/json"

    try:
        resp = httpx.get(api_url, auth=auth, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}

    return {
        "job_name": job_name,
        "build_number": data.get("number"),
        "building": data.get("building", False),
        "result": data.get("result"),  # SUCCESS / FAILURE / ABORTED / null (still building)
        "duration_ms": data.get("duration"),
        "url": data.get("url"),
        "display_name": data.get("displayName"),
        "timestamp": data.get("timestamp"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
