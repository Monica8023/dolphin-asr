#!/usr/bin/env python3
"""
dolphin-asr Jenkins 自动部署工具
用法:
  python deploy.py list                          # 列出所有 Job
  python deploy.py branches <job_name>           # 查看某个 Job 的可用分支
  python deploy.py deploy <job_name> [branch]    # 触发部署，branch 默认 origin/test_env
  python deploy.py status [job_name]             # 查看最近构建状态
"""

import sys
import json
import time
import base64
import urllib.request
import urllib.parse
import urllib.error

JENKINS_URL = "http://tools.zhulie.com/jenkins"
USER = "test"
TOKEN = "1178d3d12180f74708a145d0b3666d2f0a"
DEFAULT_BRANCH = "origin/test_env"

AUTH = base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
HEADERS = {"Authorization": f"Basic {AUTH}"}


def request(path, method="GET", data=None):
    url = f"{JENKINS_URL}{path}"
    req = urllib.request.Request(url, headers=HEADERS, method=method, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read()
            return r.status, body.decode() if body else ""
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        print(f"请求失败: {e}")
        sys.exit(1)


def list_jobs():
    status, body = request("/api/json?tree=jobs[name,color]")
    if status != 200:
        print(f"获取失败 HTTP {status}")
        return
    jobs = json.loads(body).get("jobs", [])
    print(f"{'Job 名称':<45} {'状态'}")
    print("-" * 60)
    for j in jobs:
        color = j.get("color", "")
        state = {"blue": "✓ 成功", "red": "✗ 失败", "yellow": "⚠ 不稳定",
                 "blue_anime": "⟳ 构建中", "red_anime": "⟳ 构建中"}.get(color, color)
        print(f"{j['name']:<45} {state}")


def get_branches(job_name):
    path = f"/job/{urllib.parse.quote(job_name)}/api/json?tree=property[parameterDefinitions[name,defaultParameterValue]]"
    status, body = request(path)
    if status != 200:
        print(f"获取失败 HTTP {status}")
        return
    data = json.loads(body)
    for prop in data.get("property", []):
        for param in prop.get("parameterDefinitions", []):
            default = param.get("defaultParameterValue", {}).get("value", "")
            print(f"参数: {param['name']}  默认值: {default}")


def deploy(job_name, branch=DEFAULT_BRANCH):
    print(f"触发构建: {job_name}  分支: {branch}")
    data = urllib.parse.urlencode({"branch": branch}).encode()
    status, _ = request(f"/job/{urllib.parse.quote(job_name)}/buildWithParameters", method="POST", data=data)
    if status not in (200, 201):
        print(f"触发失败 HTTP {status}")
        return

    print("构建已触发，等待启动...")
    time.sleep(3)

    # 轮询状态
    path = f"/job/{urllib.parse.quote(job_name)}/lastBuild/api/json?tree=number,result,building,url"
    for i in range(40):
        status, body = request(path)
        if status != 200:
            print(f"查询失败 HTTP {status}")
            return
        d = json.loads(body)
        num = d.get("number")
        building = d.get("building")
        result = d.get("result")
        url = d.get("url", "")
        if not building:
            icon = "✓" if result == "SUCCESS" else "✗"
            print(f"\n{icon} 构建 #{num} 完成: {result}")
            print(f"  详情: {url}")
            return
        print(f"  [{i*15}s] 构建 #{num} 进行中...", end="\r")
        time.sleep(15)
    print("\n超时，请手动查看构建状态")


def status(job_name=None):
    if job_name:
        path = f"/job/{urllib.parse.quote(job_name)}/api/json?tree=builds[number,result,building,timestamp,duration]{{0,5}}"
        s, body = request(path)
        if s != 200:
            print(f"获取失败 HTTP {s}")
            return
        builds = json.loads(body).get("builds", [])
        print(f"{'#':<6} {'结果':<12} {'耗时(s)'}")
        print("-" * 30)
        for b in builds:
            result = "构建中" if b.get("building") else (b.get("result") or "-")
            duration = round(b.get("duration", 0) / 1000)
            print(f"#{b['number']:<5} {result:<12} {duration}")
    else:
        # 列出所有 job 最近构建
        s, body = request("/api/json?tree=jobs[name,lastBuild[number,result,building]]")
        if s != 200:
            print(f"获取失败 HTTP {s}")
            return
        jobs = json.loads(body).get("jobs", [])
        print(f"{'Job 名称':<45} {'最近构建':<8} {'状态'}")
        print("-" * 65)
        for j in jobs:
            lb = j.get("lastBuild") or {}
            num = f"#{lb['number']}" if lb.get("number") else "-"
            result = "构建中" if lb.get("building") else (lb.get("result") or "-")
            print(f"{j['name']:<45} {num:<8} {result}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "list":
        list_jobs()
    elif args[0] == "branches" and len(args) >= 2:
        get_branches(args[1])
    elif args[0] == "deploy":
        job = args[1] if len(args) >= 2 else "test_dolphin-operate-service"
        branch = args[2] if len(args) >= 3 else DEFAULT_BRANCH
        deploy(job, branch)
    elif args[0] == "status":
        status(args[1] if len(args) >= 2 else None)
    else:
        print(__doc__)
