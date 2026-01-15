## Step 1: Install helm 
```
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4
chmod 700 get_helm.sh
./get_helm.sh
```

## Step:2 Create n8n deployment
```
kubectl apply -f n8n-fix.yaml
```

## Step 3: Create Alertmanager → n8n using a global AlertmanagerConfig
#### 3.1 Create the BasicAuth Secret (in monitoring namespace)
```
kubectl -n monitoring create secret generic n8n-webhook-basic-auth \
  --from-literal=username='alertmanager' \
  --from-literal=password='alertmanager'
```
#### 3.2: Create AlertmanagerConfig (in monitoring namespace)


```
nano alertmanagerconfig-aiops.yaml
```
And paste

```
apiVersion: monitoring.coreos.com/v1alpha1
kind: AlertmanagerConfig
metadata:
  name: aiops-global-webhook
  namespace: monitoring
spec:
  route:
    receiver: n8n-webhook
    groupBy: ["alertname","namespace","severity"]
    groupWait: 10s
    groupInterval: 2m
    repeatInterval: 1h
  receivers:
    - name: n8n-webhook
      webhookConfigs:
        - url: "http://n8n-svc.aiops.svc.cluster.local:5678/webhook/alertmanager"
          sendResolved: true
          httpConfig:
            basicAuth:
              username:
                name: n8n-webhook-basic-auth
                key: username
              password:
                name: n8n-webhook-basic-auth
                key: password


```

Apply 
```
kubectl apply -f alertmanagerconfig-aiops.yaml
```
#### 3.3 Point your Alertmanager CR to this global config

```
kubectl -n monitoring patch alertmanager monitoring-kube-prometheus-alertmanager \
  --type merge \
  -p '{"spec":{"alertmanagerConfiguration":{"name":"aiops-global-webhook"}}}'

```

#### 3.4 Check Alertmanager received the alert
```
kubectl -n monitoring run amcheck --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://monitoring-kube-prometheus-alertmanager.monitoring.svc.cluster.local:9093/api/v2/alerts | grep -i AIOpsWebhookTest
```

#### 3.5 Confirm Alertmanager config contains your n8n webhook receiver
```
AM_POD=$(kubectl -n monitoring get pod -l app.kubernetes.io/name=alertmanager -o jsonpath='{.items[0].metadata.name}')
kubectl -n monitoring exec -it "$AM_POD" -c alertmanager -- sh -c \
  "grep -nE 'receivers:|webhook_configs:|n8n-svc|alertmanagerConfiguration' /etc/alertmanager/config_out/alertmanager.env.yaml || true"
```

## Step 4: Deploy AI Recommendation API (RAG-ready + OpenAI + /metrics + local storage)

#### 4.1 Create configmap with python file

first create main.py
```
nano main.py
```
```
import os, json, time, sqlite3, re
from typing import Any, Dict, Optional, List
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from openai import OpenAI
import requests

client = OpenAI()  # uses OPENAI_API_KEY from env

DB_PATH = os.getenv("DB_PATH", "/data/aiops.db")
RUNBOOKS_PATH = os.getenv("RUNBOOKS_PATH", "/data/runbooks.txt")

REQ_TOTAL = Counter("aiops_reco_requests_total", "Total recommendation requests", ["status"])
LAT = Histogram("aiops_reco_request_latency_seconds", "Recommendation latency (seconds)")
OPENAI_TOTAL = Counter("aiops_reco_openai_calls_total", "OpenAI calls", ["status"])

app = FastAPI(title="AIOps Recommendation API")

# --- DB helpers (with auto migration) ---
def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            alertname TEXT,
            severity TEXT,
            namespace TEXT,
            pod TEXT,
            deployment TEXT,
            summary TEXT,
            recommendation TEXT,
            k8s_context TEXT,
            raw_json TEXT
        )
    """)
    conn.commit()

    # Auto-migrate old DBs (add missing columns)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(recommendations)")
    cols = [r[1] for r in cur.fetchall()]
    adds = {
        "pod": "TEXT",
        "deployment": "TEXT",
        "k8s_context": "TEXT",
    }
    for c, t in adds.items():
        if c not in cols:
            conn.execute(f"ALTER TABLE recommendations ADD COLUMN {c} {t}")
    conn.commit()
    return conn

# --- simple runbook retrieval ---
def load_runbooks() -> str:
    return open(RUNBOOKS_PATH, "r", encoding="utf-8").read() if os.path.exists(RUNBOOKS_PATH) else ""

def retrieve(runbooks: str, q: str, max_chars: int = 2000) -> str:
    qtok = set(re.findall(r"[a-z0-9]+", q.lower()))
    best, best_score = "", 0
    for chunk in runbooks.split("\n\n"):
        ctok = set(re.findall(r"[a-z0-9]+", chunk.lower()))
        score = len(qtok & ctok)
        if score > best_score:
            best, best_score = chunk, score
    return best[:max_chars] if best else ""

# --- extract from Alertmanager payload ---
def extract(payload: Dict[str, Any]) -> Dict[str, str]:
    alerts = payload.get("alerts") or []
    a = alerts[0] if alerts else {}
    labels = a.get("labels") or {}
    ann = a.get("annotations") or {}
    ns = labels.get("namespace") or labels.get("kubernetes_namespace") or "unknown"
    return {
        "alertname": labels.get("alertname", "unknown"),
        "severity": labels.get("severity", "unknown"),
        "namespace": ns,
        "pod": labels.get("pod", "") or labels.get("pod_name", ""),
        "deployment": labels.get("deployment", "") or labels.get("app", ""),
        "summary": (ann.get("summary") or ann.get("description") or "")[:500],
    }

# --- Kubernetes API client (in-cluster SA) ---
def k8s_headers() -> Dict[str, str]:
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(token_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    return {"Authorization": f"Bearer {token}"}

def k8s_get(path: str) -> Any:
    api = "https://kubernetes.default.svc"
    ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    url = f"{api}{path}"
    r = requests.get(url, headers=k8s_headers(), verify=ca, timeout=5)
    r.raise_for_status()
    return r.json()

def pod_summary(pod_json: Dict[str, Any]) -> str:
    status = pod_json.get("status", {})
    phase = status.get("phase", "")
    cs = (status.get("containerStatuses") or [])
    if not cs:
        return f"phase={phase}"
    c0 = cs[0]
    name = c0.get("name", "")
    restarts = c0.get("restartCount", 0)

    state = c0.get("state", {}) or {}
    last = c0.get("lastState", {}) or {}

    def fmt_state(s: Dict[str, Any], prefix: str) -> List[str]:
        out = []
        if "waiting" in s:
            out.append(f"{prefix}.waiting.reason={s['waiting'].get('reason','')}")
            out.append(f"{prefix}.waiting.message={s['waiting'].get('message','')}")
        if "terminated" in s:
            out.append(f"{prefix}.terminated.reason={s['terminated'].get('reason','')}")
            out.append(f"{prefix}.terminated.exitCode={s['terminated'].get('exitCode','')}")
        return out

    lines = [f"phase={phase}", f"container={name}", f"restartCount={restarts}"]
    lines += fmt_state(state, "state")
    # lastState is nested keys like {"terminated": {...}}
    lines += fmt_state(last, "last")
    return "\n".join([l for l in lines if l and not l.endswith("=")])

def fetch_pod_events(ns: str, pod: str) -> List[Dict[str, Any]]:
    # events v1
    try:
        ev = k8s_get(f"/api/v1/namespaces/{ns}/events?fieldSelector=involvedObject.kind%3DPod%2CinvolvedObject.name%3D{pod}")
        items = ev.get("items", []) or []
        out = []
        for e in items[-30:]:
            out.append({
                "type": e.get("type", ""),
                "reason": e.get("reason", ""),
                "message": (e.get("message", "") or "")[:400],
                "count": e.get("count", 1),
                "lastTimestamp": e.get("lastTimestamp") or e.get("eventTime") or e.get("metadata", {}).get("creationTimestamp", ""),
            })
        return out
    except Exception:
        return []

def fetch_pod_logs(ns: str, pod: str, container: Optional[str]) -> str:
    # Try: previous + container, then current + container, then without container
    attempts = []
    if container:
        attempts.append(f"/api/v1/namespaces/{ns}/pods/{pod}/log?tailLines=80&previous=true&container={container}")
        attempts.append(f"/api/v1/namespaces/{ns}/pods/{pod}/log?tailLines=80&container={container}")
    attempts.append(f"/api/v1/namespaces/{ns}/pods/{pod}/log?tailLines=80&previous=true")
    attempts.append(f"/api/v1/namespaces/{ns}/pods/{pod}/log?tailLines=80")

    api = "https://kubernetes.default.svc"
    ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    for p in attempts:
        try:
            r = requests.get(f"{api}{p}", headers=k8s_headers(), verify=ca, timeout=5)
            if r.status_code == 200 and (r.text or "").strip():
                return r.text[-5000:]
        except Exception:
            pass
    return ""

def build_k8s_context(ns: str, pod: str) -> Dict[str, Any]:
    if not ns or not pod:
        return {}
    try:
        pj = k8s_get(f"/api/v1/namespaces/{ns}/pods/{pod}")
        container = ""
        cs = (pj.get("status", {}).get("containerStatuses") or [])
        if cs:
            container = cs[0].get("name", "")
        return {
            "pod_summary": pod_summary(pj),
            "events": fetch_pod_events(ns, pod),
            "logs_tail": fetch_pod_logs(ns, pod, container),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/recommend")
def recommend(payload: Dict[str, Any]):
    t0 = time.time()
    try:
        info = extract(payload)

        # Real K8s context only if pod is provided
        kctx = build_k8s_context(info["namespace"], info["pod"]) if info.get("pod") else {}

        runbooks = load_runbooks()
        ctx = retrieve(runbooks, f"{info['alertname']} {info['summary']} {info['namespace']} {info['severity']} {info.get('pod','')}")

        system = (
            "You are a Kubernetes SRE assistant. Provide safe, step-by-step troubleshooting guidance. "
            "No self-healing actions. Include kubectl commands and what to validate in Prometheus/Grafana. "
            "Use the provided Kubernetes context (pod status, events, logs) as evidence."
        )

        user = f"""Alert:
- alertname: {info['alertname']}
- severity: {info['severity']}
- namespace: {info['namespace']}
- pod: {info.get('pod','')}
- deployment: {info.get('deployment','')}
- summary: {info['summary']}

Kubernetes context (evidence):
{json.dumps(kctx, indent=2) if kctx else "(no pod context provided)"}

Runbook context:
{ctx if ctx else "(no matching runbook snippet)"}

Return:
1) Probable cause (based on evidence)
2) Immediate checks (commands)
3) Mitigation steps
4) What confirms recovery
"""

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        try:
            resp = client.responses.create(model=model, instructions=system, input=user)
            text = resp.output_text
            OPENAI_TOTAL.labels("success").inc()
        except Exception as e:
            text = f"OpenAI call failed: {e}"
            OPENAI_TOTAL.labels("error").inc()

        conn = db()
        conn.execute(
            "INSERT INTO recommendations(ts, alertname, severity, namespace, pod, deployment, summary, recommendation, k8s_context, raw_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                int(time.time()),
                info["alertname"],
                info["severity"],
                info["namespace"],
                info.get("pod", ""),
                info.get("deployment", ""),
                info["summary"],
                text,
                json.dumps(kctx)[:200000],
                json.dumps(payload)[:200000],
            ),
        )
        conn.commit()
        conn.close()

        REQ_TOTAL.labels("success").inc()
        LAT.observe(time.time() - t0)
        return {"alert": info, "k8s_context": kctx, "recommendation": text}

    except Exception as e:
        REQ_TOTAL.labels("error").inc()
        LAT.observe(time.time() - t0)
        return {"error": str(e)}
```
```
kubectl -n aiops create configmap ai-reco-app --from-file=main.py
```
#### 4.1.1 Create secret 
```
kubectl -n aiops create secret generic ai-reco-openai \
  --from-literal=OPENAI_API_KEY='PASTE_YOUR_OPENAI_KEY_HERE' \
  --from-literal=OPENAI_MODEL='gpt-5-nano'
```

#### 4.2 Create deployment
```
cat > ai-reco-deploy.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-reco
  namespace: aiops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-reco
  template:
    metadata:
      labels:
        app: ai-reco
    spec:
      containers:
        - name: api
          image: python:3.12-slim
          ports:
            - name: http
              containerPort: 8080
          envFrom:
            - secretRef:
                name: ai-reco-openai
          env:
            - name: DB_PATH
              value: /data/aiops.db
            - name: RUNBOOKS_PATH
              value: /data/runbooks.txt
          volumeMounts:
            - name: app
              mountPath: /app
            - name: data
              mountPath: /data
          command: ["/bin/sh","-lc"]
          args:
            - |
              pip install --no-cache-dir fastapi uvicorn prometheus-client openai && \
              if [ ! -f /data/runbooks.txt ]; then echo "Paste runbooks here (blank line between topics)." > /data/runbooks.txt; fi && \
              uvicorn main:app --app-dir /app --host 0.0.0.0 --port 8080
      volumes:
        - name: app
          configMap:
            name: ai-reco-app
        - name: data
          persistentVolumeClaim:
            claimName: ai-reco-data
---
apiVersion: v1
kind: Service
metadata:
  name: ai-reco
  namespace: aiops
  labels:
    app: ai-reco
spec:
  selector:
    app: ai-reco
  ports:
    - name: http
      port: 8080
      targetPort: http
EOF

 

kubectl apply -f ai-reco-deploy.yaml
kubectl -n aiops rollout status deploy/ai-reco
```

#### 4.3 Test the API
```
kubectl -n monitoring run recotest --rm -it --restart=Never --image=curlimages/curl -- \
  sh -c 'curl -sS -XPOST http://ai-reco.aiops.svc.cluster.local:8080/recommend \
  -H "Content-Type: application/json" \
  -d "{\"status\":\"firing\",\"alerts\":[{\"labels\":{\"alertname\":\"Demo\",\"severity\":\"warning\",\"namespace\":\"default\"},\"annotations\":{\"summary\":\"pod crashloop\"}}]}" | head -c 700; echo'

```

#### 4.4 Verify that the configmap present in AI-deployment
```
kubectl -n aiops get cm ai-reco-app -o jsonpath='{.data.main\.py}' | head -n 3
echo
```
#### 4.5 Make AI-Reco “smart” by enriching with Kubernetes context- create service account and rolebinding
Right now your AI-Reco is giving general advice because it only sees:

alertname + summary

To make it “better”, AI-Reco should (based on labels):

   - if alert has pod → fetch pod describe + events + logs

   - if alert has deployment → fetch rollout status + events

   - if alert has container → fetch container state and --previous logs

#### 4.6 migrate the existing DB (no data loss)
```
POD=$(kubectl -n aiops get pod -l app=ai-reco -o jsonpath='{.items[0].metadata.name}')
echo "POD=$POD"

kubectl -n aiops exec -it "$POD" -- sh -lc 'python3 - <<'"'"'PY'"'"'
import os, sqlite3

db = os.getenv("DB_PATH", "/data/aiops.db")
conn = sqlite3.connect(db)
cur = conn.cursor()

cur.execute("PRAGMA table_info(recommendations)")
cols = [r[1] for r in cur.fetchall()]
print("Existing columns:", cols)

need = {
  "pod": "TEXT",
  "deployment": "TEXT",
  "k8s_context": "TEXT",
}
for c,t in need.items():
    if c not in cols:
        print(f"Adding column: {c} {t}")
        cur.execute(f"ALTER TABLE recommendations ADD COLUMN {c} {t}")

conn.commit()

cur.execute("PRAGMA table_info(recommendations)")
print("Updated columns:", [r[1] for r in cur.fetchall()])

conn.close()
print("✅ Migration complete")
PY'

```
#### 4.6 retry test
```
kubectl -n monitoring run recotest --rm -i --restart=Never --image=curlimages/curl -- \
  sh -c 'curl -sS -XPOST http://ai-reco.aiops.svc.cluster.local:8080/recommend \
  -H "Content-Type: application/json" \
  -d "{\"status\":\"firing\",\"alerts\":[{\"labels\":{\"alertname\":\"PodCrashLoop\",\"severity\":\"warning\",\"namespace\":\"ai-test\",\"pod\":\"badpod\"},\"annotations\":{\"summary\":\"badpod is crashing\"}}]}"; echo'

```

