"""Send realistic random traffic to the API so Grafana has something to show.

python scripts/generate_traffic.py --url http://localhost:8080 --n 300
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def patient():
    return {
        "age": random.randint(30, 77), "sex": random.randint(0, 1), "cp": random.randint(1, 4),
        "trestbps": random.randint(95, 190), "chol": random.randint(150, 380),
        "fbs": random.randint(0, 1), "restecg": random.choice([0, 2]),
        "thalach": random.randint(90, 200), "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 4.5), 1), "slope": random.randint(1, 3),
        "ca": random.choice([0, 0, 0, 1, 2, 3, None]), "thal": random.choice([3, 3, 6, 7, None]),
    }


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--bad-ratio", type=float, default=0.05, help="share of invalid requests")
    a = ap.parse_args()
    codes = {}
    for _ in range(a.n):
        body = {"age": -1} if random.random() < a.bad_ratio else patient()
        c = post(a.url.rstrip("/") + "/predict", body)
        codes[c] = codes.get(c, 0) + 1
        time.sleep(a.delay)
    print("status codes:", codes)


if __name__ == "__main__":
    main()
