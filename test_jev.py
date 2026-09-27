import json
import time
import urllib.request
import urllib.error

from config import JEV_API_KEY, JEV_ENDPOINT

def test_jev():
    print("=== Testing Jev (TypeSafe AI) System One API ===")
    headers = {
        "Authorization": f"Bearer {JEV_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "jev-latest",
        "state": "The user wants to cancel subscription due to high billing charges.",
        "questions": {
            "is_cancellation": {
                "type": "noul",
                "instructions": "Is the user requesting a cancellation?"
            },
            "category": {
                "type": "choice",
                "instructions": "What is the primary category?",
                "criteria": {
                    "pricing": "Issues regarding cost or billing",
                    "technical": "Issues regarding product features or bugs",
                    "other": "Other general feedback"
                }
            },
            "churn_risk": {
                "type": "score",
                "instructions": "Rate the churn risk level",
                "criteria": ["low", "medium", "high", "critical"]
            }
        }
    }
    
    req = urllib.request.Request(
        JEV_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            latency = (time.time() - t0) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Status: HTTP {resp.status} (Completed in {latency:.1f}ms)")
            print("\nResponse:")
            print(json.dumps(data, indent=2))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_jev()
