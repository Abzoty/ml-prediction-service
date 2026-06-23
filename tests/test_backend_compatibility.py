"""Simulates PredictionServiceClient.java end-to-end."""
import requests

URL = "http://localhost:5002/predict"


def main():
    payload = {
        "courses": [
            {"code": "CS301",  "term_work": 35.0, "exam_work": 50.0,
             "result": 85.0,   "grade": "A",      "points": 4.0},
            {"code": "MATH201","term_work": 30.0, "exam_work": 45.0,
             "result": 75.0,   "grade": "B+",     "points": 3.3},
            {"code": "AI400",  "term_work": None, "exam_work": None,
             "result": None,   "grade": None,     "points": None},  # pending
        ]
    }
    r = requests.post(URL, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()

    # Must match PythonServiceResponse.java
    assert "probabilities"   in data
    assert "model_version"   in data
    assert isinstance(data["probabilities"], dict)

    total = sum(data["probabilities"].values())
    assert 0.99 <= total <= 1.01, f"probabilities do not sum to 1: {total}"

    print(f"model_version = {data['model_version']}")
    for dept, p in sorted(data["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {dept:>4}  {p:.4f}")
    print(f"  Σ     = {total:.4f}  ✓")


if __name__ == "__main__":
    main()