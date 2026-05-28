"""
Day-1 script: creates a dummy model.pkl that Role 3 can use to build
the serving infrastructure before the real model is ready.

Run once:
    python models/create_dummy_model.py
"""

import pickle
import random


class DummyModel:
    """Dummy model that returns a random prediction."""

    def predict(self, features: dict) -> dict:
        direction = random.choice(["high", "low", "flat"])
        confidence = round(random.uniform(0.3, 0.9), 2)
        return {"direction": direction, "confidence": confidence}


if __name__ == "__main__":
    model = DummyModel()
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Dummy model saved to models/model.pkl")
