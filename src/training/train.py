import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.env.swim_meet_env import SwimMeetEnv
from src.models.roster import Roster
from src.models.meet_schedule import MeetSchedule, MeetEvent
from src.data.pipeline import DataPipeline


RULES_PATH = Path("data/league_rules.json")
ROSTER_PATH = Path("data/raw/roster.csv")
MODEL_PATH = Path("models/policy.zip")


def make_env(rules: dict, roster_csv: str) -> SwimMeetEnv:
    pipeline = DataPipeline(rules)
    df = pipeline.load(roster_csv)
    df = pipeline.impute_missing_times(df)
    swimmers = pipeline.build_roster(df)

    events = [
        MeetEvent(
            event_id=e,
            is_relay="relay" in e.lower(),
            slots=4 if "relay" in e.lower() else 1,
            scoring_table=rules["scoring"]["relay" if "relay" in e.lower() else "individual"],
        )
        for e in rules["events"]
    ]
    schedule = MeetSchedule(events=events)
    return SwimMeetEnv(Roster(swimmers), schedule)


def mask_fn(env: SwimMeetEnv):
    return env.action_masks()


def train(total_timesteps: int = 500_000):
    rules = json.loads(RULES_PATH.read_text())
    env = ActionMasker(make_env(rules, str(ROSTER_PATH)), mask_fn)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_PATH)
    print(f"Saved policy to {MODEL_PATH}")


def infer(model_path: str):
    rules = json.loads(RULES_PATH.read_text())
    env = ActionMasker(make_env(rules, str(ROSTER_PATH)), mask_fn)

    model = MaskablePPO.load(model_path, env=env)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Simulated meet points: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.timesteps)
    else:
        infer(args.model)
