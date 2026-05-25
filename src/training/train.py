# MaskablePPO training + inference entry point
# build_schedule(rules, opponent_projections) -> MeetSchedule
#   iterates age_groups x genders x events from league_rules.json
# make_env(rules) -> SwimMeetEnv
# train(total_timesteps) — wraps env with ActionMasker, runs MaskablePPO.learn()
# infer(model_path) — loads saved policy and runs one full meet episode
