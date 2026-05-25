# DataPipeline class
# Methods:
#   load(path) -> DataFrame
#   _clean(df) -> DataFrame
#   impute_missing_times(df) -> DataFrame
#   build_roster(df) -> list[Swimmer]
#   load_opponent_times(path) -> dict[(age_group, gender, event), (mean, std)]
#     groups historical times by key, fits mean + std per group
#     (mean, std) are passed to MeetSchedule for Gaussian opponent sampling
