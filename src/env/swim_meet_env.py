# SwimMeetEnv — Gymnasium environment
# Models lineup assignment as a sequential decision process
# Each step assigns one swimmer to the current event
# action_masks() filters by age_group + gender + event count limits (MaskablePPO)
# Reward is total simulated meet points, returned only at episode termination
