# Streamlit GUI — Swim Meet Lineup Optimizer
# Owner: Naomi
#
# Sidebar:
#   - Upload roster CSV (fed through DataPipeline)
#   - Choose solver: OR-Tools (deterministic) or RL policy (models/policy.zip)
#
# Main panel:
#   - Roster table grouped by age group
#   - "Generate Lineup" button -> runs solver -> displays results
#   - Results table: age_group, gender, event, assigned swimmer, best time
#   - Highlights events with no eligible swimmer
