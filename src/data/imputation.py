# StrokeProfileImputer class
# Uses linear regression to fill missing event times within the correct distance tier
# PREDICTOR_MAP: maps target event -> predictor event(s)
#   25-yard: butterfly, backstroke, breaststroke <- freestyle
#   50-yard: butterfly, backstroke, breaststroke, 100_im <- freestyle
# Methods: fit(), transform(), fit_transform()
