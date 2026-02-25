# AI_USAGE.md

## AI Tools Used
I used AI coding assistants (ChatGPT) to:
- brainstorm project structure and workflow
- review anomaly detection model options and tradeoffs (e.g., Isolation Forest vs alternatives)
- help draft/review boilerplate code patterns and documentation structure
- improve wording and organization for `REPORT.md` / `README.md`
- cross-check edge cases for feature engineering and inference script behavior

## What I Implemented / Validated Myself
I personally:
- implemented and ran the EDA, feature engineering, training, and inference workflow in my environment
- debugged environment and execution issues during development
- verified saved artifacts and generated prediction CSV outputs
- ran final smoke tests for both training and inference after documentation updates
- selected the final submission approach and documented the tradeoffs/results

## Validation Steps Performed
- Verified dataset loading/parsing and missing values
- Verified timestamp parsing and feature generation
- Ran training pipeline successfully (`python -m src.train ...`)
- Ran inference successfully (`python predict.py --input ... --output ...`)
- Confirmed output CSV row count matches input row count
- Confirmed output includes `anomaly_score` and `is_anomaly` columns
- Checked anomaly count and basic output consistency on final runs

## Limitations of AI Assistance
AI suggestions included alternative model ideas, code-review guidance, and report phrasing support. Final model selection, implementation, debugging, validation, and submission decisions were made based on my own execution/testing and judgment.