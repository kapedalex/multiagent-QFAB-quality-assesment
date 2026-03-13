
## Understanding the Problem

FRI regularly collects forecasts and the accompanying explanations from experts. Manually analyzing thousands of these texts to identify the highest-quality and most reliable reasoning is an extremely time-consuming process.

My task was to develop an automated system, utilizing LLMs, to assess and synthesize the quality of these rationales while maintaining high standards for accuracy.

## My Approach

I developed a multi-stage system based on advanced evaluation techniques, including LLM-as-a-Judge, the Quantitative Bipolar Argumentation Framework (QBAF), and ensemble principles.

Key Stages of Work:

1.  **Gathering Initial Forecasts**: I took a small set of questions from the ForecastBench dataset and used three different Gemini LLMs to generate forecasts and their associated rationales (explanations) based on a predefined prompt.
2.  **Quantitative Assessment (QBAF)**: This was the most crucial stage. I created a separate LLM function (the "Judge") that took each generated rationale and was forced to convert it into a formal QBAF structure. QBAF decomposes the reasoning into:
    *   Arguments (Atoms): Individual statements.
    *   Strength ($\tau$): A numerical rating of confidence in each individual argument (from 0.0 to 1.0).
    *   Relationships (Attacks/Supports): How arguments relate to each other.
3.  **Reasoning Quality Evaluation**: The final strength of the conclusion (the forecast) is calculated based on the QBAF. Arguments that remain strong within this structure are considered the highest quality in terms of logic and soundness.
4.  **Synthesis of Findings**: I used an LLM to analyze these high-scoring arguments from 4 different questions to identify common, cross-cutting themes that define high-quality forecasting.

## Key Results

1.  **Individual Model Performance**: The gemini-2.5-flash model showed the best forecast accuracy (lowest Brier Score) among the models tested.
2.  **Importance of Structure**: The QBAF system allowed me to see that reasoning quality does not always perfectly align with forecast accuracy, but the strongest arguments in the QBAF structure (those that retain high strength after considering all attacks and supports) are the most concrete and technically sound. For example, in the analysis of the S&P 500 forecast, the strongest arguments related to a specific requirement (low growth hurdle, March closing price) rather than general macroeconomic concerns.
3.  **Cross-Cutting Quality Themes**: Synthesizing the high-quality arguments revealed that the best reasoning always:
    *   Is Balanced: It acknowledges both historical patterns and current high volatility/uniqueness of the situation.
    *   Focuses on Specific "Disruptor" Factors: Successful arguments clearly identify what might break the general trend (e.g., a single strong athlete, a sudden economic shock).
    *   Considers Lagged Effects and Unforeseen Variables.

## System Strengths and Weaknesses

What my system does well:
*   **Structuring**: Converting free-form text into a rigorous, visualizable framework (QBAF) is helpful compared to simple text scoring.
*   **Identifying** **Important Arguments**: The system successfully isolates arguments that carry the most weight in the overall logical structure.
*   **Synthesis**: The LLM was able to extract common, non-trivial themes from analyzing different topics (sports forecasts, economics, politics).

What can be improved (Next Steps):
*   **Quality Scoring**: The criteria for the "Judge" need finer tuning to better distinguish between deep technical reasoning and superficial but well-articulated claims.
*   **Scaling**: A full-scale experiment must be conducted on a larger dataset.
*   **QFAB Ensemble**: Results from QFAB should be used to create more reliable ensemble forecasts by combining structures from different models (the idea of Multi-Agent QFAB).
*   **Visualization**: QBAF visualizations should be made dynamically adjustable (e.g., hiding weakly connected arguments).

## Conclusion

I have successfully developed and demonstrated an LLM-based system for evaluating forecasting quality that goes beyond simple text analysis. The integration of QBAF allows for an assessment of the logical soundness of the reasoning, which is a key step toward increasing the reliability of automated analysis.