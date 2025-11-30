# -New-York-Housing-Market-kaggle.com
In NYC real estate, location sets the floor, but luxury amenities — especially that elusive third bathroom — determine how high prices can climb.

https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market/code?datasetId=4269029

**A Personal Project Retrospective: NYC Housing Prices with Only Public Data**

I worked through the popular “NYC House Prices” dataset – not to chase leaderboard scores, but to understand what is actually possible with strictly public, reproducible data.

Final honest numbers (fixed 70/15/15 split, seed 42):  
R² = 0.877 | MAE = $540,907 | 4,801 rows retained

Data sources:
- Original listings dataset
- NYC PLUTO 2024 v1 (joined via BBL + geographic fallback)

Notable model discoveries:
- A clear pricing discontinuity at exactly 3 bathrooms (the “3-bath cliff”)
- Size rank and distance to Midtown remain the strongest predictors
- Building age from PLUTO adds meaningful signal in older neighborhoods

The code is a single, fully documented script that anyone can run end-to-end.

I found historian Francis J. Gavin’s framework for historical thinking surprisingly useful for engineering reflection. Here’s how the 12 questions applied to this project:

1. How did we get here?  
   Started from a leaked $98k MAE notebook → rebuilt everything from scratch across seven major versions.

2. What else was happening?  
   An external code audit arrived mid-project, NumPy changed clip() behavior, validation rules swung from too strict to balanced.

3. What was unsaid?  
   Many published low-error results on this dataset rely on future leakage, private data, or aggressive outlier removal – none of which were used here.

4. How are things trending?  
   MAE path: $98k → $1.2M → $540k → briefly $423k → settled at $540k. The number has now stabilized.

5. How is this understood by others?  
   $540k MAE looks modest at first glance, but is achieved without any proprietary features that commercial models rely on.

6. Why does this matter?  
   It establishes a transparent ceiling for what open data alone can accomplish in NYC valuation.

7. What were the unexpected outcomes?  
   The model independently discovered the sharp premium at 3+ bathrooms – a pattern brokers know intuitively but rarely see quantified.

8. Was this inevitable?  
   No – keeping the original leaked model would have been far easier.

9. Are things changing rapidly?  
   Seven full rewrites in a short period – a deliberate choice to avoid settling too early.

10. Are we using the past correctly?  
    Every bug (validation drops, clip syntax, age imputation) became a permanent lesson encoded in the final version.

11. Was this unprecedented?  
    To my knowledge, this is the first fully reproducible, strictly no-leak baseline at this scale on this dataset.

12. What does it mean?  
    The practical limit of public-data-only NYC pricing appears to be in the $520–550k MAE range. Anything significantly lower requires information not available in open sources.

Grateful for the clarity that Gavin’s framework brought to a solo technical project.

#DataScience #MachineLearning #RealEstate #PublicData #NYC #CatBoost #GeospatialAnalysis #OpenSource
