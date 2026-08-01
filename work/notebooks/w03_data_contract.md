# ML-04 — Search Intelligence Data Contract

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w03_data_contract.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

> This notebook uses the local starter dataset (`content_refresh_anonymized.csv`) instead of the Hugging Face warehouse dataset to complete the data contract exercise.



## 1. Contract (Markdown)

### What does one row mean?

One row represents one content page with its aggregated search performance metrics over the last 90 days.

---

### Which dataset am I using?

Dataset:
data/raw/content_refresh_anonymized.csv

> This notebook uses the local starter dataset (`content_refresh_anonymized.csv`) instead of the Hugging Face warehouse dataset to complete the data contract exercise.

---

### Time Window

The dataset contains aggregated metrics for the previous 90 days along with last 30-day and previous 30-day statistics.

---

### What am I predicting?

I want to rank content pages according to their refresh priority.

---

### What am I deliberately excluding?

I will not use future performance information or manually created labels because they would introduce data leakage.

# 2. Verification Queries (Code)

### Query 1


```python
from pathlib import Path
import pandas as pd

candidate_paths = [
    Path.cwd() / "data/raw/content_refresh_anonymized.csv",
    Path.cwd().parent / "data/raw/content_refresh_anonymized.csv",
    Path.cwd().parent.parent / "data/raw/content_refresh_anonymized.csv",
]
data_path = next((p for p in candidate_paths if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Starter data not found in the expected repo locations.")

df = pd.read_csv(data_path)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

```

    Rows: 30000
    Columns: 44
    

The dataset contains 30,000 rows and 44 columns.

### Query 2


```python
print(df["content_id"].nunique())
```

    30000
    

Each row represents one content page.

### Query 3


```python
print(df.isnull().sum())
```

    content_id                    0
    client_id                     0
    search_volume              2468
    competition                2468
    competition_level          2610
    cpc                        2468
    content_type                  0
    main_intent                2374
    word_count                 7699
    char_count                 7699
    provider_used             21438
    model_used                 5733
    impressions_90d               0
    clicks_90d                    0
    pageviews_90d                 0
    sessions_90d                  0
    users_90d                     0
    engaged_sessions_90d          0
    ai_sessions_90d               0
    scroll_events_90d             0
    days_with_impressions         0
    days_with_sessions            0
    impressions_last_30d          0
    clicks_last_30d               0
    sessions_last_30d             0
    impressions_prev_30d          0
    clicks_prev_30d               0
    sessions_prev_30d             0
    content_age_days              0
    age_tier                      0
    age_tier_order                0
    days_since_last_update        0
    freshness_tier                0
    word_count_tier            7699
    char_count_tier            7699
    ctr                           0
    avg_position                  0
    engagement_rate               0
    scroll_rate                 125
    ai_traffic_pct                0
    impression_tier               0
    position_tier                 0
    trend_direction               0
    trend_pct                  3388
    dtype: int64
    

Some columns such as search_volume, competition and word_count contain missing values.








```python
# This cell is for CODE (numbers, a query, a check).
# Write your text answer in the cell ABOVE this one — typing sentences here breaks Run All.
print(df.columns.tolist())
df.info()
```

    ['content_id', 'client_id', 'search_volume', 'competition', 'competition_level', 'cpc', 'content_type', 'main_intent', 'word_count', 'char_count', 'provider_used', 'model_used', 'impressions_90d', 'clicks_90d', 'pageviews_90d', 'sessions_90d', 'users_90d', 'engaged_sessions_90d', 'ai_sessions_90d', 'scroll_events_90d', 'days_with_impressions', 'days_with_sessions', 'impressions_last_30d', 'clicks_last_30d', 'sessions_last_30d', 'impressions_prev_30d', 'clicks_prev_30d', 'sessions_prev_30d', 'content_age_days', 'age_tier', 'age_tier_order', 'days_since_last_update', 'freshness_tier', 'word_count_tier', 'char_count_tier', 'ctr', 'avg_position', 'engagement_rate', 'scroll_rate', 'ai_traffic_pct', 'impression_tier', 'position_tier', 'trend_direction', 'trend_pct']
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 44 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   content_id              30000 non-null  object 
     1   client_id               30000 non-null  object 
     2   search_volume           27532 non-null  float64
     3   competition             27532 non-null  float64
     4   competition_level       27390 non-null  object 
     5   cpc                     27532 non-null  float64
     6   content_type            30000 non-null  object 
     7   main_intent             27626 non-null  object 
     8   word_count              22301 non-null  float64
     9   char_count              22301 non-null  float64
     10  provider_used           8562 non-null   object 
     11  model_used              24267 non-null  object 
     12  impressions_90d         30000 non-null  int64  
     13  clicks_90d              30000 non-null  int64  
     14  pageviews_90d           30000 non-null  int64  
     15  sessions_90d            30000 non-null  int64  
     16  users_90d               30000 non-null  int64  
     17  engaged_sessions_90d    30000 non-null  int64  
     18  ai_sessions_90d         30000 non-null  int64  
     19  scroll_events_90d       30000 non-null  int64  
     20  days_with_impressions   30000 non-null  int64  
     21  days_with_sessions      30000 non-null  int64  
     22  impressions_last_30d    30000 non-null  int64  
     23  clicks_last_30d         30000 non-null  int64  
     24  sessions_last_30d       30000 non-null  int64  
     25  impressions_prev_30d    30000 non-null  int64  
     26  clicks_prev_30d         30000 non-null  int64  
     27  sessions_prev_30d       30000 non-null  int64  
     28  content_age_days        30000 non-null  int64  
     29  age_tier                30000 non-null  object 
     30  age_tier_order          30000 non-null  int64  
     31  days_since_last_update  30000 non-null  int64  
     32  freshness_tier          30000 non-null  object 
     33  word_count_tier         22301 non-null  object 
     34  char_count_tier         22301 non-null  object 
     35  ctr                     30000 non-null  float64
     36  avg_position            30000 non-null  float64
     37  engagement_rate         30000 non-null  float64
     38  scroll_rate             29875 non-null  float64
     39  ai_traffic_pct          30000 non-null  float64
     40  impression_tier         30000 non-null  object 
     41  position_tier           30000 non-null  object 
     42  trend_direction         30000 non-null  object 
     43  trend_pct               26612 non-null  float64
    dtypes: float64(11), int64(19), object(14)
    memory usage: 10.1+ MB
    

## 3. Feature Frame

Create only FIVE features.


```python
features = df[
[
"impressions_90d",
"clicks_90d",
"ctr",
"avg_position",
"days_since_last_update"
]]

display(features.head())
```



  <div id="df-44b49f99-51f6-4a53-badd-35676a77d7e3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>impressions_90d</th>
      <th>clicks_90d</th>
      <th>ctr</th>
      <th>avg_position</th>
      <th>days_since_last_update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3803</td>
      <td>29</td>
      <td>0.76</td>
      <td>10.6</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15320</td>
      <td>7</td>
      <td>0.05</td>
      <td>20.3</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12581</td>
      <td>11</td>
      <td>0.09</td>
      <td>36.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11751</td>
      <td>58</td>
      <td>0.49</td>
      <td>6.2</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19140</td>
      <td>24</td>
      <td>0.13</td>
      <td>44.0</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-44b49f99-51f6-4a53-badd-35676a77d7e3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-44b49f99-51f6-4a53-badd-35676a77d7e3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-44b49f99-51f6-4a53-badd-35676a77d7e3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



## 4. Leakage Experiment

Create a deliberately bad feature.


```python
df["bad_feature"] = df["trend_direction"]
display(df[["trend_direction", "bad_feature"]].head())
```



  <div id="df-8b6c4846-d298-4a01-a53f-d9ff167e6023" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trend_direction</th>
      <th>bad_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>down</td>
      <td>down</td>
    </tr>
    <tr>
      <th>1</th>
      <td>down</td>
      <td>down</td>
    </tr>
    <tr>
      <th>2</th>
      <td>down</td>
      <td>down</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stable</td>
      <td>stable</td>
    </tr>
    <tr>
      <th>4</th>
      <td>down</td>
      <td>down</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8b6c4846-d298-4a01-a53f-d9ff167e6023')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8b6c4846-d298-4a01-a53f-d9ff167e6023 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8b6c4846-d298-4a01-a53f-d9ff167e6023');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



This feature contains information very close to the prediction target. Using it would give the model unfair future knowledge, producing unrealistically high accuracy. Therefore it was removed.


```python
df.drop(columns=["bad_feature"], inplace=True)
```

### impressions_90d
`impressions_90d` is a historical metric reflecting past search visibility, thus it's available before making any predictions.

### clicks_90d
`clicks_90d` summarizes historical clicks, making it available as a feature before prediction.

### ctr
`ctr` (Click-Through Rate) is calculated from historical clicks and impressions, so it is already known before making a prediction.

### avg_position
`avg_position` comes directly from historical Google Search performance data, so it is available before prediction.

### days_since_last_update
`days_since_last_update` is known because the page's last update date is a pre-existing fact, available before prediction.




```python

```



## 4. Data limits

*What can this data never tell you? Unbalanced history, GSC-only early rows, window overlaps.*
## Limitations

This dataset is anonymized and contains aggregated historical metrics. It does not include actual page content or future observations, so conclusions are limited to decision support rather than proving causal effects.
*   Dataset is anonymized.
*   Missing values exist in several columns.
*   The dataset cannot explain why traffic changes.
*   Results may not generalize to every website.


```python

```



## Self-check

Before you submit, confirm each line honestly:

- [x] Every section above is filled — markdown thinking AND the code that backs it
- [x] The notebook runs top to bottom with no errors (Runtime → Run all)
- [x] No client names, URLs, or private queries anywhere
- [x] My claims use careful words: observed, measured, directional, decision-support
- [x] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.

✔ One row defined

✔ Dataset defined

✔ Five features selected

✔ Leakage demonstrated

✔ Limitation explained
