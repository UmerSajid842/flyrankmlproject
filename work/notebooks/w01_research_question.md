# ML-02 — Research Question and Provisional Lane

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w01_research_question.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. My lane (or freestyle) and why

# 1. My Lane

I chose the Content Refresh Prioritization lane.

The goal is to identify webpages that should be refreshed first to improve their search performance. Instead of reviewing hundreds or thousands of webpages manually, machine learning can help prioritize which pages deserve attention first based on their historical SEO performance.


```python
# This cell is for CODE (numbers, a query, a check).
# Write your text answer in the cell ABOVE this one — typing sentences here breaks Run All.

```

## 2. The question: decision, action, cost of a wrong call

# 2. Research Question

Research Question:
How can machine learning help identify which webpages should be refreshed first to improve search performance?

Decision:
Which webpages should the SEO team refresh first?

Who takes the action?
SEO specialists and content managers.

Action:
Refresh the webpages that the model identifies as high priority.

Cost of a wrong decision:
If the wrong pages are selected, time and resources are wasted while important pages continue losing traffic.


```python
# This cell is for CODE (numbers, a query, a check).
# Write your text answer in the cell ABOVE this one — typing sentences here breaks Run All.
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

df.head()
```





  <div id="df-5b49e679-c6d4-43dc-8d4c-30296314f2b2" class="colab-df-container">
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
      <th>content_id</th>
      <th>client_id</th>
      <th>search_volume</th>
      <th>competition</th>
      <th>competition_level</th>
      <th>cpc</th>
      <th>content_type</th>
      <th>main_intent</th>
      <th>word_count</th>
      <th>char_count</th>
      <th>...</th>
      <th>char_count_tier</th>
      <th>ctr</th>
      <th>avg_position</th>
      <th>engagement_rate</th>
      <th>scroll_rate</th>
      <th>ai_traffic_pct</th>
      <th>impression_tier</th>
      <th>position_tier</th>
      <th>trend_direction</th>
      <th>trend_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>content_304f48230142</td>
      <td>client_f369cb89fc</td>
      <td>10.0</td>
      <td>0.67</td>
      <td>HIGH</td>
      <td>2.05</td>
      <td>keyword article</td>
      <td>transactional</td>
      <td>3221.0</td>
      <td>20457.0</td>
      <td>...</td>
      <td>15000-25000</td>
      <td>0.76</td>
      <td>10.6</td>
      <td>5.88</td>
      <td>4.55</td>
      <td>0.0</td>
      <td>good</td>
      <td>striking</td>
      <td>down</td>
      <td>-41.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>content_a1fb4e703a9e</td>
      <td>client_4e07408562</td>
      <td>90.0</td>
      <td>0.01</td>
      <td>LOW</td>
      <td>0.05</td>
      <td>keyword article</td>
      <td>informational</td>
      <td>2481.0</td>
      <td>15562.0</td>
      <td>...</td>
      <td>15000-25000</td>
      <td>0.05</td>
      <td>20.3</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>0.0</td>
      <td>good</td>
      <td>page_3_5</td>
      <td>down</td>
      <td>-57.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>content_9aa793d4d895</td>
      <td>client_7f2253d7e2</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>LOW</td>
      <td>0.00</td>
      <td>keyword article</td>
      <td>informational</td>
      <td>3515.0</td>
      <td>23643.0</td>
      <td>...</td>
      <td>15000-25000</td>
      <td>0.09</td>
      <td>36.5</td>
      <td>0.00</td>
      <td>28.57</td>
      <td>0.0</td>
      <td>good</td>
      <td>page_3_5</td>
      <td>down</td>
      <td>-60.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>content_331d6c4de07b</td>
      <td>client_19581e27de</td>
      <td>10.0</td>
      <td>0.00</td>
      <td>LOW</td>
      <td>0.00</td>
      <td>keyword article</td>
      <td>commercial</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.49</td>
      <td>6.2</td>
      <td>1.28</td>
      <td>3.45</td>
      <td>0.0</td>
      <td>good</td>
      <td>page_1</td>
      <td>stable</td>
      <td>-13.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>content_d99b7a2d90ca</td>
      <td>client_3fdba35f04</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>LOW</td>
      <td>0.00</td>
      <td>keyword article</td>
      <td>informational</td>
      <td>2803.0</td>
      <td>17469.0</td>
      <td>...</td>
      <td>15000-25000</td>
      <td>0.13</td>
      <td>44.0</td>
      <td>0.00</td>
      <td>24.29</td>
      <td>0.0</td>
      <td>good</td>
      <td>page_3_5</td>
      <td>down</td>
      <td>-34.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5b49e679-c6d4-43dc-8d4c-30296314f2b2')"
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
        document.querySelector('#df-5b49e679-c6d4-43dc-8d4c-30296314f2b2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5b49e679-c6d4-43dc-8d4c-30296314f2b2');
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





```python
df.shape
df.info()
df.describe()
```

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
    





  <div id="df-6566ba18-25b8-4033-8d94-ce54b8327032" class="colab-df-container">
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
      <th>search_volume</th>
      <th>competition</th>
      <th>cpc</th>
      <th>word_count</th>
      <th>char_count</th>
      <th>impressions_90d</th>
      <th>clicks_90d</th>
      <th>pageviews_90d</th>
      <th>sessions_90d</th>
      <th>users_90d</th>
      <th>...</th>
      <th>sessions_prev_30d</th>
      <th>content_age_days</th>
      <th>age_tier_order</th>
      <th>days_since_last_update</th>
      <th>ctr</th>
      <th>avg_position</th>
      <th>engagement_rate</th>
      <th>scroll_rate</th>
      <th>ai_traffic_pct</th>
      <th>trend_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27532.000000</td>
      <td>27532.000000</td>
      <td>27532.000000</td>
      <td>22301.000000</td>
      <td>22301.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>...</td>
      <td>30000.000000</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>29875.000000</td>
      <td>30000.000000</td>
      <td>26612.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>158.882391</td>
      <td>0.146954</td>
      <td>0.485342</td>
      <td>3107.760325</td>
      <td>20665.277835</td>
      <td>5200.366300</td>
      <td>16.097333</td>
      <td>49.942467</td>
      <td>37.066633</td>
      <td>35.937700</td>
      <td>...</td>
      <td>10.283000</td>
      <td>256.16780</td>
      <td>4.786533</td>
      <td>46.098300</td>
      <td>0.510733</td>
      <td>16.34238</td>
      <td>2.534520</td>
      <td>18.212921</td>
      <td>0.768196</td>
      <td>-4.785969</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1518.270825</td>
      <td>0.285241</td>
      <td>2.101560</td>
      <td>1452.382598</td>
      <td>10115.344042</td>
      <td>16838.019547</td>
      <td>75.076958</td>
      <td>152.101430</td>
      <td>107.069131</td>
      <td>103.748185</td>
      <td>...</td>
      <td>42.578003</td>
      <td>132.70793</td>
      <td>0.790392</td>
      <td>42.078709</td>
      <td>3.279162</td>
      <td>15.21679</td>
      <td>8.310096</td>
      <td>29.472768</td>
      <td>7.429454</td>
      <td>473.861780</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>90.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-100.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2413.000000</td>
      <td>15644.000000</td>
      <td>81.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>132.00000</td>
      <td>4.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>6.20000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-62.600000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2877.000000</td>
      <td>19116.000000</td>
      <td>731.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>236.00000</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.070000</td>
      <td>10.80000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>-33.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20.000000</td>
      <td>0.130000</td>
      <td>0.000000</td>
      <td>3666.000000</td>
      <td>24011.000000</td>
      <td>3615.250000</td>
      <td>7.000000</td>
      <td>33.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>333.00000</td>
      <td>5.000000</td>
      <td>104.000000</td>
      <td>0.290000</td>
      <td>22.30000</td>
      <td>1.350000</td>
      <td>23.530000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>74000.000000</td>
      <td>1.000000</td>
      <td>100.360000</td>
      <td>9546.000000</td>
      <td>111158.000000</td>
      <td>517715.000000</td>
      <td>4178.000000</td>
      <td>5998.000000</td>
      <td>4345.000000</td>
      <td>4913.000000</td>
      <td>...</td>
      <td>4247.000000</td>
      <td>564.00000</td>
      <td>6.000000</td>
      <td>373.000000</td>
      <td>100.000000</td>
      <td>245.00000</td>
      <td>100.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>44900.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6566ba18-25b8-4033-8d94-ce54b8327032')"
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
        document.querySelector('#df-6566ba18-25b8-4033-8d94-ce54b8327032 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6566ba18-25b8-4033-8d94-ce54b8327032');
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




## 3. Quick look at the data (2-3 real numbers)

# 3. Quick Look at the Data

The starter dataset contains **30,000 webpages (rows)** and **44 features (columns)**.

Some important features include:

- impressions_90d
- clicks_90d
- ctr
- avg_position
- search_volume
- trend_direction

From the summary statistics:

- The average number of impressions over the last 90 days is **5,200.37**.
- The average number of clicks over the last 90 days is **16.10**.
- The average click-through rate (CTR) is **0.51%**.
- The average Google search position is **16.34**.

These statistics show that the dataset contains useful SEO and content performance information that can be used to prioritize webpages for content refresh.


```python
# This cell is for CODE (numbers, a query, a check).
# Write your text answer in the cell ABOVE this one — typing sentences here breaks Run All.
## Quick Look at the Data


```

## 4. Careful words: what I can and can't claim

*Write what your work will be able to say (observed, directional, decision-support) — and what it never will (causal proof, 'predicting Google').*

# 4. Careful Words

This project is intended to support SEO specialists in identifying webpages that may need content updates.

The results should be used as decision support rather than automatic decisions.

The model cannot guarantee better Google rankings because search performance depends on many factors outside this dataset.


```python
# This cell is for CODE (numbers, a query, a check).
# Write your text answer in the cell ABOVE this one — typing sentences here breaks Run All.

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
