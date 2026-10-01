# Smart-Credit-Automating-Financial-Risk-Brackets-with-Predictive-Machine-Learning-Models

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Score Classification API</title>
<style>
body{max-width:860px;margin:2rem auto;padding:0 1rem;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.6;color:#1f2328}
h1,h2{border-bottom:1px solid #d0d7de;padding-bottom:.3em}
code{background:#f6f8fa;padding:.2em .4em;border-radius:6px;font-size:90%}
pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto}
pre code{background:none;padding:0}
table{border-collapse:collapse;display:block;overflow-x:auto}
th,td{border:1px solid #d0d7de;padding:6px 13px}
th{background:#f6f8fa}
a{color:#0969da}
</style>
</head>
<body>
<h1 id="credit-score-classification-api">Credit Score Classification
API</h1>
<p>A machine learning model that classifies a customer's credit score as
<strong>Poor</strong>, <strong>Standard</strong>, or
<strong>Good</strong> from their banking and credit data, served through
a REST API built with FastAPI.</p>
<p>This project was built as a final certification assignment: acting as
a data scientist at a global finance company, the goal is to turn raw
credit-related records into a reliable credit score classifier.</p>
<h2 id="features">Features</h2>
<ul>
<li>Data cleaning and feature preparation for messy real-world bank
data</li>
<li>Comparison of several classifiers (Random Forest, XGBoost, LightGBM,
Logistic Regression)</li>
<li>Model explainability with feature importance and SHAP</li>
<li>REST API for single and batch predictions, returning class
probabilities</li>
<li>Interactive API docs (Swagger UI) out of the box</li>
<li>Optional Docker deployment</li>
</ul>
<h2 id="tech-stack">Tech Stack</h2>
<p>Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost,
LightGBM, SHAP, FastAPI, Uvicorn</p>
<h2 id="project-structure">Project Structure</h2>
<pre><code>.
├── Model.ipynb        # Data cleaning, EDA, model comparison, explainability
├── main.py            # FastAPI application
├── requirements.txt   # Python dependencies
├── train.csv          # Training data (not included in the repo, see below)
├── test.csv           # Test data (not included in the repo, see below)
└── model.joblib       # Trained model, created on first run</code></pre>
<h2 id="the-model">The Model</h2>
<p>The notebook walks through the full workflow:</p>
<ol type="1">
<li><strong>Cleaning:</strong> removes junk characters and placeholder
values, fixes impossible values, and fills gaps.</li>
<li><strong>EDA:</strong> distributions by credit class, categorical
breakdowns, and a correlation heatmap.</li>
<li><strong>Modelling:</strong> class imbalance is handled with SMOTE,
and four models are compared on validation macro F1 using a
customer-grouped train/validation split so the same customer never
appears in both sets.</li>
<li><strong>Explainability:</strong> feature importance, SHAP values,
and a confusion matrix show what drives each rating.</li>
</ol>
<p>The model served by the API is a Logistic Regression classifier
(<code>C=0.1</code>, balanced class weights) trained on the full
training set with label-encoded categorical features and median
imputation, as in the final cell of the notebook.</p>
<table>
<thead>
<tr class="header">
<th>Model</th>
<th>Validation Macro F1</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Add your results here</td>
<td></td>
</tr>
</tbody>
</table>
<h2 id="getting-started">Getting Started</h2>
<h3 id="prerequisites">Prerequisites</h3>
<ul>
<li>Python 3.12 or 3.13 recommended</li>
<li><code>train.csv</code> placed in the project root</li>
</ul>
<h3 id="installation">Installation</h3>
<div class="sourceCode" id="cb2"><pre
class="sourceCode bash"><code class="sourceCode bash"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a><span class="fu">git</span> clone https://github.com/<span class="op">&lt;</span>your-username<span class="op">&gt;</span>/<span class="op">&lt;</span>your-repo<span class="op">&gt;</span>.git</span>
<span id="cb2-2"><a href="#cb2-2" aria-hidden="true" tabindex="-1"></a><span class="bu">cd</span> <span class="op">&lt;</span>your-repo<span class="op">&gt;</span></span>
<span id="cb2-3"><a href="#cb2-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb2-4"><a href="#cb2-4" aria-hidden="true" tabindex="-1"></a><span class="ex">python</span> <span class="at">-m</span> venv venv</span>
<span id="cb2-5"><a href="#cb2-5" aria-hidden="true" tabindex="-1"></a><span class="ex">venv\Scripts\activate</span>          <span class="co"># Windows</span></span>
<span id="cb2-6"><a href="#cb2-6" aria-hidden="true" tabindex="-1"></a><span class="co"># source venv/bin/activate     # Linux / macOS</span></span>
<span id="cb2-7"><a href="#cb2-7" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb2-8"><a href="#cb2-8" aria-hidden="true" tabindex="-1"></a><span class="ex">pip</span> install <span class="at">-r</span> requirements.txt</span></code></pre></div>
<h3 id="run-the-api">Run the API</h3>
<div class="sourceCode" id="cb3"><pre
class="sourceCode bash"><code class="sourceCode bash"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a><span class="ex">uvicorn</span> main:app <span class="at">--reload</span></span></code></pre></div>
<p>On the first start the model is trained from <code>train.csv</code>
and saved to <code>model.joblib</code>. Later starts load the saved
model. Set <code>RETRAIN=1</code> to force retraining.</p>
<p>The API runs at <code>http://127.0.0.1:8000</code>, and the
interactive docs are at <code>http://127.0.0.1:8000/docs</code>.</p>
<h2 id="api-reference">API Reference</h2>
<table>
<thead>
<tr class="header">
<th>Method</th>
<th>Endpoint</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>GET</td>
<td><code>/health</code></td>
<td>Check that the service and model are loaded</td>
</tr>
<tr class="even">
<td>POST</td>
<td><code>/predict</code></td>
<td>Predict the credit score for one record</td>
</tr>
<tr class="odd">
<td>POST</td>
<td><code>/predict/batch</code></td>
<td>Predict for up to 1000 records at once</td>
</tr>
</tbody>
</table>
<p>Fields you leave out are treated as missing and filled with the
training median.</p>
<h3 id="example-request">Example request</h3>
<div class="sourceCode" id="cb4"><pre
class="sourceCode bash"><code class="sourceCode bash"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a><span class="ex">curl</span> <span class="at">-X</span> POST http://127.0.0.1:8000/predict <span class="dt">\</span></span>
<span id="cb4-2"><a href="#cb4-2" aria-hidden="true" tabindex="-1"></a>  <span class="at">-H</span> <span class="st">&quot;Content-Type: application/json&quot;</span> <span class="dt">\</span></span>
<span id="cb4-3"><a href="#cb4-3" aria-hidden="true" tabindex="-1"></a>  <span class="at">-d</span> <span class="st">&#39;{</span></span>
<span id="cb4-4"><a href="#cb4-4" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Month&quot;: &quot;January&quot;,</span></span>
<span id="cb4-5"><a href="#cb4-5" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Age&quot;: 28,</span></span>
<span id="cb4-6"><a href="#cb4-6" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Occupation&quot;: &quot;Scientist&quot;,</span></span>
<span id="cb4-7"><a href="#cb4-7" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Annual_Income&quot;: 19114.12,</span></span>
<span id="cb4-8"><a href="#cb4-8" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Monthly_Inhand_Salary&quot;: 1824.84,</span></span>
<span id="cb4-9"><a href="#cb4-9" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Num_Bank_Accounts&quot;: 3,</span></span>
<span id="cb4-10"><a href="#cb4-10" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Num_Credit_Card&quot;: 4,</span></span>
<span id="cb4-11"><a href="#cb4-11" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Interest_Rate&quot;: 3,</span></span>
<span id="cb4-12"><a href="#cb4-12" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Num_of_Loan&quot;: 4,</span></span>
<span id="cb4-13"><a href="#cb4-13" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Delay_from_due_date&quot;: 3,</span></span>
<span id="cb4-14"><a href="#cb4-14" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Num_of_Delayed_Payment&quot;: 7,</span></span>
<span id="cb4-15"><a href="#cb4-15" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Changed_Credit_Limit&quot;: 11.27,</span></span>
<span id="cb4-16"><a href="#cb4-16" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Num_Credit_Inquiries&quot;: 4,</span></span>
<span id="cb4-17"><a href="#cb4-17" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Credit_Mix&quot;: &quot;Good&quot;,</span></span>
<span id="cb4-18"><a href="#cb4-18" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Outstanding_Debt&quot;: 809.98,</span></span>
<span id="cb4-19"><a href="#cb4-19" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Credit_Utilization_Ratio&quot;: 26.82,</span></span>
<span id="cb4-20"><a href="#cb4-20" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Credit_History_Age&quot;: &quot;22 Years and 9 Months&quot;,</span></span>
<span id="cb4-21"><a href="#cb4-21" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Payment_of_Min_Amount&quot;: &quot;No&quot;,</span></span>
<span id="cb4-22"><a href="#cb4-22" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Total_EMI_per_month&quot;: 49.57,</span></span>
<span id="cb4-23"><a href="#cb4-23" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Amount_invested_monthly&quot;: 80.42,</span></span>
<span id="cb4-24"><a href="#cb4-24" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Payment_Behaviour&quot;: &quot;High_spent_Small_value_payments&quot;,</span></span>
<span id="cb4-25"><a href="#cb4-25" aria-hidden="true" tabindex="-1"></a><span class="st">    &quot;Monthly_Balance&quot;: 312.49</span></span>
<span id="cb4-26"><a href="#cb4-26" aria-hidden="true" tabindex="-1"></a><span class="st">  }&#39;</span></span></code></pre></div>
<h3 id="example-response">Example response</h3>
<div class="sourceCode" id="cb5"><pre
class="sourceCode json"><code class="sourceCode json"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a><span class="fu">{</span></span>
<span id="cb5-2"><a href="#cb5-2" aria-hidden="true" tabindex="-1"></a>  <span class="dt">&quot;credit_score&quot;</span><span class="fu">:</span> <span class="st">&quot;Standard&quot;</span><span class="fu">,</span></span>
<span id="cb5-3"><a href="#cb5-3" aria-hidden="true" tabindex="-1"></a>  <span class="dt">&quot;class_id&quot;</span><span class="fu">:</span> <span class="dv">1</span><span class="fu">,</span></span>
<span id="cb5-4"><a href="#cb5-4" aria-hidden="true" tabindex="-1"></a>  <span class="dt">&quot;probabilities&quot;</span><span class="fu">:</span> <span class="fu">{</span></span>
<span id="cb5-5"><a href="#cb5-5" aria-hidden="true" tabindex="-1"></a>    <span class="dt">&quot;Poor&quot;</span><span class="fu">:</span> <span class="fl">0.1832</span><span class="fu">,</span></span>
<span id="cb5-6"><a href="#cb5-6" aria-hidden="true" tabindex="-1"></a>    <span class="dt">&quot;Standard&quot;</span><span class="fu">:</span> <span class="fl">0.5127</span><span class="fu">,</span></span>
<span id="cb5-7"><a href="#cb5-7" aria-hidden="true" tabindex="-1"></a>    <span class="dt">&quot;Good&quot;</span><span class="fu">:</span> <span class="fl">0.3041</span></span>
<span id="cb5-8"><a href="#cb5-8" aria-hidden="true" tabindex="-1"></a>  <span class="fu">}</span></span>
<span id="cb5-9"><a href="#cb5-9" aria-hidden="true" tabindex="-1"></a><span class="fu">}</span></span></code></pre></div>
<p>The probabilities above are illustrative. Real values depend on the
trained model.</p>
<h2 id="docker">Docker</h2>
<div class="sourceCode" id="cb6"><pre
class="sourceCode dockerfile"><code class="sourceCode dockerfile"><span id="cb6-1"><a href="#cb6-1" aria-hidden="true" tabindex="-1"></a><span class="kw">FROM</span> python:3.12-slim</span>
<span id="cb6-2"><a href="#cb6-2" aria-hidden="true" tabindex="-1"></a><span class="kw">WORKDIR</span> /app</span>
<span id="cb6-3"><a href="#cb6-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb6-4"><a href="#cb6-4" aria-hidden="true" tabindex="-1"></a><span class="kw">COPY</span> requirements-api.txt .</span>
<span id="cb6-5"><a href="#cb6-5" aria-hidden="true" tabindex="-1"></a><span class="kw">RUN</span> <span class="ex">pip</span> install <span class="at">--no-cache-dir</span> <span class="at">-r</span> requirements-api.txt</span>
<span id="cb6-6"><a href="#cb6-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb6-7"><a href="#cb6-7" aria-hidden="true" tabindex="-1"></a><span class="kw">COPY</span> main.py model.joblib ./</span>
<span id="cb6-8"><a href="#cb6-8" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb6-9"><a href="#cb6-9" aria-hidden="true" tabindex="-1"></a><span class="kw">EXPOSE</span> 8000</span>
<span id="cb6-10"><a href="#cb6-10" aria-hidden="true" tabindex="-1"></a><span class="kw">CMD</span> [<span class="st">&quot;uvicorn&quot;</span>, <span class="st">&quot;main:app&quot;</span>, <span class="st">&quot;--host&quot;</span>, <span class="st">&quot;0.0.0.0&quot;</span>, <span class="st">&quot;--port&quot;</span>, <span class="st">&quot;8000&quot;</span>]</span></code></pre></div>
<p>Run the app once locally first so <code>model.joblib</code> exists,
then:</p>
<div class="sourceCode" id="cb7"><pre
class="sourceCode bash"><code class="sourceCode bash"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a><span class="ex">docker</span> build <span class="at">-t</span> credit-score-api .</span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a><span class="ex">docker</span> run <span class="at">-p</span> 8000:8000 credit-score-api</span></code></pre></div>
<h2 id="data">Data</h2>
<p>The dataset (<code>train.csv</code>, <code>test.csv</code>) contains
customer-month banking records with fields such as income, debt, number
of loans, payment delays, credit mix, and credit history age. The target
is <code>Credit_Score</code> (Poor, Standard, Good). The data files are
not included in this repository. Place them in the project root before
running.</p>
<h2 id="notes-and-limitations">Notes and Limitations</h2>
<ul>
<li>The model is for learning and demonstration. It should not be used
to make real lending decisions.</li>
<li>Categories not seen during training are encoded as unknown, and
missing values are filled with the training median.</li>
<li>The model and the saved <code>model.joblib</code> should be built
with the same scikit-learn version you deploy with.</li>
</ul>
<h2 id="license">License</h2>
<p>Add a license of your choice (for example MIT) and state it here.</p>
<h2 id="author">Author</h2>
<p><strong>Mohsin Mubarok</strong></p>

</body>
</html>
#Exploratory Data Analysis

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-9.png)
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
