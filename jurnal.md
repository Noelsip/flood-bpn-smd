Flood Probability Estimation with Anomaly Detection using Automated Machine Learning in Balikpapan and Samarinda
> "Estimating Flood probability estimation using automated machine learning (AutoML) in balikpapan and Samarinda,"

Abstract
Keywords: artificial intelligence; machine learning; deep learning;
informatics
Introduction
As the primary buffer zones for the Nusantara Capital City
(IKN), Balikpapan and Samarinda play a strategic role in
supporting mobility, logistics, and development activities for
the new capital. The official decision to relocate the national
capital to East Kalimantan in 2019 firmly established these
two cities as indispensable supporting regions [1]. The
urgency of this development is currently challenged by severe
disaster threats. The disaster risk profile of East Kalimantan
Province identifies flooding as one of the most significant
hazards. It is classified under Sector IV Level 6 in the
Interpretative Structural Modelling analysis, indicating high
hazard, vulnerability, and risk levels across almost all local
municipalities, including Balikpapan and Samarinda [2]. This
situation is further aggravated by land cover changes resulting
from ongoing IKN construction. A hydrological study in the
Pemaluan area within the IKN development zone observed a
3.15% expansion in flood inundation areas between 2019 and
2023, directly caused by reduced soil infiltration and increased
surface runoff [3]. The flood vulnerability of Balikpapan
and Samarinda transcends local concerns and represents a
systemic threat that could impede the overall progress of the
IKN project.
Flooding in East Kalimantan, including Balikpapan and
Samarinda, is a highly dynamic phenomenon that fluctuates in
close correlation with weather conditions and extreme rainfall.
Climatologically, the province receives an average rainfall of
174.8 mm and is heavily influenced by tropical convection
processes alongside intensive global climate change. Extreme
precipitation events ultimately serve as the primary trigger for
local flooding [2]. This research supported by a hydrological
analysis of the Pemaluan watershed, which reveals that daily
rainfall data from 2004 to 2023 directly correlates with the
annual flood inundation extent. The study demonstrates that
increased rainfall intensity is proportional to the expansion of
flood-affected areas [3]. Recent research on the sustainable
development of IKN also highlights that flood management
in cities like Samarinda and Balikpapan remains unresolved.
This condition is expected to deteriorate further due to
ongoing large-scale land conversions [1]. Establishing a clear
correlation between daily weather data and historical flood
records is an essential step in understanding the temporal
patterns of these environmental triggers. This correlation
provides a reliable analytical foundation to support early
warning systems and disaster mitigation strategies across the
IKN buffer zones.
Using Traditional Machine Learning in flood modeling also
has some problems. It needs people to get involved, which
includes choosing the features picking the right algorithm and
adjusting the settings manually [4]. This not takes a long time
but also means that peoples personal opinions can affect the
results, which can make the model not work as well as it could.
Traditional Machine Learning also has trouble with the kind
of data we use for flood prediction modeling. This data is
often very complicated. Has a lot of information and it is not
balanced with many more days without floods than days with
floods. This means the model tends to favor the days without
floods, which makes it hard for it to find the flood events [4].
To fix these problems some new methods like Gradient
Boosting, Random Forest and XGBoost have become popular
for modeling flood risk [5]. These methods are good at
showing the relationships between weather and floods, give
us a better understanding of how floods work in different
environments.
This study uses three algorithms because they have unique
features and advantages.
First, Random Forest is used as a model because it is very
stable and can handle noisy data. It is also very good to
Volume 1, Issue 1 2
Author et al.: Short version of title
avoiding overfitting, which makes it a good model to compare
with the models [4].
Second, XGBoost is used because it is very fast and
efficient. It also has a built-in way to handle missing
information, which makes it very strong when working with
real-world data [5].
Third, CatBoost is chosen because it can avoid the model
from overfitting. It does by calculating some values which
makes it less likely to have problems when working with
limited data [5].
To get rid of bias and make Machine Learning more efficient
Automated Machine Learning offer a solution. Models like
TPOT and FLAML can fully automate the process of making
a Machine Learning model [6], [4]. This includes getting the
data ready, choosing the features, and finding the best model.
By minimizing involvement Automated Machine Learning
makes research more reliable and allows us to explore more
ideas in less time. This is very important for flood modeling
because it means we can make models faster which can
help save times. Flood models like these can really help us
understand how floods work and make predictions, which is
crucial.
In response to this gap research, our study proposes a
data-driven method that integrates disaster event records data
from the National Disaster Management Agency (BNPB) with
weather data from Open-Meteo. To minimize subjective bias,
Automated Machine Learning (AutoML) is implemented to
design an optimal pipeline architecture. This study compares
three state-of-the-art tree-ensemble algorithms: Random For-
est, XGBoost, and CatBoost that are designed to handle imbal-
anced tabular environmental data. The performance of these
models is then enhanced through a targeted hyperparameter
optimization scheme to ensure optimal generalization.
This research objectively implements machine learning
models to predict the probability of daily flood events in
Balikpapan and Samarinda, which are strategic buffer zones
for the Nusantara Capital City (IKN). By integrating historical
weather data and disaster event records, the study aims to
develop a robust predictive model that can provide early
warnings for flood events. The specific objectives of this
research are:
1. To integrate datasets by combining weather data from
Open-Meteo with event-based flood disaster records from
the National Disaster Management Agency (BNPB) into a
structured tabular environmental dataset.
2. To compare the performance of three tree-ensemble
algorithms (Random Forest, XGBoost, and CatBoost) in
predicting flood events based on the integrated dataset.
3. Automate the classification pipeline using an AutoML
framework, ensuring an objective and efficient modeling
process that includes data preprocessing and handling class
imbalance.
Related Works
In recent developments within disaster prediction research
specially for flood prediction, the way we look at weather
data has shifted quite fundamentally. In the past, weather
data was often treated like independent rows of isolated
records snapshots of atmospheric conditions at a single
point in time. But today, that approach is considered
insufficient, because floods rarely happen suddenly based
solely on weather conditions on the day itself. More often,
flooding is the result of a cumulative process: rainfall from
several previous days, how water seeps into or runs off the
ground, and the saturation levels of the soil over time. To
capture this kind of dynamics, researchers have turned to time-
series analysis. They use sliding window techniques that allow
models to "look back". For example, over daily to weekly
historical data, while continuously updating themselves as
new data streams in. During training, they also strictly
apply chronological splitting (TimeSeriesSplit) to prevent data
leakage, meaning past data cannot be used to peek into the
future during testing. This is crucial to keep predictions valid
and reliable.
Findings from various studies confirm that sequential ap-
proaches are far more effective than static ones. In research
by Atashi et al.[7] and Li et al.[8], sequential networks like
LSTM which use memory cells to track time lags consistently
outperformed conventional algorithms such as Random For-
est or SARIMA, both of which still treat each row of data in-
dependently. Similarly, Weng et al.[9] demonstrated that the
ability of time-series neural networks to learn and retain long-
range sequential dependencies significantly reduces prediction
errors. In short, recognizing and integrating temporal depen-
dence is no longer just an added bonus, it’s the key to identify-
ing extreme weather anomaly patterns earlier than ever before.
When evaluating models for weather and disaster predic-
tion, a number of studies point to a pretty wide performance
gap between basic classification algorithms and more ad-
vanced tree-based ensemble methods (Bagging versus Boost-
ing). Work by Ma et al. [10] and by Grzesiak & Thakkar [11],
for instance, shows that single classifiers or old-school algo-
rithms such as K-Nearest Neighbor (KNN) and Naive Bayes
often run into trouble. They struggle to map out environmental
data that tends to be highly fluctuating, high-dimensional, and
complex in pattern. As a result, their accuracy and ability to
generalize end up being pretty poor, especially when dealing
with large-scale datasets.
To get around these weaknesses, ensemble approaches have
proven to give a dramatic boost in performance. Modalavalasa
[12], as well as Hsu et al. [13], found that Random Forest,
which represents the Bagging side of things, offers strong
generalization stability and holds up well against overfitting.
This is mainly because it works by averaging out the results
from many independent decision trees that aren’t correlated
with each other.
On the other hand, Boosting algorithms like XGBoost
and CatBoost take a different route. They aim for extreme
accuracy by correcting errors in a sequential manner. Ma et
al. [10] explain that XGBoost chains together weak learners
step by step to form one very strong model. It also throws
in a regularization function that sharpens predictions while
keeping model complexity in check. Meanwhile, Hsu et al.
[13] add that CatBoost has a built-in advantage when handling
categorical weather data, it doesn’t need messy preprocessing.
Their findings empirically show that these top-tier tree-based
ensemble methods clearly outperform conventional algorithms
when it comes to reliable weather forecasting.
A study by Wang et al.[14] brings up a core problem in
disaster prediction modeling: historical data is often heavily
imbalanced. What that means in practice is that rare but high-
3 This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY) Volume 1, Issue 1
Author et al.: Short version of title
risk flood events, the minority class tend to get overlooked or
completely missed by models, simply because normal, non-
flood conditions dominate the dataset.
To stop models from being so biased toward the majority
class, Wang et al. [14] argue that algorithm-level approaches
like cost-sensitive learning are the way to go. This method
tweaks the learning algorithm internally so it becomes more
sensitive to spotting the minority class. They actually
recommend this over data-level tricks like resampling (for
example, SMOTE oversampling), because those come with
the unavoidable risk of making the model overfit.
Once we start adjusting how classes are treated, the whole
evaluation changes. Wang et al. [14] show through real data
that conventional accuracy is basically a broken metric when
dealing with imbalanced classes, it just fails to capture how
well the model is really performing. Instead, they insist that
evaluation should focus squarely on metrics like F1-score,
recall, and G-mean, because these actually reflect whether the
model can correctly detect real crisis events.
Adding another layer to the discussion, Esparza et al. [15]
point out that data imbalance isn’t just about numbers, it
can also show up as demographic and spatial bias. In many
flood reporting systems, minority populations and vulnerable
communities tend to be underrepresented in the data. So even
if a model looks good on paper, it might still fail the people
who need protection the most.
Current progress in disaster prediction modeling shows
a notable methodological shift: researchers are moving
away from manually tuning hyperparameters and toward
fully automated pipeline searches using Automated Machine
Learning (AutoML).
A study by Nam et al. [4] points out that conventional
methods like Grid Search or Random Search don’t just
take forever to run, they also tend to carry a fair amount
of human bias. To go beyond these limitations, Nam
et al. [4] proposed a hybrid framework. It combines
TPOT (Tree-based Pipeline Optimization Tool), which relies
on evolutionary algorithms to design the overall pipeline
architecture, with Optuna, which uses Bayesian optimization
to fine-tune hyperparameters efficiently and precisely.
Taking a similar direction, research by Guo et al. [5] and by
Han & Bae [16] applied TPOT to simulate genetic selection
mechanisms—mutation and crossover, for instance—in order
to automate preprocessing steps and feature selection from
weather and hydrological time-series data. The advantage here
is that the whole process becomes more objective, with less
risk of researcher bias sneaking in.
Adding to this automation ecosystem, Gao et al. [17] used
the Auto-Sklearn framework, backed by a Bayesian optimizer,
to reduce the heavy reliance on expert knowledge during
model design.
Taken together, these studies provide strong evidence that
AutoML not only speeds up computations significantly but
also ends up producing hyperparameter combinations and
complex ensemble structures that routinely beat manually
designed architectures in terms of performance.
Methods
This section describes the overall workflow used to estimate
flood probability in Balikpapan and Samarinda, organised to
ensure the reproducibility of the research. The methodol-
ogy consists of five main stages: data collection, dataset con-
struction, time-series transformation and forecast-target def-
inition, data preprocessing and feature engineering, machine
learning modelling, and model evaluation. An optional spatial
susceptibility-mapping stage using DEM and RBI layers may
be included to attribute the city-level forecast to flood-prone
sub-districts (kecamatan). Each stage is presented in detail in
the following subsections, covering the data sources, the con-
struction of the labelled point dataset, the preparation of the
conditioning factors, the training and tuning of the classifica-
tion models, and the metrics used to assess their performance.
Data Collection
This study utilized two primary data sources: flood event
records obtained from DIBI-BNPB and meteorological data
collected from Open-Meteo. The data used in this study
was obtained from the Data and Information on Disasters
in Indonesia (DIBI), a disaster information data reference
managed by the National Disaster Management Authority
(BNPB). DIBI is the primary national repository for historical
disaster records in Indonesia, and is recognized as the first and
most comprehensive disaster information center in Southeast
Asia.
This dataset covers disaster occurrence records across all
regions in Indonesia, including disaster information, disaster
type, date of occurrence, geographic location, number of
casualties, and infrastructure damage. Data collection in DIBI
is conducted by the Data and Information Center.
This dataset was selected because it covers records of
flood events in Balikpapan and Samarinda that are relevant
as training data for the prediction model, and provides spatial
disaster impact information down to the regency/city level,
enabling analysis focused on the buffer zone of the Nusantara
Capital City (IKN). This data is also an official government
source and has been widely used as a reference for research in
Indonesia.
To analyze the triggers of flooding due to extreme
weather, this study downloaded daily meteorological datasets
(2016–2026) in Balikpapan and Samarinda via the Open-
Meteo API. Crucial variables collected include rainfall,
temperature (2 m), relative humidity (2 m), MSLP, and
wind speed (10 m) due to their strong correlation with
hydrometeorological dynamics. Open-Meteo itself integrates
various proven global climate models for environmental
studies. This API-based data structure not only accelerates the
information collection process but also ensures transparency
and consistent data reprocessing.
Data Construction
Flood event data sourced from the National Disaster Mitiga-
tion Agency (BNPB) was integrated with daily meteorological
records from Open-Meteo, using date parameters as a refer-
ence for alignment. The resulting combination formed a final
dataset representing daily dynamics, with each entry contain-
ing weather parameter information along with relevant flood
event validation labels.
Time-Series Transformation and Forecast Target
In order to understand the time-related changes that occur
before flooding, the daily dataset was converted into a time
series format. For every weather-related factor (precipitation,
Volume 1, Issue 1 4
Author et al.: Short version of title
rainfall, highest and lowest temperature, wind velocity, and
soil moisture), features were created from t − 1 to t − 7
with a time lag, concurrently with accumulation characteristics
like 3, 7, and 14 day rainfall totals and 3 and 7 day soil-
moisture averages, as flooding is primarily influenced by total
precipitation than by an observation made in one day. The
target for prediction was established. as the flood incidence
three days prior (Floodt+3), structuring the three-day-ahead
flood prediction classification using time-series data instead
of a prediction of uninterrupted values. To stop any future
data from seeping into the training procedure, each feature
utilizes only data up to day t, and sequence windows were
created explicitly following dataset partitioning—an approach
inspired by findings that Pre-split sequence formation leads to
temporal leakage and results in excessive positive performance
projections [18]. In line with this In principle, the dataset was
divided by time, assigning the initial 70% for training and
the latest 30% for testing, instead of using random mixing.
This stepwise division maintains chronological causality and
guarantees that the model is never presented with future
data while training [18]. The location coordinates (latitude
and longitude) of every city was kept unchanged as location
identifiers assigned to the forecast result.
Flood and Non-Flood Point Sampling (Optional Spatial
Component)
The following point-based sampling describes an optional
spatial susceptibility-mapping component, applied only when
DEM and RBI layers are available, that complements the time-
series model by locating flood-prone sub-districts.
Machine learning flood probability estimation is formulated
as a binary classification problem in which every observation
corresponds to a geographic location labelled as either a flood
point (1) or a non-flood point (0). This point-based sampling
strategy is widely adopted in data-driven flood susceptibility
studies, where a flood inventory is paired with an equal
number of non-flood locations to train and validate spatial
prediction models [19, 4, 17].
The flood points were generated from the historical flood
records of Balikpapan and Samarinda obtained from DIBI.
Each documented flood event with an identifiable location
within the study area was converted into a point feature,
forming the positive class of the dataset. To address the class
imbalance that is characteristic of flood data, where non-flood
conditions vastly outnumber flood conditions [4], the non-
flood points were generated through random sampling within
areas that have no recorded flood history. These points were
constrained to higher-elevation and low-susceptibility zones to
reduce the likelihood of mislabelling, and were sampled in a
balanced 1:1 ratio with the flood points, following common
practice in flood susceptibility modelling [19, 4].
Each sampled point was then assigned a set of flood
conditioning factor values extracted from the spatial datasets
through point overlay in a Geographic Information System
(GIS). Topographic factors such as elevation and slope were
derived from the SRTM 30 m Digital Elevation Model [?
]. Land cover, the river network, and the road network
were obtained from the Rupabumi Indonesia (RBI) vector
basemap, from which hydrological factors such as distance
to the nearest river were computed [? ]. The anthropogenic
factor of population density was derived from the official
population statistics of Balikpapan and the East Kalimantan
regencies/cities published by BPS. The resulting labelled point
dataset, in which each point carries both its conditioning factor
values and its flood/non-flood label, serves as the input for the
subsequent preprocessing and modelling stages.
Data Preprocessing & Feature Engineering
Before the labelled point dataset could be used for modelling,
a series of preprocessing steps was applied to ensure data
quality and consistency. Points with missing or invalid
conditioning factor values, which may arise from gaps in
the spatial layers or from locations outside the coverage of
the source datasets, were identified and removed to prevent
bias during model training. Because the conditioning factors
were extracted from heterogeneous sources with different
spatial resolutions, all layers were reprojected into a common
coordinate reference system and resampled to a uniform grid
prior to point overlay, following the spatial harmonisation
procedures commonly applied in flood susceptibility studies
[19, 4].
The conditioning factors used in this study consist of both
numerical and categorical variables, and therefore required
different handling before modelling. Numerical factors such
as elevation, slope, distance to the nearest river, and popu-
lation density were standardised so that variables measured
on different scales would contribute comparably to the learn-
ing process, while the categorical land cover factor was trans-
formed into numerical form through encoding [4]. This step is
particularly important because flood conditioning factors span
widely different ranges, and unscaled inputs can cause tree-
ensemble and gradient-based models to behave inconsistently
during training [5].
Feature engineering was directed at deriving informative
hydrological and topographic predictors from the base spatial
datasets rather than using only the raw layers. Secondary
factors such as slope and distance to the river were computed
from the SRTM Digital Elevation Model and the RBI river
network, as these derived variables are known to be strong
indicators of flood susceptibility [19]. To retain only the most
relevant predictors and reduce redundancy, feature importance
analysis was used to assess the contribution of each factor,
which is consistent with the interpretable feature selection
approaches reported in recent flood modelling research [4, 6].
Anomaly Detection
In analyzing weather conditions related to flooding, the
anomaly detection stage aims to validate the significance of
normal behavior in the dataset.
This study uses the Isolation Forest algorithm as an anomaly
detection method. Isolation Forest can detect abnormal behav-
ior by recursively separating features and randomly selected
separator values. This is because anomalous observations are
relatively rare and significantly different from the general pop-
ulation. Previous studies have demonstrated the effectiveness
of the Isolation Forest algorithm in observing anomalous pat-
terns in environmental and time series datasets, maintaining
optimal performance compared to alternative anomaly detec-
tion techniques [20].
The algorithm was applied to variables collected from
Open-Meteo, including rainfall, temperature, relative humid-
ity, atmospheric pressure, and wind speed. Anomaly informa-
5 This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY) Volume 1, Issue 1
Author et al.: Short version of title
tion is maintained by generating an anomaly score for each ob-
servation. This anomaly score is implemented as an additional
factor in an Automated Machine Learning (AutoML) frame-
work. By incorporating anomaly information into the learning
process, the recommended approach can improve the detection
of unusual conditions that may contribute to flooding.
Machine Learning Modeling
This study estimates flood probability through supervised
machine learning, framing the task as a binary classification
of flood and non-flood events based on daily meteorological
observations. Three ensemble algorithms, namely Random
Forest, XGBoost, and CatBoost, were selected and compared
because they have repeatedly demonstrated strong and robust
performance in data-driven flood modelling [5, 4]. Random
Forest was adopted as the primary model owing to its
stability, resistance to overfitting, and ability to handle noisy
environmental data, while XGBoost and CatBoost were
included as competitive gradient-boosting baselines [21, 19].
The selection of these algorithms is motivated by their
complementary strengths in handling the complex and imbal-
anced nature of flood data. XGBoost is highly efficient and
includes a built-in mechanism for handling missing values,
which makes it robust when working with real-world spatial
data, whereas CatBoost reduces the risk of overfitting on lim-
ited samples through its ordered boosting scheme [5]. Random
Forest complements these models by aggregating multiple de-
cision trees to produce stable predictions, making it a reliable
benchmark for comparison across the three approaches [4].
To minimise human bias and improve reproducibility, the
training of the three algorithms was carried out within an Au-
tomated Machine Learning (AutoML) framework rather than
through manual configuration. AutoML automates feature
selection, model selection, and hyperparameter optimisation,
which reduces the subjective decisions that commonly affect
conventional flood modelling and allows a fairer comparison
between Random Forest, XGBoost, and CatBoost [4, 5]. This
approach follows recent studies that successfully applied Au-
toML to flood susceptibility and waterlogging prediction and
reported performance exceeding that of manually tuned base-
lines [? 17].
The preprocessed point dataset was divided into training
and testing subsets so that model performance could be
assessed on unseen data. Within the AutoML pipeline,
each algorithm was trained on the training subset and its
hyperparameters were automatically tuned to obtain the best
configuration for the flood classification task [4, 6]. The
trained models then produce a flood probability value for
each location, which can be aggregated into a daily flood
occurrence prediction for Balikpapan and Samarinda [19].
Model Evaluation
The performance of the trained models was evaluated to de-
termine how accurately they distinguish flood and non-flood
events based on daily meteorological conditions. Evaluation
was carried out on the testing subset using standard classi-
fication metrics, namely accuracy, precision, recall, and the
F1-score, which together provide a balanced view of model
performance on imbalanced flood data [17, 4]. These metrics
were chosen because relying on accuracy alone can be mis-
leading when the classes are uneven, so precision and recall
Table 1: Example of table format
head1 head2 head3
content1 content2 content3
content1 content2 content3
are required to capture how well the model identifies actual
flood events. Given the severe class imbalance, where flood
days represent less than one percent of all observations, par-
ticular emphasis was placed on recall and the area under the
precision–recall curve (AUC-PR), as these better reflect the
model’s ability to detect rare flood events than accuracy or
AUC-ROC alone. Class imbalance was handled only on the
training subset, after the chronological split, using SMOTE or
class weighting to avoid data leakage.
In addition to the threshold-based metrics, the area under the
receiver operating characteristic curve (AUC-ROC) was used
as a complementary indicator of overall discriminative ability.
The AUC-ROC summarises the trade-off between the true
positive rate and the false positive rate across all classification
thresholds, and it is widely regarded as a reliable measure for
comparing flood susceptibility models [4, 17]. The model
achieving the highest AUC-PR together with consistently
strong recall, precision, and F1-scores was selected as the
best model for estimating flood probability in the study area
[19, 5].