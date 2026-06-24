Abstract
The new Nusantara Capital City (IKN) and the areas surrounding Ba-
likpapan and Samarinda, known as buffer zones, are in serious danger
of suffering damaging flood events, which could stall IKN’s develop-
ment. Machine Learning has demonstrated its capacity to assist in
anticipating such events but conventional manual flood forecasting
relies on human expertise, is poorly equipped to handle the heavily
class imbalance in the historical weather data. Our team developed
an objective and automated model that predicts one to three days
ahead of urban floods in Balikpapan and Samarinda using weather
data and floodlogs collected at BNPB to achieve the necessary lev-
els of robustness and predictive ability. We combined Daily weather
parameters (open-Meteo.com) with archivedflood logs data (BNPB).
Framing the problem as time-series prediction with extracted sliding
windowsfeatures and computedanomaly scoresusing Isolation Forest.
After applying the SMOTE techniques to correct classimbalancein
training set, theAutoML methodology
Keywords: Automated Machine Learning (AutoML); Flood Fore-
casting and Prediction; Time-Series Classification; Anomaly Detec-
tion.
Introduction
Strategically positioned as the buffer zones to the Nusantara
Capital City (IKN), Balikpapan and Samarinda are significant
to bolster any mobility, logistical, and development efforts
required for the new capital city. With official approval
of the move of the national capital to East Kalimantan in
2019, both cities were designated as inseparable supporting
regions for IKN development [1]. However, at this critical
development stage, these cities have now been seriously
threatened by a multitude of severe risks to a disaster that
would possibly impact IKN’s development plans. Based
on the disaster risk profile for East Kalimantan Province,
floods were determined as among the most severe disaster
hazards and were classified as Sector IV level 6 in the
Interpretative Structural Modelling analysis; showing the risks
associated with the high levels of hazard and vulnerability
faced by the regions almost throughout the entire area,
including Balikpapan and Samarinda [2]. Compounding with
IKN construction, land cover changes driven by the ongoing
development also lead to increased risks. Hydrological
assessments conducted on the Pemaluan watershed located
within the IKN development zone also indicated increased
flood areas by 3.15% between 2019 and 2023 due to lack
of water permeability and increased surface runoff due to
the reduction of surface water capacity [3]. Therefore,
vulnerability to floods in Balikpapan and Samarinda poses not
only local risks, but also systemic risks that can thwart the
entire development plan of IKN.
Flooding in East Kalimantan which includes Balikpapan
and Samarinda as the two largest cities in East Kalimantan are
considered to be highly dynamic, fluctuating as a consequence
of changes in weather conditions and extreme rainfall .
Climatologically, the province averages 174.8 mm rainfall,
where it is also characterized by topical convective systems
along with an intensity of global climate change . Eventually,
high extreme rainfall is identified as the cause of flooding
within an area [2]. The study has a scientific backing
from a hydrological study of the Pemaluan watershed that
show a very significant correlation between daily rainfall data
between 2004 to 2023 and area of annual flood inundation.
This study reveals the relationship between rainfall and
expanded flood inundated area is increasing in proportional
to increase in Rainfall intensity [3]. Further to add, based on
recent study concerning the sustainability of IKN development
that flood control measures in cities like Samarinda and
Balikpapan were never completed and are predicted to get
worse due to widespread land conversion [1].
The use of traditional machine learning is limited for flood
prediction as it works with particular types of data. The
climate data that are generated at regional levels to develop
predictions in case of extreme events is particularly compli-
cated, has enormous dimensions, highly insufficient data of
Volume 1, Issue 1 2
Author et al.: Short version of title
the minority class and predominantly uses the dominant class,
which consequently creates significant difficulties in identi-
fying flood signals, which are critically important [4]. We
present our approach to with the integration of anomaly detec-
tion with unsupervised machine learning to auto-capture the
extremely entropic deviations in weather, and then applying
the SMOTE (Synthetic Minority Over-sampling Technique)
algorithm that balances the dataset in a synthetic manner with-
out data loss.
Also, traditional machine learning approaches are also
extremely rely on human input and manual manipulation like
selecting features, choosing a relevant algorithm, or tuning its
parameters. This not only becomes a rather time consuming
process, but also leaves room for human bias within the
model. In order to address human bias and model the highly
non-linear dependencies on real-world environments in an
efficient and automated way, we selected three popular tree-
based ensemble learning algorithms which Random Forest is
utilized because of its remarkable stability, tolerance to noisy
data, and prevention of over-fitting. XGBoost is preferred
for its speed, efficiency, and the incorporation of features for
handling missing data, which are often an issue in the real
world, and CatBoost for its inherent strengths in avoiding
over-fitting, especially for limited data [5, 6]. .
To remove human bias and render the research process
efficient, there is a fully-automated approach with Automated
Machine Learning. The full predictive pipeline (data prep,
feature extraction and model selection) is optimized using a
Fast and Lightweight Automated Machine Learning (FLAML)
system [7], to reduce or ideally fully eliminate human
intervention.
To bridge this gap, our research offers a data-driven
approach using disaster event logs from the National Disaster
Management Agency (BNPB) and weather data from Open-
Meteo to predict the daily possibility of flood events occurring
in Balikpapan and Samarinda between 1–3 days ahead. Our
approach focuses on the following objectives:
1. To construct a sequential time-series environmental
dataset by integrating Open-Meteo weather data with
BNPB disaster logs, enhanced by unsupervised anomaly
scores and balanced using SMOTE.
2. To compare the performance of three robust tree-
ensemble algorithms (Random Forest, XGBoost, and
CatBoost) tailored for predicting complex, imbalanced
flood events.
3. To automate the classification pipeline using the FLAML
framework, ensuring an objective and efficient modeling
process that eliminates subjective human bias.
Related Works
Parray and Owais Ahmad [8] investigate flood frequency and
magnitude predictions, demonstrating the superiority of Ma-
chine Learning method over conventional statistical models,
such as Gumbel Extreme Value Type I, and Log-Pearson Type
III (LP-III), especially for modeling non-linear variability and
interplay of hydrological factors. The results of compari-
son prove the high accuracy (R=0.9996) of data-driven mod-
els like Polynomial Regression with an extremely low er-
ror (RMSE)=860 m/s. In stark contrast, the performance of
classical statistics has fatal drawbacks like underestimation
of extreme flood discharge for high return periods and fail-
ing to reflect dynamics of climate change due to its assump-
tion of stationarity of the series. Kumar et al. [9] empha-
sized, in their survey on applications of AI for floods detec-
tion and management, that different ML algorithms including
Random Forest (RF), Support Vector Machine (SVM), Ar-
tificial Neural Networks (ANN) were efficient in extracting
complex patterns that trigger floods from huge hydrological
and weather information. These methods are efficient in the
early warning systems for real-time application, however ap-
plication ofML to flood Forecasting has significant disadvan-
tages like the black-box phenomenon of an algorithm (mak-
ing its decisionmaking difficult to interprete physicallly), its
inclination towards overfitting, and the large volume of histor-
ical data is required for its development. Meanwhile, Motta
et al. [10] developed a framework based on an ML classi-
fier combined with Geographical Information System (GIS)
for predicting urban floods to overcome the poor spatial rep-
resentation and poortheoretical understanding characteristic to
standalone ML algorithms. The study reveals RF (Acc=0.96)
as the best model, while the two-hour rainfall moving aver-
age is the most important predictive factor. The combined ML
and GIS Hot Spot analysis methodology resulted in a strongly
consistent predictive system successfully detecting 97 out of
116 flooding events and achieving a sensitivityof 0.84. The
cost of improved sensitivity however has to be paid with an in-
creased frequency of alarm (high false alarm rate), and weak
explanatory relationships between flood causes and effects.
Recent progresses in flood prediction are shifting towards
time-series methods, furthest towards the sliding window/lag
features to cater for the needs of real-time processing and
to track the long-term memory of water flow duration.
The advantages of the methodology above are demonstrated
by Atashi et al. [11] who proved that Long Short-Term
Memory (LSTM) consistently outperforms SARIMA and RF
by capturing non-linear dynamics of peak flow unaffected by
conventional models. To tackle the challenges of scarceness
of extremities data for deep learning prediction, Weng et al.
[12] leverage the time-series variant of Generative Adversarial
Networks (GAN) to generate synthetic data in order to
significantly enhance the accuracy of Gradient Boosting
Regression Tree (GBRT), while to a small degree, marginally
improving LSTM. Meanwhile, Li et al. [13] introduce a hybrid
alternative by combining outputs from a hydrological water-
influx model (MIKE 11) and historical observations to train
LSTM. Trained via sliding window on a optimal ratio of 2:1,
the hybrid framework minimizes the error values significantly
although heavily reliant on the quality of simulation data and
less adaptive to rugged topographical applications.
Data imbalanced (class imbalance) induces a fatal bias
to noise-sensitive extreme flood detection. The real-world
consequence of data imbalance was proven by Esparza et
al. [14], as they showed bias of crowdsourced reporting
to create "blind spots" in the spatial domain that rendered
community areas vulnerable to missing emergency mapping.
In order to correct the classifaction bias caused by the
imbalance ratio, Wang et al. [4] justified the hybrid framework
combining Borderline-SMOTE and K-Means undersampling
by asserting its significant improvement on Recall and F1-
3 This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY) Volume 1, Issue 1
Author et al.: Short version of title
scores with no scope to lose information. Confirming the
superiority of SMOTE techniques, Matharaarachchi et al. [15]
prove that Dirichlet ExtSMOTE was successful in removing
noise in minority classes; the weighted synthetic samples
extracted were proved to be far more representative and
remarkably enhanced the Random Forest classifier model
without damaging original data structure.
The selection of tree-based ensemble algorithms (Random
Forest, XGBoost and Cat Boost) has been undoubtedly
proved most effective to handle complex and imbalanced
tabular environmental data due to their solid mathematical
architecture. Empirical evidences of this reliability are
presented by Hsu et al. [5] in their extreme weather forecast
study where the RF has achieved a peak accuracy of 70%
with an AUC of 76%. This accuracy was achieved owing
to the successful implementation of the Bagging technique on
RF which helped reducing the model variance and classifying
non-linear data boundaries avoiding the model to lean towards
predicting the majority class. The same study also proposed
the Cat Boost algorithm as most innovative due to its
ability to work with categorical variables in a native manner
thus preserving data integrity without sparking the usual
infiltrations with data leakage caused by one-hot encoding. To
round-up the performances of these ensemble architectures,
Ma et al. [6] reinforced the superiority of XG Boost to
plot flood risk; this model has strongly outperformed the
classical LSSVM method with test accuracy of 0.84 and high
precision to identified 63,15% of high risk locations. The
aforementioned state-of the-art performance of XG Boost
that was made possible by the adoption of the regularization
term that penalized hard the model complexity to beat down
the overfitting to the dominant classes supported by the
column sampling property and the parallel computing oriented
parallelization made this algorithm extremely agile in tackling
missing values within the huge observation set.
Moving towards automation in flood modeling has been
driven by the efficiency of AutoML in overcoming the
human biases and intricacies of manual parameter tuning.
This efficiency is demonstrated by Guo et al. [16] who
use the TPOT framework to test thousands of pipeline
combinations; this method successfully achieved the most
highly correlated flood prediction (NSE and R >0.95) and
substantially outperformed manually tuned models. The
prowess of AutoML’s complex pipeline architecture is further
corroborated by Han et al. [17] on the stacking model set
of regressor models which has been shown to outperform
deep learning algorithms (LSTM) in predicting water level
accurately. In the realm of risk mapping, Gao et al. [18] utilize
the Auto-Sklearn framework to objectively and autonomously
test various models to select CatBoost as the model of highest
accuracy (0.9030) suggesting that pure data-driven model
selection approaches for optimal models are extremely robust.
To overcome the slow run-time issue of conventional AutoML,
Chu et al. [7] have demonstrated the efficiency of the Fast and
Lightweight AutoML (FLAML) framework; this resource-
efficient searching approach has been successful in optimizing
a very high accuracy XGBoost model (R= 0.963) capable of
executing flood predictions 2000 times faster than the physical
hydrodynamic model. Such an application of FLAML is
significant not only in having autonomously identified a better
predictor ensemble architecture but also in delivering the
computational agility was essential for implementation of
early warning systems in real time.
In environmental anomaly detection, the unsupervised al-
gorithms such as the Isolation Forest is considered far more
rational than the supervised classifier since its capability to
operate autonomously on the manually-unlabeled raw obser-
vations in real-world setting with a view to detect extremely
sparse anomalies. This efficacy was empirically proven by
Agmeyang [19] evidenced that Isolation Forest relatively can
reach the highest state-of-the-art performance (F1 score of
64.41% and a Recall of 95%) at isolating the outliers, amortiz-
ing the problem of poor Recall on models like OCSVM. The
operational reliability of this algorithm in real-life field set-
ting was also corroborated by Shi et al. [20] to monitor the
soil water microdynamics; to be specific, the Isolation Forest
model was evidenced to precisely localize the anomalous hy-
drological cluster coincident with the actual landslide move-
ment event in Chengdu. On the technical perspective, the ad-
vantage of Isolation Forest can be justified from the fact that it
does not attempt to build the profile of the majority class, us-
ing the random binary partitioning trees to isolate anomalies,
where this cut mechanism frees the computationally-expensive
calculation of distance, rendering Isolation Forest very nimble,
computationally efficient, and suitable for high-dimensional
environmental sensor data streams processing.
Methods
This section provides a systematic, data-based approach
utilized to compute daily flood likelihood in Balikpapan and
Samarinda. In an effort to improve reproducibility and uphold
the integrity of the experiment, the method is designed around
a rigid chronological framework.
Data Collection and Time-Series Transformation
Our methods of data collection involved combining two
main open source data streams based on temporal indexes
in order to record the context of the environment per day.
Records of historical flood events were collected from the
registry of the Data and Information on Disasters in Indonesia
(DIBI) database managed by the Indonesian National Disaster
Management Authority (BNPB). To classify weather triggers,
daily meteorological data between the years of 2016 and 2026
were downloaded through the Open-Meteo API.
Features downloaded included precipitation (i.e. Rainfall),
maximum/minimum temperature, wind speed, and soil mois-
ture parameters (i.e. Soil percentiles, rolling soil mean).
Raw daily data was aggregated over running windows
to model the aggregational effects of hydrological cycles.
Additionally, explicit time-lagged covariates for temperature,
rainfall and soil outputs were created from day to 7, and 3, 7
and 14-day rolling sums. The target phenomenon was set three
days ahead (F loodt+3) so as to create a time-series predictive
classification task.
To avoid a disastrous temporal data leakage with future
records leading to the unnoticed prediction, we explicitly do
not shuffle data randomly. Instead, the formatted sequence was
bifurcated by a strict chronological time cut (TimeSeriesSplit),
reserving the initial 70% of the historic order for training, with
the remaining 30% held out as a entirely unseen test set to
emulate the realisic operation of forecasting.
Volume 1, Issue 1 4
Author et al.: Short version of title
Additionally, cyclical calendar features were included, such
that the seasonality of the sequence was preserved (month_sin,
month_cos, doy_sin, and doy_cos).
Unsupervised Anomaly Detection
In environmental anomaly detection, unsupervised algorithms
would make much more reasonable choices than supervised
classifiers since they could just automatically work on the
manually-untabled raw observations for extremely sparse
anomalies detection [19], [20]. It had been shown in earlier
researches that the isolation forest algorithm would be more
capable of isolating the anomalies by for instance not giving
a profile of the big class and through the use of randomized
binary partitioning system hence eliminatethe toil of costly
distance computation processes [19].
Based on this assumption, days of high magnitude variation
were predicted using this unsupervised anomaly detection
approach, before running the main classifiers. An Isolation
Forest method was added across all the earlier performed
environmental and soil features. The days of extreme
anomalies in environmental conditions bear both a low
probability from a statistical perspective, and a high intrinsic
discrepancy in structure compared to normal observations.
Thus, the algorithm proceeds to repeatedly perform very
many random binary splits to find highly entropy anomalies
in the dataset without regard to the underlying distribution of
the dataset. The anomalous deviation score is given as a value
on a 0 to 1 spectrum for each day observed. The score is an
input into the supervised learning algorithm to compensate for
the high shiftness of atmospheric conditions within each day
observed.
Results and Discussion
Time-Series Feature Extraction and Anomaly Detection
Both the raw data from Open-Meteo and BNPB have been
reformatted via sliding window. This process makes sure that
we handle weather events on right timing. Features from lag
variables, and rolling sums over 3 days, 7 days, 14 days of
accumulated rainfall help us to estimate level of humidity on
ground right before that day. For initial unsupervised check,
we apply Isolation Forest algorithm where each entry of each
day gets an anomaly score. If you review the data exploration
section you see that the anomaly scores readily divided the
two classes. The flooded days (Class 1) received far higher
values than the healthy day (Class 0). This trend suggests the
hypothesis is reasonable that the cities generally experienced
extreme shifts before a flooding. Given the evidence of strong
positive correlation we used the anomaly score as one of the
initial inputs for the supervised models that follows below.
Figure 1: Distribution of the anomaly scores by class
Class Imbalance Handling
A big problem we encountered when processing our data
was that class was severely unbalanced. In both Balikpapan
and Samarinda, we get many more normal days compared to
floods. If we had not taken action, any models we developed
would have just predicted ‘no flood’ all the time. We solved
this with the use of SMOTE: the Synthetic Minority Over-
sampling Technique.
We applied SMOTE only to training data (X_train) follow-
ing TimeSeriesSplit (Table 1). Test set was left undisturbed
entirely. Without touch to test data we prevent any sort of time
leak and we demonstrate what is likely in real-world scenarios
Table 1: Class Distribution Before and After SMOTE on Training Data
Dataset Phase Non-Flood Days (Class 0) Flood Days (Class 1)
Before SMOTE 18380 1906
After SMOTE 18380 18380
AutoML Pipeline Exploration and SOTA Model Compari-
son
Following this adjustment to the trainset data, we compared
three widely-used tree-based methods, namely Random For-
est, XGBoost and CatBoost to the FLAML machine learning
platform. We allocated the AutoML process a time budget of
120 seconds and the Average Precision (AUC-PR) score as a
key indicator since it captures imbalanced distributions.
Table 2 Visualizes the performance of all classifiers on the
hidden test set. The analysis performed by FLAML showed
that CatBoost with 0.1 learning_rate and 8192 n_estimators
was the optimal set-up for boosting models. With a boost
inperformance in all the boosting types when comparing AUC-
PR scores against the traditional Random Forest, it can be
argued that boosting approaches perform significantly better
at the complex non-linear data relationship in weather forecast
datasets.
Table 2: Performance Comparison of State-of-the-Art Classifiers
Model Accuracy Precision Recall F1-Score AUC-ROC AUC-PR
AutoML (CatBoost) 0.9727 0.0000 0.0000 0.0000 0.4545 0.0410
CatBoost 0.9735 0.0000 0.0000 0.0000 0.4230 0.0453
Random Forest 0.9618 0.1364 0.0909 0.1091 0.4997 0.0637
XGBoost 0.9548 0.0690 0.0606 0.0645 0.4233 0.0255
Isolation Forest 0.8879 0.0336 0.1212 0.0526 0.5006 0.0721
5 This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY) Volume 1, Issue 1
Author et al.: Short version of title
Figure 2: Receiver Operating Characteristic (ROC) and Precision-Recall
Curves
Figure 3: Confusion Matrix at the Calibrated Operational Threshold
Operational Threshold Calibration and Limitations
By setting the threshold to a low value, at 0.333, we had to
force the model to search for flooding events, that could also
be observed by means of Fig. 3. Yet, even with this setting,
the model was unable to find floods: out of the 33 flood events
present in the test set, the model identified 2 (True Positives);
failed to identify 31 of them (False Negatives); identified as
floods 8 out of the 1251 no-event days (False Positives).
Since the model missed many occurrences the final recall
metric was drastically low which points to a huge problem
in the model used - we cannot reliably predict floods in
the city up to a few days in advance based solely on
meteorological data. Balikpapan and Samarinda have local
problems, drainage issues, the form of the land itself, tides
(rob) for example, which significantly affect flood occurrences
and these features were nowhere captured by the Open-Meteo
dataset. Adjusting the decision threshold in an EWS may
sound good at first, in hindsight these results tell us such a
tool cannot work if it doesn’t incorporate any hydrological or
geographical information.
Feature Importance and Interpretability
We decided to actually examine precisely what decisions were
the drivers behind each of these algorithms and averaged the
feature importance scores calculated for each model (Random
Forest, XGBoost, and CatBoost). Fig. 4, once again, reveals
that the actual feature drivers were entirely unlike what we
might have intuitively assumed; we expected that accumulated
rainfall and our anomaly score would be primary factors but
the algorithms favoured basic calendar features month_sin and
month_cos.
The next most important calendar-features were also lagged
temp values like tmin_lag6, tmin_lag1. Interestingly, none
of the rainfall feature measurements appear to even be in the
top list of important features. So, the tree based model was
essentially learned to predict a flood based on what season it is
not when it’s a crazy downpour. Knowing that it essentially
uses what season it is to predict a flood that means why
there were so many FN, or False Negative from the confusion
matrix, feeding the system weather information alone made
it more or less a seasonal calendar as oppose to a real time
warning system.
Figure 4: Top 20 Feature Importances Averaged Across Ensembles
Conclusion
We initially embarked on the construction of an automated
machine learning pipeline in order to predict flood in Balik-
papan and Samarinda 1 to 3 days ahead. We achieved this
by obtaining daily open-source weather forecasts and apply-
ing rolling totals to estimate cumulative rainfall and integrat-
ing unsupervised anomaly scores in order to identify extreme
weather events. However, in a final evaluation, it was revealed
that relying purely on meteorological information simply fails
to deliver reliable forecasts for urban flood in these locales in
short time horizons.
Even after equilibrating the dataset utilizing SMOTE, along
with dropping the decision threshold down to 0.333 so as to
oblige the model to consider detecting a flood, its performance
was still inadequate. Our most efficient CatBoost model
missed 31 of 33 positive cases of urban flood in the data test.
In analyzing the process whereby our model made its decision,
we discerned the key problem - The weather conditions or
a weather anomaly were almost irrelevant when explaining
flood, except calendar based features that represent month and
year . Ultimately, our model did not grasp to predict flood on
based on meteorological events that have been triggered, but
just predicted which season is flood, Which are summer and
Rainy season.
It turn out that open-source weather API failed to incorpo-
rate enough relevant factor. Flood in city of Balikpapan and
Samarinda is extremely local issue which heavily influence by
the poor of drainage networks, terrain slope, and ocean tides.
Thus, in terms of developing EWS in the future, researchers
must extend to include information from more local sources in
order to grasp real reason of flood event that can assist with
predicting more precisely