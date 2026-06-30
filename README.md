# Patient-Adaptive Information Acquisition for Prehospital Mortality Prediction

This repository implements a reinforcement-learning framework that determines
which additional prehospital clinical information should be acquired for each
patient and when information acquisition should stop.

The framework begins with a partially observed state derived from an emergency
call. A fixed mortality-prediction model estimates the patient's mortality
risk, while a Maskable Proximal Policy Optimization (Maskable PPO) agent
selects either an unobserved clinical feature group or an explicit `STOP`
action.

The objective is to retain the predictive performance of the fixed mortality
model while minimizing unnecessary clinical information acquisition.

## Framework

### Patient-specific information acquisition

The policy does not follow a fixed question order. At each step, it evaluates
the information currently available for an individual patient and decides:

- which missing clinical feature group should be acquired next; or
- whether the current information is sufficient and acquisition should stop.

The mortality predictor remains frozen during reinforcement-learning training.
Only the information supplied to the predictor changes.

### State

The reinforcement-learning state contains:

- the currently available model-input values; and
- a binary observation mask indicating which inputs are currently known.

The state is order-independent. Acquisition paths that result in the same
observed values and feature mask are represented as the same state.

### Action space

The action space contains 22 actions:

- 21 clinical feature-group acquisition actions; and
- one `STOP` action.

Selecting a feature-group action reveals the corresponding values from the
linked all-feature record. Selecting `STOP` ends the episode using the
currently available information.

### Maskable PPO

The valid action set changes across patients and acquisition steps. A feature
group cannot be selected when it:

- was already available in the initial call-derived state;
- has already been acquired; or
- is not eligible for acquisition under the defined missingness rules.

Maskable PPO removes these invalid actions from the policy distribution. The
agent therefore chooses only among the currently available acquisition actions
and `STOP`.

### Reward

An acquisition is rewarded when the newly obtained information improves the
prediction produced by the frozen mortality model. Each acquisition also
incurs a question cost, discouraging unnecessary information collection.

The policy consequently learns a patient-specific balance between predictive
utility and information-acquisition burden.

## Trauma data preprocessing

The preprocessing pipeline converts structured prehospital trauma data into
the model-ready representation required by the frozen mortality predictor.

### Main preprocessing steps

- Retain the record key, visit date, and mortality outcome.
- Convert missing or blank raw values to predefined missing-data categories.
- Normalize age from its encoded age group.
- Convert gender into a binary male indicator.
- One-hot encode categorical clinical variables.
- Discretize six vital signs into clinically meaningful categories.
- Generate a summary indicator for uncheckable vital signs.

### Categorical variables

The following variables are one-hot encoded:

- injury time;
- intentionality;
- injury mechanism;
- injury type;
- job-related injury;
- accident location;
- hospital visit route;
- transport mode;
- insurance type;
- prehospital cardiac arrest;
- prehospital notification;
- AVPU; and
- protective-equipment use.

Each variable includes its predefined categories, such as unknown or missing,
when applicable.

Example encoded features include:

```text
Injury time - day
Injury time - night
Intentionality - accident
Intentionality - suicide
Intentionality - assult
injury_blunt
injury_penetrating
AVPU - alert
AVPU - coma
```

### Vital-sign categorization

Six prehospital vital signs are discretized and one-hot encoded.

| Vital sign | Encoded categories |
|---|---|
| Systolic blood pressure | unchecked, uncheckable, 0, 1–49, 50–75, 76–89, over 89 mmHg |
| Diastolic blood pressure | unchecked, uncheckable, 0, 1–29, 30–45, 46–59, over 59 mmHg |
| Pulse rate | unchecked, uncheckable, 0, 1–29, 30–59, 60–100, 101–119, over 119 |
| Respiratory rate | unchecked, uncheckable, 0, 1–5, 6–9, 10–29, over 29 |
| Body temperature | unchecked, uncheckable, 0, 0–24, 24–28, 28–32, 32–35, 35–37.8, over 37.8 °C |
| Oxygen saturation | unchecked, uncheckable, 0, 1–80, 81–90, 91–95, over 95% |

Each category is represented as a binary model input. The preprocessing script
also creates `total_uncheckable`, indicating whether any vital sign was
recorded as uncheckable.

## Clinical feature groups

The preprocessed model inputs are organized into the following 21 clinical
feature groups:

1. Age
2. Gender
3. Intentionality
4. Injury mechanism
5. Injury type
6. Injury time
7. Protective equipment
8. Job-related injury
9. AVPU
10. Systolic blood pressure
11. Diastolic blood pressure
12. Pulse rate
13. Respiratory rate
14. Body temperature
15. Oxygen saturation
16. Accident location
17. Hospital visit route
18. Transport mode
19. Insurance type
20. Prehospital cardiac arrest
21. Prehospital notification

Acquisition is performed at the clinical feature-group level. When the agent
selects a group, all preprocessed model inputs belonging to that group are
revealed together.

## Training workflow

The training pipeline performs the following operations:

1. Load the all-feature dataset, call-derived dataset, and frozen mortality
   predictor.
2. Align call-derived and all-feature records using the patient key and visit
   date.
3. Convert raw clinical variables into model-ready features.
4. Map the preprocessed inputs to the 21 clinical feature groups.
5. Determine which groups are initially observed or missing.
6. Construct the initial partial-observation values and masks.
7. Split the internal cohort into training, validation, and test sets.
8. Train the Maskable PPO information-acquisition policy.
9. Evaluate validation weighted log loss during training.
10. Save the best-performing and final policies.

## Source file structure

### `src/train.py`

Main entry point for reinforcement-learning training.

It:

- loads the private clinical datasets and frozen predictor;
- aligns call-derived records with all-feature records;
- runs preprocessing;
- constructs feature groups and observation masks;
- creates stratified training, validation, and test splits;
- initializes the training and validation environments;
- trains the Maskable PPO policy;
- evaluates the policy during training; and
- saves the best and final policy checkpoints.

### `src/config.py`

Defines the project paths and reinforcement-learning configuration.

It contains:

- input data and frozen-model paths;
- output policy paths;
- maximum and minimum question counts;
- acquisition cost and reward settings;
- random seed;
- total training timesteps;
- validation and early-stopping settings; and
- PPO hyperparameters.

### `src/preprocess.py`

Converts raw trauma-registry variables into the model-ready feature matrix used
by the frozen mortality predictor.

Its main functions:

- apply predefined missing-data codes;
- normalize the encoded age variable;
- encode gender;
- one-hot encode categorical clinical variables;
- discretize and encode vital signs;
- create the `total_uncheckable` feature; and
- separate the mortality outcome from the predictor inputs.

### `src/call_align.py`

Aligns the call-derived dataset with the all-feature dataset.

Records are matched using:

- the patient key; and
- the visit date.

The script returns aligned dataframes, the record match rate, and indicators
for cases in which age or gender must be supplied as required initial
information.

### `src/group_map.py`

Maps the preprocessed model-input columns to the 21 clinical feature groups.

For example:

- all `Sbp_value_*` inputs are assigned to the `SBP` group;
- all `AVPU - *` inputs are assigned to the `AVPU` group; and
- all `Injury mechanism - *` inputs are assigned to the
  `Injury mechanism` group.

This mapping allows the reinforcement-learning policy to acquire an entire
clinical concept rather than an individual one-hot column.

### `src/missing_rules.py`

Defines whether each clinical feature group is considered observed or missing
in the call-derived data.

It specifies:

- the raw columns belonging to each feature group;
- which feature groups are vital signs; and
- the category codes representing missing or unresolved information.

These rules determine which acquisition actions are initially available to the
agent.

### `src/mask_utils.py`

Constructs the initial model input and observation mask from the aligned
call-derived data.

It:

- applies the predefined values used for unavailable vital signs;
- preprocesses call-derived values into the same feature space as the
  all-feature data;
- creates the initial partially observed input; and
- creates a binary mask indicating which model inputs are initially known.

### `src/oracle.py`

Loads and runs the frozen ensemble mortality predictor.

The model file is expected to contain a list or tuple of fitted classifiers.
For each patient, the script:

- obtains the positive-class probability from every classifier; and
- averages the probabilities to produce the ensemble mortality prediction.

The predictor is not updated during reinforcement-learning training.

### `src/env_feature_acq.py`

Implements the Gymnasium environment for patient-specific clinical information
acquisition.

For each patient episode, the environment:

- initializes the call-derived partial-observation state;
- provides the currently observed values and mask to the policy;
- identifies valid feature-acquisition actions;
- reveals the selected feature group from the all-feature record;
- obtains an updated prediction from the frozen mortality model;
- calculates the acquisition reward;
- applies the cost of each question; and
- terminates when `STOP` is selected or the acquisition limit is reached.

### `src/callbacks.py`

Evaluates the learned policy on the validation set during training.

- saves the policy when validation performance improves; and
- stops training after the configured number of evaluations without
  improvement.

## Research-use notice

This repository contains retrospective research code. It is not a medical
device and must not be used for clinical diagnosis, triage, or treatment
decisions.
