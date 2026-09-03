# Paikkadata — a first data-collection exercise for an intro ML course

A single web page that lets a student collect their own machine learning training data with a
phone, in about 20 minutes, without installing anything. The page has two modes, and the choice
between them *is* the lesson:

- **Classification / Luokittelu** — the target is a name
- **Regression / Regressio** — the target is a number

Everything else about the two datasets is deliberately identical, so the difference between the
two kinds of supervised learning is the only thing left to notice.

The interface is bilingual: a **FI / EN** switch in the top right changes every label on the page,
and the choice is remembered. Column names stay in English in both languages, so the same analysis
code works for the whole group.

**Live page:** [https://datamikko.github.io/locationdata/](https://datamikko.github.io/locationdata/)

## Why this exercise

Beginners meet linear regression and decision trees before they have ever seen a dataset that
isn't already cleaned, labelled and split for them. This page makes them produce one. Because they
choose the places, walk the walk and watch the accuracy reading wobble, the noise in the data is
something they can explain rather than something that just happens to be there.

The page uses the vocabulary explicitly and marks every column with its role while the data is
being collected: the last column is always the target, the rest are features.

| English | Finnish |
|---|---|
| feature | piirre |
| target | kohde |
| class / label | luokka |
| observation / row | havainto |
| training data | opetusdata |
| test data | testidata |

## What the student collects

**Classification mode.** Pick 3–4 spots 20–60 m apart — a mailbox, a bench, some steps, a bus
stop. Name each one, stand still and record for ~20 seconds. GPS scatter means each place becomes
a cloud of points rather than a dot, which is exactly what makes the task non-trivial.

```
x_m,  y_m,  accuracy_m, altitude_m, label
34.2, 18.7, 9.0,        112.4,      bench
```

A decision tree on `x_m` and `y_m` can only split on one axis at a time, so the model it learns is
a grid of named rectangles — a picture the student can compare against their own mental map of the
neighbourhood.

**Regression mode.** Stand at a starting point, start recording, walk straight away from it for
about three minutes.

```
time_s, accuracy_m, altitude_m, distance_m
120.0,  7.4,        113.0,      158.3
```

A linear regression on `time_s` gives a slope in metres per second — the student's own walking
speed, a coefficient they can check against reality instead of taking on faith.

Both datasets download as their own CSV: `classification.csv` and `regression.csv`.

### About `altitude_m`

Altitude is included because it costs nothing to collect and is easy to drop. It is also the most
honest column in the file: many phones report no altitude at all, so the cell is simply empty, and
GPS altitude is far less accurate than horizontal position even when it is reported. That gives a
class three things to talk about that a tidy teaching dataset never does — missing values, a
feature that carries almost no signal, and the difference between "collected" and "useful". If it
just gets in the way, leave the column out of `X` and move on.

## Analysing the data

The CSVs are already in the shape scikit-learn expects: features first, target last. In Colab or
any notebook:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# --- classification ---
d = pd.read_csv("classification.csv")
X, y = d[["x_m", "y_m"]], d["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
print(tree.score(X_tr, y_tr), tree.score(X_te, y_te))

# --- regression ---
d = pd.read_csv("regression.csv")
X, y = d[["time_s"]], d["distance_m"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

line = LinearRegression().fit(X_tr, y_tr)
print(f"{line.coef_[0]:.2f} m/s", line.score(X_te, y_te))
```

Good follow-up questions: how does `max_depth` change the gap between training and test accuracy;
is the fitted walking speed believable; what does the regression intercept mean; does adding
`accuracy_m` or `altitude_m` to `X` help at all; why can't the place name be predicted with a
regression.

## Running it

The page is one self-contained file. Serve `index.html` over HTTPS — GitHub Pages is enough.
**Geolocation requires a secure context**, so a file opened straight from phone storage will never
get a fix. Test it outdoors on a phone before handing the link to a group.

Works in mobile Chrome and Safari. Rows are kept in `localStorage`, so a reload or an accidental
back-swipe doesn't lose the session, and the page asks for a screen wake lock while recording
because browsers stop the GPS when the phone locks.

## Accessibility fallback

**Add demo data / Lisää demodataa** generates a realistic dataset in the same schema for the mode
currently selected. Students who can't go outside — weather, mobility, safety, darkness — can still
complete the assignment, and the page flags demo rows so a teacher can see which submissions used
it.

## Privacy

Nothing is sent anywhere; all data stays in the browser. Exports contain metres and seconds
relative to the student's own starting point, never latitude/longitude, and the starting point
itself is never written to the file — so a submitted CSV cannot be traced back to where the student
lives. Altitude is the one absolute value in the file; it is metres above sea level, which narrows
a location down about as much as knowing the town does. Worth telling students anyway: name places
with ordinary words rather than addresses, and don't start the walk at your own front door.

## Files

| File | Purpose |
|---|---|
| `index.html` | the whole tool — no build step, no dependencies beyond Google Fonts |
| `README.md` | this file |
