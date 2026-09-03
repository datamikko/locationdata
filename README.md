# Paikkadata — a first data-collection exercise for an intro ML course

A single web page that lets a student collect their own machine learning training data with a
phone, in about 20 minutes, without installing anything. The page has two modes, and the choice
between them *is* the lesson:

- **Luokittelu** (classification) — the target is a name
- **Regressio** (regression) — the target is a number

Everything else about the two datasets is deliberately identical, so the difference between the
two kinds of supervised learning is the only thing left to notice.

**Live page:** https://datamikko.github.io/wordmap/

## Why this exercise

Beginners meet linear regression and decision trees before they have ever seen a dataset that
isn't already cleaned, labelled and split for them. This page makes them produce one. Because
they choose the places, walk the walk and watch the accuracy reading wobble, the noise in the data
is something they can explain rather than something that just happens to be there.

The page uses the vocabulary explicitly, in Finnish and English, and marks every column with its
role while the data is being collected: the last column is always the target, the rest are
features.

| English | Finnish (used in the UI) |
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
x_m, y_m, tarkkuus_m, luokka
34.2, 18.7, 9.0,      penkki
```

A decision tree on `x_m` and `y_m` can only split on one axis at a time, so the model it learns is
a grid of named rectangles — a picture the student can compare against their own mental map of the
neighbourhood.

**Regression mode.** Stand at a starting point, start recording, walk straight away from it for
about three minutes.

```
aika_s, tarkkuus_m, etaisyys_m
120.0,  7.4,        158.3
```

A linear regression on `aika_s` gives a slope in metres per second — the student's own walking
speed, a coefficient they can check against reality instead of taking on faith.

Both datasets download as their own CSV: `luokittelu.csv` and `regressio.csv`.

## Analysing the data

The CSVs are already in the shape scikit-learn expects: features first, target last. In Colab or
any notebook:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# --- luokittelu ---
d = pd.read_csv("luokittelu.csv")
X, y = d[["x_m", "y_m"]], d["luokka"]
X_op, X_te, y_op, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

puu = DecisionTreeClassifier(max_depth=3).fit(X_op, y_op)
print(puu.score(X_op, y_op), puu.score(X_te, y_te))

# --- regressio ---
d = pd.read_csv("regressio.csv")
X, y = d[["aika_s"]], d["etaisyys_m"]
X_op, X_te, y_op, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

suora = LinearRegression().fit(X_op, y_op)
print(f"{suora.coef_[0]:.2f} m/s", suora.score(X_te, y_te))
```

Good follow-up questions: how does `max_depth` change the gap between training and test accuracy;
is the fitted walking speed believable; what does the regression intercept mean; why can't the
place name be predicted with a regression.

## Running it

The page is one self-contained file. Serve `index.html` over HTTPS — GitHub Pages is enough.
**Geolocation requires a secure context**, so a file opened straight from phone storage will never
get a fix. Test it outdoors on a phone before handing the link to a group.

Works in mobile Chrome and Safari. Rows are kept in `localStorage`, so a reload or an accidental
back-swipe doesn't lose the session, and the page asks for a screen wake lock while recording
because browsers stop the GPS when the phone locks.

## Accessibility fallback

**Lisää demodataa** generates a realistic dataset in the same schema for the mode currently
selected. Students who can't go outside — weather, mobility, safety, darkness — can still complete
the assignment, and the page flags demo rows so a teacher can see which submissions used it.

## Privacy

Nothing is sent anywhere; all data stays in the browser. Exports contain metres and seconds
relative to the student's own starting point, never latitude/longitude, and the starting point
itself is never written to the file — so a submitted CSV cannot be traced back to where the
student lives. Worth telling students anyway: name places with ordinary words rather than
addresses, and don't start the walk at your own front door.

## Files

| File | Purpose |
|---|---|
| `index.html` | the whole tool — no build step, no dependencies beyond Google Fonts |
| `README.md` | this file |
