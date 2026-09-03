# Sanakartta — a what3words-inspired GPS data collection exercise

A ready-to-run assignment for the first weeks of an introductory AI/ML course, built for
online students working alone. Students collect their own GPS data with a phone, and the same
30-minute walk yields **both a classification dataset and a regression dataset** — enough to
train the two models beginners usually meet first: a decision tree and a linear regression.

No installation, no app store, no API keys. The collector is a single HTML page.

## The idea

[what3words](https://what3words.com) divides the world into 3 × 3 m squares and gives each one a
three-word name. The grid is decided in advance. This exercise turns that around: students name a
handful of real places near their home, collect GPS points at each one, and let a **decision tree
work out where one named area ends and the next begins**.

Because a decision tree splitting on two coordinates can only draw axis-aligned rectangles, the
trained model *is* a grid of named cells — a what3words grid, but learned from data. That makes it
one of the clearest visual explanations of how a decision tree actually works.

Meanwhile the walks between places give a regression target: predict how long a walk takes from how
far apart the endpoints are. The fitted slope is seconds per metre, and its reciprocal is the
student's own walking speed — a coefficient they can sanity-check against reality.

## Quick start

**For teachers.** Host `sanakartta.html` on any HTTPS URL — a course platform, GitHub Pages, or a
published Claude Artifact. Geolocation requires a secure context, so a file opened straight from the
phone's storage will *not* get a location fix. Test the page outdoors on a phone before handing out
the link. Then share the collector link, `tehtava1_datankeruu.md`, and later the notebook.

**For students.** Open the link outdoors, wait for the accuracy reading to drop below 20 m, record
five named places and the walks between them, then export the CSV.

**Running the notebook locally** instead of in Colab:

```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook sanakartta_analyysi.ipynb
```

The notebook falls back to `esimerkkidata_sanakartta.csv` when the Colab upload widget is
unavailable, so it runs end to end without any collected data.

## Data format

One row per GPS sample. Coordinates are metres relative to the session's first fix — the origin
itself is never written to the file.

| Column | Meaning |
|---|---|
| `aikaleima` | local timestamp, ISO 8601 |
| `t_s` | seconds since the session started |
| `tila` | `paikka` = standing at a named place, `matka` = walking |
| `sarja` | recording-run id; each start/stop pair gets its own number |
| `paikka` | the three-word name (empty on walking rows) |
| `ymparisto` | `ulko` / `sisa` — outdoors or indoors |
| `x_m`, `y_m` | east and north offset in metres from the session origin |
| `tarkkuus_m` | the browser's own horizontal accuracy estimate |
| `korkeus_m`, `nopeus_ms` | altitude and speed, when the device reports them |
| `kuljettu_m` | cumulative distance within the current `sarja` |
| `lahde` | `gps` or `demo` — flags simulated data |
| `lat`, `lon` | *optional*, only when the privacy setting is switched off |

Classification uses the `paikka` rows; regression uses the `matka` rows, from which the notebook
builds point pairs (straight-line distance → elapsed time).

## The collector page

- Works in mobile Chrome and Safari; nothing to install.
- Names places for you with a Finnish three-word generator.
- Keeps points in `localStorage`, so a reload or an accidental back-swipe doesn't lose the walk.
- Requests a screen wake lock while recording (browsers stop the GPS when the phone locks).
- Exports CSV by download, by clipboard, or through the Artifact `downloads` capability when
  published as an Artifact.
- **Demo mode** generates a realistic dataset in the same schema, so students who can't go outside —
  weather, mobility, safety, darkness — can still complete the assignment.
- Sends nothing anywhere. All data stays in the browser until the student clears it.

## Expected results

With the example dataset (6 places, ~10 m GPS scatter, 25–60 m apart):

- Decision tree, depth 3: 0.98 train / 0.89 test accuracy
- Full-depth tree: 1.00 train / 0.91 test — visible overfitting, as intended
- Indoor/outdoor from accuracy alone, depth 1: 0.94, threshold ≈ 25 m
- Regression: `duration_s = 0.83 · distance_m + 0.2`, R² ≈ 0.81, implied pace 1.2 m/s

Real student data varies a lot more, which is the point.

## Privacy

Exports contain metres relative to the session origin, never latitude/longitude, and the origin is
not stored — so a submitted file can't be traced back to where the student lives. The instructions
tell students to name places with words rather than addresses and not to start at their front door.
Keep the privacy toggle on; if a submission contains `lat`/`lon` columns, ask for a fresh export.

## Adapting it

The material is course-agnostic apart from the language. Obvious variants:

- **Pooled dataset.** Collect submitted CSVs, add a student column, and ask a follow-up question
  about differences in walking pace between people.
- **Travel mode classification.** Have students record walking, cycling and a bus ride separately,
  then classify on `nopeus_ms` and `tarkkuus_m`. Considerably harder than place classification.
- **k-NN comparison.** Plot a `KNeighborsClassifier` on the same map; the contrast with the tree's
  rectangles makes the inductive bias of each model obvious.

Edit `tee_esimerkkidata.py` — place positions, GPS scatter, walking speed — to make the example
dataset easier or harder for the models.
