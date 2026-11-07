# Music recommendation system

Cosine nearest-neighbor artist recommendations from the Last.fm 360K play-count
dataset. The original script was written by Bhavesh Wadibhasme on 13 December 2018
and is now `music_recommendation.py`. Running that file loads the listening
tables, keeps the first 40,000 play rows, keeps listeners whose country is
`United States`, drops duplicate user-artist rows, and prints artists nearest
to a query artist.

This repository also contains `archive/Rupesh_work.zip`. That archive is a
separate loan-amount exercise. Its notebooks are stored unchanged and are not
imported by the recommender.

No end-to-end run on the Last.fm files was done in this cleanup. The files are
not in the repository. `--demo` uses a few synthetic rows so the functions can
be checked; those rows are not listening data and the printout is not a
quality or performance result.

## Setup

Python 3.12 is the version used by CI and by the local test run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements.txt` is enough to run the recommender. `requirements-dev.txt`
adds pytest and ruff. CI installs those pins: numpy 2.5.4, pandas 3.1.1,
scipy 1.18.1, scikit-learn 1.9.1, pytest 9.2.0, ruff 0.17.4. The same tests
also passed with the Ubuntu 24.04 packages numpy 1.26.4, pandas 2.1.4,
scipy 1.11.4, and scikit-learn 1.4.1.post1.

## Run

Synthetic check (no download):

```bash
python music_recommendation.py --demo --seed 0 --query-index 0
```

Real run, after the dataset is extracted as described below:

```bash
python music_recommendation.py --seed 0
```

Useful arguments: `--plays`, `--profiles`, `--nrows` (default 20000000),
`--row-cap` (default 40000), `--neighbors` (default 6), `--country`
(default `United States`), `--query-index`, and `--seed`. Environment variables
`LASTFM_PLAYS` and `LASTFM_PROFILES` override the default paths. An explicit
argument wins over the environment variable. If neither is set, a file under
`data/lastfm-dataset-360K/` is used when it exists. Otherwise the script still
defaults to the 2018 Windows locations:

- `C:/Users/Bhavesh/Pictures/Music_recommendation/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv`
- `C:/Users/Bhavesh/Pictures/Music_recommendation/lastfm-dataset-360K/usersha1-profile.tsv`

The 2018 profiles path was written with a `file:///` prefix. The default is
now the plain path in that same directory, which is how the plays file was
already opened. A `file:///` path is still accepted.

The play file has about 17.6 million rows. The script builds a dense
artist-by-user pivot of the rows that remain after the cap and the country
filter. Memory use of that pivot was not measured here.

## Last.fm 360K dataset

The recommender expects the two tables below. They are absent from this
repository. An earlier README said to request a dataset by email at
`wadibhasme.bhavesh@gmail.com`. That address is the contact the author
published. It is not a download URL, and this repository has no Google Drive
link for the data.

### Source

| Item | URL | Checked |
| --- | --- | --- |
| Dataset page cited in the README | http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html | HTTP 200 on 7 November 2026. Page last modified 24 November 2011. |
| Download linked from that page | http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz | HTTP 403 on 7 November 2026. |
| Working archive | https://zenodo.org/records/6090214 | Record page loaded. File request returned HTTP 200, `content-length` 569202935, filename `lastfm-dataset-360K.tar.gz`. |
| DOI | https://doi.org/10.5281/zenodo.6090214 | Redirects to the Zenodo record. |
| Citation | Òscar Celma. (2010). lastfm Music Recommendation Dataset (Version 1.2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.6090214 | |
| MTG catalog page | https://www.upf.edu/web/mtg/lastfm360k | Not confirmed here; the request received a Cloudflare challenge (HTTP 403). |
| URL in the 2018 script comment | https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html | No HTTP response during this check. |

Zenodo publishes the tarball MD5 as `635e6ed3fc873aa4ba33aba0ebce02b1`. That
checksum was copied from the record. The tarball was not downloaded, so the
checksum was not recomputed and the internal directory layout was not opened.
The dataset description names these files:

| File | Published MD5 | Role |
| --- | --- | --- |
| `usersha1-artmbid-artname-plays.tsv` | `be672526eb7c69495c27ad27803148f1` | `user-mboxsha1`, `musicbrainz-artist-id`, `artist-name`, `plays`. 17,559,530 lines. |
| `usersha1-profile.tsv` | `51159d4edf6a92cb96f87768aa2be678` | `user-mboxsha1`, `gender`, `age`, `country`, `signup`. |
| `mbox_sha1sum.py` | `feb3485eace85f3ba62e324839e6ab39` | Helper shipped with the dataset. This repository does not use it. |

The script reads the first file with no header and the column names
`user`, `musicbrainz-artist-id`, `artist-name`, `plays`. It reads `users` and
`country` from the profile file. Country values in the dataset examples look
like `Mexico` and `United States`. The filter compares with `United States`.

### Rights

Òscar Celma (Music Technology Group, Universitat Pompeu Fabra) collected the
data from the Last.fm API (`user.getTopArtists`) and Last.fm permits
redistribution for non-commercial use. Commercial use of the data or the
Last.fm web services goes through `partners@last.fm`. The dataset page points
at the Last.fm API terms: https://www.last.fm/api/tos (HTTP 200 on 7 November
2026). Work that uses the dataset must reference the Last.fm website
(https://www.last.fm/). An optional citation is chapter 3 of Òscar Celma,
*Music Recommendation and Discovery in the Long Tail* (Springer, 2010).

Do not commit the download. `data/` is gitignored.

### Acquire

```bash
mkdir -p data
curl -L -o data/lastfm-dataset-360K.tar.gz \
  https://zenodo.org/records/6090214/files/lastfm-dataset-360K.tar.gz
# Published MD5: 635e6ed3fc873aa4ba33aba0ebce02b1
md5sum data/lastfm-dataset-360K.tar.gz
tar -xzf data/lastfm-dataset-360K.tar.gz -C data
```

If the tarball extracts the two `.tsv` files into `data/lastfm-dataset-360K/`,
`python music_recommendation.py` finds them. If the directory name differs,
pass `--plays` and `--profiles`.

## Archived loan exercise

`archive/Rupesh_work.zip` is the file previously stored at `Rupesh_work.zip`.
`git mv` moved it. The blob is the same (`ae8de43c88eee1d75455bcf253b321fc5016983c`,
sha256 `b2566e528cc2b4502e31e5c646f0a73a200c0a3f5304a1317c64ea582e44f4e8`).
It was not repacked. Notebooks were not extracted, executed, or reformatted.

| Path inside the zip | Uncompressed size | sha256 |
| --- | --- | --- |
| `Rupesh_work/Machine_learning_model.ipynb` | 311,246 bytes | `bd5b4ba03810acbf0593cc18a9f505d1ff7a92a59cb7c8ee06ddf6b83aa8cdc3` |
| `Rupesh_work/.ipynb_checkpoints/Machine_learning_model-checkpoint.ipynb` | 310,178 bytes | `f7d20c67e596c3210c6502304bcde6def18641738bfc046e421febf87687395b` |
| `Rupesh_work/trainingData.csv` | 4,165,680 bytes | `d7e3b5faaea2058d2dd4ea5cca6574bf95f09b71615eb9795afa8b949c901faf` |
| `Rupesh_work/Exercise (1).pdf` | 39,085 bytes | `f88c7536a900a33431858dd8cf874747370e8b6fc6e6b637f89e26399cd3d3bd` |
| `Rupesh_work/DS_Store (1)` | 6,148 bytes | `d65165279105ca6773180500688df4bdc69a2c7b771752f0a46ef120b7fd8ec3` |

The zip file itself is 971,539 bytes. `trainingData.csv` has a header plus
40,000 data rows. Columns:

`Id`, `city`, `age`, `sex`, `social_class`, `primary_business`,
`secondary_business`, `annual_income`, `monthly_expenses`, `old_dependents`,
`young_dependents`, `home_ownership`, `type_of_house`, `occupants_count`,
`house_area`, `sanitary_availability`, `water_availabity`, `loan_purpose`,
`loan_tenure`, `loan_installments`, `loan_amount`.

The header spells the water column `water_availabity`. Both notebooks call
`pd.read_csv('C:/Users/Bhavesh/Desktop/Rupesh_work/trainingData.csv')`. That
absolute path is outside this repository. Running the notebooks unchanged
requires that CSV at that Windows path. The CSV bytes are already inside the
zip at `Rupesh_work/trainingData.csv`. They were left there so the archive
stays intact. Nothing in `music_recommendation.py` reads this zip or this CSV.

The notebook and the PDF (its metadata names a Word file `Exercise.docx`)
match a widely copied student problem, "Calculating credit worthiness for
rural India": prior rural loan applications, used to discuss how large a loan
to grant. A public write-up with the same column description is
https://github.com/diptiprakashpalai/Loan_Prediction_ML_Project. This
repository does not record a publisher, license, or DOI for `trainingData.csv`.
A cleaned derivative has been indexed as Kaggle dataset
`ikjotsingh221/rural-credit-dataset-cleaned`; a request to
https://www.kaggle.com/datasets/ikjotsingh221/rural-credit-dataset-cleaned
on 7 November 2026 returned HTTP 404, so that page is not a confirmed download.
No Google Drive link for this file exists in the repository.

The notebook was read, not executed. From that reading: it explores the loan
table and defines a small logistic-regression class. One cell that references
`loan_purpose` is stored as markdown, and the frame built just above that cell
does not contain `loan_purpose`. Those notes are static observations. They are
not the result of a run, and the notebook was not changed to address them.

`DS_Store (1)` is macOS folder metadata. It stays in the zip because repacking
would change the archive bytes.

### Migration

- The zip remains in git, now at `archive/Rupesh_work.zip`. History still has
  the blob at `Rupesh_work.zip`. No history rewrite, force-push, or external
  copy was made.
- Delete or rehost that archive only after a coordinator has verified a backup.
- Scripts that looked for `Rupesh_work.zip` at the repository root need the
  new path. The recommender never used that path.
- The notebooks still depend on `C:/Users/Bhavesh/Desktop/Rupesh_work/trainingData.csv`.
  Moving the zip does not satisfy that path.

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

CI runs those three commands on Python 3.12. It does not download Last.fm
data and it does not execute notebooks.

## Known behavior carried over from the 2018 script

- `--row-cap` / `popularity_threshold` keeps the first N play rows. It does
  not keep artists whose `total_artist_plays` are at least 40000.
- Duplicate `(user, artist)` rows are removed. The play counts on the dropped
  copies are not added into the kept row.
- With no `--query-index` and no `--seed`, the query artist is a fresh random
  row of the pivot, so repeated runs differ.
- Artist totals are computed before the country filter, so listens from other
  countries still contribute to `total_artist_plays`. The row cap ignores that
  column anyway.

## Layout

```
music_recommendation.py    recommender
tests/                     synthetic-data tests and archive hash checks
requirements.txt           runtime pins
requirements-dev.txt       runtime pins plus pytest and ruff
pyproject.toml             ruff and pytest settings
.github/workflows/ci.yml
archive/Rupesh_work.zip    unchanged notebook archive and loan CSV
```

There is no installable package and no published distribution.
