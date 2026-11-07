"""The archived notebooks and loan CSV must stay byte-for-byte identical."""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "Rupesh_work.zip"

ZIP_SHA256 = "b2566e528cc2b4502e31e5c646f0a73a200c0a3f5304a1317c64ea582e44f4e8"

MEMBER_SHA256 = {
    "Rupesh_work/.ipynb_checkpoints/Machine_learning_model-checkpoint.ipynb": (
        "f7d20c67e596c3210c6502304bcde6def18641738bfc046e421febf87687395b"
    ),
    "Rupesh_work/DS_Store (1)": (
        "d65165279105ca6773180500688df4bdc69a2c7b771752f0a46ef120b7fd8ec3"
    ),
    "Rupesh_work/Exercise (1).pdf": (
        "f88c7536a900a33431858dd8cf874747370e8b6fc6e6b637f89e26399cd3d3bd"
    ),
    "Rupesh_work/Machine_learning_model.ipynb": (
        "bd5b4ba03810acbf0593cc18a9f505d1ff7a92a59cb7c8ee06ddf6b83aa8cdc3"
    ),
    "Rupesh_work/trainingData.csv": (
        "d7e3b5faaea2058d2dd4ea5cca6574bf95f09b71615eb9795afa8b949c901faf"
    ),
}

NOTEBOOK_PLAYS_PATH = b"C:/Users/Bhavesh/Desktop/Rupesh_work/trainingData.csv"
CSV_HEADER = (
    b"Id,city,age,sex,social_class,primary_business,secondary_business,"
    b"annual_income,monthly_expenses,old_dependents,young_dependents,"
    b"home_ownership,type_of_house,occupants_count,house_area,"
    b"sanitary_availability,water_availabity,loan_purpose,loan_tenure,"
    b"loan_installments,loan_amount"
)


def test_archive_blob_and_members_match_the_uploaded_zip():
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ZIP_SHA256
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        assert names == [
            "Rupesh_work/",
            "Rupesh_work/.ipynb_checkpoints/",
            *MEMBER_SHA256.keys(),
        ]
        for name, expected in MEMBER_SHA256.items():
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == expected
        notebook = archive.read("Rupesh_work/Machine_learning_model.ipynb")
        checkpoint = archive.read(
            "Rupesh_work/.ipynb_checkpoints/Machine_learning_model-checkpoint.ipynb"
        )
        assert NOTEBOOK_PLAYS_PATH in notebook
        assert NOTEBOOK_PLAYS_PATH in checkpoint
        csv_lines = archive.read("Rupesh_work/trainingData.csv").splitlines()
        assert csv_lines[0] == CSV_HEADER
        assert len(csv_lines) == 40_001


def test_notebooks_are_not_extracted_into_the_work_tree():
    notebooks = [path for path in ROOT.rglob("*.ipynb") if ".git" not in path.parts]
    assert notebooks == []
