import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aihub186_category_hours import build_category_hours


def test_build_category_hours_aggregates_by_hierarchy_and_split(tmp_path):
    orgtext_file = tmp_path / "orgtext.tsv"
    wav_info_file = tmp_path / "wav_info.tsv"

    orgtext_file.write_text(
        "file_label\tcategory1\tcategory2\tcategory3\tspeaker_type\torgtext\tdataset_type\n"
        "A1\t대학병원\t병원이용안내\t원무상담\t고객\t안녕하세요\ttrain\n"
        "A2\t대학병원\t병원이용안내\t원무상담\t상담사\t네 말씀하세요\ttrain\n"
        "A3\t대학병원\t병원이용안내\t예약\t고객\t예약할게요\tvalidation\n",
        encoding="utf-8-sig",
    )
    wav_info_file.write_text(
        "file_id\trelative_path\tduration_sec\tdataset_type\n"
        "A1\tdom/A1.wav\t10.0\ttrain\n"
        "A2\tdom/A2.wav\t20.0\ttrain\n"
        "A3\tdom/A3.wav\t5.0\tvalidation\n",
        encoding="utf-8-sig",
    )

    tables = build_category_hours(orgtext_file, wav_info_file)

    cat1 = tables["category1"]
    row = cat1[(cat1["category1"] == "대학병원") & (cat1["dataset_type"] == "train")].iloc[0]
    assert row["count"] == 2
    assert row["sum_sec"] == 30.0
    assert row["sum_hour"] == 30.0 / 3600

    cat123 = tables["category1_2_3"]
    assert set(cat123["category3"]) == {"원무상담", "예약"}
    val_row = cat123[cat123["category3"] == "예약"].iloc[0]
    assert val_row["dataset_type"] == "validation"
    assert val_row["sum_sec"] == 5.0


def test_build_category_hours_drops_duplicate_file_labels(tmp_path):
    orgtext_file = tmp_path / "orgtext.tsv"
    wav_info_file = tmp_path / "wav_info.tsv"

    # 같은 wav가 여러 문장(inputText 항목)으로 나뉘어 orgtext에 중복 행이 있는 상황
    orgtext_file.write_text(
        "file_label\tcategory1\tcategory2\tcategory3\tspeaker_type\torgtext\tdataset_type\n"
        "A1\t대학병원\t병원이용안내\t원무상담\t고객\t문장1\ttrain\n"
        "A1\t대학병원\t병원이용안내\t원무상담\t고객\t문장2\ttrain\n",
        encoding="utf-8-sig",
    )
    wav_info_file.write_text(
        "file_id\trelative_path\tduration_sec\tdataset_type\n"
        "A1\tdom/A1.wav\t10.0\ttrain\n",
        encoding="utf-8-sig",
    )

    tables = build_category_hours(orgtext_file, wav_info_file)
    cat1 = tables["category1"]
    assert cat1.iloc[0]["count"] == 1
