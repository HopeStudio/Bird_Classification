import csv
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "db/bird_reference.sqlite"
OUTPUT_PATH = ROOT_DIR / "db/bird_info.csv"


def export_bird_info() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                model_class_id,
                chinese_simplified,
                english_name,
                scientific_name,
                short_description_zh
            FROM BirdCountInfo
            ORDER BY
                model_class_id IS NULL,
                model_class_id ASC
            """
        )

        with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "model_class_id",
                "chinese_simplified",
                "english_name",
                "scientific_name",
                "short_description_zh",
            ])
            writer.writerows(rows)
    finally:
        conn.close()


if __name__ == "__main__":
    export_bird_info()
