from datetime import datetime, timedelta
from pathlib import Path

import pytest

from model_router.analytics.interfaces import RequestRecord
from model_router.analytics.sqlite_repository import SQLiteRepository


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "analytics.db"


@pytest.fixture
def repo(temp_db: Path) -> SQLiteRepository:
    return SQLiteRepository(temp_db)


def _record(idx: int, when: datetime) -> RequestRecord:
    return RequestRecord(
        id=f"{idx}",
        timestamp=when,
        model="gpt-4" if idx % 2 == 0 else "claude-3",
        cost=0.01 * idx,
        latency=0.2 * idx,
        success=idx % 3 != 0,
    )


def test_save_and_find_by_model(repo: SQLiteRepository):
    now = datetime.utcnow()
    records = [_record(i, now + timedelta(minutes=i)) for i in range(1, 4)]
    for record in records:
        repo.save(record)

    gpt_records = repo.find_by_model("gpt-4")
    assert len(gpt_records) == 1
    assert gpt_records[0].id == "2"


def test_find_by_date_filters_window(repo: SQLiteRepository):
    start = datetime(2024, 1, 1, 12, 0, 0)
    records = [_record(i, start + timedelta(minutes=i * 5)) for i in range(1, 6)]
    for record in records:
        repo.save(record)

    window_start = start + timedelta(minutes=5)
    window_end = start + timedelta(minutes=15)

    results = repo.find_by_date(window_start, window_end)
    ids = [record.id for record in results]

    assert ids == ["1", "2", "3"]
