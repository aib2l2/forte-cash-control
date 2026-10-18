"""Create a local .env without printing its key or overwriting existing settings."""

import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / ".env"
try:
    with target.open("x", encoding="utf-8") as output:
        target.chmod(0o600)
        output.write(
            (root / ".env.example").read_text(encoding="utf-8").replace(
                "replace-with-a-random-key", secrets.token_urlsafe(32)
            )
        )
except FileExistsError:
    print(".env уже существует; настройки сохранены без изменений.")
else:
    print("Создан .env с локальным API-ключом. Не добавляйте этот файл в Git.")
