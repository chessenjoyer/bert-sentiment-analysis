# Sentiment Analysis с BERT

## Описание

Учебный семидневный проект по бинарному анализу тональности англоязычных
отзывов IMDB с использованием DistilBERT. Ноутбук охватывает токенизацию,
эмбеддинги, attention, baseline-классификатор, fine-tuning, сравнение моделей,
анализ ошибок и публикацию FastAPI-демо.

## Структура

- `HW_transformers.ipynb` — полный проект за 7 дней
- `fine_tuned_model/` — обученная модель (создаётся ноутбуком и не хранится в Git)
- `api.py` — FastAPI-приложение
- `error_analysis.txt` — анализ ошибок
- `comparison_results.txt` — сравнение моделей
- `requirements.txt` — зависимости демо

## Запуск FastAPI-демо

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload
```

Откройте документацию API: <http://127.0.0.1:8000/docs>.

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This movie was absolutely fantastic!"}'
```

Приложение сначала ищет `./fine_tuned_model`. Если папки нет, для работающего
демо автоматически используется публичная модель
`distilbert-base-uncased-finetuned-sst-2-english`. Путь можно задать явно:

```bash
set MODEL_PATH=C:\path\to\fine_tuned_model
uvicorn api:app --reload
```

## Результаты

| Модель | F1 macro | Accuracy |
|---|---:|---:|
| Fine-tuned DistilBERT | 0.8100 | 0.8100 |
| Baseline | 0.7947 | 0.7950 |

Улучшение F1 относительно baseline: **1.92%**.

На тестовой выборке из 200 отзывов fine-tuned модель допустила 38 ошибок:
22 false positives и 16 false negatives. Подробности приведены в
`error_analysis.txt`.

## Использование модели в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True)
with torch.inference_mode():
    outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
```

## Требования

- Python 3.9+
- transformers
- torch
- fastapi
- uvicorn

Весовые файлы модели не включены в репозиторий из-за размера. Их можно получить,
выполнив ячейки обучения и сохранения в ноутбуке.
